"""Candidate recall@k of :class:`~sylphae.repo.index.RepoIndex` against real commit history.

This is the measurement that bounds the whole target-selection head. If the ranker cannot
put the file a commit actually changed into the top 8 candidates, no model downstream can
recover it: the target head chooses among candidates, it does not see the repository.

Method
------
For each repository, replay the commit history. For every eligible commit:

1. materialise the tree **at the parent commit** (``git archive`` + in-Python tar
   extraction) -- the agent would be looking at the repository *before* the edit, so
   indexing HEAD would leak every later file into the candidate set;
2. build a fresh :class:`RepoIndex` over that tree, timing it;
3. take the commit subject as the request and the changed file as the target;
4. record whether the target appears in the top 3 / top 8 candidates.

An eligible commit changes exactly one ``.py`` file with status ``M``. Commits that
*add* or *delete* a file are excluded and counted separately: the target cannot be in an
index built from the parent tree, so including them would measure the index's inability to
see the future, not the ranker. Merge commits, root commits and empty subjects are
excluded for the same reason.

Everything is deterministic: the same commit set is sampled the same way, and
``candidates`` has no randomness.

Usage
-----
    python sylphae/repo/bench_recall.py
    python sylphae/repo/bench_recall.py --repos click itsdangerous --max-commits 50
    python sylphae/repo/bench_recall.py --json-out results.json
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # allow `python sylphae/repo/bench_recall.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sylphae.repo.index import RepoIndex, is_test_path  # noqa: E402

DEFAULT_REPOS = ["itsdangerous", "markupsafe", "tabulate", "click", "requests"]
DEFAULT_SCRATCH = Path(__file__).resolve().parents[2] / "work" / "scratch"
K_VALUES = (3, 8)


# --------------------------------------------------------------------------------------
# Commit harvesting
# --------------------------------------------------------------------------------------


@dataclass
class CommitCase:
    repo: str
    sha: str
    parent: str
    subject: str
    target: str

    @property
    def target_is_test(self) -> bool:
        return is_test_path(self.target)


def _git(repo_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo_dir}: {proc.stderr.strip()}")
    return proc.stdout


def collect_commit_cases(repo_dir: Path, name: str) -> tuple[list[CommitCase], dict[str, int]]:
    """All eligible commits, oldest first, plus a tally of what was excluded and why."""
    raw = _git(
        repo_dir,
        "log",
        "--no-merges",
        "--reverse",
        "--format=%x1e%H%x1f%P%x1f%s",
        "--name-status",
    )

    cases: list[CommitCase] = []
    dropped = {
        "root_commit": 0,
        "empty_subject": 0,
        "no_py_change": 0,
        "multiple_py_changes": 0,
        "not_modified": 0,
    }

    for chunk in raw.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = chunk.split("\n")
        sha, _, rest = lines[0].partition("\x1f")
        parents, _, subject = rest.partition("\x1f")
        subject = subject.strip()

        changes: list[tuple[str, str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                changes.append((parts[0], parts[-1]))

        py_changes = [(status, path) for status, path in changes if path.endswith(".py")]

        if not parents.strip():
            dropped["root_commit"] += 1
            continue
        if not subject:
            dropped["empty_subject"] += 1
            continue
        if not py_changes:
            dropped["no_py_change"] += 1
            continue
        if len(py_changes) > 1:
            dropped["multiple_py_changes"] += 1
            continue
        status, path = py_changes[0]
        if not status.startswith("M"):
            dropped["not_modified"] += 1
            continue

        cases.append(
            CommitCase(
                repo=name,
                sha=sha,
                parent=parents.split()[0],
                subject=subject,
                target=path,
            )
        )
    return cases, dropped


def stride_sample(items: list, limit: int | None) -> list:
    """Evenly spaced deterministic sample, preserving order and always including the last."""
    if limit is None or len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[-1]]
    step = (len(items) - 1) / (limit - 1)
    picked = {int(round(i * step)) for i in range(limit)}
    return [items[i] for i in sorted(picked)]


# --------------------------------------------------------------------------------------
# Tree materialisation
# --------------------------------------------------------------------------------------


def materialise(repo_dir: Path, sha: str, dest: Path) -> int:
    """Extract the ``.py`` files of ``sha`` into ``dest``. Returns the file count.

    Only ``.py`` members are extracted: they are the only thing the index reads, and the
    extraction is on the measured path.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "archive", "--format=tar", sha],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {sha} failed: {proc.stderr.decode('utf-8', 'replace')}")
    count = 0
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".py")]
        count = len(members)
        archive.extractall(dest, members=members, filter="data")
    return count


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile. ``None`` for an empty sample, never ``0.0``."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((q / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def evaluate_repo(
    repo_dir: Path, name: str, max_commits: int | None, verbose: bool = True
) -> dict:
    cases, dropped = collect_commit_cases(repo_dir, name)
    sampled = stride_sample(cases, max_commits)

    hits = {k: 0 for k in K_VALUES}
    evaluated = 0
    target_missing = 0
    target_is_test = 0
    shallow_hits = {k: 0 for k in K_VALUES}
    random_expected = {k: 0.0 for k in K_VALUES}
    build_ms: list[float] = []
    materialise_ms: list[float] = []
    cold_ms: list[float] = []
    warm_ms: list[float] = []
    index_sizes: list[int] = []
    misses: list[dict] = []

    for case in sampled:
        work_dir = Path(tempfile.mkdtemp(prefix="sylphae-bench-"))
        try:
            t0 = time.perf_counter()
            materialise(repo_dir, case.parent, work_dir)
            t1 = time.perf_counter()

            index = RepoIndex.build(work_dir)
            t2 = time.perf_counter()

            if case.target not in set(index.files):
                target_missing += 1
                continue

            t3 = time.perf_counter()
            candidates = index.candidates(case.subject, k=max(K_VALUES))
            t4 = time.perf_counter()
            index.candidates(case.subject, k=max(K_VALUES))
            t5 = time.perf_counter()

            paths = [c.path for c in candidates]
            rank = paths.index(case.target) if case.target in paths else None

            evaluated += 1
            materialise_ms.append((t1 - t0) * 1000)
            build_ms.append((t2 - t1) * 1000)
            cold_ms.append((t4 - t3) * 1000)
            warm_ms.append((t5 - t4) * 1000)
            index_sizes.append(len(index))
            if case.target_is_test:
                target_is_test += 1
            for k in K_VALUES:
                if rank is not None and rank < k:
                    hits[k] += 1
                # Reference points, so the headline is interpretable. "Shallowest k" is
                # what the depth prior alone would return; "random k" is the chance floor.
                if case.target in _shallowest(index, k):
                    shallow_hits[k] += 1
                random_expected[k] += min(k, len(index)) / len(index)
            if rank is None or rank >= max(K_VALUES):
                if len(misses) < 25:
                    misses.append(
                        {
                            "sha": case.sha[:10],
                            "subject": case.subject,
                            "target": case.target,
                            "top": paths[:5],
                        }
                    )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    recall = {k: (hits[k] / evaluated if evaluated else None) for k in K_VALUES}
    result = {
        "repo": name,
        "commits_total_eligible": len(cases),
        "commits_sampled": len(sampled),
        "commits_evaluated": evaluated,
        "target_absent_from_parent_index": target_missing,
        "target_is_test_file": target_is_test,
        "excluded": dropped,
        "recall": {f"recall@{k}": recall[k] for k in K_VALUES},
        "hits": {f"hit@{k}": hits[k] for k in K_VALUES},
        "baseline_shallowest": {
            f"recall@{k}": (shallow_hits[k] / evaluated if evaluated else None) for k in K_VALUES
        },
        "baseline_random": {
            f"recall@{k}": (random_expected[k] / evaluated if evaluated else None)
            for k in K_VALUES
        },
        "index_files": _summary([float(n) for n in index_sizes]),
        "materialise_ms": _summary(materialise_ms),
        "index_build_ms": _summary(build_ms),
        "retrieval_cold_ms": _summary(cold_ms),
        "retrieval_warm_ms": _summary(warm_ms),
        "example_misses": misses,
    }
    if verbose:
        r3 = result["recall"]["recall@3"]
        r8 = result["recall"]["recall@8"]
        print(
            f"  {name:14s} n={evaluated:5d}  "
            f"recall@3={r3:.3f}  recall@8={r8:.3f}  "
            f"(shallowest8={result['baseline_shallowest']['recall@8']:.3f}  "
            f"random8={result['baseline_random']['recall@8']:.3f})  "
            f"files~{result['index_files']['mean']:.0f}  "
            f"build={result['index_build_ms']['mean']:.1f}ms  "
            f"retr={result['retrieval_cold_ms']['mean']:.2f}ms",
            flush=True,
        )
    return result


def _shallowest(index: RepoIndex, k: int) -> set[str]:
    """The ``k`` files with the fewest path components -- the depth prior, nothing else."""
    return {
        r.path
        for r in sorted(index.records, key=lambda r: (r.path.count("/"), r.path))[:k]
    }


def aggregate(results: list[dict]) -> dict:
    total = sum(r["commits_evaluated"] for r in results)
    agg: dict = {"commits_evaluated": total, "repos": len(results)}
    for k in K_VALUES:
        # Micro: pooled over every commit. Macro: unweighted mean of per-repository
        # recalls, so one large repository cannot define the headline on its own.
        agg[f"recall@{k}"] = (
            sum(r["hits"][f"hit@{k}"] for r in results) / total if total else None
        )
        scored = [r["recall"][f"recall@{k}"] for r in results if r["commits_evaluated"]]
        agg[f"macro_recall@{k}"] = sum(scored) / len(scored) if scored else None
        for base in ("baseline_shallowest", "baseline_random"):
            vals = [
                r[base][f"recall@{k}"]
                for r in results
                if r["commits_evaluated"] and r[base][f"recall@{k}"] is not None
            ]
            weights = [r["commits_evaluated"] for r in results if r["commits_evaluated"]]
            if vals:
                agg[f"{base}@{k}"] = (
                    sum(v * w for v, w in zip(vals, weights)) / sum(weights)
                    if sum(weights)
                    else None
                )
            else:
                agg[f"{base}@{k}"] = None
    all_build = [
        r["index_build_ms"][key]
        for r in results
        for key in ("mean",)
        if r["index_build_ms"]["mean"] is not None
    ]
    agg["index_build_ms_mean_of_means"] = sum(all_build) / len(all_build) if all_build else None
    # Weighted by commits so a big repository is not drowned out by a tiny one.
    for field in ("index_build_ms", "retrieval_cold_ms", "retrieval_warm_ms", "materialise_ms"):
        weighted = [
            (r[field]["mean"], r["commits_evaluated"])
            for r in results
            if r[field]["mean"] is not None and r["commits_evaluated"]
        ]
        if weighted:
            num = sum(m * n for m, n in weighted)
            den = sum(n for _m, n in weighted)
            agg[f"{field}_weighted_mean"] = num / den
        else:
            agg[f"{field}_weighted_mean"] = None
    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    parser.add_argument(
        "--max-commits",
        type=int,
        default=200,
        help="per repository, evenly sampled across history (0 or negative = all)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    limit = args.max_commits if args.max_commits and args.max_commits > 0 else None

    print("Sylphae candidate recall benchmark")
    print(f"  scratch: {args.scratch}")
    print(f"  commit sampling: {'all eligible' if limit is None else f'{limit} per repo, even stride'}")
    print()

    results: list[dict] = []
    for name in args.repos:
        repo_dir = args.scratch / name
        if not (repo_dir / ".git").exists():
            print(f"  {name:14s} SKIPPED (no git repository at {repo_dir})", flush=True)
            continue
        try:
            results.append(evaluate_repo(repo_dir, name, limit))
        except Exception as exc:  # noqa: BLE001 - one bad repo must not kill the run
            print(f"  {name:14s} FAILED: {type(exc).__name__}: {exc}", flush=True)

    if not results:
        print("\nNo repositories evaluated.")
        return 1

    agg = aggregate(results)

    print()
    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    print(f"  repositories evaluated : {agg['repos']}")
    print(f"  commits evaluated      : {agg['commits_evaluated']}")
    print(f"  recall@3 (micro)       : {agg['recall@3']:.4f}")
    print(f"  recall@8 (micro)       : {agg['recall@8']:.4f}")
    print(f"  recall@3 (macro)       : {agg['macro_recall@3']:.4f}")
    print(f"  recall@8 (macro)       : {agg['macro_recall@8']:.4f}")
    print(f"  index build (mean)     : {agg['index_build_ms_weighted_mean']:.2f} ms")
    print(f"  retrieval cold (mean)  : {agg['retrieval_cold_ms_weighted_mean']:.3f} ms")
    print(f"  retrieval warm (mean)  : {agg['retrieval_warm_ms_weighted_mean']:.3f} ms")
    print(f"  tree materialise (mean): {agg['materialise_ms_weighted_mean']:.1f} ms")
    print()
    print("  per-repository p95:")
    for r in results:
        print(
            f"    {r['repo']:14s} build p95={r['index_build_ms']['p95']:7.2f}ms  "
            f"retrieval p95={r['retrieval_cold_ms']['p95']:6.3f}ms  "
            f"files p50={r['index_files']['p50']:.0f} "
            f"p95={r['index_files']['p95']:.0f}"
        )
    print()
    print("  reference points on the same commits:")
    print(f"    always return the {max(K_VALUES)} shallowest files : "
          f"recall@3={agg['baseline_shallowest@3']:.4f}  recall@8={agg['baseline_shallowest@8']:.4f}")
    print(f"    return {max(K_VALUES)} files at random          : "
          f"recall@3={agg['baseline_random@3']:.4f}  recall@8={agg['baseline_random@8']:.4f}")
    print("  INTERFACES.md section 3 asks for >= 0.95 target presence in the candidate set.")

    payload = {"per_repo": results, "overall": agg, "config": {
        "max_commits": limit, "repos": [r["repo"] for r in results], "k_values": list(K_VALUES),
    }}
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
