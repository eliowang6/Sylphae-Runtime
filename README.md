# Sylphae Runtime

**A local, millisecond-latency reflex layer for code edits.**

Sylphae finds the file you meant, makes the edit you asked for, checks that it
still compiles, and tells you what it did. It runs entirely on your machine, and
**it needs no model weights to do any of that.**

```console
$ sylphae --request "change .navbar height from 200px to 300px" --repo ./my-site --apply
--- Reflex Result ---
Target: styles/main.css
Operation: replace_literal
Confidence: 0.90
Immediate Patch Applied: True
Verification: compiles=True, tests_passed=0/0
```

29 ms end to end on the example below, including retrieval, the edit, and
verification.

---

## What this is

Most of what a coding agent does in a session is mechanical: change this value,
rename this symbol, add this argument, bump this default. A frontier LLM can do
that, but it pays full price for a general-purpose model on every round, and it
cannot show its work.

Sylphae is the reflex layer for that work. It is designed to sit **under** a
planner LLM in a two-system loop:

```
user request
  -> planner LLM: what to find            (System 2, seconds, real money)
  -> Sylphae:    retrieve + analyse       (System 1, ~30 ms, local)
  -> planner LLM: what to do
  -> Sylphae:    locate -> edit -> compile -> test -> receipt
  -> planner LLM: accept from the receipt
```

You can also just run the CLI yourself. It does not require the loop.

### The one idea

**The output is a structured edit, not generated text.**

An LLM asked to "fix this" writes text, and text can be subtly wrong — a
transposed argument, a lost edge case, a plausible-looking diff that does not
compile. Sylphae selects from a fixed vocabulary of 26 edit operations
(§[The vocabulary](#the-vocabulary)), each of which is *detectable from a diff*
and *deterministically executable*. When it says it changed `height: 200px` to
`height: 300px`, that is literally the transformation that was applied, and it is
verified to round-trip before it is reported.

The vocabulary is closed, and that is the whole trick. It means the edit either
applies exactly or the run reports that it could not. There is no middle state
where a patch looks plausible.

### Weights are optional

The neural candidate ranker — a small Mamba-2 model that reranks files by
relevance — is **off by default and not published here.** There are no
checkpoint files in this repository, and you do not need any to use any of the
above.

That is a deliberate design choice, not a limitation of the packaging:

- The lexical and AST retrieval tools are deterministic, need no training data,
  and are what actually carry retrieval in the measured configuration.
- A component you cannot inspect the weights of is a component you have to trust.
  Not shipping one means everything here is auditable.
- It means the runtime cannot quietly degrade. With no weights present, the
  neural path is simply not taken.

If you do train one, the runtime uses it. See
[§Using your own weights](#using-your-own-weights).

---

## Install

```console
$ pip install sylphae-runtime
```

Python 3.11+. For source:

```console
$ git clone https://github.com/eliowang6/Sylphae-Runtime
$ cd Sylphae-Runtime && pip install -e ".[dev]"
```

### Optional extras

| Extra | Why |
|---|---|
| `trash` | `send2trash`. Without it, a "delete" is **refused** rather than falling back to `unlink` — see [§Safety](#safety). |
| `dev` | pytest, pytest-asyncio, ruff. |

### External tools

These are used if present, and the corresponding capability degrades or is
unavailable without them. Nothing here silently pretends to have worked.

| Tool | Used for | Without it |
|---|---|---|
| `git` | commit-history search during retrieval | that one retrieval tool is skipped; AST and lexical search still run |
| [`ripgrep`](https://github.com/BurntSushi/ripgrep) | content search across the tree | falls back to a slower in-process scan |
| `pytest` | running the affected tests | verification reports `pytest-unavailable`; compile checking still runs |
| `ruff` | format/check, for the playbooks that offer them | those playbooks are unavailable |

`pip install ripgrep` if you want the fast path. The runtime looks for `rg` on
`PATH` and in a few well-known editor bundle locations.

---

## Quickstart

### Probe a codebase

The retrieval half on its own, for feeding a planner LLM. Sub-second budget.

```console
$ sylphae --probe-context "make the navbar taller" --repo ./my-site
=== Codebase Context Probe (30.9ms) ===
Frameworks: standard
Suggested Modification Targets: styles/main.css
Relevant Code Snippets (Ranked by Importance):
  1. [P=0.92] (Base Style) styles/main.css:2-2
     .navbar { height: 200px; background: #f5f5f5; }
```

Add `--json` for machine-readable output. This is the intended integration
point: a planner LLM consumes the bundle and returns a concrete instruction.

### Make an edit

```console
$ sylphae --request "change .navbar height from 200px to 300px" --repo ./my-site --json
```

```json
{
  "target_path": "styles/main.css",
  "op_kind": "replace_literal",
  "confidence": 0.9,
  "spec": {
    "kind": "replace_literal",
    "path": "styles/main.css",
    "span": [2, 2],
    "props": { "old": "200", "new": "300" }
  },
  "escalated": false,
  "model_output_used": null,
  "explore_ms": 24.98, "patch_ms": 4.15, "reflex_ms": 29.13
}
```

Dry run by default. `--apply` writes to disk. `--effort high` spends more
exploration budget before committing to a target.

`model_output_used: null` is the zero-weight path, stated explicitly rather than
implied.

**When it is not sure, it escalates.** `confidence` and `escalated` are in the
output, the exit code is non-zero, and with `--apply` a patch that escalated is
*not* written — the diff is still shown, but the disk write waits for you. An
escalation you cannot act on is just a log line.

### Execute a plan

For a multi-step plan written by a planner:

```console
$ sylphae --execute-plan plan.json --repo ./my-site --json
$ echo "$PLAN_JSON" | sylphae --execute-plan - --repo ./my-site
```

### Start a service

```console
$ sylphae --start-service backend --repo ./my-api --apply
```

Autonomously detects the framework, starts it, and health-checks it.

---

## As a library

The CLI is a thin wrapper. The interesting entry point is the agent.

```python
import asyncio
from pathlib import Path
from sylphae.agent.reflex import ReflexAgent

agent = ReflexAgent(root=Path("./my-site"))

result = asyncio.run(agent.run(
    prompt="change .navbar height from 200px to 300px",
    dry_run=True,                      # False to write
))

print(result.target_path, result.op_kind, result.confidence)
if result.spec:
    print(result.spec.to_dict())
if result.escalated:
    print(result.escalation_reason)
```

### The op vocabulary

The one thing worth reading before building on this. Each op is detectable from
a `(before, after)` pair and applies back exactly.

```python
from sylphae.ops import detect_op, apply_op

spec = detect_op(old_source, new_source, "src/app/handlers.py")
if spec is not None and spec.kind != "freeform_patch":
    assert apply_op(old_source, spec) == new_source   # guaranteed
```

26 kinds: `replace_literal`, `rename_symbol`, `change_condition`,
`modify_call_args`, `change_default`, `add_import`, `remove_import`,
`add_guard`, `insert_statement`, `delete_statement`, `add_decorator`,
`change_return`, `modify_string`, `wrap_try_except`, `freeform_patch`,
`replace_line`, `add_call_arg`, `remove_call_arg`, `add_dict_entry`,
`remove_dict_entry`, `wrap_block`, `modify_string_multi`, `reindent`,
`add_list_element`, `remove_list_element`.

Ids 0–15 are frozen; later ones are appended. **Do not reorder them** — a corpus
or a checkpoint indexes into them, and reordering is a breaking change.

`freeform_patch` is the escape hatch and the honest failure mode: when the diff
is not one of the 25 structured forms, that is what you get, and it is counted
as coverage loss rather than quietly accepted.

`ops.polyglot` extends the same model to CSS/SCSS, JS/TS/TSX/JSX, Vue, HTML, Go,
Rust, SQL, JSON and YAML. No Node toolchain is invoked; it is text and
line-oriented structural work.

### Repository state

```python
from pathlib import Path
from sylphae.repo.index import RepoIndex

index = RepoIndex.build(Path("./my-site"))
for cand in index.candidates("where is the retry limit set", k=8):
    print(cand.path, cand.score, cand.snippet)
```

`build` walks the tree and records symbols for Python via `ast` and for other
source via a regex extractor. It never raises on a malformed file: the file is
recorded with an empty symbol list and a parse-error reason, so a syntax error in
one file cannot hide the other 400 from the index.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
   request ────────▶│  agent/         exploration, tools,  │
                    │                 the reflex policy    │
                    └───────┬──────────────────┬───────────┘
                            │                  │
              ┌─────────────▼──────┐   ┌───────▼───────────────┐
              │ repo/   world state│   │ model/  optional      │
              │   inventory, AST,  │   │   neural reranker     │
              │   retrieval        │   │   (no weights by def.)│
              └─────────────┬──────┘   └───────────────────────┘
                            │
                    ┌───────▼──────────┐    ┌──────────────────────┐
                    │ ops/  26 edit    │───▶│ exec/  apply + verify│
                    │ forms, the      │    │   compile, tests,    │
                    │ vocabulary      │    │   diagnose, self-heal│
                    └──────────────────┘    └──────────────────────┘
```

| Module | Responsibility |
|---|---|
| `ops/` | The edit vocabulary. `detect_op` / `apply_op` and the round-trip invariant. `params.py` defines what value a parameterised op emits. |
| `repo/` | World state: file inventory, AST symbol index, candidate retrieval, ignore rules. |
| `agent/` | The reflex agent: multi-round exploration with a sufficiency gate, the retrieval tools, the CLI. |
| `exec/` | Applying a structured patch to a tree; compile and test verification; failure diagnosis and bounded self-healing. |
| `coprocessor/` | The `ActPlan` protocol: context probing, parallel plan execution, snippet ranking, service launching. |
| `ops_env/` | The sandboxed command surface — playbooks, a safety gate, and trash-instead-of-delete. |
| `vision/` | Screenshot and defect inspection for image input. |
| `model/` | Pure-PyTorch Mamba-2 SSD backbone, the op/param/target heads, the tokenizer, the emission decoder. All optional at runtime. |

`INTERFACES.md` has the frozen module contracts. The docstrings reference it by
section number.

### Two invariants worth knowing

**1. Round-trip.** If `detect_op` returns a spec whose kind is not
`freeform_patch`, then `apply_op(old, spec) == new` — character for character,
including line endings. This is not aspirational; the test suite pins it.

**2. Act first, verify after.** A patch is applied optimistically, then verified.
*Mechanical* failures (syntax, import, indentation, typo) are diagnosed and fixed
locally, up to a bounded number of retries. *Semantic* failures (wrong logic,
failing assertions) escalate immediately. Only mechanical fixes are ever applied
automatically, and the tests are never edited. A semantic fix leaking into the
mechanical loop produces patches that compile and are wrong, which is worse than
no patch at all.

### Safety

- **Dry run is the default.** Writes require `--apply`.
- **Escalated patches are not written.** The diff is reported; the disk write waits.
- **Deleting goes to the trash, not to `unlink`.** If `send2trash` is missing, the
  delete is refused. There is no fallback to permanent deletion.
- **Tests are ground truth** and are never modified by the self-healing loop.
- **Scope guard.** Before writing, the agent checks that the candidate is a source
  file in the tree and that the span is in range. Out-of-range or non-source
  targets are refused.
- **Tool artifacts are never edited.** Files carrying the `.sylphae` prefix — the
  generated index cache — are excluded from candidate selection, so the tool
  cannot propose to edit its own state.

---

## Using your own weights

If you have trained a ranker, point the runtime at the run directory. It needs
`config.json` and `checkpoint.safetensors`, and a `tokenizer.json` next to them.

```console
$ export SYLPHAE_RUN_DIR=/path/to/your/run
$ sylphae --request "..." --repo . --run-dir /path/to/your/run
```

`$SYLPHAE_RUN_DIR` is a default, not a requirement — every entry point takes an
explicit `run_dir` too. If the path is missing or the checkpoint does not match
the model, the runtime reports itself unavailable and the lexical tools carry
retrieval. It never falls back to a partially-loaded model.

There is a strictness rule worth knowing about: `sylphae.model.io.load_weights`
refuses a checkpoint whose parameters do not match the model exactly, and names
the difference. "The checkpoint loaded but half of it was random" is not a
failure anyone notices downstream, so it is not allowed to happen.

To build a tokenizer with no downloads, `train_tokenizer` over the local standard
library is enough:

```python
from sylphae.model.tokenizer import stdlib_corpus_paths, train_tokenizer

tok = train_tokenizer(stdlib_corpus_paths(max_files=200), vocab_size=32768,
                      out_path="tokenizer.json")
```

---

## What is not here

This repository is the **inference half** of Sylphae. Deliberately excluded:

| Not here | Why |
|---|---|
| Corpus synthesis and harvesting | Produces training data. No dataset ships. |
| Training loops, losses, optimiser state | Produces the optional weights above. |
| The evaluation harness and benchmarks | Measurement, not product. |
| Model weights and tokenizers | See [§Weights are optional](#weights-are-optional). |

This is a boundary, not a convenience, and it is tested:
`tests/test_runtime_split.py` reads the import graph and fails if any shipped
module imports the data, training, or evaluation layer. The shared definitions
that made the split necessary — the parameter-label predicates and the
checkpoint reader — live in `sylphae/ops/params.py` and `sylphae/model/io.py`,
which are themselves shipped, so neither side has to own them.

Some docstrings refer to modules that live in the parent project (`sylphae.data`,
`sylphae.train`, `sylphae.eval`) when explaining why a definition lives where it
does. Those references are history and are not importable here.

---

## Status and honest limits

This is early. The parts that work are the parts you can test yourself.

**Measured and reproducible.** The op vocabulary round-trips exactly across the
test corpus. The zero-weight path above is what the numbers in this README are
from, measured on a two-file fixture; sub-100 ms retrieval and a full reflex
round in tens of milliseconds are the design budget, and the timing fields are in
every JSON response so you can check it on your own hardware.

**Not claimed.** That retrieval is solved. A request naming a *concept* rather
than an identifier — a concept word in one language against an English path — is
the known weak spot of lexical scoring, and that is precisely the gap the neural
ranker exists to close. Quality depends on having the right tools on `PATH`
(§[External tools](#external-tools)).

**What to expect.** A strong fit for mechanical, single-location edits: literal
and default changes, renames, argument and entry add/remove, indentation,
style tweaks, JSX/CSS class manipulation. A poor fit for new files, architectural
work, or anything requiring you to read the code and reason about it — those
escalate, and that is the intended behaviour rather than a bug.

The design rationale, the measurements, and what is currently failing are in the
parent project's `DESIGN.md` and `ROADMAP.md`.

---

## Development

```console
$ pip install -e ".[dev]"
$ pytest                      # 962 passed, 8 skipped
$ ruff check .
```

The 8 skips are the tests that need a trained tokenizer artifact, which is not
committed. They are the only tests that require weights to run at all.

Tests are invariants, not coverage. Every module is expected to ship at least one
negative case — a test that only ever prints PASS is worthless, and the ones here
are mostly about what the runtime *refuses* to do.

Contributions welcome. The closed op vocabulary is the load-bearing design
decision: adding an op is welcome, and note that ids are append-only.

## License

Not yet chosen. Add a `LICENSE` file before distributing.

## Acknowledgements

Built on [ripgrep](https://github.com/BurntSushi/ripgrep),
[PyTorch](https://pytorch.org/), and
[Hugging Face tokenizers](https://github.com/huggingface/tokenizers).
