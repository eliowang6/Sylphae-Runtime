"""CLI entry point for Sylphae System-1 Reflex Agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sylphae.agent.effort import HighEffort, LowEffort


def main() -> int:
    ap = argparse.ArgumentParser(description="Sylphae: Universal Edge Coding Reflex & Act Coprocessor Agent")
    ap.add_argument("--request", "-r", default=None, help="Natural language request (e.g. '把导航栏的高度缩短一些')")
    ap.add_argument("--image", "-i", default=None, help="Optional UI screenshot / bug image path")
    ap.add_argument(
        "--probe-context",
        "-p",
        default=None,
        help=(
            "Probe codebase context for Frontier LLM. The <150ms budget covers the "
            "probe itself, not this process's start-up or imports."
        ),
    )
    ap.add_argument("--execute-plan", default=None, help="Path to ActPlan JSON file for parallel execution")
    ap.add_argument("--start-service", default=None, help="Autonomously launch and verify service (e.g. 'backend')")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
    ap.add_argument("--repo", default=".", help="Repository root path")
    ap.add_argument("--run-dir", default=None, help="Optional checkpoint run directory")
    ap.add_argument("--effort", default="low", choices=("low", "high"), help="Effort policy")
    ap.add_argument("--apply", action="store_true", help="Apply patch to disk (default is dry-run)")
    args = ap.parse_args()

    root = Path(args.repo).resolve()

    # Mode 1: Coprocessor Codebase Context Probing
    if args.probe_context:
        from sylphae.coprocessor.collector import collect_codebase_context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            report = loop.run_until_complete(collect_codebase_context(root, args.probe_context))
        finally:
            loop.close()

        if args.json:
            import json
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.to_prompt_context())
        return 0

    # Mode 2: Direct Service Launch
    if args.start_service:
        from sylphae.coprocessor.service_runner import launch_and_verify_service
        import json
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            srv_rep = loop.run_until_complete(launch_and_verify_service(root, target_hint=args.start_service))
        finally:
            loop.close()

        if args.json:
            print(json.dumps(srv_rep.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"\n--- Service Runner: {srv_rep.service_name} ---")
            print(f"Status: {srv_rep.status.upper()}")
            if srv_rep.url:
                print(f"URL: {srv_rep.url} (PID: {srv_rep.pid})")
            if srv_rep.runtime_path:
                print(f"Runtime: {srv_rep.runtime_path}")
            if srv_rep.working_dir:
                print(f"Working Dir: {srv_rep.working_dir}")
            print(f"Startup Latency: {srv_rep.latency_ms:.1f}ms")
            if srv_rep.log_file:
                print(f"Log File: {srv_rep.log_file}")
            if srv_rep.error_summary:
                print(f"Error / Diagnosis:\n{srv_rep.error_summary}")
        return 0 if srv_rep.status in ("running", "already_running") else 1

    # Mode 3: Coprocessor Parallel Plan Execution
    if args.execute_plan:
        from sylphae.coprocessor.executor import execute_act_plan
        from sylphae.coprocessor.protocol import parse_act_plan
        import json

        plan_raw = args.execute_plan
        if plan_raw == "-":
            plan_input = sys.stdin.read().strip()
        else:
            plan_path = Path(plan_raw)
            if plan_path.is_file():
                plan_input = plan_path.read_text(encoding="utf-8", errors="replace").strip()
            else:
                plan_input = plan_raw

        try:
            plan = parse_act_plan(plan_input, root=root)
        except Exception as exc:
            print(f"Error: Failed to parse plan: {exc}", file=sys.stderr)
            return 1

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            exec_rep = loop.run_until_complete(execute_act_plan(root, plan))
        finally:
            loop.close()

        if args.json:
            print(json.dumps(exec_rep.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"\n--- Act Plan Execution: {plan.plan_id} ---")
            print(f"Macro Intent: {plan.macro_intent}")
            print(f"All Passed: {exec_rep.all_passed} ({exec_rep.total_latency_ms:.1f}ms)")
            for sr in exec_rep.step_results:
                status = "OK" if (sr.applied and sr.syntax_ok) else "FAILED"
                print(f"  [{status}] Step {sr.step_id} ({sr.op_kind}) -> {sr.target_path}")
                if sr.diff_summary:
                    print(f"       Diff: {sr.diff_summary}")
                if sr.error:
                    print(f"       Error: {sr.error}")
        return 0 if exec_rep.all_passed else 1

    # Mode 4: Autonomous Reflex Execution / NL Routing
    if not args.request:
        ap.print_help()
        return 1

    # Check if natural language request is an operational service command
    from sylphae.coprocessor.grounding import ground_natural_language_sync
    grounded_steps = ground_natural_language_sync(args.request, root=root)
    if any(s.action == "start_service" for s in grounded_steps):
        # Service launch must respect --apply gating
        if not args.apply:
            if args.json:
                import json
                print(json.dumps({
                    "plan_id": "nl_service_plan",
                    "macro_intent": args.request,
                    "dry_run": True,
                    "all_passed": True,
                    "message": "Service launch skipped in dry-run mode. Pass --apply to start background daemon.",
                    "steps": [s.to_dict() for s in grounded_steps],
                }, indent=2, ensure_ascii=False))
            else:
                print("\n--- Service Execution Plan (Dry-Run) ---")
                print(f"Intent: {args.request}")
                print("[dry-run] Service launch skipped. Pass --apply to start background daemon.")
                for s in grounded_steps:
                    print(f"  Plan: {s.action} -> target: {s.target_hint or 'default'}")
            return 0

        from sylphae.coprocessor.executor import execute_act_plan
        from sylphae.coprocessor.protocol import ActPlan
        plan = ActPlan(plan_id="nl_service_plan", macro_intent=args.request, steps=grounded_steps)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            exec_rep = loop.run_until_complete(execute_act_plan(root, plan))
        finally:
            loop.close()

        if args.json:
            import json
            print(json.dumps(exec_rep.to_dict(), indent=2, ensure_ascii=False))
        else:
            print("\n--- Service Execution Report ---")
            print(f"Intent: {args.request}")
            print(f"Success: {exec_rep.all_passed} ({exec_rep.total_latency_ms:.1f}ms)")
            for sr in exec_rep.step_results:
                status = "OK" if (sr.applied and sr.syntax_ok) else "FAILED"
                print(f"  [{status}] Step {sr.step_id} ({sr.op_kind}) -> {sr.target_path}")
                if sr.diff_summary:
                    print(f"       Details: {sr.diff_summary}")
                if sr.error:
                    print(f"       Error: {sr.error}")
        return 0 if exec_rep.all_passed else 1

    from sylphae.agent.reflex import ReflexAgent
    effort = HighEffort if args.effort == "high" else LowEffort

    model = None
    tokenizer = None
    if args.run_dir:
        run_path = Path(args.run_dir)
        import json
        from sylphae.model.config import ModelConfig
        from sylphae.model.io import load_weights
        from sylphae.model.policy import SylphaeModel
        from sylphae.model.tokenizer import CodeTokenizer
        from sylphae.ops.vocab import N_OPS
        cfgj = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
        d_ff = cfgj.get("d_ff") or cfgj.get("d_model", 384)
        mcfg = ModelConfig(
            d_model=cfgj.get("d_model", 384),
            d_ff=d_ff,
            n_layers=cfgj.get("n_layers", 6),
            d_state=cfgj.get("d_state", 128),
            n_heads=cfgj.get("n_heads", 16),
            chunk_size=cfgj.get("chunk_size", 64),
            vocab_size=cfgj.get("vocab_size", 32768),
            max_len=cfgj.get("max_len", 1024),
            max_params=cfgj.get("max_params", 64),
            n_ops=int(N_OPS),
        )
        model = SylphaeModel(mcfg)
        load_weights(run_path / "checkpoint.safetensors", model=model)
        tokenizer = CodeTokenizer.load("sylphae/model/artifacts/tokenizer.json")

    agent = ReflexAgent(root=root, model=model, tokenizer=tokenizer, effort=effort)

    def _progress(message: str) -> None:
        """Progress narration goes to stderr, never to stdout.

        ``--json`` consumers parse stdout and nothing else.  This narration used to
        be printed unconditionally, so ``-r ... --json`` emitted a mix of prose and
        fields that no parser could read, while still exiting 0 — the caller had no
        way to tell a successful run from an unreadable one.
        """
        print(message, file=sys.stderr)

    def _on_intermediate(cands):
        if cands:
            best = cands[0]
            _progress(f"[explore] current top: {best.path} (score={best.score:.3f}, tools={sorted(best.contributing_tools)})")

    _progress(f"\n[Sylphae] User Request: {args.request}")
    if args.image:
        _progress(f"[Sylphae] Image Input: {args.image}")
    _progress(f"[Sylphae] Effort: {effort.name} | Repo: {root.name}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        res = loop.run_until_complete(
            agent.run(
                prompt=args.request,
                image_path=args.image,
                dry_run=not args.apply,
                on_intermediate_candidate=_on_intermediate,
                verify_in_background=False,  # Run synchronously in CLI to report
            )
        )
    finally:
        loop.close()

    if args.json:
        import json
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("\n--- Reflex Result ---")
        print(f"Target: {res.target_path}")
        print(f"Operation: {res.op_kind}")
        print(f"Confidence: {res.confidence:.2f}")
        if args.apply:
            print(f"Immediate Patch Applied: {res.immediate_patch_ok}")
        elif res.spec is None:
            # Saying "Valid" here would claim a patch exists when none does; that is
            # the state a caller most needs to notice.
            print("Patch Status: None (no patch could be constructed from this request)")
        else:
            print("Patch Status: Valid (dry-run: not applied to disk)")
        if res.spec:
            print(f"Patch Spec: {res.spec.to_dict()}")
        if res.escalated:
            print(f"[ESCALATED] Reason: {res.escalation_reason}")
        if res.verification_report:
            vr = res.verification_report
            print(f"Verification: compiles={vr.compiles}, tests_passed={vr.tests_passed}/{vr.tests_run}")
            if vr.errors:
                print(f"Verification Errors: {vr.errors[:2]}")

    # Determine accurate exit code
    if not res.immediate_patch_ok:
        return 1
    if res.escalated:
        return 1
    if res.verification_report and not res.verification_report.compiles:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
