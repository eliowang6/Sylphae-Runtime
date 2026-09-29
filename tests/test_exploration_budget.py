"""The exploration budget has to be real, and a truncated search has to say so.

Two defects made ``ParallelExplorer``'s budget advisory rather than binding, and both
were invisible because the *result* still looked like a normal one:

1. ``ripgrep_search`` spawned its subprocess with ``asyncio.create_subprocess_exec``.
   MEASURED on this machine, spawning ``rg.exe`` blocked the event loop for 441 ms while
   ripgrep itself ran in 33 ms.  Because that block happened inside an explorer task, the
   other tools could not be scheduled until it returned -- the "parallel" explorer was
   serial -- and the budget could not fire, because the timer lives on the blocked loop.
   A 0.5 s budget was observed letting a run reach 1721 ms.

2. ``stream_explore``'s per-task ``except Exception: continue`` swallowed the
   ``TimeoutError`` that ``asyncio.as_completed`` raises on expiry (on Python 3.13 the
   timeout is raised from the awaited result, and ``asyncio.TimeoutError`` *is* the
   builtin ``TimeoutError``).  So even when the budget did expire, nothing recorded it:
   3 of 8 runs silently dropped ripgrep's matches while reporting
   ``last_exploration_truncated = False``.

The tests below are written against the *cause*, not the symptom: each one fails if the
loop is blocked or the timeout is swallowed, regardless of how fast the machine is.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path

import pytest

from sylphae.agent import explorer as explorer_mod
from sylphae.agent import tools as tools_mod
from sylphae.agent.effort import LowEffort
from sylphae.agent.explorer import ParallelExplorer
from sylphae.agent.reflex import ReflexAgent
from sylphae.agent.tools import get_ripgrep_binary, ripgrep_search


def _tiny_repo(root: Path) -> Path:
    (root / "a.py").write_text("MAX_WIDTH = 80\n", encoding="utf-8")
    return root


def _low_budget(seconds: float):
    return dataclasses.replace(LowEffort, timeout_seconds=seconds)


def _slow_ripgrep(delay: float, state: dict[str, bool] | None = None):
    """A stand-in for ripgrep that takes ``delay`` seconds, optionally switchable."""

    async def _search(*_args, **_kwargs):
        if state is None or state.get("slow", True):
            await asyncio.sleep(delay)
        return []

    return _search


@pytest.mark.asyncio
async def test_expired_budget_is_recorded(monkeypatch, tmp_path):
    """A tool that outlives the budget must leave a trace, not just vanish.

    Regression for defect 2: with the timeout swallowed by the per-task handler this
    reported ``False`` and an empty ``pending`` list.
    """
    _tiny_repo(tmp_path)
    monkeypatch.setattr(explorer_mod, "ripgrep_search", _slow_ripgrep(2.0))

    ex = ParallelExplorer(root=tmp_path, effort=_low_budget(0.2))
    await ex.explore("把最大宽度调整为120")

    assert ex.last_exploration_truncated is True
    assert "ripgrep" in ex.last_pending_tools


@pytest.mark.asyncio
async def test_finished_exploration_reports_no_truncation(tmp_path):
    """The complement: a search that finishes must not claim it was cut short.

    The budget here is deliberately generous, and that is the point of the test rather
    than a convenience.  This asserts a *reporting* invariant -- "the flag is False when
    nothing was pending" -- and that claim is only testable when the search is guaranteed
    to finish.  ``LowEffort``'s 0.5 s does not guarantee it on this machine: MEASURED,
    a single ``ripgrep_search`` against a **one-file** directory took 218-759 ms
    (20-run explore distribution 232-310 ms, but the isolated spawn has a heavy tail),
    so at 0.5 s this test failed for a reason that has nothing to do with the flag.  A
    test that goes red when the machine is busy reports a defect it cannot name.
    """
    _tiny_repo(tmp_path)
    ex = ParallelExplorer(root=tmp_path, effort=_low_budget(5.0))
    await ex.explore("把最大宽度调整为120")

    assert ex.last_exploration_truncated is False
    assert ex.last_pending_tools == []


@pytest.mark.asyncio
async def test_truncation_flag_resets_between_calls(monkeypatch, tmp_path):
    """A stale flag must not be read as the next request's outcome."""
    _tiny_repo(tmp_path)
    state = {"slow": True}
    monkeypatch.setattr(explorer_mod, "ripgrep_search", _slow_ripgrep(2.0, state))

    ex = ParallelExplorer(root=tmp_path, effort=_low_budget(0.2))
    await ex.explore("第一次")
    assert ex.last_exploration_truncated is True

    state["slow"] = False
    await ex.explore("第二次")
    assert ex.last_exploration_truncated is False
    assert ex.last_pending_tools == []


@pytest.mark.asyncio
async def test_escalation_reason_names_the_budget(monkeypatch, tmp_path):
    """When the budget is the cause, the reason must say so.

    The prompt deliberately matches nothing, so the no-match fallback supplies the
    candidate and the confidence is capped by it.  MEASURED before this change, the
    reason reported in exactly this situation was ``Confidence 0.25 < threshold 0.60``
    -- true, and useless: it sends the next reader to tune a policy threshold when the
    real cause was that the budget expired before anything was located.

    Note what this test does *not* claim: a truncated exploration is not on its own a
    reason to escalate.  When another tool did locate the target, the agent proceeds --
    see ``test_truncation_alone_does_not_block_a_located_target``.
    """
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(explorer_mod, "ripgrep_search", _slow_ripgrep(2.0))

    agent = ReflexAgent(root=tmp_path, effort=_low_budget(0.2))
    res = await agent.run(prompt="修复这个问题", dry_run=True)

    assert res.exploration_truncated is True
    assert res.pending_tools, "the pending tools should be named"
    assert res.escalated is True
    assert "Exploration budget" in (res.escalation_reason or "")
    assert "ripgrep" in (res.escalation_reason or "")


@pytest.mark.asyncio
async def test_truncation_alone_does_not_block_a_located_target(monkeypatch, tmp_path):
    """A truncated search must not veto a target another tool did find.

    Escalating here would trade one silent failure for a louder one: the budget expiring
    says the search was incomplete, not that the answer is wrong.
    """
    _tiny_repo(tmp_path)
    monkeypatch.setattr(explorer_mod, "ripgrep_search", _slow_ripgrep(2.0))

    agent = ReflexAgent(root=tmp_path, effort=_low_budget(0.2))
    res = await agent.run(prompt="把最大宽度调整为120", dry_run=True)

    assert res.exploration_truncated is True
    assert res.candidates, "path_lexical should still have offered a candidate"
    assert res.target_path == "a.py"
    assert res.escalated is False


@pytest.mark.asyncio
async def test_ripgrep_does_not_spawn_on_the_event_loop(monkeypatch, tmp_path):
    """Regression for defect 1, locked without depending on this machine's speed.

    The invariant is *where* the spawn happens.  ``create_subprocess_exec`` is replaced
    with one that blocks the loop for 300 ms -- which is what the real call did on the
    machine this was found on (441 ms to spawn ``rg.exe``, 33 ms to run it).  Then a
    concurrent ticker is asserted to have kept running.

    A worker thread passes on any machine; spawning on the loop fails on any machine,
    because the injected block does not depend on the real subprocess being slow.  An
    earlier version of this test slowed ``subprocess.run`` instead, which the blocking
    implementation never calls -- it passed against the bug, so it locked nothing.
    """
    if get_ripgrep_binary() is None:
        pytest.skip("ripgrep is not installed, so there is no subprocess to spawn")

    _tiny_repo(tmp_path)

    # Both spawn paths are slowed to the same 300 ms so the test discriminates on *where*
    # the spawn happens rather than on how fast this machine is.
    real_spawn = asyncio.create_subprocess_exec
    spawned_on_loop: list[str] = []

    async def slow_spawn(*args, **kwargs):
        spawned_on_loop.append(str(args[0]) if args else "?")
        time.sleep(0.3)  # block the loop, as the measured spawn did
        return await real_spawn(*args, **kwargs)

    real_run = tools_mod.subprocess.run

    def slow_run(*args, **kwargs):
        time.sleep(0.3)  # runs in a worker thread, so it must not starve the loop
        return real_run(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_spawn)
    monkeypatch.setattr(tools_mod.subprocess, "run", slow_run)

    ticks: list[float] = []

    async def ticker() -> None:
        while True:
            await asyncio.sleep(0.05)
            ticks.append(time.perf_counter())

    task = asyncio.create_task(ticker())
    try:
        matches = await ripgrep_search(["MAX_WIDTH"], tmp_path, max_matches=5)
    finally:
        task.cancel()

    assert not spawned_on_loop, (
        f"ripgrep was spawned on the event loop ({spawned_on_loop}); a slow spawn then "
        f"blocks the loop for its whole duration, which is what stopped the exploration "
        f"budget from firing"
    )
    assert matches, "ripgrep should still find MAX_WIDTH in a.py"
    assert len(ticks) >= 3, (
        f"the event loop was starved during the search ({len(ticks)} ticks)"
    )
