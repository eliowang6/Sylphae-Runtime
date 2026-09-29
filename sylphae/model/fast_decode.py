"""Compiled streaming decode path.

The product requirement in ``DESIGN.md`` is single-digit-millisecond latency at
``batch=1``.  Measured on the RTX 5070 Ti Laptop, the eager streaming path for the
24-layer / 295M-parameter config is **~32-35 ms** per token, and bfloat16 does not
help (34.6 ms -> 33.6 ms).  That is the signature of a *launch-bound* step, not a
compute-bound one, and the profiler confirms it: 1412 CUDA kernels and ~20 ms of
pure CPU time inside ``cudaLaunchKernel`` for a single token, against ~4 ms of
actual GPU work.

Compiling the backbone's streaming forward with ``torch.compile(mode="default")``
lets Inductor fuse the elementwise chains (the SSD step is mostly ``mul``/``add``/
``pow``/``mean``) and drops the step to **~7.8 ms**.  That is the difference
between missing and meeting the design target, so it is worth a module.

``mode="reduce-overhead"`` (CUDA graphs) does **not** work here and the failure is
structural rather than a tuning problem: the cache is updated *functionally*, so
every call returns freshly allocated tensors, and a CUDA graph replays into the
same output buffers.  The result is a genuine ``RuntimeError`` about accessing an
overwritten CUDAGraph output -- not a slowdown, a crash.  Making CUDA graphs work
would require an in-place cache, which is a larger change than this file.

Numerical equivalence with the eager path is asserted in
``tests/test_fast_decode.py``: the compiled path is only allowed to be faster, not
different.
"""

from __future__ import annotations

import torch

from .policy import SylphaeModel, MemState

__all__ = ["CompiledSylphae", "compile_streaming", "DEFAULT_COMPILE_MODE"]

#: ``default`` is the mode that actually works; see the module docstring for why
#: ``reduce-overhead`` cannot.
DEFAULT_COMPILE_MODE = "default"


class CompiledSylphae:
    """``SylphaeModel.stream_step`` with the backbone's streaming forward compiled.

    Drop-in for the decode loop::

        fast = CompiledSylphae(model)
        out, cache, mem = fast.stream_step(batch, cache)

    The first call triggers compilation and is slow (tens of seconds); warm up
    before timing anything.
    """

    def __init__(self, model: SylphaeModel, mode: str = DEFAULT_COMPILE_MODE,
                 dynamic: bool = False, compile: bool = True):
        self.model = model
        self.mode = mode
        self._compiled = compile
        if compile:
            self._forward = torch.compile(model.backbone.forward, mode=mode, dynamic=dynamic)
        else:
            self._forward = model.backbone.forward

    def stream_step(
        self,
        batch,
        cache,
        mem_write_override: float | None = None,
    ):
        """One new token.  Mirrors :meth:`SylphaeModel.stream_step` exactly."""
        m = self.model
        hidden, new_cache = self._forward(batch.input_ids, cache=cache)
        pooled = m._pool(hidden, batch.attention_mask)
        out = m._heads(batch, hidden, pooled)
        gate = m._gate(out.mem_write, mem_write_override)
        mem_state = MemState([gate * c[1] for c in new_cache])
        return out, new_cache, mem_state

    def init_cache(self, batch_size: int = 1):
        p = next(self.model.parameters())
        return self.model.backbone.init_cache(batch_size, p.device, p.dtype)


def compile_streaming(model: SylphaeModel, mode: str = DEFAULT_COMPILE_MODE,
                      warmup: int = 2) -> CompiledSylphae:
    """Compile and warm up, so the returned object is ready to be timed."""
    fast = CompiledSylphae(model, mode=mode)
    if fast._compiled:
        batch = _warmup_batch(model)
        cache = fast.init_cache(1)
        with torch.inference_mode():
            for _ in range(max(1, warmup)):
                _, cache, _ = fast.stream_step(batch, cache)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return fast


def _warmup_batch(model: SylphaeModel):
    """A single-token batch shaped like a real request, for compile warmup.

    Shape only, so it comes from :mod:`sylphae.model.collate` with the model --
    the module that owns :class:`Batch` -- rather than from a training benchmark.
    ``torch.compile`` will not specialise a graph until it has seen one, and
    making the inference half import the training half to get one is the coupling
    this split exists to remove.
    """
    from .collate import synth_batch

    cfg = model.cfg
    dev = next(model.parameters()).device
    n_cand = getattr(cfg, "n_candidates", 8)
    max_params = getattr(cfg, "max_params", 32)
    return synth_batch(1, 1, n_cand, max_params, cfg.vocab_size, dev)


def main(argv=None) -> int:  # pragma: no cover - manual benchmark entry point
    """Measure eager vs compiled single-token latency for the default config."""
    import argparse
    import json
    import statistics

    from .collate import synth_batch
    from .config import REFERENCE_CONFIG, ModelConfig

    ap = argparse.ArgumentParser(description="eager vs compiled streaming decode latency")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--mode", default=DEFAULT_COMPILE_MODE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(**REFERENCE_CONFIG)
    torch.manual_seed(0)
    model = SylphaeModel(cfg).to(dev).eval()
    batch = synth_batch(1, 1, cfg.n_candidates, cfg.max_params, cfg.vocab_size, dev)

    def time_streamer(streamer):
        cache = streamer.init_cache(1)
        with torch.inference_mode():
            for _ in range(args.warmup):
                _, cache, _ = streamer.stream_step(batch, cache)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            s = e = None
            if dev.type == "cuda":
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
            with torch.inference_mode():
                _, cache, _ = streamer.stream_step(batch, cache)
            if dev.type == "cuda":
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            else:
                ts.append(0.0)
        return ts

    eager = CompiledSylphae(model, compile=False)
    t_eager = time_streamer(eager)
    fast = compile_streaming(model, mode=args.mode, warmup=args.warmup)
    t_fast = time_streamer(fast)

    def stats(ts):
        ts = sorted(ts)
        return {
            "p50_ms": round(statistics.median(ts), 2),
            "p95_ms": round(ts[max(0, int(0.95 * len(ts)) - 1)], 2),
            "min_ms": round(ts[0], 2),
        }

    report = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "config": cfg.to_dict(),
        "n_params": model.num_parameters(),
        "mode": args.mode,
        "eager": stats(t_eager),
        "compiled": stats(t_fast),
    }
    report["speedup_p50"] = round(report["eager"]["p50_ms"] / report["compiled"]["p50_ms"], 2)

    print(f"device   : {report['device']}")
    print(f"params   : {report['n_params']:,}")
    print(f"eager    : p50 {report['eager']['p50_ms']:.2f} ms  p95 {report['eager']['p95_ms']:.2f} ms")
    print(f"compiled : p50 {report['compiled']['p50_ms']:.2f} ms  "
          f"p95 {report['compiled']['p95_ms']:.2f} ms  (mode={args.mode})")
    print(f"speedup  : {report['speedup_p50']}x")
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
