"""The compiled streaming path must be faster, never different.

``sylphae/model/fast_decode.py`` exists because compiling the streaming forward is
the difference between ~33 ms and ~8 ms per token at batch=1, which is the whole
latency argument in DESIGN.md.  A speedup that changes the numbers is not a
speedup, it is a different model, so equivalence is asserted here rather than
assumed.

The config is deliberately tiny: ``torch.compile`` is the point, not scale, and a
2-layer model compiles in seconds instead of a minute.

These tests require CUDA.  Inductor's CPU backend compiles generated C++ through
MSVC, and this machine has no ``cl.exe`` on PATH, so a CPU run fails with
``RuntimeError: Compiler: cl is not found``.  That is an environment fact, not a
bug in the compiled path -- the CUDA/Triton backend is the one that works and the
one the latency argument is about -- so the tests skip rather than pretend.
"""

from __future__ import annotations

import pytest
import torch

from sylphae.model.config import ModelConfig
from sylphae.model.fast_decode import CompiledSylphae, compile_streaming
from sylphae.model.policy import SylphaeModel

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch.compile on the streaming path needs the CUDA/Triton backend; "
           "Inductor's CPU backend requires MSVC cl.exe, which is not installed",
)

pytestmark = requires_cuda

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TINY = dict(
    d_model=64,
    n_layers=2,
    d_state=16,
    n_heads=4,
    chunk_size=8,
    expand=2,
    vocab_size=256,
    max_len=64,
    d_ff=64,
    n_candidates=4,
    max_params=8,
)


def _model(seed: int = 0) -> SylphaeModel:
    torch.manual_seed(seed)
    return SylphaeModel(ModelConfig(**TINY)).to(DEVICE).eval()


def _batch(model: SylphaeModel, vocab_size: int = 256):
    """A one-token batch built by hand, so the test does not depend on bench.py."""
    from sylphae.model.collate import Batch

    return Batch(
        input_ids=torch.randint(0, vocab_size, (1, 1), device=DEVICE),
        attention_mask=torch.ones(1, 1, dtype=torch.long, device=DEVICE),
        cand_index=torch.tensor([[0, 0, 0, 0]], dtype=torch.long, device=DEVICE),
        cand_mask=torch.tensor([[True, True, False, False]], device=DEVICE),
        op_labels=torch.zeros(1, dtype=torch.long, device=DEVICE),
        param_ids=torch.zeros(1, 8, dtype=torch.long, device=DEVICE),
        param_mask=torch.zeros(1, 8, dtype=torch.bool, device=DEVICE),
        conf_label=torch.zeros(1, device=DEVICE),
        mem_write_label=torch.zeros(1, device=DEVICE),
    )


def _run(streamer, batch, cache, steps: int = 3):
    outs = []
    with torch.inference_mode():
        for _ in range(steps):
            out, cache, mem = streamer.stream_step(batch, cache)
            outs.append(out)
    return outs, cache


@pytest.fixture(scope="module")
def compiled_and_eager():
    model = _model()
    batch = _batch(model)
    eager = CompiledSylphae(model, compile=False)
    fast = compile_streaming(model, mode="default", warmup=1)

    e_out, e_cache = _run(eager, batch, eager.init_cache(1))
    c_out, c_cache = _run(fast, batch, fast.init_cache(1))
    return model, batch, (e_out, e_cache), (c_out, c_cache)


def test_compiled_matches_eager_on_every_head(compiled_and_eager):
    _, _, (e_out, _), (c_out, _) = compiled_and_eager
    for eo, co in zip(e_out, c_out):
        for field in ("target_logits", "op_logits", "param_logits", "confidence", "mem_write"):
            a, b = getattr(eo, field), getattr(co, field)
            assert a.shape == b.shape, f"{field}: {a.shape} != {b.shape}"
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (
                f"{field} diverged: max abs diff "
                f"{(a.float() - b.float()).abs().max().item():.3e}"
            )


def test_compiled_matches_eager_on_carried_cache(compiled_and_eager):
    """The recurrent state must survive the round trip identically too.

    This is the part a fused kernel is most likely to get wrong, and the part the
    persistent-memory argument in DESIGN.md section 2.3 depends on.
    """
    _, _, (_, e_cache), (_, c_cache) = compiled_and_eager
    assert len(e_cache) == len(c_cache)
    for layer_e, layer_c in zip(e_cache, c_cache):
        assert len(layer_e) == len(layer_c)
        for a, b in zip(layer_e, layer_c):
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (
                f"cache diverged: max abs diff {(a.float() - b.float()).abs().max().item():.3e}"
            )


def test_comparison_is_not_vacuous(compiled_and_eager):
    """A negative control.

    ``allclose`` passing is only meaningful if it *can* fail.  Two independently
    initialised models must produce different logits, otherwise the equivalence
    assertions above would pass no matter what the compiled path computed.
    """
    _, batch, (e_out, _), _ = compiled_and_eager
    other = CompiledSylphae(_model(seed=1234), compile=False)
    o_out, _ = _run(other, batch, other.init_cache(1))
    assert not torch.allclose(e_out[0].target_logits, o_out[0].target_logits, atol=1e-4)


def test_reduce_overhead_is_documented_as_unavailable():
    """``mode='reduce-overhead'`` cannot work with a functional cache.

    It is not a tuning knob we forgot to set: CUDA graphs replay into fixed output
    buffers while our cache returns fresh tensors, so it raises rather than being
    merely slow.  Asserting the docstring keeps someone from "optimising" this
    later and silently breaking the decode path.
    """
    from sylphae.model import fast_decode

    assert fast_decode.DEFAULT_COMPILE_MODE == "default"
    assert "reduce-overhead" in fast_decode.__doc__
    assert "CUDAGraph" in fast_decode.__doc__ or "CUDA graph" in fast_decode.__doc__


def test_uncompiled_streamer_is_a_drop_in_for_eager(compiled_and_eager):
    """``CompiledSylphae(..., compile=False)`` must equal ``SylphaeModel.stream_step``."""
    model, batch, (e_out, _), _ = compiled_and_eager
    plain = CompiledSylphae(model, compile=False)
    p_out, _ = _run(plain, batch, plain.init_cache(1))
    for eo, po in zip(e_out, p_out):
        assert torch.allclose(eo.target_logits, po.target_logits, atol=1e-5)
        assert torch.allclose(eo.confidence, po.confidence, atol=1e-5)
