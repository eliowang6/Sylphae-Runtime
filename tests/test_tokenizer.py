"""Invariants for ``sylphae/model/tokenizer.py``.

The seven special tokens at ids 0..6 are the contract; ``decode(encode(x)) == x``
is what makes the corpus lossless; truncation must respect ``max_len``.
"""

from __future__ import annotations

import pytest

from sylphae.model.tokenizer import (
    BOS_ID,
    CAND_ID,
    ENDCAND_ID,
    EOS_ID,
    MASK_ID,
    N_SPECIAL,
    PAD_ID,
    SEP_ID,
    SPECIAL_TOKENS,
    CodeTokenizer,
    stdlib_corpus_paths,
    train_tokenizer,
)

ASCII_SAMPLES = [
    "def f(x):\n    return x + 1\n",
    "import os\nfrom pathlib import Path\n\n\nclass A(B):\n    pass\n",
    "if __name__ == '__main__':\n    main()\n",
    "# fix typo in the docstring\n",
    "value = {'a': 1, 'b': [2, 3]}\n",
    "Fix crash when broker is instantiated (#1234)\n",
    "\n\t   \n",
    "",
    "x" * 5000,
]


def test_special_token_ids_are_what_we_claim(tiny_tokenizer):
    tok = tiny_tokenizer
    assert SPECIAL_TOKENS == [
        "<pad>", "<bos>", "<eos>", "<sep>", "<cand>", "</cand>", "<mask>",
    ]
    assert (PAD_ID, BOS_ID, EOS_ID, SEP_ID, CAND_ID, ENDCAND_ID, MASK_ID) == tuple(range(7))
    assert N_SPECIAL == 7

    vocab = tok._tok.get_vocab()
    for i, name in enumerate(SPECIAL_TOKENS):
        assert vocab[name] == i, f"{name} has id {vocab[name]}, expected {i}"

    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2


@pytest.mark.parametrize("text", ASCII_SAMPLES)
def test_round_trip(tiny_tokenizer, text):
    tok = tiny_tokenizer
    ids = tok.encode(text, max_len=10_000)
    assert tok.decode(ids) == text


def test_round_trip_unicode(tiny_tokenizer):
    tok = tiny_tokenizer
    for text in ["café — naïve", "日本語のコメント", "emoji: 🚀", "\x00\x01\x02 raw bytes"]:
        assert tok.decode(tok.encode(text, 1000)) == text


def test_lone_surrogate_is_rejected(tiny_tokenizer):
    """A lone surrogate is not valid UTF-8, so it cannot be byte-level encoded.

    Rejecting loudly is correct -- silently substituting a replacement character
    would corrupt the corpus.
    """
    with pytest.raises(TypeError):
        tiny_tokenizer.encode("bad bytes: \udcff", 100)


def test_encode_never_emits_special_ids(tiny_tokenizer):
    """Ordinary text must never collide with the special ids."""
    tok = tiny_tokenizer
    for text in ASCII_SAMPLES:
        ids = tok.encode(text, 10_000)
        if ids:
            assert min(ids) >= N_SPECIAL


def test_truncation_respects_max_len(tiny_tokenizer):
    tok = tiny_tokenizer
    text = "def function_name(argument_one, argument_two):\n    return argument_one\n" * 50
    full = tok.encode(text, 10_000)
    for m in (1, 2, 7, 32, 100):
        got = tok.encode(text, m)
        assert len(got) <= m
        assert got == full[:m]
    assert len(tok.encode(text, 0)) == 0


def test_encode_batch_matches_encode(tiny_tokenizer):
    tok = tiny_tokenizer
    batch = tok.encode_batch(ASCII_SAMPLES, 64)
    assert len(batch) == len(ASCII_SAMPLES)
    for text, ids in zip(ASCII_SAMPLES, batch):
        assert ids == tok.encode(text, 64)


def test_save_load_round_trip(tiny_tokenizer, tmp_path):
    p = tmp_path / "tok.json"
    tiny_tokenizer.save(p)
    other = CodeTokenizer.load(p)
    assert other.vocab_size == tiny_tokenizer.vocab_size
    for text in ASCII_SAMPLES:
        assert other.encode(text, 10_000) == tiny_tokenizer.encode(text, 10_000)
        assert other.decode(other.encode(text, 10_000)) == text


# --------------------------------------------------------------------------
# negative cases
# --------------------------------------------------------------------------


def test_negative_truncation_is_lossy(tiny_tokenizer):
    """If truncation round-tripped, ``test_truncation_respects_max_len`` would be
    checking nothing."""
    tok = tiny_tokenizer
    text = "def function_name(argument_one, argument_two):\n    return argument_one\n"
    truncated = tok.decode(tok.encode(text, 4))
    assert truncated != text


def test_negative_corrupted_ids_do_not_round_trip(tiny_tokenizer):
    tok = tiny_tokenizer
    text = "def f(x):\n    return x\n"
    ids = tok.encode(text, 1000)
    assert ids, "sample must not encode to nothing"
    ids[0] = ids[0] + 1 if ids[0] + 1 < tok.vocab_size else ids[0] - 1
    assert tok.decode(ids) != text


def test_negative_different_text_differs(tiny_tokenizer):
    tok = tiny_tokenizer
    a = tok.encode("alpha = 1\n", 100)
    b = tok.encode("beta = 2\n", 100)
    assert a != b


def test_negative_vocab_size_too_small_rejected(tmp_path):
    files = stdlib_corpus_paths(max_files=2, max_bytes=20_000)
    with pytest.raises(ValueError):
        train_tokenizer(files, vocab_size=7, out_path=tmp_path / "x.json")
    with pytest.raises(ValueError):
        train_tokenizer([], vocab_size=256, out_path=tmp_path / "y.json")


def test_negative_wrong_special_order_rejected(tiny_tokenizer, tmp_path):
    """A file claiming a different special-token order must be refused."""
    import json

    p = tmp_path / "bad.json"
    tiny_tokenizer.save(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["special_tokens"] = ["<pad>", "<bos>", "<eos>"]
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        CodeTokenizer.load(p)


def test_encode_rejects_non_string(tiny_tokenizer):
    with pytest.raises(TypeError):
        tiny_tokenizer.encode(b"bytes", 10)


def test_vocab_size_is_respected(tiny_tokenizer):
    assert tiny_tokenizer.vocab_size == 768
