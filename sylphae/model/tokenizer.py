"""Byte-level BPE tokenizer for code + commit text.

Design notes
------------
* **Byte level.**  Every byte has a token, so ``decode(encode(x)) == x`` holds
  for arbitrary input including source, diffs, unicode and mojibake.  There is
  no UNK, which matters because the request text comes from real commit
  subjects and will contain whatever the author typed.
* **No regex pre-tokenisation** (``use_regex=False``).  The GPT-2 regex splits
  on whitespace and punctuation, which prevents merges such as ``"def "`` or
  ``"self."`` that are exactly the units code wants.  Without it, BPE sees one
  long word per document and learns code-shaped merges.
* Seven special tokens occupy ids 0..6 in the order fixed by INTERFACES.md
  section 4.  Byte tokens start at 7, so ordinary text can never produce a
  special id.

This module is the inference side: it loads and runs a vocabulary, and trains
one from a caller-supplied file list.  Assembling a corpus out of the harvested
JSONL is dataset work and lives in :mod:`sylphae.data.tokenizer_corpus`, so
nothing here knows the corpus format.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

__all__ = [
    "SPECIAL_TOKENS",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "SEP_ID",
    "CAND_ID",
    "ENDCAND_ID",
    "MASK_ID",
    "N_SPECIAL",
    "CodeTokenizer",
    "train_tokenizer",
    "stdlib_corpus_paths",
]

# Frozen by INTERFACES.md section 4, in this exact order.
SPECIAL_TOKENS: list[str] = ["<pad>", "<bos>", "<eos>", "<sep>", "<cand>", "</cand>", "<mask>"]
PAD_ID, BOS_ID, EOS_ID, SEP_ID, CAND_ID, ENDCAND_ID, MASK_ID = range(7)
N_SPECIAL = len(SPECIAL_TOKENS)

_FORMAT = "sylphae-code-tokenizer-v1"


class CodeTokenizer:
    """Thin wrapper over a ``tokenizers.Tokenizer`` with the Sylphae special ids."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.sep_id = SEP_ID
        self.cand_id = CAND_ID
        self.endcand_id = ENDCAND_ID
        self.mask_id = MASK_ID

    # -- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    @property
    def special_tokens(self) -> list[str]:
        return list(SPECIAL_TOKENS)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CodeTokenizer(vocab_size={self.vocab_size})"

    # -- encoding ----------------------------------------------------------

    def encode(self, text: str, max_len: int) -> list[int]:
        """Byte-level BPE ids, truncated to at most ``max_len``.  No specials added."""
        if not isinstance(text, str):
            raise TypeError(f"encode expects str, got {type(text).__name__}")
        if max_len <= 0:
            return []
        ids = self._tok.encode(text).ids
        return ids[:max_len]

    def encode_batch(self, texts, max_len: int) -> list[list[int]]:
        if max_len <= 0:
            return [[] for _ in texts]
        texts = list(texts)
        encs = self._tok.encode_batch(texts)
        return [e.ids[:max_len] for e in encs]

    def decode(self, ids) -> str:
        """Inverse of :meth:`encode` for id sequences it produced.

        Special ids (0..6) are not bytes, so they are dropped rather than
        decoded -- that keeps ``decode(encode(x)) == x`` exact.
        """
        clean = [int(i) for i in ids if int(i) >= N_SPECIAL]
        return self._tok.decode(clean)

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _FORMAT,
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "tokenizer": json.loads(self._tok.to_str()),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CodeTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != _FORMAT:
            raise ValueError(
                f"{path} is not a {_FORMAT} artifact (found {payload.get('format')!r}); "
                "retrain with train_tokenizer()."
            )
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError(
                f"{path} was written with special tokens {payload.get('special_tokens')!r}; "
                f"this build expects {SPECIAL_TOKENS!r}. Refusing to load."
            )
        return cls(Tokenizer.from_str(json.dumps(payload["tokenizer"])))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def _build_untrained() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(corpus_paths, vocab_size: int, out_path) -> CodeTokenizer:
    """Train a byte-level BPE tokenizer and write it to ``out_path``.

    ``corpus_paths`` is an iterable of files to train on (Python source plus
    commit-message text).  ``vocab_size`` counts the seven special tokens.
    """
    paths = [str(Path(p)) for p in corpus_paths]
    if not paths:
        raise ValueError("train_tokenizer needs at least one corpus file")
    if vocab_size <= N_SPECIAL:
        raise ValueError(
            f"vocab_size must exceed the {N_SPECIAL} special tokens, got {vocab_size}"
        )

    tok = _build_untrained()
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train(paths, trainer)
    out = CodeTokenizer(tok)
    _check_special_ids(out)
    out.save(out_path)
    return out


def _check_special_ids(tok: CodeTokenizer) -> None:
    vocab = tok._tok.get_vocab()
    for i, name in enumerate(SPECIAL_TOKENS):
        got = vocab.get(name)
        if got != i:
            raise RuntimeError(
                f"tokenizer assigned id {got} to {name!r}, expected {i}. "
                "The special-token order is frozen by INTERFACES.md section 4."
            )


def stdlib_corpus_paths(
    root: str | Path | None = None,
    max_files: int = 400,
    max_bytes: int = 8_000_000,
) -> list[Path]:
    """Deterministic slice of the local CPython standard library.

    The download-free Python corpus.  Files are taken in sorted order so the
    corpus -- and hence the tokenizer -- is reproducible.  This ships with the
    tokenizer because it is how a user builds one offline, with nothing but the
    interpreter they are already running.

    Assembling a corpus that also includes commit-message text lives in
    :mod:`sylphae.data.tokenizer_corpus`, which reads the harvested corpus and so
    belongs with the rest of the data layer.
    """
    if root is None:
        import sysconfig

        root = Path(sysconfig.get_paths()["stdlib"])
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"stdlib root {root} is not a directory")

    out: list[Path] = []
    total = 0
    for p in sorted(root.rglob("*.py")):
        if "site-packages" in p.parts or "test" in p.parts:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        out.append(p)
        total += size
        if len(out) >= max_files or total >= max_bytes:
            break
    return out
