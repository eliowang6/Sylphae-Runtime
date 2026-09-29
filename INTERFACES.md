# INTERFACES.md — frozen module contracts

These signatures and schemas are the contract between the modules of this
package. **If you need to change one, change it here first and say so** — do not
silently diverge, because everything else is coded against the current text, and a
signature that moves underneath four call sites is a bug nobody can bisect.

Conventions for every module:

- Python 3.11+. Dependencies are declared in `pyproject.toml` and are
  deliberately few: `torch`, `tokenizers`, `pillow`, `safetensors`. No `datasets`,
  no `libcst`, no `tree-sitter` — the AST work uses the standard library, and a
  repository index that can fall over because a native dependency failed to
  build is a repository index nobody trusts.
- Every module ships with tests that are runnable from a fresh clone. **A module
  is not done until its tests pass.** Tests must include at least one *negative*
  case — a check that only ever prints PASS is worthless.
- Never silently rewrite a hyperparameter or coerce bad input into something
  plausible. Reject and say what to change.
- Missing metrics are `None`, never `0.0`.

---

## 1. `sylphae/ops/` — edit-operation vocabulary

The vocabulary is the ceiling of the whole approach: only operations that are *detectable
from a diff* and *deterministically executable* are in it.

```python
# sylphae/ops/vocab.py
OP_KINDS: list[str] = [
    "no_op",              # 0
    "replace_literal",    # 1  a numeric/string literal changes value, structure identical
    "rename_symbol",      # 2  an identifier is renamed consistently
    "change_condition",   # 3  a line inside if/while/assert changes
    "modify_call_args",   # 4  arguments of a call change
    "change_default",     # 5  a parameter default value changes
    "add_import",         # 6
    "remove_import",      # 7
    "add_guard",          # 8  an early-return / raise guard is inserted
    "insert_statement",   # 9  statements inserted, nothing removed
    "delete_statement",   # 10 statements removed, nothing inserted
    "add_decorator",      # 11
    "change_return",      # 12
    "modify_string",      # 13 only a message string changes
    "wrap_try_except",    # 14
    "freeform_patch",     # 15 anything else — the escape hatch, and the coverage loss
    # -- appended when the sequence detector landed.  Ids 0..15 above are UNCHANGED;
    #    the corpus and the model op head index into them, so reordering is a
    #    breaking change and appending is not.
    "replace_line",         # 16 one whole line replaced by exactly one other whole line
    "add_call_arg",         # 17 an argument line added to a multi-line call
    "remove_call_arg",      # 18 an argument line removed from a multi-line call
    "add_dict_entry",       # 19 a `key: value` entry added to a dict literal
    "remove_dict_entry",    # 20 a `key: value` entry removed from a dict literal
    "wrap_block",           # 21 a region indented under a newly inserted header line
    "modify_string_multi",  # 22 several string literals change in one hunk
    "reindent",             # 23 only indentation changes, content identical
    "add_list_element",     # 24 an element line added to a list/set/tuple literal
    "remove_list_element",  # 25 an element line removed from a list/set/tuple literal
]
OP_TO_ID: dict[str, int]
ID_TO_OP: dict[int, str]
N_OPS: int  # == len(OP_KINDS) == 26

# The distinction that actually matters.  A *parameterised* op is one the model
# emits by choosing a short value or name (a literal, an identifier, a
# keyword-argument name, a dict key) — the reflex domain the design is about.  A
# *generative* op requires writing new code text, which a small model does badly.
# These two partition EXECUTABLE_KINDS (everything but `freeform_patch`/`no_op`),
# and both coverages are reported separately.
PARAMETERIZED_OPS: frozenset[str]
GENERATIVE_OPS: frozenset[str]
```

```python
# sylphae/ops/spec.py
@dataclass
class OpSpec:
    kind: str                    # must be in OP_KINDS
    path: str                    # repo-relative posix path of the edited file
    span: tuple[int, int]        # 1-based inclusive line span in the ORIGINAL source
    props: dict[str, str]        # op-specific payload; values are plain strings
    def to_dict(self) -> dict    # JSON-serialisable
    @staticmethod
    def from_dict(d: dict) -> "OpSpec"

class OpApplyError(Exception): ...
```

```python
# sylphae/ops/detect.py
def detect_op(old_src: str, new_src: str, path: str) -> OpSpec | None:
    """Classify one file's change as a SINGLE op.

    Returns None when old_src == new_src.
    Returns an OpSpec of kind 'freeform_patch' when no structured op reproduces new_src.
    MUST satisfy the round-trip invariant below for every non-freeform result.
    """

def detect_sequence(old_src: str, new_src: str, path: str) -> list[OpSpec]:
    """Decompose one file's change into an ordered sequence of ops.

    A real commit is often two to five independent small edits, and forcing the
    whole change to be ONE op was the core modelling error behind the first
    (25.4%) coverage measurement.  `detect_op` is the length-one special case:
    whenever it finds a structured op, `detect_sequence` returns exactly
    `[that_op]`.

    Returns [] when old_src == new_src, and [freeform_patch] when no sequence of
    structured ops reproduces new_src.

    Ordering / application contract (this is part of the interface, not an
    implementation detail): ops are emitted top-to-bottom (ascending original
    line number), and every span is in the ORIGINAL source, exactly as for
    `detect_op`.  The sequence is therefore applied BACK-TO-FRONT — the lowest op
    first, so it cannot shift the line numbers of the ops above it.  Equivalently
    `apply_sequence(old, seq) == functools.reduce(apply_op, reversed(seq), old)`.
    """

# sylphae/ops/apply.py
def apply_op(old_src: str, spec: OpSpec) -> str:
    """Reproduce the edited source. Raises OpApplyError if spec does not fit old_src."""

def apply_sequence(old_src: str, seq: list[OpSpec]) -> str:
    """Apply a `detect_sequence` result back-to-front. Raises OpApplyError."""
```

**Hard invariants** (these are the tests that matter):

1. **Round-trip, single op**: for any `(old, new)` where `detect_op` returns a non-`freeform_patch`
   spec, `apply_op(old, spec) == new`. Exactly. No whitespace tolerance.
2. **Round-trip, sequence**: for any `(old, new)` where `detect_sequence` returns a sequence
   whose ops are not all `freeform_patch`, `apply_sequence(old, seq) == new`. Exactly.
   A sequence that does not replay is never returned; the result degrades to
   `[freeform_patch]` instead.
3. **Idempotent rejection**: `apply_op` on a spec whose `span` is out of range raises
   `OpApplyError`, it does not clamp.
4. **No-op detection**: identical sources → `None` (single) / `[]` (sequence).
5. Every `props` value round-trips through JSON as a string.
6. `PARAMETERIZED_OPS` and `GENERATIVE_OPS` partition `EXECUTABLE_KINDS`, and neither
   contains `freeform_patch` or `no_op`.

`detect_op` / `detect_sequence` may only use the Python standard library (`ast`, `tokenize`,
`difflib`). They must be *conservative*: a wrong structured label is worse than
`freeform_patch`, because it teaches the model a lie.

Training data (section 2, in the parent project) additionally carries an additive
`op_sequence` key — the `detect_sequence` result as `[OpSpec.to_dict(), ...]` —
alongside the unchanged required `op` key, so that a consumer of this contract can
tell a one-step edit from a multi-step one without re-deriving it.

---

## 3. `sylphae/repo/` — world state and retrieval

```python
# sylphae/repo/index.py
@dataclass
class Candidate:
    path: str
    score: float
    symbols: list[str]
    snippet: str          # <= 600 chars, the region most likely to be edited
    snippet_span: tuple[int, int]

class RepoIndex:
    files: list[str]
    @classmethod
    def build(cls, root: Path) -> "RepoIndex"
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> "RepoIndex"
    def candidates(self, request: str, k: int = 8,
                   exclude_tests: bool = False) -> list[Candidate]
```

`build` walks a source tree, parses every `.py` with `ast`, and records: relative path,
top-level and nested symbol qualified names, imports, line count. Files that fail to parse
are recorded with `symbols=[]` — never dropped, never fatal.

`candidates` must be deterministic for a given `(index, request, k)`. It is a lexical +
structural ranker (identifier overlap with the request, symbol-name match, import graph
proximity, path-token match). It is **not** allowed to be an embedding model — there is no
GPU budget for it and it must be reproducible. It must return *fewer* than `k` only when the
index has fewer files, and the target file must be present in the candidate set for at
least 95% of Tier-B examples (measure this, report it — it is the recall ceiling of the
whole target head).

---

## 4. `sylphae/model/` — SSD backbone and heads

```python
# sylphae/model/ssd.py
@dataclass
class SSDCfg:
    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    n_heads: int = 16
    chunk_size: int = 64
    expand: int = 2
    d_conv: int = 4
    vocab_size: int = 32768
    max_len: int = 2048

class Mamba2Block(nn.Module):
    def forward(self, x: Tensor, cache: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]
    # x: (B, L, d_model) -> (B, L, d_model)
```

The SSD mixer must be implemented in pure PyTorch. Two implementations are required:

- a **naive recurrent** reference (`ssd_recurrent`), obviously correct, O(L) loop;
- a **chunked parallel** version (`ssd_chunked`), the one training uses.

and a test asserting they agree to `atol=1e-4` on random inputs. The chunked path is the
only one allowed to be fast; the reference is the only one allowed to be trusted.

`cache` carries the recurrent state so that a second call with one new token reproduces the
same output as a full forward — the streaming property the design depends on. Test it.

```python
# sylphae/model/collate.py
@dataclass
class Batch:            # every field a tensor
    input_ids: Tensor       # (B, L) long
    attention_mask: Tensor  # (B, L) long
    cand_index: Tensor      # (B, C) long   token position where candidate c starts
    cand_mask: Tensor       # (B, C) bool
    op_labels: Tensor       # (B,) long
    param_ids: Tensor       # (B, P) long, P <= 32
    param_mask: Tensor      # (B, P) bool
    conf_label: Tensor      # (B,) float
    mem_write_label: Tensor # (B,) float

def collate(examples: list[dict], tok, index_provider, cfg: SSDCfg) -> Batch

# sylphae/model/tokenizer.py
class CodeTokenizer:
    vocab_size: int
    pad_id: int; bos_id: int; eos_id: int
    def encode(self, text: str, max_len: int) -> list[int]
    def encode_batch(self, texts, max_len) -> list[list[int]]
    def decode(self, ids: list[int]) -> str
    def save(self, path); @classmethod load(cls, path)

def train_tokenizer(corpus_paths: list[Path], vocab_size: int, out_path: Path) -> CodeTokenizer
```

Special tokens, in this order, ids 0..6: `<pad> <bos> <eos> <sep> <cand> </cand> <mask>`.

The tokenizer is trained on Python source (the standard library on this machine is a
perfect corpus, no download needed) plus commit-message text.

```python
# sylphae/model/policy.py
@dataclass
class PolicyOutput:
    target_logits: Tensor   # (B, C)
    op_logits: Tensor       # (B, N_OPS)
    param_logits: Tensor    # (B, P, vocab_size)
    confidence: Tensor      # (B,) in [0, 1]
    mem_write: Tensor       # (B,) in [0, 1]

class SylphaeModel(nn.Module):
    def forward(self, batch: Batch, mem: "MemState | None" = None
                ) -> tuple[PolicyOutput, "MemState"]
    def num_parameters(self) -> int
```

`MemState` must expose `detach()`, `to(device)`, and a `zeros(batch_size, cfg, device)`
constructor. It is the design's persistent latent memory — it is the SSM state, carried
across requests, and the memory-write gate scales what gets written into it.

The candidate encoding is **not** a second encoder: candidate `c`'s tokens occupy a segment
of the same sequence, delimited by `<cand>` / `</cand>`, and `cand_index[b, c]` is the
position of that segment's opening token. `target_logits` is read out from those positions.

---

## 5. `sylphae/exec/` — execution and verification

```python
# sylphae/exec/patch.py
@dataclass
class ApplyResult:
    ok: bool
    new_source: str | None
    error: str | None

def apply_to_tree(root: Path, target_path: str, spec: OpSpec, dry_run: bool = True) -> ApplyResult

# sylphae/exec/verify.py
@dataclass
class VerifyReport:
    compiles: bool
    tests_run: int
    tests_passed: int
    errors: list[str]

def verify(root: Path, changed_paths: list[str], test_paths: list[str],
           timeout_s: int = 120) -> VerifyReport

def affected_tests(root: Path, changed_paths: list[str], all_tests: list[str]) -> list[str]
```

`affected_tests` maps a changed source file to the test files that exercise it, via the
import graph. When it cannot tell, it returns the tests whose path tokens overlap the
changed path, and says so by returning a *superset* rather than an empty list — an empty
list would silently mean "verified".

`verify` must never claim success it did not observe: if `pytest` is unavailable or times
out, `tests_run` is 0 and the caller must be able to distinguish that from "0 tests failed".

---


---

## Sections that are not here

Sections **2** (`sylphae/data/`), **6** (`sylphae/eval/`) and **7** (`sylphae/train/`) are
omitted. They are the corpus-synthesis, measurement and training layers, which live in
the parent project and are deliberately not published here.

The numbering is preserved rather than compacted, so a reference in a docstring to
"INTERFACES.md section 5" resolves to the same contract it did before the split. The
gaps are the point: a missing section is a statement that nothing here depends on it.

