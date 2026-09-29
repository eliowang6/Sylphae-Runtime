"""The identifier in a request must survive a file's own low-information header lines.

This is a regression guard for a *retrieval* failure that no amount of reweighting
can recover.  ``ripgrep_search`` passes its keywords to ``rg`` as one alternation
under ``--max-count 3``, which caps matches **per file**.  When a file's first
three matches are header comments that contain only a common subtoken, the line
that actually carries the request's identifier is truncated away before the
ranking layer ever sees it.

The measured instance: ``work/repos/encode__starlette`` for
``把 HTTP_502_BAD_GATEWAY 修改为 503``.  ``starlette/status.py`` matches ``http``
on its first three lines -- ``HTTP codes``, ``See HTTP Status Code Registry``, the
IANA URL -- while ``HTTP_502_BAD_GATEWAY = 502`` sits 145 lines further down.  The
gold file entered ranking with the keyword set ``{'http'}``, tied 97 unrelated
files at the same idf-degraded score, and ranked 34th.  Raising ``--max-count``
does not fix it: that file defines 60-odd ``HTTP_*`` constants, so any per-file
budget is spent on the quoted ``"HTTP_XXX",`` lines first -- measured across
max-count 3/5/8/12/20, the gold's keyword set stayed ``{'http'}`` and its rank
stayed 34-42.

The fix searches identifiers apart from subtokens, so the identifier's matches do
not compete with the common token for the per-file budget.  On the real repo that
moved the gold from rank 34 to rank 1 of 98, and the ladder's ``content`` rung for
class A from 42/86 to 84/86.

These tests build the same shape synthetically so they fail without a starlette
checkout: a file whose header matches the common token and whose body defines the
identifier, plus a distractor that matches the common token only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The shape of ``starlette/status.py``: header comments that match only the
#: common token, then many quoted names, then the definition the request wants.
STATUS_FILE = """\
# HTTP codes
#
# See the HTTP Status Code Registry at
# https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml
#
# And RFC 9110 - https://www.rfc-editor.org/rfc/rfc9110

HTTP_100_CONTINUE = 100
HTTP_101_SWITCHING_PROTOCOLS = 101
HTTP_102_PROCESSING = 102
HTTP_103_EARLY_HINTS = 103
HTTP_200_OK = 200
HTTP_201_CREATED = 201
HTTP_204_NO_CONTENT = 204
HTTP_304_NOT_MODIFIED = 304
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE = 431
HTTP_502_BAD_GATEWAY = 502
HTTP_503_SERVICE_UNAVAILABLE = 503
HTTP_505_HTTP_VERSION_NOT_SUPPORTED = 505
"""

#: A distractor that matches the common token ``http`` many times and the
#: identifier never: the shape of the 97 files that used to outrank the gold.
DISTRACTOR_FILE = """\
import http.client


def send_http(request: str) -> str:
    \"\"\"Send an http request and return the http response.\"\"\"
    client = http.client.HTTPConnection("localhost")
    client.request("GET", "/")
    resp = client.getresponse()
    return resp.read().decode("utf-8")


class HttpMiddleware:
    def __init__(self, http_client):
        self.http = http_client
"""


def _write_repo(root: Path) -> None:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "status.py").write_text(STATUS_FILE, encoding="utf-8")
    (src / "client.py").write_text(DISTRACTOR_FILE, encoding="utf-8")


@pytest.mark.asyncio
async def test_the_defining_line_is_not_truncated_by_header_matches(tmp_path: Path):
    """The file that defines the identifier must outrank a file that only mentions
    the common subtoken, even when the header lines match first."""
    from sylphae.agent.tools import ripgrep_search

    _write_repo(tmp_path)
    matches = await ripgrep_search(
        ["http_502_bad_gateway", "gateway", "http", "bad", "502", "503"],
        tmp_path, max_matches=15,
    )
    paths = [m.path for m in matches]
    assert "src/status.py" in paths, f"the defining file was not retrieved at all: {paths}"
    assert paths[0] == "src/status.py", (
        f"the defining file should rank first, got {paths[:3]}"
    )


@pytest.mark.asyncio
async def test_the_file_keeps_the_identifier_as_a_matched_keyword(tmp_path: Path):
    """The retrieval failure is invisible in the ranking layer, so the guard has to
    assert on the keyword evidence the ranker actually received.

    Before the fix ``src/status.py`` entered ranking with ``{'http'}`` only, because
    the ``--max-count`` budget was spent on the header lines.  The distractor then
    tied it on that single common term and won on order.  The fix is verified by
    checking that the identifier is now part of the file's evidence.
    """
    from sylphae.agent import tools as _tools

    captured: dict[str, set[str]] = {}
    original = _tools._rank_content_matches

    def spy(best_per_file, file_kws, position, max_matches):
        captured.update({p: set(k) for p, k in file_kws.items()})
        return original(best_per_file, file_kws, position, max_matches)

    _tools._rank_content_matches = spy
    try:
        _write_repo(tmp_path)
        await _tools.ripgrep_search(
            ["http_502_bad_gateway", "gateway", "http", "bad", "502", "503"],
            tmp_path, max_matches=15,
        )
    finally:
        _tools._rank_content_matches = original

    assert "src/status.py" in captured, "the defining file never reached ranking"
    kws = captured["src/status.py"]
    assert "http_502_bad_gateway" in kws, (
        f"the identifier was truncated away; status.py only matched {sorted(kws)}"
    )


@pytest.mark.parametrize(
    ("keyword", "expected"),
    [
        ("http_502_bad_gateway", True),
        ("DEFAULT_TIMEOUT", True),
        ("parse_response_headers", True),
        ("HTTP_101_SWITCHING_PROTOCOLS", True),
        # Subtokens: common enough that searching them alone matches a large share
        # of a web framework's files, so they must stay in the shared group.
        ("http", False),
        ("bad", False),
        ("502", False),
        ("gateway", False),
        ("navbar", False),
        ("port", False),
    ],
)
def test_identifier_classification(keyword: str, expected: bool):
    from sylphae.agent.tools import _is_identifier_keyword

    assert _is_identifier_keyword(keyword) is expected


def test_the_identifier_group_is_not_the_whole_keyword_set():
    """If every keyword were an identifier the split would be a no-op, and the guard
    above could pass without the second invocation doing anything.  A request always
    carries subtokens alongside its identifier."""
    from sylphae.agent.tools import _is_identifier_keyword

    kws = ["http_502_bad_gateway", "gateway", "http", "bad", "502", "503"]
    ids = [k for k in kws if _is_identifier_keyword(k)]
    rest = [k for k in kws if not _is_identifier_keyword(k)]
    assert ids == ["http_502_bad_gateway"]
    assert rest and "http" in rest
