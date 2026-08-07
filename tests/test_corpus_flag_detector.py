"""The flag detector against REAL PDFs, plus the version guard that stands in for them on CI.

`test_smoke.py` covers `native_flag_strikes` thoroughly, but every PDF it uses is drawn by fitz in
the test itself. Those synthetic pages did **not** catch this: PyMuPDF 1.26.3-1.26.5 raise a native
access violation inside `JM_make_textpage_dict` when `get_text("dict", ...)` is passed
`TEXT_COLLECT_VECTORS` -- the flag the strikeout detector requires -- and they do so on **page 0 of
every document in the public corpus**. The whole suite passed while the flag path crashed on real
input.

The gap was real-world PDF structure, so the first two tests point the detector at whatever corpus
is on disk. That corpus is git-ignored and re-downloadable, so they skip when it is absent -- which
means they guard a dev machine and not CI. `test_flag_detector_*_version_*` carry the CI half:
they pin the guard that turns the crash into an exception, and run everywhere.

⚠ A failure of the crash kind is a **process crash**, not an assertion: it takes the whole pytest
run down with SIGSEGV / 0xC0000005 and no traceback. If the suite dies partway with no report,
check the PyMuPDF version first (`pyproject.toml` documents the floor and why).
"""
from __future__ import annotations

import pathlib

import pytest

import pdf_strikethrough as st
from pdf_strikethrough import native

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "corpus"
PDFS = sorted(CORPUS.glob("*.pdf")) if CORPUS.is_dir() else []

needs_corpus = pytest.mark.skipif(
    not PDFS, reason="benchmarks/corpus/ not populated (git-ignored; see benchmarks/README.md)")


@needs_corpus
@pytest.mark.parametrize("pdf", PDFS or [None], ids=lambda p: p.stem if p else "none")
def test_flag_detector_survives_real_pdf(pdf):
    """The styled-span pass must complete on a real page rather than crashing the interpreter."""
    import pymupdf

    with pymupdf.open(pdf) as doc:
        out = st.native_flag_strikes(doc[0], 0)
    assert isinstance(out, list)
    # Records are the same shape the vector detector emits; a struck one is final.
    for rec in out:
        assert rec["tier"] == "flag"
        assert rec["final"] is True
        assert len(rec["bbox_frac"]) == 4


@needs_corpus
def test_flag_detector_survives_every_page_and_the_public_dispatch():
    """Page 0 is where the crash landed, but the guard should not depend on that -- nor on callers
    reaching the detector directly, since `method='both'` is how most of them do."""
    import pymupdf

    # The smallest corpus document, so this stays cheap; any real one exercises the same path.
    pdf = min(PDFS, key=lambda p: p.stat().st_size)
    with pymupdf.open(pdf) as doc:
        for i in range(doc.page_count):
            assert isinstance(st.native_flag_strikes(doc[i], i), list)
        assert isinstance(st.page_strikes(doc[0], 0, method="both"), list)


def test_installed_pymupdf_clears_the_flag_floor():
    """The environment running this suite must not be one of the crashing versions -- otherwise
    every corpus test above is a coin flip on whether pytest reports at all."""
    assert native._PYMUPDF_VERSION is not None, "could not parse the PyMuPDF version"
    assert native._PYMUPDF_VERSION >= native.FLAG_MIN_PYMUPDF


def test_flag_detector_refuses_a_crashing_pymupdf(monkeypatch):
    """On a bad version the detector must raise before extracting, naming the fix. This is the one
    behaviour that cannot be tested for real: the failure it prevents kills the process."""
    import pymupdf

    monkeypatch.setattr(native, "_PYMUPDF_VERSION", (1, 26, 3))
    doc = pymupdf.open()
    doc.new_page().insert_text((40, 60), "text")
    with pytest.raises(RuntimeError, match=r"1\.26\.6"):
        st.native_flag_strikes(doc[0], 0)
    # 'vector' and 'annot' never collect vectors, so the guard must not touch them
    assert st.page_strikes(doc[0], 0, method="vector") == []
    assert st.page_strikes(doc[0], 0, method="annot") == []
    doc.close()


@pytest.mark.parametrize("version", [(1, 26, 6), (1, 27, 0), (2, 0, 0)])
def test_flag_detector_accepts_versions_at_or_above_the_floor(monkeypatch, version):
    """A tuple compare, not a string one: "1.26.10" must not sort below "1.26.6"."""
    import pymupdf

    monkeypatch.setattr(native, "_PYMUPDF_VERSION", version)
    doc = pymupdf.open()
    doc.new_page().insert_text((40, 60), "text")
    assert st.native_flag_strikes(doc[0], 0) == []
    doc.close()


def test_unparseable_pymupdf_version_does_not_disable_the_detector(monkeypatch):
    """`_pymupdf_version` returning None means "unknown", which must fail open -- a version string
    this cannot read is not evidence of a bad build."""
    import pymupdf

    monkeypatch.setattr(native, "_PYMUPDF_VERSION", None)
    doc = pymupdf.open()
    doc.new_page().insert_text((40, 60), "text")
    assert st.native_flag_strikes(doc[0], 0) == []
    doc.close()


@pytest.mark.parametrize("raw,want", [
    ("1.26.6", (1, 26, 6)),
    ("1.26.10", (1, 26, 10)),
    ("1.28.2", (1, 28, 2)),
    ("1.26.6rc1", (1, 26, 6)),      # a suffix is not a reason to give up on the numbers
    ("", None),
    ("nonsense", None),
])
def test_pymupdf_version_parser(monkeypatch, raw, want):
    import pymupdf

    monkeypatch.setattr(pymupdf, "VersionBind", raw, raising=False)
    monkeypatch.setattr(pymupdf, "__version__", raw, raising=False)
    assert native._pymupdf_version() == want
