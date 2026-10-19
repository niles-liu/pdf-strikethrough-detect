"""The `[markdown]` extra: clean_markdown() through pymupdf4llm.

Nothing else in the suite calls clean_markdown(), so when pymupdf4llm 2.0 became an empty
placeholder (PyMuPDF 2.0 ships the module), the extra broke on Python 3.10, where 2.0 cannot
install, and CI stayed green. These tests skip where pymupdf4llm is not installed; on Python 3.11+
PyMuPDF 2.0 installs it even without the extra.
"""
import importlib
import importlib.metadata
import subprocess
import sys

import pytest

import pdf_strikethrough as st


def _pymupdf4llm_installed():
    try:
        importlib.metadata.distribution("pymupdf4llm")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_markdown = pytest.mark.skipif(not _pymupdf4llm_installed(),
                                    reason="pymupdf4llm (the [markdown] extra) is not installed")


def _struck_pdf(path):
    """One line of text with a vector strike through 'deleted text', saved to `path`."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "keep this deleted text here", fontsize=12)
    words = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}
    r1, r2 = words["deleted"], words["text"]
    ymid = (r1.y0 + r1.y1) / 2
    page.draw_line(pymupdf.Point(r1.x0 - 1, ymid), pymupdf.Point(r2.x1 + 1, ymid), width=1.0)
    doc.save(path)
    doc.close()
    return path


@needs_markdown
def test_installed_pymupdf4llm_imports():
    """pymupdf4llm 2.0 holds no module (PyMuPDF 2.0 ships it), so beside a PyMuPDF below 2.0 the
    import fails and clean_markdown() asks for the extra that is already installed. The extra's
    markers in pyproject.toml keep that pair from resolving; this catches an environment where it
    did."""
    importlib.import_module("pymupdf4llm")


@needs_markdown
def test_clean_markdown_drops_the_struck_span(tmp_path):
    md = st.clean_markdown(str(_struck_pdf(tmp_path / "struck.pdf")))
    assert "keep this" in md and "here" in md
    assert "deleted" not in md and "~~" not in md


@needs_markdown
def test_clean_markdown_writes_nothing_to_stdout(tmp_path):
    """pymupdf4llm's parser banner ("=== Document parser messages ===", on every call under
    PyMuPDF 2.0 without RapidOCR) goes to stderr. Run in a subprocess: pymupdf.message() writes to
    the stream PyMuPDF captured at import, which pytest's capture fixtures do not see."""
    pdf = _struck_pdf(tmp_path / "struck.pdf")
    run = subprocess.run(
        [sys.executable, "-c",
         "import sys, pdf_strikethrough as st; st.clean_markdown(sys.argv[1])", str(pdf)],
        capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    assert run.stdout == ""
