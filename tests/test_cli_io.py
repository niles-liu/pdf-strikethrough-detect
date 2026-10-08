"""CLI input and output: what reaches stdout (only results, as UTF-8), when batch mode applies,
which arguments count as files, and how errors and interruptions end a run."""
import io
import json
import os
import subprocess
import sys
import zipfile

import pymupdf
import pytest

from pdf_strikethrough import __main__ as cli


def _struck_pdf(path, word="deleted"):
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), f"keep {word} text here", fontsize=12)
    r = {w[4]: w[:4] for w in page.get_text("words")}[word]
    y = (r[1] + r[3]) / 2
    page.draw_line((r[0], y), (r[2], y), width=1.0)
    doc.save(str(path))
    return path


def _docx(path, struck):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml",
                   f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}"><w:body>'
                   f'<w:p><w:r><w:rPr><w:strike/></w:rPr><w:t>{struck}</w:t></w:r></w:p>'
                   f'</w:body></w:document>')
    path.write_bytes(buf.getvalue())
    return path


# ----------------------------------------------------------------------- stdout is for results

def test_json_to_stdout_is_utf8_whatever_the_console_encoding(tmp_path):
    src = _docx(tmp_path / "greek.docx", "Ωμέγα→€")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}            # a redirected Windows stdout
    out = subprocess.run([sys.executable, "-m", "pdf_strikethrough", "detect", str(src),
                          "--json", "-"], capture_output=True, env=env, check=True).stdout
    assert json.loads(out.decode("utf-8"))["words"][0]["chars"] == "Ωμέγα→€"


def test_status_lines_go_to_stderr(tmp_path, capsys):
    src = _struck_pdf(tmp_path / "a.pdf")
    assert cli.main(["detect", str(src), "--markdown", str(tmp_path / "a.md"), "--json", "-"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["n_struck_final"] == 1          # nothing else on stdout
    assert "wrote struck-aware markdown" in captured.err


def test_human_output_numbers_pages_like_pages_flag(tmp_path, capsys):
    cli.main(["detect", str(_struck_pdf(tmp_path / "a.pdf"))])
    assert "  p1 " in capsys.readouterr().out                       # --pages is 1-based too


# ----------------------------------------------------------------------- batch mode

def test_jsonl_is_honored_for_a_folder_with_one_file(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    _struck_pdf(folder / "only.pdf")
    out = tmp_path / "out.jsonl"
    assert cli.main(["detect", str(folder), "--jsonl", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_with_one_literal_file_is_a_batch_of_one(tmp_path):
    out = tmp_path / "out.jsonl"
    cli.main(["detect", str(_struck_pdf(tmp_path / "a.pdf")), "--jsonl", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["n_struck_final"] == 1


def test_interrupted_batch_keeps_finished_lines(tmp_path, monkeypatch):
    for i in range(4):
        _struck_pdf(tmp_path / f"d{i}.pdf")
    real = cli._detect_payload
    done = []

    def payload(path, opts):
        if len(done) == 2:
            raise KeyboardInterrupt
        done.append(path)
        return real(path, opts)
    monkeypatch.setattr(cli, "_detect_payload", payload)
    out = tmp_path / "out.jsonl"
    assert cli.main(["detect", str(tmp_path), "--jsonl", str(out)]) == 130
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_unwritable_batch_output_fails_before_any_work(tmp_path, monkeypatch):
    _struck_pdf(tmp_path / "a.pdf")
    monkeypatch.setattr(cli, "_detect_payload", lambda *a: pytest.fail("ran before the check"))
    assert cli.main(["detect", str(tmp_path), "--jsonl", str(tmp_path / "no" / "x.jsonl")]) == 1


# ----------------------------------------------------------------------- what counts as a file

def test_filename_with_brackets_is_a_file_not_a_pattern(tmp_path):
    assert cli.main(["detect", str(_struck_pdf(tmp_path / "report [final].pdf"))]) == 0


def test_glob_keeps_supported_files_and_recurses(tmp_path):
    _struck_pdf(tmp_path / "a.pdf")
    (tmp_path / "notes.txt").write_text("not a document")
    (tmp_path / "sub").mkdir()
    _struck_pdf(tmp_path / "sub" / "b.pdf")
    assert cli._expand_inputs([str(tmp_path / "*")]) == [str(tmp_path / "a.pdf")]
    deep = cli._expand_inputs([str(tmp_path / "**" / "*.pdf")])
    assert sorted(os.path.basename(p) for p in deep) == ["a.pdf", "b.pdf"]


def test_stdin_cannot_join_other_inputs(tmp_path):
    assert cli._expand_inputs(["-", str(_struck_pdf(tmp_path / "a.pdf"))]) is None


# ----------------------------------------------------------------------- errors and exit codes

@pytest.mark.parametrize("argv", [["--method", "bogus"], ["--dpi", "0"], ["--jobs", "-1"]])
def test_usage_errors_exit_1(tmp_path, argv):
    with pytest.raises(SystemExit) as e:
        cli.main(["detect", str(_struck_pdf(tmp_path / "a.pdf")), *argv])
    assert e.value.code == 1                       # 2 means "encrypted / OCR required" here


def test_pages_past_the_end_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["detect", str(_struck_pdf(tmp_path / "a.pdf")), "--pages", "99"]) == 1
    assert capsys.readouterr().err.startswith("error: --pages: page 99 is past the end")


def test_unwritable_output_is_a_clean_error(tmp_path, capsys):
    src = _struck_pdf(tmp_path / "a.pdf")
    assert cli.main(["detect", str(src), "--json", str(tmp_path / "no" / "x.json")]) == 1
    assert "cannot write output" in capsys.readouterr().err


def test_unreadable_image_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 20)
    assert cli.main(["detect", str(bad), "--ocr", "none"]) in (1, 2)
    assert "Traceback" not in capsys.readouterr().err


# ----------------------------------------------------------------------- review follow-ups

def test_stdout_output_keeps_the_replace_error_handler(monkeypatch):
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="replace")
    monkeypatch.setattr(sys, "stdout", stream)
    with cli._open_out("-") as f:
        f.write("name_\udce9.pdf")             # an undecodable file name, as Python sees it
    assert stream.encoding == "utf-8" and stream.errors == "replace"


def test_failed_precondition_leaves_previous_output(tmp_path, monkeypatch):
    for i in range(2):
        _struck_pdf(tmp_path / f"d{i}.pdf")
    out = tmp_path / "out.jsonl"
    out.write_text('{"previous": "run"}\n', encoding="utf-8")

    def missing(name):
        raise SystemExit("--ocr tesseract requires the tesseract extra")
    monkeypatch.setattr(cli, "_check_ocr_available", missing)
    with pytest.raises(SystemExit):
        cli.main(["detect", str(tmp_path), "--jsonl", str(out)])
    assert out.read_text(encoding="utf-8") == '{"previous": "run"}\n'


def test_output_closed_mid_batch_is_a_clean_error(tmp_path, monkeypatch, capsys):
    import contextlib
    for i in range(3):
        _struck_pdf(tmp_path / f"d{i}.pdf")

    class ClosedPipe(io.StringIO):
        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")
    monkeypatch.setattr(cli, "_open_out", lambda path: contextlib.nullcontext(ClosedPipe()))
    assert cli.main(["detect", str(tmp_path), "--jsonl", "-"]) == 1
    assert "cannot write output" in capsys.readouterr().err


class _FakePool:
    """A process pool that runs work in-process and "crashes" on files named crash*.pdf, the way a
    native crash breaks a real pool: deterministic, and the same on every platform."""

    def __init__(self, max_workers=None):
        pass

    def submit(self, fn, item):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        fut = Future()
        if os.path.basename(item[0]).startswith("crash"):
            fut.set_exception(BrokenProcessPool("a worker died"))
        else:
            fut.set_result({"schema_version": 1, "source": item[0], "n_struck_final": 0})
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


def test_a_crashing_file_costs_one_error_line_with_jobs(tmp_path, monkeypatch):
    import concurrent.futures
    names = ["a.pdf", "crash.pdf", "b.pdf", "c.pdf"]
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _FakePool)
    paths = [str(tmp_path / n) for n in names]
    payloads = list(cli._iter_payloads(paths, {}, jobs=2))
    assert [os.path.basename(p["source"]) for p in payloads] == names
    assert [os.path.basename(p["source"]) for p in payloads if "error" in p] == ["crash.pdf"]


def test_image_too_large_to_decode_is_a_clean_error(tmp_path, monkeypatch, capsys):
    from PIL import Image
    big = tmp_path / "big.png"
    Image.new("L", (400, 400), 255).save(big)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)   # 160k px is far past 2x the limit
    assert cli.main(["detect", str(big), "--ocr", "none"]) == 1
    assert capsys.readouterr().err.startswith("error: cannot read")
