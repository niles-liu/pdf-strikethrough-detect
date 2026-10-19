"""Paths, hashing, JSONL, config and the guards every stage shares.

Records (PREREG.md, Records), all JSONL, one per line:

- A ground-truth page, in `data/gt/<set>.jsonl`: `page_id` (`<set>.<doc>.p<page>`), `set` (G0, G1,
  G2, G5), `split` (dev, test), `doc`, `page` (0-based), `cluster` (the statistics' unit: a G0
  page, a G1 document, a G2 bill, a G5 page), `pdf` (the source file relative to the benchmark
  root: a PDF, or on G5 the page's scan, a PNG) and `pdf_sha256`, and `words` in PyMuPDF's
  extraction order (G5: the transcription's), each `{text, bbox, struck, masked, furniture, line}`
  (+ `strata` on G0): `bbox` in page fractions (none on G5), `struck` a list of `[start, end)`
  character spans, `line` the word's `[block, line]` (G5: `[paragraph, line]`). G2's pages also
  name their XML (`xml`, `xml_sha256`) and G5's their transcription, the ground truth's sources.
- A response (one per attempt), in `runs/<run>/responses.jsonl`; see `harness.py`.
- A score (one per arm and page), in `runs/<run>/scores.jsonl`; see `score.py`.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pymupdf

# pymupdf4llm turns PyMuPDF's quad corrections off when it is imported, which moves word boxes by
# about a point. Every module imports this one, so every entry point sees the same boxes.
pymupdf.TOOLS.unset_quad_corrections(True)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GT = DATA / "gt"
INPUTS = DATA / "inputs"
RUNS = ROOT / "runs"
CONFIG = ROOT / "config"
PREREG1 = CONFIG / "prereg1.json"  # the data prereg-1 tags, written by `plans freeze`
PREREG2 = CONFIG / "prereg2.json"  # what prereg-2 tags, written by `plans freeze --stage 2`
PROMPTS = ROOT / "prompts"
SETS = ("G0", "G1", "G2", "G5")  # the page sets the harness runs


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rel(path: Path) -> str:
    """A path as stored in records: relative to the benchmark root, with forward slashes."""
    return path.resolve().relative_to(ROOT).as_posix()


def iter_jsonl(path: Path) -> Iterator[dict]:
    """The file's records. A last line a killed process left unfinished (no newline) is skipped,
    with a warning, even if it ends inside a character; a bad line anywhere else is an error."""
    if not path.exists():
        return
    lines = [line for line in path.read_bytes().split(b"\n")]
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line.decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            if k < len(lines) - 1:  # a complete line: the file is damaged
                raise
            print(f"warning: skipping the unfinished last line of {path}", file=sys.stderr)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Replace `path` atomically, so an interrupted write never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for attempt in range(5):  # Windows refuses while another process has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def append_jsonl(path: Path, row: dict) -> None:
    """Append one record durably. An unfinished last line (a killed write) is cut off first, so
    it never ends up in the middle of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.seek(0)
                data = f.read()
                f.truncate(data.rfind(b"\n") + 1)
                print(f"warning: cut the unfinished last line of {path}", file=sys.stderr)
        f.seek(0, os.SEEK_END)
        f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def config(name: str) -> dict:
    with (CONFIG / f"{name}.toml").open("rb") as f:
        return tomllib.load(f)


def seed(name: str) -> int:
    return config("seeds")[name]


def page_id(set_: str, doc: str, page: int) -> str:
    return f"{set_}.{doc}.p{page}"


def load_pages(*sets: str, split: str | None = None) -> list[dict]:
    """Ground-truth pages of the named sets, in file order, optionally of one split."""
    pages = [p for s in sets for p in read_jsonl(GT / f"{s}.jsonl")]
    if not pages:
        raise FileNotFoundError(f"no ground truth for {', '.join(sets)}; build it first")
    return [p for p in pages if split is None or p["split"] == split]


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else ""


def dirty(exclude: tuple[str, ...] = ()) -> str:
    """What `git status` lists under the benchmark's folder (git-ignored files apart), leaving out
    `exclude`, paths relative to the folder."""
    return git("status", "--porcelain", "--", ".", *(f":(exclude){p}" for p in exclude))


def harness_commit(exclude: tuple[str, ...] = ()) -> str:
    """The commit the code ran at, marked dirty when the benchmark's files, `exclude` apart,
    differ from it."""
    commit = git("rev-parse", "HEAD") or "unknown"
    return commit + ("-dirty" if dirty(exclude) else "")


def tag_holds(tag: str, path: str) -> bool:
    """Whether the commit `tag` names holds `path`, relative to the benchmark's folder."""
    out = subprocess.run(["git", "cat-file", "-e", f"{tag}:./{path}"], cwd=ROOT,
                         capture_output=True, check=False)  # fmt: skip
    return out.returncode == 0


def tags_in_force() -> list[str]:
    return [
        t for t in ("prereg-1", "prereg-2") if git("rev-parse", "-q", "--verify", f"refs/tags/{t}")
    ]


# what may change after prereg-2 without touching a rule: the price list in force, and the drops
# (the budget's other values, its caps and drop order, are compared with what prereg-2 froze)
UNFROZEN = ("config/prices.toml", "config/budget.toml")
BUDGET_CHANGED = "config/budget.toml (beyond `dropped`)"


def changed_since_prereg_2() -> list[str] | None:
    """The code, prompts and config that differ from the prereg-2 tag, committed or not (the
    price list apart, and the budget unless more than its `dropped` list differs from
    prereg2.json's); None before the tag exists. Recorded with scores and reports, so a change
    after the tag is visible, never silent (PREREG.md, Deviations)."""
    if "prereg-2" not in tags_in_force():
        return None
    out = git("diff", "--name-only", "--relative", "prereg-2", "--", "bench", "prompts", "config")
    changed = [f for f in out.splitlines() if f and f not in UNFROZEN]
    if PREREG2.exists():
        frozen = json.loads(PREREG2.read_text(encoding="utf-8")).get("budget", {})
        now = config("budget")
        if {k: v for k, v in now.items() if k != "dropped"} != (
            {k: v for k, v in frozen.items() if k != "dropped"}
        ):
            changed.append(BUDGET_CHANGED)
    return changed


def provenance() -> dict:
    """What a score or a report records about the code it ran at."""
    return {"commit": harness_commit(), "tags": tags_in_force(),
            "changed_since_prereg_2": changed_since_prereg_2()}  # fmt: skip


def guard_test_pages(pages: Iterable[dict]) -> None:
    """PREREG.md: no arm, free or paid, touches a test page before `prereg-2`."""
    if any(p["split"] == "test" for p in pages) and "prereg-2" not in tags_in_force():
        raise PermissionError("test pages are locked until the prereg-2 tag exists")


@contextmanager
def lock(path: Path) -> Iterator[None]:
    """An exclusive lock file, so a second run on the same stage stops instead of sending twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"{path} exists: another run holds this stage (delete it if none does)")
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def load_env() -> None:
    """Read the repo's git-ignored `.env` into the environment, without overriding what is set."""
    path = ROOT.parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))
