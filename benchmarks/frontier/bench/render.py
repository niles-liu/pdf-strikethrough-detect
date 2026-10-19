"""Page inputs (PREREG.md, Arms and inputs), built once per page into `data/inputs/<page_id>/`
and hashed in `data/inputs/manifest.jsonl`:

- `c1.pdf`, C1: the page as a single-page PDF, annotations kept. A tagged PDF's soft hyphens
  (U+00AD, held in its structure tree) extract as the printed "-", which the scorer treats alike.
- `c2.png`, C2: the page at 200 dpi, RGB, annotations rendered; `band-<k>.png`, C2-tiled.
- `text.txt`, C1′'s text block: the page's `page.get_text()`.
- `scan.pdf`, P-scan's input: C2's rendering in a PDF with no text layer.

A G5 page is a scan, so it has no C1, C1′ text or born-digital page: its C2 is the scan in RGB,
widened to 1,700 px (a US Letter page's width at 200 dpi, as every other set's C2) with Lanczos
resampling when narrower, its aspect kept; its bands and `scan.pdf` come from C2 as on any page.

    python -m bench render build [--set G0 G1 G2 G5]
    python -m bench render check      # stage B1 step 5

Each manifest row names the source PDF's sha256 and page, so inputs built from an earlier ground
truth are refused (`require_current`) rather than sent, as are test pages that differ from what
`plans freeze` froze for prereg-1.
"""

import functools
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pymupdf
from PIL import Image

from bench.common import (
    GT,
    INPUTS,
    PREREG1,
    PREREG2,
    ROOT,
    SETS,
    load_pages,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    tag_holds,
    tags_in_force,
    write_jsonl,
)

DPI = 200  # C2's resolution, and the package detector's calibration point
BAND, STEP = 1000, 850  # C2-tiled: bands 1,000 px tall, starting every 850 px
SCAN_WIDTH = 1700  # G5's C2 is at least this wide: US Letter at 200 dpi
MANIFEST = INPUTS / "manifest.jsonl"


def _scan_pdf(pages: list[tuple[pymupdf.Rect, bytes]]) -> bytes:
    """Each (page size, PNG) as an image-only page, so the package treats it as scanned:
    byte-identical on a rerun, and at the page's size, so page-fraction boxes carry over."""
    with pymupdf.open() as out:
        for rect, png in pages:
            scan = out.new_page(width=rect.width, height=rect.height)
            scan.insert_image(scan.rect, stream=png)
        # MuPDF otherwise writes a random /ID, so the same pages would hash differently.
        return out.tobytes(no_new_id=True)


def band_starts(height: int) -> list[int]:
    """Top rows of C2-tiled's bands: every 850 px, the last ending at the bottom edge."""
    if height <= BAND:
        return [0]
    starts = list(range(0, height - BAND, STEP))
    return starts + [height - BAND]


@functools.cache
def source_sha256(pdf: str) -> str:
    """A source PDF's sha256, read once per process."""
    return sha256_file(ROOT / pdf)


def check_sources(pages: list[dict]) -> None:
    """Refuse pages whose source PDF is no longer the one their ground truth was built from."""
    if changed := sorted({p["pdf"] for p in pages if source_sha256(p["pdf"]) != p["pdf_sha256"]}):
        raise ValueError(f"{changed[0]} differs from its sha256 in the ground truth "
                         f"({len(changed)} files); rebuild the ground truth")  # fmt: skip


def page_inputs(page: dict) -> dict[str, bytes]:
    check_sources([page])
    if page["set"] == "G5":
        png = scan_c2(ROOT / page["pdf"])
        with Image.open(io.BytesIO(png)) as image:
            rect = pymupdf.Rect(0, 0, image.width * 72 / DPI, image.height * 72 / DPI)
        files = {"c2.png": png}
    else:
        with pymupdf.open(ROOT / page["pdf"]) as src:
            p = src[page["page"]]
            png = p.get_pixmap(dpi=DPI).tobytes("png")
            text, rect = p.get_text(), p.rect
            with pymupdf.open() as one:  # not select(): it keeps the whole structure tree, 18 MB
                one.insert_pdf(src, from_page=page["page"], to_page=page["page"], annots=True)
                c1 = one.tobytes(no_new_id=True, garbage=3, deflate=True)
        files = {"c1.pdf": c1, "c2.png": png, "text.txt": text.encode("utf-8")}
    image = Image.open(io.BytesIO(png))
    for k, y in enumerate(band_starts(image.height)):
        buf = io.BytesIO()
        image.crop((0, y, image.width, min(y + BAND, image.height))).save(buf, "PNG")
        files[f"band-{k}.png"] = buf.getvalue()
    files["scan.pdf"] = _scan_pdf([(rect, png)])  # C2's pixels, with no text layer
    return files


def scan_c2(path: Path) -> bytes:
    """G5's C2: the scan in RGB, widened to SCAN_WIDTH with Lanczos when narrower."""
    with Image.open(path) as scan:
        image = scan.convert("RGB")
    if image.width < SCAN_WIDTH:
        size = (SCAN_WIDTH, round(image.height * SCAN_WIDTH / image.width))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def build(pages: list[dict], out: Path = INPUTS) -> list[dict]:
    rows = []
    for page in pages:
        files = page_inputs(page)
        folder = out / page["page_id"]
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        for name, data in files.items():
            (folder / name).write_bytes(data)
        with Image.open(io.BytesIO(files["c2.png"])) as image:
            size = list(image.size)
        rows.append(
            {
                "page_id": page["page_id"],
                "pdf_sha256": page["pdf_sha256"],
                "page": page["page"],
                "files": {name: sha256_bytes(data) for name, data in files.items()},
                "c2_px": size,
                "bands": [[y, min(y + BAND, size[1])] for y in band_starts(size[1])],
            }
        )
    return rows


@functools.cache
def manifest() -> dict[str, dict]:
    return {r["page_id"]: r for r in read_jsonl(MANIFEST)}


@functools.cache
def path(page_id: str, name: str) -> Path:
    """An input file, checked against its manifest hash before anything sends it (once per
    process: a stage builds thousands of requests from the same files)."""
    row = manifest().get(page_id)
    if row is None or name not in row["files"]:
        raise FileNotFoundError(
            f"no input {name} for {page_id}; run `python -m bench render build`"
        )
    file = INPUTS / page_id / name
    if sha256_file(file) != row["files"][name]:
        raise ValueError(f"{file} differs from its hash in {MANIFEST.name}")
    return file


def check_frozen(pages: list[dict]) -> None:
    """Once `plans freeze` has written what prereg-1 tags, refuse a test page it doesn't list, or
    whose sources (the PDF, and G2's XML or G5's transcription) are not the ones it froze; once
    `plans freeze --stage 2` has written what prereg-2 tags, refuse a test page whose ground truth
    differs from the one it froze, G1's relabelled pages apart (PREREG.md, What each tag fixes).
    A tag without its lists is refused too, never passed over, and so is a tag on a commit that
    doesn't hold them: they are committed first, then tagged."""
    tags = tags_in_force()
    if not PREREG1.exists():
        if tags:
            raise FileNotFoundError(f"{PREREG1.name} is missing, though prereg-1 is tagged")
        return
    frozen = json.loads(PREREG1.read_text(encoding="utf-8"))
    if not PREREG2.exists() and "prereg-2" in tags:
        raise FileNotFoundError(f"{PREREG2.name} is missing, though prereg-2 is tagged")
    for tag, lists in (("prereg-1", PREREG1), ("prereg-2", PREREG2)):
        if tag in tags and not tag_holds(tag, f"config/{lists.name}"):
            raise FileNotFoundError(f"the {tag} tag's commit doesn't hold config/{lists.name}: "
                                    "commit it, then tag that commit")  # fmt: skip
    truth = json.loads(PREREG2.read_text(encoding="utf-8"))["ground_truth"] if (
        PREREG2.exists()
    ) else None  # fmt: skip
    relabelled = {c["page_id"] for c in read_jsonl(GT / "G1-corrections.jsonl")}

    def sources(p: dict) -> bool:
        if p["set"] == "G2":
            return p["page_id"] in frozen["g2_pages"] and (
                frozen["g2_bills"].get(p["doc"]) == [p["pdf_sha256"], p["xml_sha256"]]
            )
        if p["set"] == "G5":
            return frozen.get("g5_pages", {}).get(p["page_id"]) == [
                p["pdf_sha256"], p["transcription_sha256"],
            ]  # fmt: skip
        if p["set"] == "G1":
            return p["page_id"] in frozen["g1_test_pages"] and (
                frozen["g1_documents"].get(p["doc"]) == p["pdf_sha256"]
            )
        return frozen["g0_test"].get(p["pdf"]) == p["pdf_sha256"]

    def labels(p: dict) -> bool:
        if truth is None or p["page_id"] in relabelled:
            return True
        return truth.get(p["page_id"]) == sha256_bytes(
            json.dumps(p, sort_keys=True, ensure_ascii=False).encode()
        )

    tests = [p for p in pages if p["split"] == "test"]
    if bad := [p["page_id"] for p in tests if not sources(p)]:
        raise ValueError(f"{len(bad)} test pages differ from {PREREG1.name} (first {bad[0]}): "
                         "run `plans freeze` again before prereg-1, or log a deviation after it")  # fmt: skip
    if bad := [p["page_id"] for p in tests if not labels(p)]:
        raise ValueError(f"{len(bad)} test pages' ground truth differs from {PREREG2.name} "
                         f"(first {bad[0]}): log a deviation, with the results both ways")  # fmt: skip


def require_current(pages: list[dict]) -> None:
    """Refuse pages whose source PDF has changed, that differ from what prereg-1 froze, or whose
    inputs are missing or were built from another source PDF or page."""
    check_sources(pages)
    check_frozen(pages)
    have = manifest()
    stale = [p["page_id"] for p in pages
             if (r := have.get(p["page_id"])) is None
             or (r.get("pdf_sha256"), r.get("page")) != (p["pdf_sha256"], p["page"])]  # fmt: skip
    if stale:
        raise ValueError(f"{len(stale)} pages' inputs are missing or stale (first {stale[0]}); "
                         f"run `python -m bench render build`")  # fmt: skip


def check(pages: list[dict]) -> bool:
    """Stage B1 step 5: identical hashes on a rebuild; C2 at 200 dpi (G5: its scan, widened); the
    bands cover every row and the last ends at the bottom edge."""
    have = manifest()
    with tempfile.TemporaryDirectory() as tmp:
        rows = build(pages, Path(tmp))
    ok = True
    for page, row in zip(pages, rows):
        problems = []
        if have.get(page["page_id"]) != row:
            problems.append("differs from the manifest")
        if page["set"] == "G5":
            with Image.open(ROOT / page["pdf"]) as scan:
                w, h = scan.size
            want = [max(w, SCAN_WIDTH), h if w >= SCAN_WIDTH else round(h * SCAN_WIDTH / w)]
        else:
            with pymupdf.open(ROOT / page["pdf"]) as src:
                rect = src[page["page"]].rect
            want = [round(rect.width * DPI / 72), round(rect.height * DPI / 72)]
        if any(abs(a - b) > 1 for a, b in zip(row["c2_px"], want)):
            problems.append(f"C2 is {row['c2_px']}, not {want} px")
        covered = set()
        for y0, y1 in row["bands"]:
            covered |= set(range(y0, y1))
        if covered != set(range(row["c2_px"][1])) or row["bands"][-1][1] != row["c2_px"][1]:
            problems.append("bands miss rows")
        if problems:
            ok = False
            print(f"FAIL {page['page_id']}: {'; '.join(problems)}")
    print(f"{'ok  ' if ok else 'FAIL'} {len(pages)} pages' inputs rebuild identically")
    return ok


def _build(args) -> None:
    pages = load_pages(*args.set)
    rows = {**manifest(), **{r["page_id"]: r for r in build(pages)}}
    current = {p["page_id"] for p in load_pages(*SETS)}
    for gone in set(rows) - current:  # pages a ground-truth rebuild no longer holds
        shutil.rmtree(INPUTS / gone, ignore_errors=True)
        del rows[gone]
    write_jsonl(MANIFEST, rows.values())
    manifest.cache_clear()
    path.cache_clear()
    print(f"inputs for {len(pages)} pages -> {INPUTS}")


def register(sub) -> None:
    p = sub.add_parser("render", help="page inputs: C1, C1′'s text, C2, C2-tiled, P-scan's scan")
    cmd = p.add_subparsers(required=True, metavar="action")
    b = cmd.add_parser("build")
    b.add_argument("--set", nargs="+", default=list(SETS))
    b.set_defaults(func=_build)
    c = cmd.add_parser("check", help="stage B1 step 5")
    c.add_argument("--set", nargs="+", default=list(SETS))
    c.set_defaults(func=lambda a: sys.exit(0 if check(load_pages(*a.set)) else 1))
