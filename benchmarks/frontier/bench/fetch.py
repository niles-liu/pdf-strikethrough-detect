"""Downloads: G1's corpus, G2's bills and G5's BRESSAY, each pinned by sha256.

    python -m bench fetch corpus
    python -m bench fetch bills --since 2026-07-01
    python -m bench fetch bressay

`corpus` fetches the manifest's PDFs into `data/corpus/`, skipping a file already present with the
right hash and deleting a download whose hash differs (its source has changed). `bills` lists the
BILLS collection in GovInfo's bulk data (no API key), keeps the Senate reported versions (`rs`)
issued on or after `--since` whose XML marks struck text, and saves each one's XML and PDF to
`data/bills/` with a sha256 manifest. House reported versions leave struck text out, so they are
skipped. Choosing the bills and aligning XML to PDF is `g2.py`'s job. `bressay` downloads
BRESSAY's archive (1.5 GB, resuming an interrupted download) into `data/bressay/`, checks it
against Zenodo's md5 and the sha256 pinned here, and extracts the test split's page images and
transcriptions with a sha256 manifest; the archive stays for anything else that reads it.
"""

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bench.common import DATA, sha256_bytes, sha256_file, write_jsonl
from bench.corpus import CORPUS, MANIFEST

# copyright.gov and gretnala.com refuse urllib's default User-Agent.
USER_AGENT = (
    "strikethrough-benchmark research (+https://github.com/niles-liu/pdf-strikethrough-detect)"
)
GOVINFO = "https://www.govinfo.gov"
SENATE_REPORTED = re.compile(r"BILLS-\d+([a-z]+?\d+)rs\.xml$")
DC_DATE = "{http://purl.org/dc/elements/1.1/}date"
BILLS = DATA / "bills"
BRESSAY = DATA / "bressay"
BRESSAY_URL = "https://zenodo.org/api/records/11637681/files/bressay.zip/content"
BRESSAY_BYTES = 1_496_017_418
BRESSAY_MD5 = "e4a4093304efff8322b605c8b79aa957"  # as Zenodo lists the file
BRESSAY_SHA256 = "7731d7ff4c08b0fc238058e95c254aac4f9d59913985247abb0992829b77cdc4"


def get(url: str, accept: str = "*/*", tries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def fetch_corpus() -> list[str]:
    """Download G1's PDFs; returns the files that failed."""
    CORPUS.mkdir(parents=True, exist_ok=True)
    failed = []
    for entry in MANIFEST["pdfs"]:
        path = CORPUS / entry["file"]
        if path.exists() and sha256_file(path) == entry["sha256"]:
            continue
        try:
            path.write_bytes(get(entry["url"], tries=1))
        except OSError as e:
            print(f"failed  {entry['file']}: {e}", file=sys.stderr)
            failed.append(entry["file"])
            continue
        if sha256_file(path) != entry["sha256"]:
            path.unlink()
            print(f"failed  {entry['file']}: sha256 differs from the manifest", file=sys.stderr)
            failed.append(entry["file"])
        else:
            print(f"fetched {entry['file']}")
    return failed


def _listing(path: str) -> list[dict]:
    return json.loads(get(f"{GOVINFO}/bulkdata/json/{path}", "application/json"))["files"]


def _candidates(congress: int, session: int, since: dt.date) -> list[dict]:
    """Senate reported versions whose bulk file changed on or after `since`. A version issued
    since then cannot have changed earlier; its issue date itself comes from the XML."""
    root = f"BILLS/{congress}/{session}"
    files = [
        f for kind in _listing(root) if kind["folder"] for f in _listing(f"{root}/{kind['name']}")
    ]
    return [
        f
        for f in files
        if SENATE_REPORTED.match(f["name"])
        and dt.datetime.strptime(f["formattedLastModifiedTime"], "%d-%b-%Y %H:%M").date() >= since
    ]


def describe(xml: bytes) -> dict:
    """A bill version's issue and introduction dates, and the struck text its XML marks. A House
    bill's Senate version opens with its receipt in the Senate, not its introduction, so its
    introduction date is left unknown."""
    root = ET.fromstring(xml)
    house = root.findtext(".//legis-num", "").strip().startswith("H.")
    dates = [] if house else [d.get("date") for d in root.iter("action-date") if d.get("date")]
    struck = {"struck_words": 0, "struck_phrases": 0, "struck_elements": 0}

    def walk(el: ET.Element) -> None:
        if el.tag == "deleted-phrase" or el.get("changed") == "deleted":
            struck["struck_phrases" if el.tag == "deleted-phrase" else "struck_elements"] += 1
            struck["struck_words"] += len("".join(el.itertext()).split())
        else:
            for child in el:
                walk(child)

    walk(root)
    introduced = dt.datetime.strptime(dates[0], "%Y%m%d").date().isoformat() if dates else None
    return {
        "issued": root.findtext(f".//{DC_DATE}", "").strip(),
        "introduced": introduced,
        **struck,
    }


def fetch_bills(congress: int, session: int, since: dt.date) -> list[dict]:
    records, files = [], {}
    for f in _candidates(congress, session, since):
        xml = get(f["link"])
        info = describe(xml)
        if info["issued"] < since.isoformat() or not info["struck_words"]:
            continue
        package = f["name"].removesuffix(".xml")
        pdf = get(f"{GOVINFO}/content/pkg/{package}/pdf/{package}.pdf")
        if not pdf.startswith(b"%PDF"):
            raise ValueError(f"{package}: the PDF link did not return a PDF")
        files[f"xml/{package}.xml"], files[f"pdf/{package}.pdf"] = xml, pdf
        records.append(
            {
                "package": package,
                "bill": SENATE_REPORTED.match(f["name"]).group(1),
                **info,
                "xml_sha256": sha256_bytes(xml),
                "pdf_sha256": sha256_bytes(pdf),
            }
        )
    shutil.rmtree(BILLS, ignore_errors=True)
    for rel_path, body in files.items():
        (BILLS / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (BILLS / rel_path).write_bytes(body)
    write_jsonl(BILLS / "manifest.jsonl", sorted(records, key=lambda r: r["package"]))
    return records


def download(url: str, path: Path, size: int, tries: int = 30) -> None:
    """Stream `url` to `path` through a `.part` file, resumed by byte range after an interruption
    (and by a rerun), so a slow 1.5 GB download never starts over."""
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={have}-"}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=120) as resp, part.open("ab") as f:  # fmt: skip
                if resp.status != 206:
                    raise ValueError(f"{url} ignored the byte range")
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            print(f"interrupted at {part.stat().st_size:,} bytes ({e}); resuming", file=sys.stderr)
            time.sleep(min(2**attempt, 60))
    if not part.exists() or part.stat().st_size != size:
        raise OSError(f"{url}: incomplete after {tries} tries; rerun to resume")
    os.replace(part, path)


def fetch_bressay() -> list[dict]:
    """BRESSAY's archive, checked against Zenodo's md5 and the pinned sha256, then its test
    split's page images and transcriptions extracted into `data/bressay/pages/`, with a sha256
    manifest whose first row is the archive's."""
    archive = BRESSAY / "bressay.zip"
    if archive.exists() and sha256_file(archive) != BRESSAY_SHA256:
        archive.unlink()  # damaged, or another file under its name
    if not archive.exists():
        download(BRESSAY_URL, archive, BRESSAY_BYTES)
        with archive.open("rb") as f:
            md5 = hashlib.file_digest(f, "md5").hexdigest()
        if md5 != BRESSAY_MD5 or sha256_file(archive) != BRESSAY_SHA256:
            archive.unlink()
            raise ValueError("bressay.zip differs from Zenodo's md5 or the pinned sha256")
    files = {}
    with zipfile.ZipFile(archive) as z:  # each member's CRC is checked as it is read
        files["test.txt"] = z.read("bressay/sets/test.txt")
        for name in files["test.txt"].decode("utf-8").split():
            for ext in ("png", "txt"):
                files[f"pages/{name}.{ext}"] = z.read(f"bressay/data/pages/{name}/{name}.{ext}")
    shutil.rmtree(BRESSAY / "pages", ignore_errors=True)
    for rel_path, body in files.items():
        (BRESSAY / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (BRESSAY / rel_path).write_bytes(body)
    rows = [{"file": "bressay.zip", "sha256": BRESSAY_SHA256, "md5": BRESSAY_MD5}]
    rows += [{"file": k, "sha256": sha256_bytes(v)} for k, v in files.items()]
    write_jsonl(BRESSAY / "manifest.jsonl", rows)
    return rows


def _corpus(args) -> None:
    failed = fetch_corpus()
    n = len(MANIFEST["pdfs"])
    print(f"{n - len(failed)} of {n} files ready in {CORPUS}")
    if failed:
        sys.exit(1)


def _bills(args) -> None:
    records = fetch_bills(args.congress, args.session, args.since)
    strict = {r["bill"] for r in records if (r["introduced"] or "") >= args.since.isoformat()}
    print(
        f"{len(records)} bills with struck text issued on or after {args.since}, {len(strict)} of "
        f"them also introduced since; {sum(r['struck_words'] for r in records):,} struck words "
        f"-> {BILLS}"
    )


def _bressay(args) -> None:
    rows = fetch_bressay()
    print(f"BRESSAY: {(len(rows) - 2) // 2} test pages, image and transcription -> {BRESSAY}")


def register(sub) -> None:
    p = sub.add_parser("fetch", help="download G1's corpus, G2's bills or G5's BRESSAY")
    what = p.add_subparsers(required=True, metavar="what")
    what.add_parser("corpus", help="G1's PDFs").set_defaults(func=_corpus)
    b = what.add_parser("bills", help="G2's Senate reported bills from GovInfo")
    b.add_argument("--since", type=dt.date.fromisoformat, default=dt.date(2026, 7, 1))
    b.add_argument("--congress", type=int, default=119)
    b.add_argument("--session", type=int, default=2)
    b.set_defaults(func=_bills)
    what.add_parser("bressay", help="G5's BRESSAY archive and its test pages").set_defaults(
        func=_bressay
    )
