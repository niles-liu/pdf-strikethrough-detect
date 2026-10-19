# Frontier benchmark

A pre-registered benchmark of whether frontier models find struck-through text as well as
`pdf-strikethrough-detect` 0.12.1, and at what cost. [RESULTS.md](RESULTS.md) holds the test
run's results, [PREREG.md](PREREG.md) the rules (frozen at the `prereg-2` tag),
[PLAN.md](PLAN.md) the reasons behind them, and [DEVIATIONS.md](DEVIATIONS.md) each change to a
rule since `prereg-1`. This folder also holds the harness and scorer that RESULTS.md lists as to
follow, and this README is the setup PREREG's Part B refers to. The pre-registrations and results
are archived on Zenodo under the concept DOI
[10.5281/zenodo.23425895](https://doi.org/10.5281/zenodo.23425895); the requests, responses,
scores and labels are in the Hugging Face dataset
[`niles-liu/strikethrough-benchmark`](https://huggingface.co/datasets/niles-liu/strikethrough-benchmark),
under `frontier/`.

## Known issues

Two scorer errors were found after results 1 was published. A tie between equal output tokens
ignores their marks: on five bills' first pages it gave a struck word the live copy in the page's
slug, so the package's PDF path seemed to leak 0.24% of the bills' struck text when it leaked none.
An answer that opens inside a struck passage, closing a strike it never opened, has every mark
after that closer reversed. Neither changes an endpoint's verdict, but the first changes the README
summary's clause for Claude Opus 5.5 (it left no less struck text live than the package, not less)
and two per-bill clauses. The fix and the rescored results will be published with the benchmark's
next results.

## What is here

| Path | What it holds |
|---|---|
| `bench/` | The code: downloads (`fetch`, `corpus`), ground truth (`g0`, `g1`, `g2`, `g5`), page inputs (`render`), the free arms (`arms`), requests (`prompts`, `plans`), one adapter per provider and a fake (`providers/`), sending and scoring (`harness`, `score`, `stats`), the hand audit (`audit`) and the tables (`report`). |
| `figures.py` | The leakage-against-cost scatter and the failure gallery. |
| `tests/` | Tests for every module but the downloads; they need no downloads and no keys. |
| `config/` | The models, prices, seeds, budget, the choices made on dev, the dev pages, and what each tag fixed (`prereg1.json`, `prereg2.json`). |
| `prompts/` | Every prompt template the run used. |
| `results/` | The tables behind RESULTS.md. |
| `corpus_manifest.json` | G1's source documents, each with its URL and sha256. |
| `requirements.in`, `requirements.txt` | The environment, every package pinned; the package under test comes from PyPI. |

G0 to G5 are the page sets (PREREG.md, Ground truth): G0 synthetic redlines, G1 public redlined
documents, G2 US Senate bills, G5 BRESSAY's handwritten pages. The free arms are the package's own
paths and two text-extraction baselines, which call no API. The code is under the repository's
MIT licence; the downloaded documents and datasets keep their own terms.

## The code against its pins

`config/prereg2.json` lists the sha256 of every code, prompt and config file as pre-registered
(`files`). The code came into this repository one commit per change;
`git log --oneline -- benchmarks/frontier/bench benchmarks/frontier/figures.py`, from the
repository root, lists them:

- **"the code as pre-registered (prereg-2)"** is the code `prereg-2` froze. Every file matches
  its sha256 except `config/budget.toml`, from which a comment on account spending limits was
  removed; its values are unchanged, and `prereg2.json` freezes them too (`budget`).
- **The changes since,** in order: deviation 4; the figures; and three parts of deviation 5 (the
  scorer; the audit page, with the audit's key errors into G1's relabel; the relabel refusing an
  ambiguous correction). Four more pinned files differ as a result, `bench/audit.py`,
  `bench/g1.py`, `bench/report.py` and `bench/score.py`: the same four the test run's provenance
  records list as changed since `prereg-2`.
- **What scored what:** the test run's stages (test, fable, qa, latency and the canaries) were
  scored with the code of "audit pages upright, ..."; the last commit, the relabel's refusal,
  changes no score. The dev stages behind RESULTS.md's choices (the ladder, paraphrases, repeats,
  ablation and flagship check) were scored before `prereg-2` with earlier code, so re-scoring them
  with this code gives different scores.

The published records name the commits of the repository the benchmark was developed in:
`config/prereg2.json` names `da4437f`, the code as pre-registered; the test run's responses name
`75574f6` (that code plus `prereg2.json`); its scores name `c29474f`, the code of "audit pages
upright, ...". To check a checkout against the pins, from this folder:

```bash
python -c "import hashlib, json, pathlib; pins = json.loads(pathlib.Path('config/prereg2.json').read_text(encoding='utf-8'))['files']; [print('same' if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == h else 'DIFFERS', p) for p, h in pins.items()]"
```

The repository's `.gitattributes` checks this folder out with no line-ending conversion, so the
hashes hold on Windows too.

## Setup

Clone the repository with its tags (a full `git clone`, not a ZIP or a shallow clone). On Windows,
clone to a short path such as `C:\src\`: onnxruntime's DLLs fail to load from deep ones. From the
repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv benchmarks/frontier/.venv --python 3.14
uv pip sync --python benchmarks/frontier/.venv benchmarks/frontier/requirements.txt
cd benchmarks/frontier
source .venv/bin/activate        # Git Bash: .venv/Scripts/activate; PowerShell: .venv\Scripts\Activate.ps1
python -m pytest tests           # about a minute
```

Everything below runs from `benchmarks/frontier/` in that environment, as
`python -m bench <module> <command>`; `python -m bench -h` lists the modules, and `-h` on a
command lists its options.

## Re-scoring the run

No keys are needed: the harness locks the test pages until the `prereg-2` tag exists, which the
full clone brings. The Hugging Face dataset's `frontier/<stage>/` folders mirror `runs/<stage>/`
here (`frontier/free/` holds the free arms' outputs), and the ground truth and page inputs rebuild
from the public sources. All of it takes about 75 minutes and 13 GB, 10 GB of them the page inputs.
Download the stages to re-score into `runs/`, at the revision the run published:

```bash
uvx --from huggingface_hub hf download niles-liu/strikethrough-benchmark --repo-type dataset \
  --revision ba024df9b38fd3849c6cfeb458bb4687a33442bc --include "frontier/*" --local-dir data/hf
mkdir -p runs data/gt && cp -r data/hf/frontier/test data/hf/frontier/free data/hf/frontier/latency runs/
rm runs/test/scores.jsonl runs/test/*.csv runs/test/endpoints.json runs/test/routes.txt \
  runs/latency/scores.jsonl          # the published outputs, so the comparison below is real
cp data/hf/frontier/test/g1-relabel/G1-corrections.jsonl data/gt/   # G1's relabel
```

Then rebuild; each `check` exits non-zero on failure:

```bash
python -m bench fetch corpus           # G1's documents
python -m bench fetch bills            # Senate reported bills since the cutoff: by now more than
                                       # the run had, and G2 takes the newest first, so keep only
                                       # the ones prereg-1 froze:
python -c "import json, pathlib; p = json.loads(pathlib.Path('config/prereg1.json').read_text(encoding='utf-8')); keep = {**p['g2_bills'], **p['g2_dev_bills']}; m = pathlib.Path('data/bills/manifest.jsonl'); rows = [x for x in m.read_text(encoding='utf-8').splitlines() if json.loads(x)['package'] in keep]; assert len(rows) == len(keep), 'a frozen bill is missing'; m.write_text(''.join(x + '\n' for x in rows), encoding='utf-8')"
python -m bench fetch bressay          # BRESSAY's archive (1.5 GB) and its test pages
python -m bench g0 build && python -m bench g0 check
python -m bench g1 build && python -m bench g1 check   # about 20 minutes, silent
python -m bench g2 build && python -m bench g2 check
python -m bench g5 build && python -m bench g5 check
python -m bench render build           # the page inputs, about 12 minutes
python -m bench harness score test     # about 20 minutes, silent; likewise fable, qa and the
                                       # canaries, once copied into runs/
python -m bench harness score latency  # the seconds per page the routes report
python -m bench report endpoints test > runs/test/endpoints.json   # each pre-registered claim
python -m bench report tables test     # every metric per arm, set and cluster
python -m bench report routes test > runs/test/routes.txt   # the routes per page kind; also
                                       # routes.csv and route_guards.csv
```

Then compare with what was published (the code writes CSVs with CRLF line endings, the repository
holds them with LF):

```bash
cmp runs/test/scores.jsonl data/hf/frontier/test/scores.jsonl && echo same scores
for f in results clusters strata calibration calibration_bins cascade routes route_guards; do
  diff -q --strip-trailing-cr runs/test/$f.csv results/test/$f.csv; done
```

A test page whose ground truth differs from what `prereg-2` froze is refused (G1's relabelled
pages apart, which DEVIATIONS.md covers), so a rebuild that scores at all scores the
pre-registered pages. Here the tags hold only the documents, not the code, so a re-score's
provenance lists every code and prompt file, and every config file but the price list, the budget
and the two prereg files, as changed since `prereg-2`; the pin check above is the comparison that
holds, and the endpoints' `provenance` block differs for the same reason.

A fresh clone of this repository, followed through this section with real downloads, reproduced
the run exactly: G0's pages and the 48 frozen bills (45 test, 3 dev) byte-identical, and every one
of the test run's 13,004 scores and the latency sample's 402, its endpoints, tables and routes the
same as published. In an earlier rehearsal all 11,833 test requests also rebuilt with the same IDs,
each hashing its prompt and its input's bytes, so `render build` remakes the inputs the models were
sent.

## Running it again

Paid calls read their keys from a `.env` at the repository root (git-ignored):

| Provider | Variables |
|---|---|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI, on Azure (GPT-6, on Global Standard deployments; the endpoint ends in `/openai/v1`) | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_GPT6_ASTRA`, `AZURE_OPENAI_DEPLOYMENT_SOL` |
| Google, on Vertex AI (Application Default Credentials; batches go through a Cloud Storage bucket) | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `BENCH_GCS_BUCKET` |
| xAI | `XAI_API_KEY` |

The stages and their order are PREREG's Part B. `python -m bench render build` builds the page
inputs, `python -m bench arms run` runs the free arms, and
`python -m bench harness dry-run <stage>` prices a stage from local files before anything is
sent; `harness submit` refuses a stage whose price passes the cap in `config/budget.toml`, and a
rerun sends only the requests without a final answer. The models pinned in `config/models.toml`
may since have been updated or retired: a response from another version is recorded as
`model_mismatch` and stops the stage until it is re-pinned.

G0's held-back pages are not here by design: only the sha256 of their seed is published, so a
model trained on the public pages can still be tested on pages it has never seen.
