"""The benchmark's command line: `python -m bench <command>`, run from `benchmarks/frontier/`.

Each module adds its own commands through `register(subparsers)`; `python -m bench -h` lists them.
"""

import argparse
import importlib
import sys

from bench.common import load_env

MODULES = (
    "fetch", "g0", "g1", "g2", "g5", "render", "arms", "prompts", "plans", "harness", "audit",
    "report",
)  # fmt: skip


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: C1′, ⟦ ⟧
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True, metavar="command")
    for name in MODULES:
        importlib.import_module(f"bench.{name}").register(sub)
    args = parser.parse_args()
    load_env()
    args.func(args)


if __name__ == "__main__":
    main()
