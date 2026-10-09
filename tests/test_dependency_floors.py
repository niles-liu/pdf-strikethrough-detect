"""CI's `lowest-bounds` job installs exactly the dependency floors pyproject.toml declares.

The job copies the floors into `.github/workflows/ci.yml` by hand and installs the package with
`--no-deps`, so pip never checks the copies against pyproject: a floor raised in one file and not
the other leaves the job testing versions the package no longer allows, or no longer testing its
floor. Both files are read with regexes: the suite supports Python 3.10, which has no tomllib, and
needs no YAML parser.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _source(rel):
    path = ROOT / rel
    if not path.is_file():
        pytest.skip("running outside a source checkout")
    return path.read_text(encoding="utf-8")


def test_lowest_bounds_installs_the_declared_floors():
    deps = re.search(r"^dependencies = \[(.*?)^\]", _source("pyproject.toml"), re.M | re.S)
    assert deps, "no [project] dependencies list in pyproject.toml"
    declared = {name.lower(): floor for name, floor
                in re.findall(r'"([A-Za-z0-9_.-]+)\s*>=\s*([0-9][0-9.]*)[^"]*"', deps.group(1))}
    job = re.search(r"^  lowest-bounds:\n(.*?)(?=^  [\w-]+:\n|\Z)",
                    _source(".github/workflows/ci.yml"), re.M | re.S)
    assert job, "no lowest-bounds job in .github/workflows/ci.yml"
    # a minor-level floor is pinned as its series ("1.24.*"), a patch-level one exactly
    pinned = {name.lower(): pin.removesuffix(".*") for name, pin
              in re.findall(r'"([A-Za-z0-9_.-]+)==([0-9][0-9.]*(?:\.\*)?)"', job.group(1))}
    assert declared and pinned == declared, (
        f"ci.yml's lowest-bounds job installs {pinned}, but pyproject.toml declares the floors "
        f"{declared}; keep the two equal")
