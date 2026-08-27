#!/usr/bin/env python3
"""Validate a benchmark submission (results JSON + raw log).

    python3 bench/validate.py [--json] results/<gpu>/<handle>-<date>.json [...]

Checks, in order:
  1. JSON Schema conformance (bench/schema.json; uses the `jsonschema`
     package when available, otherwise a built-in structural subset).
  2. Metric keys and units against bench/metrics.json; every value against
     that metric's hard `bounds` (error) and `warn_bounds` (warning).
  3. Raw-log cross-check: each reported number must appear verbatim in the
     sibling .raw.txt program output.
  4. Plausibility (bench/plausibility.py): hardware ceilings derived from
     the submission's fingerprint, cross-metric coherence, raw-log hardware
     evidence, and a full re-derivation of the doc from the raw log with
     the bench/suites.py parsers.

Findings come in two tiers: ERRORS (physically impossible, contradictory,
or malformed -- the submission is invalid) and WARNINGS (suspicious or
unverifiable -- surfaced for the human reviewer). Exit status is 0 iff no
file has errors; warnings never fail the run, so legacy submissions that
predate the hardware fingerprint keep validating. Run by CI on submission
pull requests and by scripts/bench_review.py (website repo) at review time;
--json emits [{"file", "errors", "warnings"}] for the latter.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import plausibility  # noqa: E402

SCHEMA = json.loads((BENCH / "schema.json").read_text())
METRICS = json.loads((BENCH / "metrics.json").read_text())


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def check_schema(doc: dict, errors: list[str]) -> None:
    try:
        import jsonschema
        try:
            jsonschema.validate(doc, SCHEMA)
        except jsonschema.ValidationError as e:
            fail(f"schema: {e.message} (at {'/'.join(map(str, e.path))})",
                 errors)
        return
    except ImportError:
        pass
    # Reduced built-in check (CI installs jsonschema; this is the fallback).
    for key in ("schema", "date", "environment", "code", "results", "raw"):
        if key not in doc:
            fail(f"schema: missing required key '{key}'", errors)
    if doc.get("schema") != 1:
        fail("schema: 'schema' must be 1", errors)
    gpu = doc.get("environment", {}).get("gpu", {})
    for key in ("name", "sm", "driver"):
        if key not in gpu:
            fail(f"schema: environment.gpu missing '{key}'", errors)


def check_metrics(doc: dict, errors: list[str],
                  warnings: list[str]) -> None:
    for key, meas in doc.get("results", {}).items():
        spec = METRICS.get(key)
        if spec is None:
            fail(f"metrics: unknown metric key '{key}'", errors)
            continue
        if meas.get("unit") != spec["unit"]:
            fail(f"metrics: {key} unit '{meas.get('unit')}' != "
                 f"'{spec['unit']}'", errors)
        lo, hi = spec["bounds"]
        wlo, whi = spec.get("warn_bounds", (lo, hi))
        values = ([meas["value"]] if "value" in meas
                  else list(meas.get("series", {}).values()))
        if not values:
            fail(f"metrics: {key} has neither value nor series", errors)
        for v in values:
            if not lo <= v <= hi:
                fail(f"metrics: {key} value {v} outside plausible "
                     f"[{lo}, {hi}] {spec['unit']}", errors)
            elif not wlo <= v <= whi:
                warnings.append(
                    f"metrics: {key} value {v} outside the expected "
                    f"[{wlo}, {whi}] {spec['unit']} -- reviewer, please "
                    f"sanity-check the hardware")


def _forms(v: float) -> list[str]:
    """String forms a parsed number may have taken in the program output."""
    out = [f"{v:g}"]
    if v == int(v):
        out += [str(int(v)), f"{v:.1f}", f"{v:.2f}"]
    else:
        out += [f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"]
    return out


def check_location(doc: dict, json_path: Path, errors: list[str]) -> None:
    """Inside a results/ tree, the parent directory must be the GPU slug."""
    parts = json_path.resolve().parts
    if "results" not in parts:
        return                          # ad-hoc validation of a local file
    name = doc.get("environment", {}).get("gpu", {}).get("name", "")
    slug = name.lower()
    for prefix in ("nvidia ", "geforce ", "tesla ", "quadro "):
        if slug.startswith(prefix):          # not str.removeprefix: Py3.8 ok
            slug = slug[len(prefix):]
    slug = "-".join(re.sub(r"[^a-z0-9]+", " ", slug).split())
    if json_path.parent.name != slug:
        fail(f"location: expected directory results/{slug}/ for GPU "
             f"'{name}', found '{json_path.parent.name}/'", errors)


def check_raw(doc: dict, raw: str, errors: list[str]) -> None:
    for key, meas in doc.get("results", {}).items():
        values = ([meas["value"]] if "value" in meas
                  else list(meas.get("series", {}).values()))
        for v in values:
            if not any(f in raw for f in _forms(v)):
                fail(f"raw: {key} value {v} not found in {doc['raw']}",
                     errors)


def validate(json_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        doc = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"unreadable: {e}"], warnings
    check_schema(doc, errors)
    check_metrics(doc, errors, warnings)
    check_location(doc, json_path, errors)
    raw_path = json_path.parent / doc.get("raw", "")
    if not raw_path.is_file():
        fail(f"raw: {doc.get('raw')} not found next to the JSON", errors)
    else:
        raw = raw_path.read_text(errors="replace")
        check_raw(doc, raw, errors)
        plausibility.check_all(doc, raw, errors, warnings)
    return errors, warnings


def main() -> None:
    args = sys.argv[1:]
    as_json = "--json" in args
    paths = [Path(p) for p in args if p != "--json"]
    if not paths:
        sys.exit(__doc__.strip())
    bad = 0
    report = []
    for p in paths:
        errors, warnings = validate(p)
        report.append({"file": str(p), "errors": errors,
                       "warnings": warnings})
        if not as_json:
            if errors:
                status = f"{len(errors)} error(s)"
                if warnings:
                    status += f", {len(warnings)} warning(s)"
            elif warnings:
                status = f"OK ({len(warnings)} warning(s))"
            else:
                status = "OK"
            print(f"{p}: {status}")
            for e in errors:
                print(f"  - {e}")
            for w in warnings:
                print(f"  ~ {w}")
        bad += bool(errors)
    if as_json:
        print(json.dumps(report, indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
