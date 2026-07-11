#!/usr/bin/env python3
"""Run the CUDA Handbook benchmark suites and emit a results JSON.

Usage:
    python3 bench/run.py --bin-dir build --suite all --device 0 \
                         --out results/my-gpu/handle-2026-07-15.json

The JSON conforms to bench/schema.json and is what you submit (by pull
request, together with the .raw.txt log written alongside it) to have your
machine's numbers appear on https://cudahandbook.com/benchmarks/.

Requires: Python 3.8+, nvidia-smi on PATH, and a CMake build of this repo
(pass its binary directory via --bin-dir).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Suite name -> list of ProgramSpec (binary, args, stdout parser).
from suites import SUITES  # noqa: E402  (sibling module, see bench/suites.py)


# ---------------------------------------------------------------- environment

def _run(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True,
                          **kw).stdout.strip()


def gpu_info(device: int) -> dict:
    fields = "name,driver_version,memory.total,compute_cap"
    out = _run(["nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits", "-i", str(device)])
    name, driver, mem, cc = [f.strip() for f in out.split(",")]
    return {"name": name, "sm": cc, "vram_mib": int(float(mem)),
            "driver": driver}


def cuda_version() -> str:
    try:
        out = _run(["nvcc", "--version"])
        m = re.search(r"release (\d+\.\d+)", out)
        if m:
            return m.group(1)
    except (OSError, subprocess.CalledProcessError):
        pass
    # Fall back to the driver's reported CUDA version.
    out = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    return m.group(1) if m else "unknown"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def os_name() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def code_state() -> dict:
    commit = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", str(REPO), "status",
                       "--porcelain", "-uno"]))
    return {"commit": commit[:12], "dirty": dirty}


# ------------------------------------------------------------------- running

def find_binary(bin_dir: Path, spec) -> Path | None:
    """Locate a built benchmark: flat bin dir or per-source-dir subdirectory."""
    for cand in (bin_dir / spec.binary,
                 bin_dir / spec.subdir / spec.binary,
                 bin_dir / "bin" / spec.binary):
        if cand.is_file():
            return cand
    return None


def merge(results: dict, parsed: dict) -> None:
    """Fold one run's metrics in; series metrics accumulate across runs
    (e.g. histogram is invoked once per --random values count)."""
    for key, meas in parsed.items():
        prior = results.get(key)
        if prior is None:
            results[key] = meas
        elif ("series" in prior and "series" in meas
              and prior["unit"] == meas["unit"]):
            prior["series"].update(meas["series"])
        else:
            sys.exit(f"internal error: duplicate metric key '{key}'")


def run_suites(names: list[str], bin_dir: Path, device: int,
               raw_log) -> dict:
    # Most of the book's benchmarks have no --device switch; steer them all
    # to the chosen GPU via the environment. PCI_BUS_ID ordering keeps CUDA's
    # device numbering consistent with nvidia-smi's (used for gpu_info).
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": str(device)}
    results: dict = {}
    missing, failed = [], []
    for suite in names:
        for spec in SUITES[suite]:
            exe = find_binary(bin_dir, spec)
            if exe is None:
                missing.append(spec.binary)
                continue
            for param, extra in spec.runs:
                argv = [str(exe)] + extra
                print(f"  {suite}: {' '.join([exe.name] + extra)} ...",
                      flush=True)
                raw_log.write(f"$ {' '.join([exe.name] + extra)}\n")
                try:
                    proc = subprocess.run(argv, capture_output=True,
                                          text=True, timeout=spec.timeout,
                                          check=True, env=env)
                except (subprocess.CalledProcessError,
                        subprocess.TimeoutExpired) as e:
                    out = (getattr(e, "output", "") or "") + \
                          (getattr(e, "stderr", "") or "")
                    raw_log.write(str(out) + f"\n[FAILED: {e}]\n\n")
                    failed.append(spec.binary)
                    break
                raw_log.write(proc.stdout + "\n")
                try:
                    parsed = spec.parse(proc.stdout, param)
                except Exception as e:           # parser bug or format drift
                    raw_log.write(f"[PARSE ERROR: {e}]\n\n")
                    failed.append(spec.binary)
                    break
                merge(results, parsed)
    if missing:
        print(f"  skipped (not built): {', '.join(missing)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    return results


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", type=Path, required=True,
                    help="CMake build directory containing the benchmark binaries")
    ap.add_argument("--suite", default="all",
                    help="comma-separated suites, or 'all' "
                         f"(available: {', '.join(sorted(SUITES))})")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSON path; the raw log is written alongside "
                         "with a .raw.txt suffix")
    ap.add_argument("--name", help="your name, for credit on the site")
    ap.add_argument("--github", help="your GitHub handle, for credit")
    ap.add_argument("--platform", dest="plat", default="bare-metal",
                    help='e.g. "aws g6e.xlarge", "wsl2" (default: bare-metal)')
    ap.add_argument("--submit", action="store_true",
                    help="after the run, submit the results to "
                         "cudahandbook.com (requires an active membership)")
    args = ap.parse_args()

    suites = sorted(SUITES) if args.suite == "all" else args.suite.split(",")
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        sys.exit(f"unknown suite(s): {', '.join(unknown)}")

    env = {"gpu": gpu_info(args.device), "cuda": cuda_version(),
           "cpu": cpu_model(), "platform": args.plat, "os": os_name()}
    print(f"GPU {args.device}: {env['gpu']['name']} (SM {env['gpu']['sm']}, "
          f"driver {env['gpu']['driver']}), CUDA {env['cuda']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix("").with_suffix(".raw.txt") \
        if args.out.suffix == ".json" else args.out.with_name(args.out.name + ".raw.txt")
    with open(raw_path, "w") as raw_log:
        results = run_suites(suites, args.bin_dir, args.device, raw_log)
    if not results:
        sys.exit("no results collected — check --bin-dir and the build")

    doc = {
        "schema": 1,
        "date": datetime.date.today().isoformat(),
        "environment": env,
        "code": code_state(),
        "results": results,
        "raw": raw_path.name,
    }
    submitter = {k: v for k, v in
                 (("name", args.name), ("github", args.github)) if v}
    if submitter:
        doc = {**doc, "submitter": submitter}
        doc["results"], doc["raw"] = results, raw_path.name  # keep key order

    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n{len(results)} metrics -> {args.out}\n  raw log -> {raw_path}")
    print("Validate with: python3 bench/validate.py", args.out)

    if args.submit:
        from submit import submit
        submit(args.out)


if __name__ == "__main__":
    main()
