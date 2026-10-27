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

# Suite name -> list of ProgramSpec (binary, args, stdout parser); merge
# folds parsed metrics and lives beside the parsers so plausibility.py can
# re-derive a doc from a raw log with identical folding.
from suites import (SUITES, SUITE_META, merge, run_fields,   # noqa: E402
                    FINGERPRINT_SPEC)                # (see bench/suites.py)


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


def smi_fingerprint(device: int) -> tuple[dict, str]:
    """Max clocks and PCIe link caps -- optional environment.gpu fields the
    plausibility checks derive hardware ceilings from. Returns (fields,
    raw-log block). One combined query: an old driver that rejects any of
    the fields fails the whole query, and absence is fine -- the submission
    just carries a reduced fingerprint."""
    fields = ("clocks.max.sm,clocks.max.memory,"
              "pcie.link.gen.max,pcie.link.width.max")
    cmd = ["nvidia-smi", f"--query-gpu={fields}",
           "--format=csv,noheader,nounits", "-i", str(device)]
    try:
        out = _run(cmd)
        vals = [f.strip() for f in out.split(",")]
        keys = ("clock_mhz", "mem_clock_mhz", "pcie_gen", "pcie_width")
        fp = {k: int(float(v)) for k, v in zip(keys, vals)
              if v and not v.startswith("[")}       # "[N/A]" -> omit field
        return fp, f"$ {' '.join(cmd)}\n{out}\n"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {}, ""


def c2c_present() -> bool:
    """True when the driver reports a chip-to-chip link, as on Grace Hopper
    and its successors, where the CPU/GPU path is NVLink-C2C rather than
    PCIe. Best effort: `nvidia-smi -q` grows a C2C section on those parts."""
    try:
        return bool(re.search(r"^\s*C2C(\s|:)", _run(["nvidia-smi", "-q"]),
                              re.M))
    except (OSError, subprocess.CalledProcessError):
        return False


def capture_fingerprint(bin_dir: Path, device: int, raw_log) -> dict:
    """The optional hardware fingerprint: nvidia-smi clock/PCIe caps plus
    cudaGetDeviceProperties fields from microbench/deviceProps. Both are
    echoed into the raw log as ordinary '$ command' blocks, making the
    fingerprint raw-log-backed evidence for bench/validate.py. Everything
    here is best-effort: a missing deviceProps binary or an old driver
    yields a reduced fingerprint, which validate treats as a warning."""
    fp, block = smi_fingerprint(device)
    if block:
        raw_log.write(block)
    exe = find_binary(bin_dir, FINGERPRINT_SPEC)
    if exe is None:
        print("  deviceProps not built -- submission will carry a "
              "reduced fingerprint")
        return fp
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": str(device)}
    raw_log.write(f"$ {exe.name}\n")
    try:
        proc = subprocess.run([str(exe)], capture_output=True, text=True,
                              timeout=FINGERPRINT_SPEC.timeout, check=True,
                              env=env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        out = (getattr(e, "output", "") or "") + \
              (getattr(e, "stderr", "") or "")
        raw_log.write(str(out) + f"\n[FAILED: {e}]\n\n")
        print("  deviceProps failed -- submission will carry a "
              "reduced fingerprint")
        return fp
    raw_log.write(proc.stdout + "\n")
    props = FINGERPRINT_SPEC.parse(proc.stdout, None)
    # nvidia-smi's name/sm/vram_mib/driver stay authoritative; validate.py
    # cross-checks them against the deviceProps raw block instead.
    fp.update({k: v for k, v in props.items()
               if k in ("sm_count", "bus_width_bits",
                        "l2_kib", "copy_engines")})
    # How host memory reaches the GPU. The transfer benchmarks are bounded by
    # this link, and it is not always PCIe: an integrated part (Tegra) shares
    # physical memory with the CPU, and a Grace-class host reaches a discrete
    # GPU over NVLink-C2C at several times any PCIe rate. Recording the link
    # keeps the plausibility ceilings from rejecting those platforms.
    if props.get("integrated"):
        fp["link"] = "integrated"
    elif c2c_present():
        fp["link"] = "c2c"
    elif "pcie_gen" in fp:
        fp["link"] = "pcie"
    return fp


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


def cpu_info() -> dict:
    """Host CPU, including its architecture: the launch-overhead and memcpy
    benchmarks are as much a measurement of the host as of the GPU, and an
    Arm host (Grace, Tegra) is not comparable with an x86 one."""
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if model in ("", "aarch64", "arm64", "unknown"):
        # Arm kernels usually omit "model name". The device tree names the
        # board on Tegra ("NVIDIA Jetson AGX Orin"), and Grace hosts report
        # a part number through the CPU implementer fields.
        try:
            dt = Path("/proc/device-tree/model").read_bytes()
            model = dt.decode("utf-8", "replace").strip("\x00").strip() or model
        except OSError:
            pass
    if model in ("", "aarch64", "arm64", "unknown"):
        try:
            fields = {}
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                key, sep, val = line.partition(":")
                if sep:
                    fields[key.strip()] = val.strip()
            part = fields.get("CPU part")
            impl = fields.get("CPU implementer")
            if part:
                model = (f"Arm CPU part {part}"
                         + (f" (implementer {impl})" if impl else ""))
        except OSError:
            pass
    info = {"model": model, "arch": platform.machine(),
            "cores": os.cpu_count()}
    try:  # kHz -> MHz; absent on some platforms, then omitted
        khz = int(Path("/sys/devices/system/cpu/cpu0/cpufreq"
                       "/cpuinfo_max_freq").read_text())
        info["max_mhz"] = round(khz / 1000)
    except (OSError, ValueError):
        pass
    return info


def ram_gib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1048576)
    except (OSError, ValueError):
        pass
    return None


def gpu_count() -> int:
    try:
        return len(_run(["nvidia-smi", "-L"]).splitlines())
    except (OSError, subprocess.CalledProcessError):
        return 1


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
    # results/ is data, not the code being measured — a fresh output file
    # overwriting a committed submission must not mark the run dirty.
    dirty = bool(_run(["git", "-C", str(REPO), "status",
                       "--porcelain", "-uno", "--", ".", ":!results"]))
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


def run_suites(names: list[str], bin_dir: Path, device: int,
               raw_log, n_gpus: int = 1) -> dict:
    # Most of the book's benchmarks have no --device switch; steer them all
    # to the chosen GPU via the environment. PCI_BUS_ID ordering keeps CUDA's
    # device numbering consistent with nvidia-smi's (used for gpu_info).
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": str(device)}
    results: dict = {}
    missing, failed, skipped = [], [], []
    for suite in names:
        for spec in SUITES[suite]:
            if spec.min_gpus > n_gpus:
                # Needing hardware this machine does not have is not a
                # failure: most submitters have one GPU, and the multi-GPU
                # programs exit non-zero rather than measuring nothing.
                skipped.append(f"{spec.binary} (needs {spec.min_gpus} GPUs)")
                continue
            exe = find_binary(bin_dir, spec)
            if exe is None:
                missing.append(spec.binary)
                continue
            for run in spec.runs:
                param, extra, need = run_fields(run)
                if need is not None and need > n_gpus:
                    # Name the run by its parameter, not its whole command
                    # line: a sweep skips a dozen runs on a small machine and
                    # a dozen command lines bury everything else.
                    label = (f"{param[0]}@{param[1]}"
                             if isinstance(param, tuple) else str(param))
                    skipped.append(f"{spec.binary} {label}")
                    continue
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
    if skipped:
        print(f"  skipped (hardware): {', '.join(skipped)}")
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
    groups = sorted({m["group"] for m in SUITE_META.values()})
    ap.add_argument("--suite", default="all",
                    help="comma-separated suite or group names, or 'all' "
                         f"(suites: {', '.join(sorted(SUITES))}; "
                         f"groups: {', '.join(groups)})")
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

    if args.suite == "all":
        suites = sorted(SUITES)
    else:
        suites, unknown = [], []
        for name in args.suite.split(","):
            if name in SUITES:
                suites.append(name)
            elif name in {m["group"] for m in SUITE_META.values()}:
                suites += sorted(s for s, m in SUITE_META.items()
                                 if m["group"] == name)
            else:
                unknown.append(name)
        if unknown:
            sys.exit(f"unknown suite(s)/group(s): {', '.join(unknown)}")

    env = {"gpu": gpu_info(args.device), "gpu_count": gpu_count(),
           "cuda": cuda_version(), "cpu": cpu_info(),
           "platform": args.plat, "os": os_name()}
    mem = ram_gib()
    if mem:
        env["ram_gib"] = mem
    print(f"GPU {args.device}: {env['gpu']['name']} (SM {env['gpu']['sm']}, "
          f"driver {env['gpu']['driver']}), CUDA {env['cuda']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix("").with_suffix(".raw.txt") \
        if args.out.suffix == ".json" else args.out.with_name(args.out.name + ".raw.txt")
    with open(raw_path, "w") as raw_log:
        env["gpu"].update(
            capture_fingerprint(args.bin_dir, args.device, raw_log))
        results = run_suites(suites, args.bin_dir, args.device, raw_log,
                             env["gpu_count"])
    if not results:
        if env["gpu_count"] < 2 and all(
                s.min_gpus > env["gpu_count"]
                for name in suites for s in SUITES[name]):
            sys.exit("nothing to measure: every program in the selected "
                     "suite(s) needs more GPUs than this machine has")
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
