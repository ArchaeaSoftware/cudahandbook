"""Benchmark suite specifications: which programs run.py executes, with what
canonical arguments, and how each program's stdout parses into metrics.

Metric keys are the contract with bench/metrics.json and with the rendering
pipeline on cudahandbook.com — treat them as append-only. Canonical arguments
are likewise part of the contract: results are only comparable across
machines because everyone runs the same workload.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Callable


@dataclass
class ProgramSpec:
    subdir: str          # source (and CMake binary) directory in the repo
    binary: str          # executable name
    runs: list           # [(param, [extra argv...])] — param is passed to parse
    parse: Callable      # (stdout: str, param) -> {metric key: measurement}
    timeout: int = 600   # seconds, per run


_NUM = r"(\d+(?:\.\d+)?)"


# ------------------------------------------------------------------ histogram
# One invocation per --random <values> count; each reports every GPU variant.
# The values sweep reproduces the rows of the book's Tables 16-3 and 16-4.

HIST_VALUES = [256, 128, 64, 32, 16, 8, 4, 2, 1]

_HIST_NAMES = {
    "GPUhistogramPerGrid":                    "histogram.per-grid",
    "GPUhistogramPerBlock":                   "histogram.per-block",
    "GPUhistogramPerBlock4x":                 "histogram.per-block-4x",
    "GPUhistogramPerBlockOffset":             "histogram.per-block-offset",
    "GPUhistogramPerBlock4xOffset":           "histogram.per-block-4x-offset",
    "GPUhistogramPerBlockReduce":             "histogram.per-block-reduce",
    "GPUhistogramPerBlockReduceOffset":       "histogram.per-block-reduce-offset",
    "GPUhistogramPerThread64":                "histogram.per-thread-64",
    "GPUhistogramPerThread4x64":              "histogram.per-thread-4x64",
    "GPUhistogramPerThread4x64_PeriodicMerge": "histogram.per-thread-4x64-periodic-merge",
    "GPUhistogramPerThread4x32":              "histogram.per-thread-4x32",
    "GPUhistogramPerThread4x32_PeriodicMerge": "histogram.per-thread-4x32-periodic-merge",
    "GPUhistogramNPP":                        "histogram.npp",
}
_HIST_LINE = re.compile(rf"^([A-Za-z0-9_]+): {_NUM} Mpix/s", re.M)
_HIST_ST = re.compile(rf"^Single-threaded: {_NUM} Mpix/s", re.M)
_HIST_MT = re.compile(rf"^Multithreaded \(\d+ cores\) is {_NUM}x faster "
                      rf"\({_NUM} Mpix/s\)", re.M)


def parse_histogram(out: str, values: int) -> dict:
    res = {}

    def put(key, v):
        res[key] = {"unit": "Mpix/s", "series": {str(values): float(v)}}

    m = _HIST_ST.search(out)
    if m:
        put("histogram.cpu.single", m.group(1))
    m = _HIST_MT.search(out)
    if m:
        put("histogram.cpu.multi", m.group(2))
    for m in _HIST_LINE.finditer(out):
        key = _HIST_NAMES.get(m.group(1))
        if key:                       # unmapped variants stay in the raw log
            put(key, m.group(2))
    return res


# --------------------------------------------------------------------- nbody
# Must pass --iterations (0 means run until keypress). Without a tty, nbody
# reports one algorithm per run (space cycles them interactively), so the
# parser keys metrics off whatever algorithm name each line reports.
# 65536 bodies is the canonical workload; --nocpu skips the (slow) CPU path.

_NBODY_LINE = re.compile(
    rf"([A-Za-z_][A-Za-z0-9_]*):\s*{_NUM} ms = \s*{_NUM}x10\^([69]) "
    rf"interactions/s")


def parse_nbody(out: str, param) -> dict:
    per_algo: dict[str, list[float]] = {}
    for m in _NBODY_LINE.finditer(out):
        # groups: (1) algorithm, (2) ms, (3) rate, (4) exponent 6|9
        giga = float(m.group(3)) * (1.0 if m.group(4) == "9" else 1e-3)
        per_algo.setdefault(m.group(1), []).append(giga)
    return {
        f"nbody.{name.lower().replace('_', '-')}": {
            "unit": "Ginteractions/s",
            "value": round(statistics.median(vals), 3),
        }
        for name, vals in per_algo.items()
    }


# -------------------------------------------------------------------- memcpy
# Small-transfer latency (the ch6 numbers), breakeven sweeps (Figs 6-4/6-5),
# and pinned bandwidth (Table 5-1's pinned column; the pageable column needs
# concurrency/pageableMemcpyHtoD*.cu — a follow-up suite).

_US = re.compile(rf"{_NUM} us")
_ROW2 = re.compile(rf"^(\d+)\t{_NUM}\s*$", re.M)


def parse_null(key: str) -> Callable:
    def parse(out: str, param) -> dict:
        m = _US.search(out)
        return {key: {"unit": "us", "value": float(m.group(1))}} if m else {}
    return parse


def parse_breakeven(direction: str) -> Callable:
    key = f"memcpy.{direction}.breakeven"

    def parse(out: str, param) -> dict:
        series = {m.group(1): float(m.group(2)) for m in _ROW2.finditer(out)}
        return {key: {"unit": "us", "series": series}} if series else {}
    return parse


_BW_ROW = re.compile(rf"^\d+\t{_NUM}\t{_NUM}\s*$", re.M)


def parse_pinned_bandwidth(out: str, param) -> dict:
    # CUDA_VISIBLE_DEVICES pins the program to one GPU, so one row.
    m = _BW_ROW.search(out.split("Device\t", 1)[-1])
    if not m:
        return {}
    return {
        "memcpy.htod.pinned-bandwidth": {"unit": "GB/s",
                                         "value": float(m.group(1))},
        "memcpy.dtoh.pinned-bandwidth": {"unit": "GB/s",
                                         "value": float(m.group(2))},
    }


# --------------------------------------------------------------- concurrency
# The ch6 memcpy/kernel-overlap shmoo. The cycles range 4..64 step 4 matches
# the book's published table; times are ms, speedup = sequential/concurrent.

_CONC_COLS = ("htod", "kernel", "dtoh", "total", "concurrent", "speedup")
_CONC_ROW = re.compile(
    r"^(\d+)\t" + r"\t".join([_NUM] * 6) + r"\s*$", re.M)


def parse_concurrency_memcpy_kernel(out: str, param) -> dict:
    rows = _CONC_ROW.findall(out.split("Cycles\t", 1)[-1])
    if not rows:
        return {}
    res = {f"concurrency.memcpy-kernel.{c}":
           {"unit": "x" if c == "speedup" else "ms", "series": {}}
           for c in _CONC_COLS}
    for row in rows:
        for col, v in zip(_CONC_COLS, row[1:]):
            res[f"concurrency.memcpy-kernel.{col}"]["series"][row[0]] = float(v)
    return res


# -------------------------------------------------------------------- suites

SUITES: dict[str, list[ProgramSpec]] = {
    "histogram": [
        ProgramSpec("histogram", "histogram",
                    [(v, ["--random", str(v),
                          "--padWidth", "4096", "--padHeight", "4096"])
                     for v in HIST_VALUES],
                    parse_histogram, timeout=300),
    ],
    "nbody": [
        ProgramSpec("nbody", "nbody",
                    [(None, ["--numbodies", "64", "--iterations", "12",
                             "--nocpu"])],
                    parse_nbody, timeout=900),
    ],
    "memcpy": [
        ProgramSpec("concurrency", "nullHtoDMemcpyAsync", [(None, [])],
                    parse_null("memcpy.htod.null-latency"), timeout=300),
        ProgramSpec("concurrency", "nullDtoHMemcpySync", [(None, [])],
                    parse_null("memcpy.dtoh.null-latency-sync"), timeout=300),
        ProgramSpec("concurrency", "nullDtoHMemcpyAsync", [(None, [])],
                    parse_null("memcpy.dtoh.null-latency"), timeout=300),
        ProgramSpec("concurrency", "breakevenHtoDMemcpy", [(None, [])],
                    parse_breakeven("htod"), timeout=600),
        ProgramSpec("concurrency", "breakevenDtoHMemcpy", [(None, [])],
                    parse_breakeven("dtoh"), timeout=600),
        ProgramSpec("memory", "pinnedBandwidth",
                    [(None, ["--MB", "64", "--iterations", "100"])],
                    parse_pinned_bandwidth, timeout=300),
    ],
    "concurrency": [
        ProgramSpec("concurrency", "concurrencyMemcpyKernel",
                    [(None, ["--minCycles", "4", "--maxCycles", "64",
                             "--stepCycles", "4"])],
                    parse_concurrency_memcpy_kernel, timeout=1800),
    ],
}
