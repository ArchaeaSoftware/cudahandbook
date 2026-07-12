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


# -------------------------------------------------------------- launch/memcpy
# Chapter 6 microbenchmarks: kernel launch overhead, small-transfer latency,
# breakeven sweeps (Figs 6-4/6-5), pinned bandwidth (Table 5-1), and the
# book's staged pageable memcpy. (eventRecord/managedOverhead: future work.)

_US = re.compile(rf"{_NUM} us")
_MBS = re.compile(rf"{_NUM} MB/s")
_ROW2 = re.compile(rf"^(\d+)\t{_NUM}\s*$", re.M)


def parse_null(key: str) -> Callable:
    def parse(out: str, param) -> dict:
        m = _US.search(out)
        return {key: {"unit": "us", "value": float(m.group(1))}} if m else {}
    return parse


def parse_mbs(key: str) -> Callable:
    def parse(out: str, param) -> dict:
        m = _MBS.search(out)
        return {key: {"unit": "MB/s", "value": float(m.group(1))}} if m else {}
    return parse


# managedOverhead.cu: synchronous null-launch time with N pages of managed
# memory allocated — first section touches every page between launches
# (forcing migration), the section after "Without touching memory:" doesn't.
_PAGES_ROW = re.compile(rf"^\s*{_NUM} us \((\d+) pages\)", re.M)


def parse_managed_overhead(out: str, param) -> dict:
    touched_part, _, untouched_part = out.partition("Without touching memory:")
    res = {}
    for key, part in (("managed.launch-touched", touched_part),
                      ("managed.launch-untouched", untouched_part)):
        series = {m.group(2): float(m.group(1))
                  for m in _PAGES_ROW.finditer(part)}
        if series:
            res[key] = {"unit": "us", "series": series}
    return res


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
# The ch6 memcpy/kernel-overlap shmoo. Canonical range widened for modern
# GPUs (Nick, 2026-07-12) to match the modernized program defaults: the
# book's 4..64 sweep sits in the flat region on current hardware; times are
# ms, speedup = sequential/concurrent.

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


# ------------------------------------------------------------- global memory
# memory/global{Read,Write,Copy}.cu print, per operand size, an unroll x
# block-size bandwidth matrix whose rows end with maxBW and maxThreads.
# We report the peak bandwidth per operand size (the ch5 headline numbers).

_OPERAND_RE = re.compile(r"^Operand size: (\d+) byte", re.M)
_MATRIX_ROW = re.compile(rf"^\d+\t(?:{_NUM}\t)+(\d+)\s*$", re.M)


def parse_global_peak(key: str) -> Callable:
    def parse(out: str, param) -> dict:
        series = {}
        sections = _OPERAND_RE.split(out)
        # split() yields [preamble, size1, body1, size2, body2, ...]
        for size, body in zip(sections[1::2], sections[2::2]):
            peaks = [float(row.split("\t")[-2])
                     for row in body.splitlines()
                     if _MATRIX_ROW.match(row)]
            if peaks:
                series[size] = max(peaks)
        return {key: {"unit": "GB/s", "series": series}} if series else {}
    return parse


# -------------------------------------------------------------------- suites
#
# Suites are organized into groups; run.py --suite accepts suite names,
# group names, or 'all'. SUITE_META also drives the grouping on the
# cudahandbook.com/benchmarks index.

SUITE_META: dict[str, dict] = {
    "launch":      {"title": "Kernel launch overhead",
                    "group": "microbench", "chapter": 6},
    "memcpy":      {"title": "Memcpy latency and bandwidth",
                    "group": "microbench", "chapter": 6},
    "global":      {"title": "Global memory bandwidth",
                    "group": "microbench", "chapter": 5},
    "concurrency": {"title": "Copy/kernel overlap",
                    "group": "microbench", "chapter": 6},
    "histogram":   {"title": "Histograms",
                    "group": "apps", "chapter": 16},
    "nbody":       {"title": "N-body",
                    "group": "apps", "chapter": 14},
}

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
    "launch": [
        ProgramSpec("concurrency", "nullKernelAsync", [(None, [])],
                    parse_null("launch.async"), timeout=300),
        ProgramSpec("concurrency", "nullKernelSync", [(None, [])],
                    parse_null("launch.sync"), timeout=300),
        ProgramSpec("concurrency", "managedOverhead", [(None, [])],
                    parse_managed_overhead, timeout=1800),
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
        ProgramSpec("concurrency", "pageableMemcpyHtoD", [(None, [])],
                    parse_mbs("memcpy.htod.pageable-staged"), timeout=600),
    ],
    "global": [
        ProgramSpec("memory", "globalRead", [(None, [])],
                    parse_global_peak("global.read.peak"), timeout=1800),
        ProgramSpec("memory", "globalWrite", [(None, [])],
                    parse_global_peak("global.write.peak"), timeout=1800),
        ProgramSpec("memory", "globalCopy", [(None, [])],
                    parse_global_peak("global.copy.peak"), timeout=1800),
    ],
    "concurrency": [
        ProgramSpec("concurrency", "concurrencyMemcpyKernel",
                    [(None, ["--minCycles", "0", "--maxCycles", "4096",
                             "--stepCycles", "64"])],
                    parse_concurrency_memcpy_kernel, timeout=1800),
    ],
}
