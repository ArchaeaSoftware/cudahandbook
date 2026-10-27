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
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass
class ProgramSpec:
    subdir: str          # source (and CMake binary) directory in the repo
    binary: str          # executable name
    runs: list           # [(param, [extra argv...])] — param is passed to parse
    parse: Callable      # (stdout: str, param) -> {metric key: measurement}
    timeout: int = 600   # seconds, per run
    min_gpus: int = 1    # skipped, not failed, on a machine with fewer


_NUM = r"(\d+(?:\.\d+)?)"


def run_fields(run):
    """A runs entry is (param, args), or (param, args, min_gpus) when that one
    run needs more devices than the rest of the program's runs. Keeping the
    shape in one place lets run.py and plausibility.py agree about it."""
    param, extra = run[0], run[1]
    need = run[2] if len(run) > 2 else None
    return param, extra, need


def merge(results: dict, parsed: dict) -> None:
    """Fold one run's metrics in; series metrics accumulate across runs
    (e.g. histogram is invoked once per --random values count). Shared by
    run.py (building a results doc) and plausibility.py (re-deriving one
    from a raw log) so both fold identically."""
    for key, meas in parsed.items():
        prior = results.get(key)
        if prior is None:
            results[key] = meas
        elif ("series" in prior and "series" in meas
              and prior["unit"] == meas["unit"]):
            prior["series"].update(meas["series"])
        else:
            sys.exit(f"internal error: duplicate metric key '{key}'")


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


# nbody reports one algorithm per non-interactive run, so each is selected by
# name with --algorithm. The list stops at GPU_Shuffle because that is where
# nbody.cu clamps g_maxAlgorithm, and it matches the nbody keys in
# bench/metrics.json. CPU algorithms are excluded by --nocpu: at 64K bodies
# they take far longer than the GPU paths.
# The single-device algorithms. The multi-GPU decompositions are measured by
# the scaling sweep below instead, which reports each one as a series keyed by
# device count: running them here as well would give two suites the same
# command line, and re-derivation maps a command line to one parser.
NBODY_ALGORITHMS = ["GPU_AOS", "GPU_Shared", "GPU_Const", "GPU_Shuffle"]


def _nbody_per_algo(out: str) -> dict:
    per_algo: dict[str, list[float]] = {}
    for m in _NBODY_LINE.finditer(out):
        # groups: (1) algorithm, (2) ms, (3) rate, (4) exponent 6|9
        giga = float(m.group(3)) * (1.0 if m.group(4) == "9" else 1e-3)
        per_algo.setdefault(m.group(1), []).append(giga)
    return per_algo


def parse_nbody(out: str, param) -> dict:
    per_algo = _nbody_per_algo(out)
    return _nbody_result(per_algo, statistics.median_low)


def _nbody_result(per_algo: dict, aggregate) -> dict:
    return {
        f"nbody.{name.lower().replace('_', '-')}": {
            "unit": "Ginteractions/s",
            "value": round(aggregate(vals), 3),
        }
        for name, vals in per_algo.items()
    }


def _parse_nbody_median(out: str, param) -> dict:
    """The aggregator used before 2026-10: an even sample count makes
    statistics.median interpolate between the two middle measurements, so the
    reported figure is a number the program never printed and cannot be found
    in the raw log. New runs report median_low, an actual measurement; this
    remains so that submissions made under the old contract still re-derive."""
    per_algo = _nbody_per_algo(out)
    return _nbody_result(per_algo, statistics.median)


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


# ---------------------------------------------------------------- multi-GPU
# memory/peer2peerMemcpy reports one line per directed pair:
#
#   Peer-to-peer memcpy... 4 devices detected
#   1 <- 0...18234.56 MB/s
#   2 <- 0...Not enabled
#
# The pairs cannot be a series: schema.json restricts series keys to numbers,
# and "1<-0" is not one. The matrix therefore stays in the raw log, where the
# chapter's topology discussion wants it anyway, and the metric reports the
# median over the directed pairs that P2P actually reached.

_P2P_PAIR = re.compile(rf"^(\d+) <- (\d+)\.\.\.(?:{_NUM} MB/s|(Not enabled))",
                       re.M)


def parse_peer2peer(out: str, param) -> dict:
    rates, enabled, total = [], 0, 0
    for m in _P2P_PAIR.finditer(out):
        total += 1
        if m.group(3) is not None:
            rates.append(float(m.group(3)) / 1000.0)   # MB/s -> GB/s
            enabled += 1
    if not total:
        return {}            # fewer than two devices: nothing to report
    res = {"p2p.pairs-enabled": {"unit": "pairs", "value": float(enabled)}}
    if rates:
        res["p2p.memcpy.bandwidth"] = {
            "unit": "GB/s", "value": round(statistics.median_low(rates), 2)}
    return res


# ------------------------------------------------------- multi-GPU scaling
# The scaling sweep is what a multi-GPU run is for: the same decomposition at
# 1, 2, 4 and 8 devices, so the curve shows what the decomposition's
# communication costs as devices are added. Each algorithm becomes one metric
# whose series is keyed by device count, the shape the managed-memory and
# bandwidth sweeps already use.
#
# A run is tagged with the device count it needs, so a four-GPU machine
# measures three points and skips the fourth rather than failing: nbody does
# not clamp --numgpus, it fails on the first device that is not there.

NBODY_SCALING_GPUS = [1, 2, 4, 8]

NBODY_SCALING_ALGORITHMS = [
    "multiGPU_SingleCPUThread",
    "multiGPU_MultiCPUThread",
    "multiGPU_Ring",
    "multiGPU_2D",
]


def parse_nbody_scaling(out: str, param) -> dict:
    """One point of one algorithm's scaling curve."""
    algorithm, n_gpus = param
    vals = _nbody_per_algo(out).get(algorithm)
    if not vals:
        return {}
    name = algorithm.lower().replace("_", "-")
    return {
        f"nbody.scaling.{name}": {
            "unit": "Ginteractions/s",
            "series": {str(n_gpus): round(statistics.median_low(vals), 3)},
        }
    }


NBODY_SCALING_RUNS = [
    ((algorithm, n),
     ["--numbodies", "64", "--iterations", "12", "--nocpu",
      "--numgpus", str(n), "--algorithm", algorithm],
     n)
    for algorithm in NBODY_SCALING_ALGORITHMS
    for n in NBODY_SCALING_GPUS
]

# --------------------------------------------------------------- fingerprint
# microbench/deviceProps prints the cudaGetDeviceProperties fields the
# plausibility checks need but nvidia-smi cannot report (bus width, L2 size,
# copy-engine count, SM count). Its printf contract is append-only — this
# parser and validate.py both depend on the exact "key: value" lines.
#
# Unlike the suite parsers, parse_device_props returns ENVIRONMENT field
# names (merged into doc["environment"]["gpu"] by run.py), not metric
# measurements — FINGERPRINT_SPEC is deliberately NOT in SUITES, so the
# program is never treated as a benchmark and its output never lands in
# doc["results"].

_PROPS_FIELDS = {
    "name":                ("name", str),
    "computeCapability":   ("sm", str),
    "multiProcessorCount": ("sm_count", int),
    "memoryBusWidth":      ("bus_width_bits", int),   # "192 bits"
    "l2CacheSize":         ("l2_kib", int),           # "2304 KiB"
    "asyncEngineCount":    ("copy_engines", int),
    "totalGlobalMem":      ("vram_mib", int),         # "12043 MiB"
    "integrated":          ("integrated", int),       # 0 | 1
}


def parse_device_props(out: str, param) -> dict:
    """deviceProps stdout -> environment.gpu field names. Lines that are
    missing (a future CUDA release dropping a property, as CUDA 13 dropped
    the clock fields) are simply omitted."""
    res = {}
    for line in out.splitlines():
        key, sep, val = line.partition(":")
        field = _PROPS_FIELDS.get(key.strip())
        if not sep or field is None:
            continue
        name, conv = field
        val = val.strip()
        if conv is int:                 # strip a units suffix ("192 bits")
            m = re.match(r"(\d+)", val)
            if not m:
                continue
            res[name] = int(m.group(1))
        else:
            res[name] = val
    return res


FINGERPRINT_SPEC = ProgramSpec("microbench", "deviceProps", [(None, [])],
                               parse_device_props, timeout=60)


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
    "multigpu":    {"title": "Multiple GPUs",
                    "group": "microbench", "chapter": 9},
    "histogram":   {"title": "Histograms",
                    "group": "apps", "chapter": 16},
    "nbody":       {"title": "N-body",
                    "group": "apps", "chapter": 14},
}

# Command lines retired from SUITES, kept so that results submitted under them
# still re-derive from their raw logs. Canonical arguments are part of the
# contract with past submissions as much as with future ones: when a suite's
# invocation changes, the old one moves here rather than disappearing.
LEGACY_RUNS: dict = {
    # nbody ran one algorithm per submission until --algorithm was added.
    "nbody --numbodies 64 --iterations 12 --nocpu": (_parse_nbody_median, None),
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
        # --numgpus 1 is pinned so that a multi-GPU machine reports the
        # same thing here that a single-GPU machine does.
        ProgramSpec("nbody", "nbody",
                    [(a, ["--numbodies", "64", "--iterations", "12",
                          "--nocpu", "--numgpus", "1", "--algorithm", a])
                     for a in NBODY_ALGORITHMS],
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
    "multigpu": [
        ProgramSpec("memory", "peer2peerMemcpy", [(None, [])],
                    parse_peer2peer, timeout=600, min_gpus=2),
        # The scaling sweep runs on one device too: the single-device point is
        # the baseline the curve is measured against.
        ProgramSpec("nbody", "nbody", NBODY_SCALING_RUNS,
                    parse_nbody_scaling, timeout=1800, min_gpus=1),
    ],
    "concurrency": [
        ProgramSpec("concurrency", "concurrencyMemcpyKernel",
                    [(None, ["--minCycles", "0", "--maxCycles", "4096",
                             "--stepCycles", "64"])],
                    parse_concurrency_memcpy_kernel, timeout=1800),
    ],
}
