"""Plausibility checks for benchmark submissions (see bench/validate.py).

A submitter builds the binaries and runs them on hardware we never see, so
nothing here can *prove* a submission honest. What it can do is force a
fabricated one to be jointly self-consistent -- hardware fingerprint,
metrics, and raw log all agreeing -- which is far harder than editing a
number. Three families of checks, each reporting into two tiers:

  errors    -- physically impossible or internally contradictory; the
               submission is rejected.
  warnings  -- suspicious but possible (or unverifiable, e.g. a legacy
               submission with no fingerprint); surfaced for the human
               reviewer, who approves with an explicit override.

  1. Ceilings: hardware limits derived FROM the submission's own
     fingerprint (PCIe link caps, DRAM bandwidth, FP32 throughput).
  2. Coherence: cross-metric identities that real runs satisfy and
     fabricated tables rarely maintain.
  3. Raw-log evidence + re-derivation: the raw log is re-parsed with the
     same bench/suites.py parsers that produced the doc; every reported
     number must re-derive identically, and the hardware evidence the
     programs print must match the claimed environment.
"""
from __future__ import annotations

import re

import suites

# PCIe payload ceiling per lane, GB/s (encoding + protocol overhead already
# deducted: 8b/10b for gen1/2, 128b/130b for gen3+).
PCIE_GBPS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938,
                      6: 7.877}

# FP32 CUDA cores per SM by compute capability -- exact CC first, then the
# major version, else 128 (the common modern value).
CC_FP32_CORES = {
    "6.0": 64, "6.1": 128, "6.2": 128,
    "7.0": 64, "7.2": 64, "7.5": 64,
    "8.0": 64, "8.6": 128, "8.7": 128, "8.9": 128,
    "9.0": 128,
}
CC_MAJOR_FP32_CORES = {"5": 128, "6": 128, "7": 64, "8": 128, "9": 128,
                       "10": 128, "12": 128}

# nbody's body-body interaction is ~20 flops.
NBODY_FLOPS_PER_INTERACTION = 20

# histogram's canonical workload: --padWidth 4096 --padHeight 4096, 1 B/pixel.
HIST_IMAGE_BYTES = 4096 * 4096


def _values(meas: dict) -> list[float]:
    return ([meas["value"]] if "value" in meas
            else list(meas.get("series", {}).values()))


def _scalar(doc: dict, key: str) -> float | None:
    meas = doc.get("results", {}).get(key)
    return meas.get("value") if meas else None


# ------------------------------------------------------------------ ceilings

def _fp32_cores(cc: str) -> int | None:
    if cc in CC_FP32_CORES:
        return CC_FP32_CORES[cc]
    return CC_MAJOR_FP32_CORES.get(cc.split(".")[0])


def check_ceilings(doc: dict, errors: list, warnings: list) -> None:
    """Hardware ceilings derived from the submission's own fingerprint.
    Each sub-check is skipped when the fields it needs are absent (legacy
    submissions -- check_raw_evidence emits the one covering warning)."""
    gpu = doc.get("environment", {}).get("gpu", {})
    results = doc.get("results", {})

    # --- Host/device link: transfer bandwidth cannot beat the interconnect ---
    #
    # Only a PCIe-attached discrete GPU is bounded by the PCIe rate. An
    # integrated part (Tegra) shares physical memory with the CPU, and a
    # Grace-class host reaches its GPU over NVLink-C2C at several times any
    # PCIe generation, so applying the PCIe ceiling there would reject the
    # platform rather than a bad measurement. Treat the PCIe rate as a hard
    # ceiling only when the link is known to be PCIe; where the link is
    # unstated and the host is Arm -- every coherent-CPU NVIDIA platform so
    # far pairs the GPU with an Arm CPU -- report the excess for the reviewer
    # instead of rejecting it.
    gen, width = gpu.get("pcie_gen"), gpu.get("pcie_width")
    link = gpu.get("link")
    if link is None:
        link = "integrated" if gpu.get("integrated") else None
    arch = (doc.get("environment", {}).get("cpu", {}) or {}).get("arch", "")
    fatal = link == "pcie" or (link is None and gen and width
                               and not arch.startswith(("aarch64", "arm")))
    if link in (None, "pcie") and gen in PCIE_GBPS_PER_LANE and width:
        ceil = width * PCIE_GBPS_PER_LANE[gen]
        checks = [("memcpy.htod.pinned-bandwidth", 1.0),
                  ("memcpy.dtoh.pinned-bandwidth", 1.0),
                  ("memcpy.htod.pageable-staged", 1e-3)]   # MB/s -> GB/s
        for key, scale in checks:
            v = _scalar(doc, key)
            if v is None:
                continue
            gbps = v * scale
            if gbps > 1.05 * ceil:
                msg = (f"ceiling: {key} {gbps:.1f} GB/s exceeds the PCIe "
                       f"gen{gen} x{width} ceiling of {ceil:.1f} GB/s")
                if fatal:
                    errors.append(msg)
                else:
                    warnings.append(
                        msg + " -- a faster host link (NVLink-C2C or an "
                        "integrated part) would explain it; confirm the "
                        "platform")
            elif gbps > 0.95 * ceil:
                warnings.append(
                    f"ceiling: {key} {gbps:.1f} GB/s is implausibly close "
                    f"to the PCIe gen{gen} x{width} ceiling ({ceil:.1f})")
    elif link in ("c2c", "integrated"):
        # No bus-rate ceiling applies; the metric bounds in metrics.json
        # still cap these at physically absurd values.
        for key in ("memcpy.htod.pinned-bandwidth",
                    "memcpy.dtoh.pinned-bandwidth"):
            v = _scalar(doc, key)
            if v is not None and v > 0.95 * (width or 16) * PCIE_GBPS_PER_LANE.get(gen or 5, 3.938):
                warnings.append(
                    f"ceiling: {key} {v:.1f} GB/s exceeds PCIe rates, "
                    f"consistent with the reported {link} link")

    # --- DRAM: global-memory and histogram rates cannot beat the DRAM ---
    mem_clock, bus = gpu.get("mem_clock_mhz"), gpu.get("bus_width_bits")
    if mem_clock and bus:
        dram = 2 * mem_clock * bus / 8 / 1000        # GB/s (DDR)
        for key in ("global.read.peak", "global.write.peak"):
            for v in _values(results.get(key, {})):
                if v > 1.02 * dram:
                    errors.append(
                        f"ceiling: {key} {v:.0f} GB/s exceeds the DRAM "
                        f"ceiling of {dram:.0f} GB/s")
                elif v > 0.95 * dram:
                    warnings.append(
                        f"ceiling: {key} {v:.0f} GB/s is implausibly close "
                        f"to the DRAM ceiling ({dram:.0f})")
        for v in _values(results.get("global.copy.peak", {})):
            if v > 1.02 * dram:
                errors.append(
                    f"ceiling: global.copy.peak {v:.0f} GB/s exceeds the "
                    f"DRAM ceiling of {dram:.0f} GB/s")
        # histogram: Mpix/s at 1 byte/pixel -> 1 Mpix/s = 0.001 GB/s.
        # If the whole 16 Mpix image fits in L2 (Hopper/Blackwell-class),
        # rates above DRAM are legitimate -- warn only well above it.
        l2_resident = (gpu.get("l2_kib") or 0) * 1024 >= HIST_IMAGE_BYTES
        for key, meas in results.items():
            if not key.startswith("histogram.") or key.startswith(
                    "histogram.cpu."):
                continue
            for v in _values(meas):
                if l2_resident:
                    if v > 3 * dram * 1000:
                        warnings.append(
                            f"ceiling: {key} {v:.0f} Mpix/s far exceeds "
                            f"DRAM bandwidth (L2-resident image assumed)")
                elif v > 1.05 * dram * 1000:
                    errors.append(
                        f"ceiling: {key} {v:.0f} Mpix/s exceeds the DRAM "
                        f"ceiling of {dram * 1000:.0f} Mpix/s at 1 B/pixel")
                elif v > 0.9 * dram * 1000:
                    warnings.append(
                        f"ceiling: {key} {v:.0f} Mpix/s is implausibly "
                        f"close to DRAM bandwidth")

    # --- FP32: nbody interaction rates cannot beat the arithmetic ---
    sm_count, clock = gpu.get("sm_count"), gpu.get("clock_mhz")
    if sm_count and clock:
        cores = _fp32_cores(gpu.get("sm", ""))
        if cores is None:
            cores = 128
            if any(k.startswith(("nbody.gpu-", "nbody.multigpu-"))
                   for k in results):
                warnings.append(
                    f"ceiling: unknown compute capability "
                    f"'{gpu.get('sm')}' -- assuming 128 FP32 cores/SM")
        gflops = 2 * sm_count * cores * clock / 1000
        ceil_gi = gflops / NBODY_FLOPS_PER_INTERACTION
        n_gpus = max(1, doc.get("environment", {}).get("gpu_count", 1))
        for key, meas in results.items():
            if not key.startswith(("nbody.gpu-", "nbody.multigpu-")):
                continue
            # multi-GPU variants may legitimately scale past one GPU's peak
            ceil = ceil_gi * (n_gpus if key.startswith("nbody.multigpu-")
                              else 1)
            for v in _values(meas):
                if v > 1.1 * ceil:
                    errors.append(
                        f"ceiling: {key} {v:.1f} Gi/s exceeds the FP32 "
                        f"ceiling of {ceil:.1f} Gi/s "
                        f"({gflops:.0f} GFLOPS / "
                        f"{NBODY_FLOPS_PER_INTERACTION} flops/interaction)")
                elif v > 0.75 * ceil:
                    warnings.append(
                        f"ceiling: {key} {v:.1f} Gi/s is implausibly close "
                        f"to the FP32 ceiling ({ceil:.1f})")


# ----------------------------------------------------------------- coherence

def check_coherence(doc: dict, errors: list, warnings: list) -> None:
    """Cross-metric identities every real run satisfies. Fingerprint-free,
    so these run on legacy submissions too. All warnings: none of these is
    physically impossible, merely reviewer-worthy."""
    results = doc.get("results", {})

    # concurrency: total = htod+kernel+dtoh, speedup = total/concurrent,
    # and the concurrent time cannot beat its longest component.
    cols = {c: results.get(f"concurrency.memcpy-kernel.{c}", {})
            .get("series", {}) for c in suites._CONC_COLS}
    common = set.intersection(*(set(s) for s in cols.values())) \
        if all(cols.values()) else set()
    bad_sum = bad_speedup = bad_conc = 0
    for k in common:
        htod, kern, dtoh = cols["htod"][k], cols["kernel"][k], cols["dtoh"][k]
        total, conc = cols["total"][k], cols["concurrent"][k]
        speedup = cols["speedup"][k]
        if abs(total - (htod + kern + dtoh)) > 0.05 * total:
            bad_sum += 1
        if conc and abs(speedup - total / conc) > 0.05 * max(speedup, 0.01):
            bad_speedup += 1
        if conc < 0.95 * max(htod, kern, dtoh):
            bad_conc += 1
    if bad_sum:
        warnings.append(
            f"coherence: concurrency total != htod+kernel+dtoh in "
            f"{bad_sum}/{len(common)} rows")
    if bad_speedup:
        warnings.append(
            f"coherence: concurrency speedup != total/concurrent in "
            f"{bad_speedup}/{len(common)} rows")
    if bad_conc:
        warnings.append(
            f"coherence: concurrent time beats its longest component in "
            f"{bad_conc}/{len(common)} rows")

    htod = _scalar(doc, "memcpy.htod.pinned-bandwidth")
    dtoh = _scalar(doc, "memcpy.dtoh.pinned-bandwidth")
    if htod and dtoh and not 0.5 <= htod / dtoh <= 2.0:
        warnings.append(
            f"coherence: pinned htod/dtoh bandwidth ratio "
            f"{htod / dtoh:.2f} outside [0.5, 2.0]")

    la, ls = _scalar(doc, "launch.async"), _scalar(doc, "launch.sync")
    if la and ls and la > 1.1 * ls:
        warnings.append(
            f"coherence: async launch ({la} us) slower than sync ({ls} us)")

    if doc.get("code", {}).get("dirty"):
        warnings.append(
            "coherence: built from a dirty tree -- the pinned commit does "
            "not fully describe the code that ran")


# --------------------------------------------------- raw log: split + parse

def split_blocks(raw: str) -> list[tuple[str, str]]:
    """The raw log is a concatenation of '$ <command>' lines each followed
    by that program's stdout. Returns [(command, stdout)]."""
    blocks: list[tuple[str, str]] = []
    cmd, buf = None, []
    for line in raw.splitlines(keepends=True):
        if line.startswith("$ "):
            if cmd is not None:
                blocks.append((cmd, "".join(buf)))
            cmd, buf = line[2:].strip(), []
        elif cmd is not None:
            buf.append(line)
    if cmd is not None:
        blocks.append((cmd, "".join(buf)))
    return blocks


def _block_map() -> dict:
    """'<binary> <canonical args>' -> (parse, param), from bench/suites.py."""
    out = {}
    for specs in suites.SUITES.values():
        for spec in specs:
            for param, extra in spec.runs:
                key = " ".join([spec.binary] + extra)
                out[key] = (spec.parse, param)
    return out


def _normalize_cmd(cmd: str) -> str:
    argv = cmd.split()
    if argv and argv[0].endswith(".exe"):        # Windows builds
        argv[0] = argv[0][:-4]
    return " ".join(argv)


def rederive(raw: str, warnings: list) -> dict:
    """Re-parse the raw log with the suite parsers, folding exactly as
    run.py did. Unrecognized or failed blocks become warnings."""
    block_map = _block_map()
    derived: dict = {}
    for cmd, out in split_blocks(raw):
        norm = _normalize_cmd(cmd)
        prog = norm.split()[0] if norm else ""
        if prog in ("nvidia-smi", "deviceProps"):
            continue                    # fingerprint blocks, handled below
        if "[FAILED" in out or "[PARSE ERROR" in out:
            warnings.append(f"raw: block '{prog}' marked failed")
            continue
        entry = block_map.get(norm)
        if entry is None:
            warnings.append(f"raw: unrecognized program block '{cmd}'")
            continue
        parse, param = entry
        try:
            parsed = parse(out, param)
            suites.merge(derived, parsed)
        except SystemExit:              # duplicate block in a crafted log
            warnings.append(f"raw: duplicate metrics from block '{cmd}'")
        except Exception as e:
            warnings.append(f"raw: re-parse of '{cmd}' failed: {e}")
    return derived


def check_rederive(doc: dict, raw: str, errors: list,
                   warnings: list) -> None:
    """Every reported number must re-derive from the raw log via the same
    parser that produced it -- the tamper trap. (The plain substring check
    in validate.py passes an edited digit whenever the new string happens
    to appear elsewhere in the log; an exact re-parse does not.)"""
    derived = rederive(raw, warnings)
    for key, meas in doc.get("results", {}).items():
        d = derived.get(key)
        if d is None:
            errors.append(f"rederive: {key} not derivable from the raw log")
            continue
        if d.get("unit") != meas.get("unit"):
            errors.append(f"rederive: {key} unit mismatch "
                          f"('{meas.get('unit')}' vs '{d.get('unit')}')")
        if "value" in meas:
            dv = d.get("value")
            if dv is None or abs(meas["value"] - dv) > 1e-6 * max(
                    1.0, abs(dv)):
                errors.append(f"rederive: {key} value {meas['value']} != "
                              f"{dv} re-derived from the raw log")
        else:
            dseries = d.get("series", {})
            for k, v in meas.get("series", {}).items():
                if k not in dseries:
                    errors.append(f"rederive: {key}[{k}] not in the raw log")
                elif abs(v - dseries[k]) > 1e-6 * max(1.0, abs(dseries[k])):
                    errors.append(f"rederive: {key}[{k}] value {v} != "
                                  f"{dseries[k]} re-derived")
    # The converse: numbers the raw log yields that the doc dropped.
    for key, d in derived.items():
        meas = doc.get("results", {}).get(key)
        if meas is None:
            warnings.append(f"rederive: {key} present in the raw log but "
                            f"dropped from the doc")
            continue
        for k in d.get("series", {}):
            if k not in meas.get("series", {}):
                warnings.append(f"rederive: {key}[{k}] present in the raw "
                                f"log but dropped from the doc")


# ------------------------------------------------------- raw log: evidence

_SM_EVIDENCE = re.compile(r"\((\d+) SMs\)|GPU with (\d+) SMs")
_HIST_BANNER = re.compile(r"Testing histogram on (.+?) \(\d+ SMs\)")
_SMI_FIELDS = ("clock_mhz", "mem_clock_mhz", "pcie_gen", "pcie_width")


def check_raw_evidence(doc: dict, raw: str, errors: list,
                       warnings: list) -> None:
    """The hardware evidence in the raw log -- what the programs themselves
    printed -- must match the claimed environment."""
    gpu = doc.get("environment", {}).get("gpu", {})

    # SM counts printed by the benchmark programs
    found = {int(a or b) for a, b in _SM_EVIDENCE.findall(raw)}
    m = re.search(r"^multiProcessorCount:\s*(\d+)", raw, re.M)
    if m:
        found.add(int(m.group(1)))
    if gpu.get("sm_count"):
        for n in found:
            if n != gpu["sm_count"]:
                errors.append(f"evidence: raw log reports {n} SMs but the "
                              f"fingerprint claims {gpu['sm_count']}")
    elif len(found) > 1:
        errors.append(f"evidence: raw log disagrees with itself on the SM "
                      f"count ({sorted(found)}) -- spliced log?")
    elif not found:
        warnings.append("evidence: no SM-count evidence in the raw log")

    # GPU name printed by histogram
    m = _HIST_BANNER.search(raw)
    if m and m.group(1) != gpu.get("name"):
        errors.append(f"evidence: histogram ran on '{m.group(1)}' but the "
                      f"submission claims '{gpu.get('name')}'")

    # deviceProps block vs the doc's fingerprint
    props_block = next((out for cmd, out in split_blocks(raw)
                        if _normalize_cmd(cmd).split()[:1] == ["deviceProps"]),
                       None)
    doc_props = {k: gpu[k] for k in ("sm_count", "bus_width_bits", "l2_kib",
                                     "copy_engines", "integrated")
                 if k in gpu}
    if props_block is not None:
        props = suites.parse_device_props(props_block, None)
        for field in ("name", "sm"):
            if field in props and props[field] != gpu.get(field):
                errors.append(f"evidence: deviceProps {field} "
                              f"'{props[field]}' != environment.gpu "
                              f"'{gpu.get(field)}'")
        for field, want in doc_props.items():
            if field in props and props[field] != want:
                errors.append(f"evidence: deviceProps {field} "
                              f"{props[field]} != fingerprint {want}")
        if "vram_mib" in props and gpu.get("vram_mib"):
            if abs(props["vram_mib"] - gpu["vram_mib"]) > \
                    0.15 * gpu["vram_mib"]:
                errors.append(
                    f"evidence: deviceProps totalGlobalMem "
                    f"{props['vram_mib']} MiB differs from vram_mib "
                    f"{gpu['vram_mib']} by more than 15%")
    elif doc_props:
        errors.append("evidence: fingerprint fields present but no "
                      "deviceProps block in the raw log")
    else:
        warnings.append("evidence: fingerprint absent (legacy submission) "
                        "-- hardware ceilings not checked")

    # nvidia-smi block vs the doc's clock/PCIe fields
    smi_block = next((("".join(out.splitlines()[:1]), out)
                      for cmd, out in split_blocks(raw)
                      if _normalize_cmd(cmd).split()[:1] == ["nvidia-smi"]),
                     None)
    doc_smi = {k: gpu[k] for k in _SMI_FIELDS if k in gpu}
    if smi_block is not None:
        line = smi_block[0]
        vals = [f.strip() for f in line.split(",")]
        for field, val in zip(_SMI_FIELDS, vals):
            try:
                got = int(float(val))
            except ValueError:
                continue                       # "[N/A]" -- field omitted
            if field in doc_smi and got != doc_smi[field]:
                errors.append(f"evidence: nvidia-smi reports {field}={got} "
                              f"but the fingerprint claims "
                              f"{doc_smi[field]}")
    elif doc_smi:
        errors.append("evidence: clock/PCIe fingerprint fields present but "
                      "no nvidia-smi block in the raw log")


# -------------------------------------------------------------------- entry

def check_all(doc: dict, raw: str, errors: list, warnings: list) -> None:
    check_ceilings(doc, errors, warnings)
    check_coherence(doc, errors, warnings)
    check_raw_evidence(doc, raw, errors, warnings)
    check_rederive(doc, raw, errors, warnings)
