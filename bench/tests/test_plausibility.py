"""Plausibility-engine tests — run with:  python3 -m unittest discover bench/tests

The honest fixture is the committed rtx-3060 seed submission (checked out by
CI alongside the code), so these tests permanently pin the properties the
review pipeline depends on: an untampered doc re-derives exactly, and each
class of tamper flips to its designed outcome.
"""
import copy
import json
import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))

import plausibility  # noqa: E402
import suites        # noqa: E402

SEED_DIR = BENCH.parent / "results" / "rtx-3060"
SEED_JSON = SEED_DIR / "nwilt-2026-07-12-2.json"
SEED_RAW = SEED_DIR / "nwilt-2026-07-12-2.raw.txt"

# The RTX 3060 fingerprint the harness would capture (used to synthesize
# fingerprinted variants of the legacy seed doc).
FP = {"sm_count": 28, "clock_mhz": 1852, "mem_clock_mhz": 7501,
      "bus_width_bits": 192, "l2_kib": 2304, "copy_engines": 2,
      "pcie_gen": 4, "pcie_width": 16}
PROPS_BLOCK = ("$ deviceProps\n"
               "name: NVIDIA GeForce RTX 3060\n"
               "computeCapability: 8.6\n"
               "multiProcessorCount: 28\n"
               "memoryBusWidth: 192 bits\n"
               "l2CacheSize: 2304 KiB\n"
               "asyncEngineCount: 2\n"
               "totalGlobalMem: 12043 MiB\n\n")
SMI_BLOCK = ("$ nvidia-smi --query-gpu=clocks.max.sm,clocks.max.memory,"
             "pcie.link.gen.max,pcie.link.width.max "
             "--format=csv,noheader,nounits -i 0\n"
             "1852, 7501, 4, 16\n")


def run_checks(doc, raw):
    errors, warnings = [], []
    plausibility.check_all(doc, raw, errors, warnings)
    return errors, warnings


class TestSeed(unittest.TestCase):
    """The committed seed submission is the honest baseline."""

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(SEED_JSON.read_text())
        cls.raw = SEED_RAW.read_text(errors="replace")

    def test_honest_doc_has_no_errors(self):
        errors, warnings = run_checks(self.doc, self.raw)
        self.assertEqual(errors, [])
        # exactly the two known conditions: dirty tree, legacy fingerprint
        self.assertEqual(len(warnings), 2, warnings)

    def test_split_blocks_count(self):
        self.assertEqual(len(plausibility.split_blocks(self.raw)), 24)

    def test_fingerprinted_seed_passes(self):
        """The seed doc plus a consistent fingerprint and its raw blocks."""
        doc = copy.deepcopy(self.doc)
        doc["environment"]["gpu"].update(FP)
        errors, warnings = run_checks(doc, SMI_BLOCK + PROPS_BLOCK + self.raw)
        self.assertEqual(errors, [])
        # only the dirty-tree warning remains
        self.assertEqual(len(warnings), 1, warnings)


class TestTampering(unittest.TestCase):
    """Each class of tamper must flip to its designed outcome."""

    def setUp(self):
        self.doc = json.loads(SEED_JSON.read_text())
        self.raw = SEED_RAW.read_text(errors="replace")

    def test_edited_value_fails_rederive(self):
        key = "global.read.peak"
        sk = next(iter(self.doc["results"][key]["series"]))
        self.doc["results"][key]["series"][sk] *= 1.25
        errors, _ = run_checks(self.doc, self.raw)
        self.assertTrue(any("rederive: global.read.peak" in e
                            for e in errors), errors)

    def test_shrunken_dram_claim_fails_ceiling(self):
        self.doc["environment"]["gpu"].update(
            {**FP, "mem_clock_mhz": 3750, "bus_width_bits": 96})
        errors, _ = run_checks(
            self.doc, SMI_BLOCK + PROPS_BLOCK + self.raw)
        self.assertTrue(any(e.startswith("ceiling: global.")
                            for e in errors), errors[:3])
        # the fingerprint also contradicts the raw nvidia-smi/deviceProps
        self.assertTrue(any(e.startswith("evidence:") for e in errors))

    def test_wrong_sm_count_fails_evidence(self):
        self.doc["environment"]["gpu"].update({**FP, "sm_count": 14})
        errors, _ = run_checks(
            self.doc, SMI_BLOCK + PROPS_BLOCK + self.raw)
        self.assertTrue(any("raw log reports 28 SMs" in e
                            for e in errors), errors)

    def test_impossible_speedup_fails_bounds(self):
        # bounds live in metrics.json (validate.check_metrics); here confirm
        # the coherence identity also flags the forged series row.
        key = "concurrency.memcpy-kernel.speedup"
        sk = next(iter(self.doc["results"][key]["series"]))
        self.doc["results"][key]["series"][sk] = 3.5
        errors, warnings = run_checks(self.doc, self.raw)
        self.assertTrue(any("rederive" in e and key in e for e in errors),
                        errors)
        self.assertTrue(any("speedup != total/concurrent" in w
                            for w in warnings), warnings)

    def test_deleted_raw_block_fails_rederive(self):
        blocks = plausibility.split_blocks(self.raw)
        raw = "".join(f"$ {c}\n{o}" for c, o in blocks
                      if not c.startswith("pinnedBandwidth"))
        errors, _ = run_checks(self.doc, raw)
        self.assertTrue(any(
            "memcpy.htod.pinned-bandwidth not derivable" in e
            for e in errors), errors)

    def test_dropped_metric_warns(self):
        del self.doc["results"]["memcpy.htod.pinned-bandwidth"]
        errors, warnings = run_checks(self.doc, self.raw)
        self.assertEqual(errors, [])
        self.assertTrue(any("dropped from the doc" in w for w in warnings),
                        warnings)

    def test_fingerprint_without_raw_backing_fails(self):
        self.doc["environment"]["gpu"].update(FP)
        errors, _ = run_checks(self.doc, self.raw)   # no fp blocks in raw
        self.assertTrue(any("no deviceProps block" in e for e in errors))
        self.assertTrue(any("no nvidia-smi block" in e for e in errors))

    def test_spliced_log_sm_disagreement(self):
        raw = self.raw + "\n$ histogram --random 1\n" \
            "Testing histogram on NVIDIA GeForce RTX 3060 (96 SMs)\n"
        errors, _ = run_checks(self.doc, raw)
        self.assertTrue(any("disagrees with itself" in e for e in errors),
                        errors)


class TestCeilingFormulas(unittest.TestCase):
    def test_dram_formula(self):
        # 2 * clock * width / 8 / 1000: known parts
        for clock, width, gbps in ((7501, 192, 360), (10501, 384, 1008),
                                   (1593, 5120, 2039)):
            self.assertAlmostEqual(2 * clock * width / 8 / 1000, gbps,
                                   delta=gbps * 0.01)

    def test_pcie_gen4_x16(self):
        self.assertAlmostEqual(
            16 * plausibility.PCIE_GBPS_PER_LANE[4], 31.5, delta=0.1)

    def test_fp32_cores_lookup(self):
        self.assertEqual(plausibility._fp32_cores("8.6"), 128)
        self.assertEqual(plausibility._fp32_cores("8.0"), 64)
        self.assertEqual(plausibility._fp32_cores("12.0"), 128)
        self.assertIsNone(plausibility._fp32_cores("99.9"))


if __name__ == "__main__":
    unittest.main()
