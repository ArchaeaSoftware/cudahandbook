"""Parser unit tests — run with:  python3 -m unittest discover bench/tests

Fixtures are captured or format-string-faithful program outputs, so these
tests run without a GPU. histogram.gt200.txt is a real capture from the
book's original GT200 runs; the synthesized fixtures reproduce the exact
printf formats in the current sources (see file:line notes inline).
"""
import json
import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))

import suites  # noqa: E402
from suites import (parse_histogram, parse_nbody, parse_null,          # noqa: E402
                    parse_breakeven, parse_pinned_bandwidth,
                    parse_concurrency_memcpy_kernel)

FIXTURES = Path(__file__).parent / "fixtures"


class TestHistogram(unittest.TestCase):
    def test_gt200_capture(self):
        out = (FIXTURES / "histogram.gt200.txt").read_text()
        res = parse_histogram(out, 256)
        self.assertEqual(res["histogram.per-block"]["series"]["256"], 6706.17)
        self.assertEqual(res["histogram.per-grid"]["series"]["256"], 57.97)
        self.assertEqual(res["histogram.npp"]["series"]["256"], 97.18)
        self.assertEqual(res["histogram.cpu.multi"]["series"]["256"], 1.49)
        # GT200 (SM 1.x) runs skip the per-thread variants entirely
        self.assertNotIn("histogram.per-thread-4x64", res)
        # this old capture predates the Single-threaded line
        self.assertNotIn("histogram.cpu.single", res)

    def test_modern_output(self):
        # histogram.cu:400,407-410,429-430 format strings
        out = ("Testing histogram on NVIDIA GeForce RTX 5090 (170 SMs)\n"
               "16777216 pixels, random, 64 values with stride 1\n"
               "Single-threaded: 21.51 Mpix/s\n"
               "Multithreaded (32 cores) is 24.10x faster (518.43 Mpix/s)\n"
               "GPUhistogramPerBlock: 152300.55 Mpix/s\n"
               "GPUhistogramPerThread4x64: 201417.00 Mpix/s\n")
        res = parse_histogram(out, 64)
        self.assertEqual(res["histogram.cpu.single"]["series"]["64"], 21.51)
        self.assertEqual(res["histogram.cpu.multi"]["series"]["64"], 518.43)
        self.assertEqual(res["histogram.per-thread-4x64"]["series"]["64"],
                         201417.00)

    def test_miscompare_not_parsed(self):
        res = parse_histogram(
            "GPUhistogramPerBlock: Histograms miscompare\n", 8)
        self.assertEqual(res, {})


class TestNbody(unittest.TestCase):
    def test_giga_and_mega_lines(self):
        # nbody.cu:832-846 — leading \r, %8.2f/%8.3f padding, both exponents
        out = ("Running simulation with 65536 particles, "
               "crosscheck disabled, CPU disabled\n"
               "\rmultiGPU_SingleCPUThread:    12.40 ms =  346.420x10^9 "
               "interactions/s (Rel. error: 0.000000E+00)\n"
               "\rmultiGPU_SingleCPUThread:    12.10 ms =  354.930x10^9 "
               "interactions/s (Rel. error: 0.000000E+00)\n"
               "\rmultiGPU_SingleCPUThread:    12.20 ms =  350.100x10^9 "
               "interactions/s (Rel. error: 0.000000E+00)\n")
        res = parse_nbody(out, None)
        m = res["nbody.multigpu-singlecputhread"]
        self.assertEqual(m["unit"], "Ginteractions/s")
        self.assertEqual(m["value"], 350.1)          # median of three

        out_mega = ("\rCPU_SOA:  4100.00 ms =  262.144x10^6 "
                    "interactions/s (Rel. error: 1.900000E-05)\n")
        res = parse_nbody(out_mega, None)
        self.assertAlmostEqual(res["nbody.cpu-soa"]["value"], 0.262)


class TestMemcpy(unittest.TestCase):
    def test_null_latency(self):
        # nullHtoDMemcpyAsync.cu:51,70 — prefix and result share one line
        parse = parse_null("memcpy.htod.null-latency")
        res = parse("NULL host->device memcpy's... 1.86 us\n", None)
        self.assertEqual(res["memcpy.htod.null-latency"]["value"], 1.86)

    def test_null_latency_sync_with_iterations(self):
        # nullDtoHMemcpySync.cu:51,74
        parse = parse_null("memcpy.dtoh.null-latency-sync")
        res = parse("Measuring NULL device->host memcpy's (with sync)... "
                    "7.90 us (65536 iterations)\n", None)
        self.assertEqual(res["memcpy.dtoh.null-latency-sync"]["value"], 7.90)

    def test_breakeven_rows(self):
        # breakevenHtoDMemcpy.cu:54-55,67,79 (header says D->H in both files)
        out = ("D->H memcpy times...\nSize\tTime (us)\n"
               "4096\t2.20\n8192\t2.41\n65536\t7.62\n")
        res = parse_breakeven("htod")(out, None)
        s = res["memcpy.htod.breakeven"]["series"]
        self.assertEqual(s, {"4096": 2.20, "8192": 2.41, "65536": 7.62})

    def test_pinned_bandwidth(self):
        # pinnedBandwidth.cu:83-88
        out = ("Transferring 64 MB 100 times... (all bandwidths in GB/s)\n"
               "Device\tHtoD\tDtoH\n0\t54.77\t55.21\n")
        res = parse_pinned_bandwidth(out, None)
        self.assertEqual(res["memcpy.htod.pinned-bandwidth"]["value"], 54.77)
        self.assertEqual(res["memcpy.dtoh.pinned-bandwidth"]["value"], 55.21)


class TestConcurrency(unittest.TestCase):
    def test_shmoo_table(self):
        # concurrencyMemcpyKernel.cu:93-122 — dots, then 7-column rows
        out = ("Using 16 blocks per SM on GPU with 170 SMs = 2720 blocks\n"
               "32M integers\n"
               "Timing sequential operations................\n"
               "Timing concurrent operations................\n"
               "\nCycles\tHtoD\tKernel\tDtoH\tTotal\tConcurrent\tSpeedup\n"
               "4\t8.92\t1.10\t8.20\t18.22\t17.31\t1.05\n"
               "8\t8.92\t1.76\t8.20\t18.88\t17.34\t1.09\n")
        res = parse_concurrency_memcpy_kernel(out, None)
        self.assertEqual(res["concurrency.memcpy-kernel.total"]["series"],
                         {"4": 18.22, "8": 18.88})
        self.assertEqual(res["concurrency.memcpy-kernel.speedup"]["series"],
                         {"4": 1.05, "8": 1.09})
        self.assertEqual(res["concurrency.memcpy-kernel.speedup"]["unit"], "x")

    def test_no_table_is_empty(self):
        self.assertEqual(parse_concurrency_memcpy_kernel("garbage\n", None),
                         {})


class TestContracts(unittest.TestCase):
    """suites.py, metrics.json, and schema.json must agree."""

    METRICS = json.loads((BENCH / "metrics.json").read_text())

    def test_fixture_metrics_are_registered(self):
        emitted = set(parse_histogram(
            (FIXTURES / "histogram.gt200.txt").read_text(), 256))
        for key in emitted:
            self.assertIn(key, self.METRICS)

    def test_all_static_keys_registered_and_valid(self):
        import re as _re
        key_re = _re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
        static = (["histogram.cpu.single", "histogram.cpu.multi"]
                  + list(suites._HIST_NAMES.values())
                  + [f"concurrency.memcpy-kernel.{c}" for c in suites._CONC_COLS]
                  + ["memcpy.htod.null-latency", "memcpy.dtoh.null-latency",
                     "memcpy.dtoh.null-latency-sync",
                     "memcpy.htod.breakeven", "memcpy.dtoh.breakeven",
                     "memcpy.htod.pinned-bandwidth",
                     "memcpy.dtoh.pinned-bandwidth"])
        for key in static:
            self.assertIn(key, self.METRICS, key)
            self.assertRegex(key, key_re)   # matches schema.json's pattern
        for key, spec in self.METRICS.items():
            self.assertRegex(key, key_re)
            lo, hi = spec["bounds"]
            self.assertLess(lo, hi, key)


if __name__ == "__main__":
    unittest.main()
