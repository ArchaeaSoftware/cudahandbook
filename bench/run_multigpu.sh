#!/usr/bin/env bash
#
# Everything a rented multi-GPU instance is for, in one command.
#
# The order matters.  The topology and the probe come first, because they
# explain every number that follows and they are the one thing that cannot be
# reconstructed afterward from a results file.  A machine whose peer matrix
# reads "host" rather than "P2P" has explained its own disappointing
# bandwidth before anyone profiles it.
#
# Then the scaling sweep, then the collective comparison: the row all-reduce
# is the step NVSwitch accumulators accelerate, and NCCL_ALGO selects between
# a ring reduction and a switch-resident one, so the comparison is this
# program run twice.
#
# Everything lands in one output directory, including the logs, so a single
# scp retrieves the whole session.  Terminating the instance with the logs
# still on it is the expensive mistake this guards against.
#
# Usage: bench/run_multigpu.sh [output-directory]
#
set -o errexit
set -o nounset
set -o pipefail

OUTDIR="${1:-multigpu-$(date -u +%Y%m%dT%H%M%SZ)}"
BIN_DIR="${BIN_DIR:-build}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

echo "repository: $REPO"
echo "binaries:   $BIN_DIR"
echo "output:     $OUTDIR"
echo

if [ ! -d "$BIN_DIR" ]; then
    echo "no such build directory: $BIN_DIR" >&2
    echo "configure and build first, or set BIN_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------- topology
# Captured before anything is measured, and tolerated when absent: nvidia-smi
# reports no NVLink on a machine that has none, which is a result, not an
# error.  A set -e script would otherwise stop here on a PCIe-only instance.

echo "=== topology ==="
{
    echo "## date"
    date -u +%Y-%m-%dT%H:%M:%SZ
    echo
    echo "## nvidia-smi"
    nvidia-smi || true
    echo
    echo "## nvidia-smi topo -m"
    nvidia-smi topo -m || true
    echo
    echo "## nvidia-smi nvlink -s"
    nvidia-smi nvlink -s || true
    echo
    echo "## commit"
    git -C "$REPO" rev-parse HEAD || true
    git -C "$REPO" status --porcelain || true
} > "$OUTDIR/topology.txt" 2>&1
sed -n '/## nvidia-smi topo/,/^$/p' "$OUTDIR/topology.txt" | head -20
echo "  -> $OUTDIR/topology.txt"
echo

# ------------------------------------------------------------------- probe
# Reports the peer matrix and proves a collective produces the right answer
# on this machine.  If this fails, nothing measured afterward means anything,
# so it is the one step that stops the script.

echo "=== NCCL probe ==="
if [ -x "$BIN_DIR/multigpu/ncclProbe" ]; then
    "$BIN_DIR/multigpu/ncclProbe" 2>&1 | tee "$OUTDIR/ncclProbe.txt"
else
    echo "ncclProbe was not built; NCCL was not found at configure time." \
         | tee "$OUTDIR/ncclProbe.txt"
    echo "Reconfigure with -DNCCL_HOME=<prefix> to measure the 2-D" \
         "decomposition." | tee -a "$OUTDIR/ncclProbe.txt"
fi
echo

# ------------------------------------------------------------ verification
# Check the decompositions against the CPU golden values before spending an
# hour measuring them.  The timing runs pass --nocpu, which switches the
# golden off, so they establish speed and say nothing about correctness; a
# geometry that drops a block of interactions still reports a plausible
# interaction rate.  Here the golden is on, every geometry the sweep will
# time is exercised at both a small and the measured problem size, and a
# disagreement exits nonzero before the sweep starts.
#
# Both sizes are needed.  The small one is where a wrong answer is cheap to
# find; the large one is where an offset that overflows, or a grid that
# exceeds a launch limit, shows up at all.

echo "=== verification against CPU golden ==="
VERIFY_LOG="$OUTDIR/verify.txt"
: > "$VERIFY_LOG"
nGPUsPresent="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)"
echo "devices present: $nGPUsPresent" | tee -a "$VERIFY_LOG"
cVerifyFailures=0
# Take the algorithm names from the sweep rather than repeating them here.  A
# second hard-coded list is a second thing to get wrong, and this stage exists
# to catch exactly that class of mistake; nbody exits nonzero on a name it does
# not recognize, so a drifted copy blocks the run instead of mismeasuring it,
# but blocking a good run at instance prices is its own kind of expensive.
VERIFY_ALGORITHMS="$(python3 -c 'import sys
sys.path.insert(0, "bench")
import suites
print(" ".join(suites.NBODY_SCALING_ALGORITHMS))')"
echo "algorithms: $VERIFY_ALGORITHMS" | tee -a "$VERIFY_LOG"

for algorithm in $VERIFY_ALGORITHMS; do
    for nGPUs in 1 2 4 8; do
        if [ "$nGPUs" -gt "$nGPUsPresent" ]; then
            continue
        fi
        for nBodies in 4 64; do
            printf -- "--- %s on %d GPU(s), %dK bodies\n" \
                "$algorithm" "$nGPUs" "$nBodies" | tee -a "$VERIFY_LOG"
            if "$BIN_DIR/nbody/nbody" \
                    --algorithm "$algorithm" \
                    --numgpus "$nGPUs" \
                    --numbodies "$nBodies" \
                    --iterations 2 >> "$VERIFY_LOG" 2>&1; then
                echo "    ok" | tee -a "$VERIFY_LOG"
            else
                echo "    FAILED" | tee -a "$VERIFY_LOG"
                cVerifyFailures=$((cVerifyFailures+1))
            fi
        done
    done
done
if [ "$cVerifyFailures" -ne 0 ]; then
    echo "$cVerifyFailures verification run(s) failed; see $VERIFY_LOG" >&2
    echo "Refusing to spend the sweep on results nothing has checked." >&2
    exit 1
fi
echo "all verification runs agree with the golden values"
echo

# ----------------------------------------------------------------- metrics
# The harness skips the runs that need more devices than this machine has, so
# the same command is correct on four GPUs and on eight.

echo "=== scaling sweep and peer-to-peer matrix ==="
python3 bench/run.py \
    --bin-dir "$BIN_DIR" \
    --suite multigpu,nbody \
    --out "$OUTDIR/results.json" \
    "${@:2}" 2>&1 | tee "$OUTDIR/run.log"
echo

echo "=== validation ==="
python3 bench/validate.py "$OUTDIR/results.json" 2>&1 \
    | tee "$OUTDIR/validate.txt"
echo

# ------------------------------------------------------- collective choice
# The row all-reduce again, with NCCL's algorithm forced.  NVLS is the
# switch-resident reduction; Ring is the one that moves every partial between
# devices.  A machine without NVSwitch refuses NVLS and falls back, which the
# log records, so running it costs nothing but tells us whether the hardware
# is doing what the chapter says it does.

echo "=== row all-reduce, by NCCL algorithm ==="

# NVLS performs the reduction inside the switch, and NVIDIA limits NVLink
# SHARP to third-generation NVSwitch systems (NVLink 4) with Hopper or later
# GPUs.  On anything older the NVLS arm below does not fail: with
# NCCL_NVLS_ENABLE at its default, initialization succeeds and NCCL quietly
# selects another algorithm, so the run lands in a file named NVLS having
# measured a ring.  Record which machine this is, so the files can be read for
# what they are.  An A100 instance is the case that matters here: it has
# NVSwitch, and it still has no NVLS.
ccMajor="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | cut -d. -f1)"
ccMajor="${ccMajor:-0}"
if [ "$ccMajor" -ge 9 ]; then
    echo "compute capability ${ccMajor}.x: NVLS available" \
        | tee "$OUTDIR/nvls-availability.txt"
else
    echo "compute capability ${ccMajor}.x: NVLS needs 9.0 (Hopper) on a" \
         "third-generation NVSwitch. The NVLS run below will fall back to" \
         "another algorithm and is not a measurement of in-switch reduction." \
        | tee "$OUTDIR/nvls-availability.txt"
fi
echo

for algo in Ring NVLS Tree; do
    echo "--- NCCL_ALGO=$algo"
    NCCL_ALGO="$algo" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS \
        "$BIN_DIR/nbody/nbody" --numbodies 64 --iterations 12 --nocpu \
            --algorithm multiGPU_2D 2>&1 \
        | tee "$OUTDIR/allreduce-$algo.txt" \
        | grep -aE "multiGPU_2D:|device\(s\) as a|NVLS|nvls" \
        | head -8 || true
    echo
done

# -------------------------------------------------------------- retrieval
# Printed last and loudly, because the one unrecoverable mistake is
# terminating the instance with the only copy of the results on it.

echo "=== done ==="
ls -la "$OUTDIR"
echo
echo "Retrieve before terminating the instance:"
echo "  scp -r <this-host>:$OUTDIR ."
