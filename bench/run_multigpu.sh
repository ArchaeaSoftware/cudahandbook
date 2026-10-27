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
for algo in Ring NVLS Tree; do
    echo "--- NCCL_ALGO=$algo"
    NCCL_ALGO="$algo" NCCL_DEBUG=INFO \
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
