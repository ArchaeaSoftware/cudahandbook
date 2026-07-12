# Benchmark harness — submit your GPU's results to cudahandbook.com

The performance tables in *The CUDA Handbook* were measured on 2012–2013
hardware. This harness lets you run the book's benchmarks on **your** GPU and
submit the results, which appear alongside the originals on
[cudahandbook.com/benchmarks](https://cudahandbook.com/benchmarks/).

## Run

```sh
git clone https://github.com/ArchaeaSoftware/cudahandbook.git
cd cudahandbook
cmake -B build && cmake --build build -j     # build the benchmark programs

python3 bench/run.py --bin-dir build --suite all --device 0 \
    --name "Your Name" --github yourhandle \
    --out results/$(hostname)-$(date +%F).json
```

`run.py` detects your GPU, driver, and CUDA version automatically, runs each
suite, and writes two files:

- `<name>.json` — the parsed results (see `bench/schema.json`)
- `<name>.raw.txt` — the verbatim program output, kept for provenance

Use `--suite histogram,nbody` to run specific suites, or a group — `--suite microbench` (launch overhead, memcpy, global-memory bandwidth, copy/kernel overlap) or `--suite apps` (histograms, N-body); `--platform "aws g6e.xlarge"`
if you're on a cloud instance. Pass `--device N` once per GPU on multi-GPU
machines (one JSON per GPU).

## Check

```sh
python3 bench/validate.py results/<your-file>.json
```

This is the same check CI runs on your pull request: schema conformance,
plausible value ranges, and that every reported number appears in the raw log.

## Submit

Submission is a perk of [cudahandbook.com membership](https://cudahandbook.com/membership/):

```sh
python3 bench/submit.py results/<your-file>.json
```

(or add `--submit` to the `run.py` command). Enter your membership email,
then the 6-digit code it receives — your results are queued for review and
appear on the site, credited to you if you passed `--name`/`--github`, once
approved. Your email is used only to verify membership; it is never
published.

Alternatively, open a pull request adding your two files under
`results/<gpu-slug>/` (e.g. `results/rtx-3060/`; the slug is the lowercased,
hyphenated GPU name — `validate.py` will tell you if it's off).

Please benchmark on an otherwise-idle machine, and don't submit runs from a
dirty working tree (`code.dirty` must be `false`).

## Layout

- `run.py` — orchestrator: environment capture, suite execution, raw-log tee
- `submit.py` — member submission client (email + one-time code)
- `suites.py` — per-program specs: binary, canonical arguments, stdout parser
- `metrics.json` — metric key → unit + plausibility bounds (the contract with
  the website's rendering pipeline)
- `schema.json` — JSON Schema for submissions
- `validate.py` — submission checker (run by CI)
- `tests/` — parser unit tests against captured program outputs
