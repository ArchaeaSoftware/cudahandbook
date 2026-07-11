#!/usr/bin/env python3
"""Submit a benchmark results JSON to cudahandbook.com (members only).

    python3 bench/submit.py results/rtx-3060/me-2026-07-15.json

or as part of a run:  python3 bench/run.py ... --submit

Flow: enter the email on your cudahandbook.com membership, receive a
6-digit code by email, enter it, done. Results appear on
https://cudahandbook.com/benchmarks/ once reviewed. Membership:
https://cudahandbook.com/membership/
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://cudahandbook.com"


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", "")
        except Exception:
            detail = ""
        sys.exit(f"submission failed ({e.code}): {detail or e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {API}: {e.reason}")


def submit(json_path: Path) -> None:
    doc = json.loads(json_path.read_text())
    raw_path = json_path.parent / doc["raw"]
    if not raw_path.is_file():
        sys.exit(f"raw log {doc['raw']} not found next to {json_path}")
    raw = raw_path.read_text(errors="replace")

    print(f"Submitting {json_path.name} "
          f"({len(doc['results'])} metrics, {doc['environment']['gpu']['name']})")
    print("Submission requires an active cudahandbook.com membership.")
    email = input("Membership email: ").strip()
    _post("/api/bench-start", {"email": email})
    code = input("6-digit code (check your email): ").strip()
    out = _post("/api/bench-submit",
                {"email": email, "code": code, "doc": doc, "raw": raw})
    print(out.get("message", "submitted"))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    submit(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
