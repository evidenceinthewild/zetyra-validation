#!/usr/bin/env python3
"""
Run every protected validation suite and verify each ACTUALLY REPORTED.

    python common/run_gate.py                        # whole manifest, prod
    python common/run_gate.py --url https://staging  # whole manifest, staging
    python common/run_gate.py --suite gsd-futility   # one suite
    python common/run_gate.py <script> <url>         # single-script mode

WHY VERIFICATION LIVES IN THE PARENT
------------------------------------
A process cannot guarantee its own output. If a suite dies before printing --
unhandled exception, `os._exit`, OOM kill, segfault, an R `stop()` -- whatever
self-check it intended never executes. It exits, possibly with status 0, having
said nothing.

That is not hypothetical. The R gsDesign benchmark halted on an unexpected 422
and printed two blank lines, which in a terminal reads exactly like a quiet
success. The sabotage matrix confirmed it: with a suite's own summary
suppressed, every in-suite protection passed and the run looked clean, because
the code that would have complained was the code that never ran.

So the parent does the checking. It is still alive when the child is not.

WHY THE MANIFEST DRIVES THE RUN
-------------------------------
CI invoking suites individually means a suite can be dropped from the pipeline
by deleting a workflow line, and nothing notices -- the remaining suites still
pass. Here the manifest declares the full expected suite_id set, and the runner
fails if what it observed differs. Removing a suite requires removing it from
the manifest, which is a reviewable diff rather than an invisible omission.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "gate_manifest.json")
ARTIFACT_DIR = os.path.join(ROOT, "results", "gate_artifacts")

TERMINAL_MARKERS = ("ALL VALIDATIONS PASSED", "GATE FAILED",
                    "SOME VALIDATIONS FAILED")
SUMMARY_MARKER = "GATE SUMMARY"
SUITE_ID_RE = re.compile(r"^SUITE-ID:\s*(\S+)\s*$", re.M)
ASSERTION_RE = re.compile(r"(\d+)/(\d+)\s+(?:assertions|tests|boundary comparisons)\s+passed")
MIN_OUTPUT_CHARS = 200

# Summary fields every suite must emit. A missing line means the suite is
# reporting a different shape than the gate believes it is.
REQUIRED_SUMMARY_FIELDS = (
    "manifest numerical cases",
    "numerical executed/passed",
    "manifest rejection cases",
    "rejections executed/passed",
    "unexpected skips",
    "unexpected HTTP errors",
    "total failures",
)


def _save_artifact(suite_id: str, out: str) -> str:
    """Child output is preserved even (especially) on failure."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, f"{suite_id}.log")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    return path


def run_one(entry: Dict, base_url: str) -> Dict:
    """Run one suite and apply the full checklist. Never raises."""
    suite_id = entry["suite_id"]
    script = os.path.join(ROOT, entry["script"])
    url = base_url.rstrip("/") + entry.get("url_suffix", "")
    timeout = entry.get("timeout_s", 900)
    expected_exit = entry.get("expected_exit", 0)
    runner = ["Rscript"] if script.endswith(".R") else [sys.executable]

    problems: List[str] = []
    out = ""
    code: Optional[int] = None

    try:
        proc = subprocess.run(runner + [script, url], cwd=ROOT,
                              capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        problems.append(
            f"exceeded its {timeout}s timeout. A suite that never finishes has "
            f"not passed."
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not be executed: {type(exc).__name__}: {exc}")

    artifact = _save_artifact(suite_id, out or "<no output captured>")

    if code is not None and code != expected_exit:
        problems.append(f"exited {code}, expected {expected_exit}")

    if len(out.strip()) < MIN_OUTPUT_CHARS:
        problems.append(
            f"produced {len(out.strip())} characters of output "
            f"(< {MIN_OUTPUT_CHARS}). A suite that says nothing has not "
            f"reported a pass -- it has failed to run."
        )

    terminal_hits = sum(out.count(m) for m in TERMINAL_MARKERS)
    if terminal_hits == 0:
        problems.append(
            f"emitted no terminal verdict ({' / '.join(TERMINAL_MARKERS)}); "
            f"it did not reach the end"
        )
    elif terminal_hits > 1:
        problems.append(
            f"emitted {terminal_hits} terminal verdicts. Exactly one is "
            f"expected -- more than one means output was concatenated or a "
            f"suite ran twice, and the reported result is ambiguous."
        )

    if SUMMARY_MARKER not in out:
        problems.append("emitted no structured summary block")
    else:
        missing = [f for f in REQUIRED_SUMMARY_FIELDS if f not in out]
        if missing:
            problems.append(f"summary is missing required fields: {missing}")

    ids = SUITE_ID_RE.findall(out)
    observed_id = None
    if not ids:
        problems.append("emitted no SUITE-ID line")
    elif len(set(ids)) > 1:
        problems.append(f"emitted conflicting SUITE-IDs: {sorted(set(ids))}")
    else:
        observed_id = ids[0]
        if observed_id != suite_id:
            problems.append(
                f"reported SUITE-ID {observed_id!r} but the manifest expects "
                f"{suite_id!r}"
            )

    floor = entry.get("min_assertions")
    if floor:
        counts = ASSERTION_RE.findall(out)
        if not counts:
            problems.append("did not report an assertion count")
        else:
            passed, total = (int(counts[-1][0]), int(counts[-1][1]))
            if total < floor:
                problems.append(
                    f"ran {total} assertions, below the manifest floor of "
                    f"{floor}. Coverage shrank -- which is exactly how "
                    f"226/226 became 200/200 without anyone noticing."
                )
            if passed != total:
                problems.append(f"{total - passed} assertions failed")

    return {"suite_id": suite_id, "observed_id": observed_id,
            "problems": problems, "artifact": artifact, "output": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("positional", nargs="*",
                    help="single-script mode: <script> <url>")
    ap.add_argument("--url", default=None)
    ap.add_argument("--suite", default=None)
    args = ap.parse_args()

    with open(MANIFEST) as fh:
        manifest = json.load(fh)

    # Single-script mode, used by the sabotage harness.
    if len(args.positional) >= 2:
        script, url = args.positional[0], args.positional[1]
        rel = os.path.relpath(os.path.abspath(script), ROOT)
        entry = next((e for e in manifest["suites"] if e["script"] == rel), None)
        if entry is None:
            entry = {"suite_id": "ad-hoc", "script": rel, "url_suffix": "",
                     "timeout_s": 900, "expected_exit": 0}
            # An ad-hoc script has no declared identity, so identity and floor
            # checks cannot apply; everything else still does.
        base = url
        if entry.get("url_suffix") and url.endswith(entry["url_suffix"]):
            base = url[: -len(entry["url_suffix"])]
        res = run_one(entry, base)
        sys.stdout.write(res["output"])
        if res["problems"]:
            print(f"\nGATE RUNNER: {entry['script']} FAILED", file=sys.stderr)
            for p in res["problems"]:
                print(f"  - {p}", file=sys.stderr)
            print(f"  output preserved at {res['artifact']}", file=sys.stderr)
            return 1
        print(f"\nGATE RUNNER: {entry['script']} reported completely and passed.")
        return 0

    base_url = args.url or manifest.get("base_url_default")
    entries = manifest["suites"]
    if args.suite:
        entries = [e for e in entries if e["suite_id"] == args.suite]
        if not entries:
            print(f"unknown suite_id {args.suite!r}", file=sys.stderr)
            return 2

    expected_ids = {e["suite_id"] for e in entries}
    results = []
    for entry in entries:
        print(f"\n{'#' * 78}\n# {entry['suite_id']}  ({entry['script']})\n{'#' * 78}")
        res = run_one(entry, base_url)
        sys.stdout.write(res["output"])
        results.append(res)

    observed_ids = {r["observed_id"] for r in results if r["observed_id"]}
    failed = [r for r in results if r["problems"]]

    print("\n" + "#" * 78)
    print("# GATE RUNNER -- FINAL")
    print("#" * 78)
    print(f"  target                 : {base_url}")
    print(f"  suites declared        : {len(entries)}")
    print(f"  suites reporting       : {len(observed_ids)}")
    print(f"  suites failing         : {len(failed)}")

    problems = []
    # Missing AND extra both fail: a suite that vanishes from the pipeline is
    # invisible if only the survivors are checked.
    if observed_ids != expected_ids and not args.suite:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        if missing:
            problems.append(f"declared suites that did not report: {missing}")
        if extra:
            problems.append(f"suites reported but not declared: {extra}")

    for r in failed:
        problems.append(f"{r['suite_id']}: " + "; ".join(r["problems"])
                        + f"  [log: {r['artifact']}]")

    if problems:
        print("\nGATE RUNNER FAILURES")
        for p in problems:
            print(f"  - {p}")
        print("\nGATE RUNNER: FAILED")
        return 1

    print("\nGATE RUNNER: all declared suites reported completely and passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
