#!/usr/bin/env python3
"""
Fail if CI invokes a protected suite directly instead of through the runner.

    python common/check_gate_wiring.py

WHY
---
The parent runner is what detects a suite that dies silently, reports the wrong
identity, or shrinks its coverage. All of that protection is bypassed the
moment a workflow calls `python gsd/test_gsd_futility.py` directly -- and that
is a very natural thing for someone to write while debugging, or while
"simplifying" a workflow they find noisy.

The failure would be invisible: the suite still runs, still prints, still exits
0 on success. Only the checking disappears. So the wiring itself is asserted.

This also catches the inverse problem, which was live when this was written:
a protected suite that is in the manifest but absent from CI entirely. The
gamma and futility suites -- the two carrying the 226 and 46 assertion sets --
were never wired into the workflow at all, so the most important gate in the
repository ran only when someone ran it by hand.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
MANIFEST = os.path.join(ROOT, "gate_manifest.json")

# A line may name a protected script only if the runner is on the same line.
RUNNER_TOKENS = ("run_gate.py",)


def main() -> int:
    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    protected = {e["script"]: e["suite_id"] for e in manifest["suites"]}
    basenames = {os.path.basename(p): p for p in protected}

    if not os.path.isdir(WORKFLOWS):
        print(f"no workflow directory at {WORKFLOWS}", file=sys.stderr)
        return 1

    direct_calls = []
    seen_via_runner = set()
    files = [os.path.join(WORKFLOWS, f) for f in sorted(os.listdir(WORKFLOWS))
             if f.endswith((".yml", ".yaml"))]

    for path in files:
        for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            via_runner = any(t in line for t in RUNNER_TOKENS)
            for base, rel in basenames.items():
                if base not in line:
                    continue
                if via_runner:
                    seen_via_runner.add(rel)
                    continue
                # Naming a protected script without the runner on the same
                # line is a direct invocation.
                if re.search(r"(python3?|Rscript)\s+\S*" + re.escape(base), line):
                    direct_calls.append(
                        f"{os.path.relpath(path, ROOT)}:{lineno}: "
                        f"{stripped[:90]}"
                    )

    # Every protected suite must be reachable from CI, through the runner or
    # through a manifest-driven invocation.
    manifest_driven = any(
        "run_gate.py" in open(p, encoding="utf-8").read() for p in files
    )

    # The oracle identity override must never be set in CI. It exists so a
    # human can deliberately retire a case_id during a local regeneration,
    # printing every affected case with old and new fingerprints. Setting it in
    # a workflow would let an in-place retarget pass unreviewed -- which is the
    # precise failure the fingerprint guard was added to stop.
    override_hits = []
    for path in files:
        for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
            if "GSD_ALLOW_IDENTITY_CHANGE" in line and not line.strip().startswith("#"):
                override_hits.append(
                    f"{os.path.relpath(path, ROOT)}:{lineno}: {line.strip()[:90]}")

    problems = []
    if override_hits:
        problems.append(
            "GSD_ALLOW_IDENTITY_CHANGE appears in a workflow. That override "
            "disables the oracle case-identity guard and belongs only in a "
            "deliberate local regeneration:\n    " + "\n    ".join(override_hits)
        )
    if direct_calls:
        problems.append(
            "protected suites invoked directly, bypassing the runner "
            "(and therefore bypassing silence, identity and coverage "
            "detection):\n    " + "\n    ".join(direct_calls)
        )
    if not manifest_driven:
        problems.append(
            "no workflow invokes common/run_gate.py at all. The protected "
            "suites are unguarded in CI."
        )

    if problems:
        print("GATE WIRING CHECK FAILED\n")
        for p in problems:
            print(f"  - {p}\n")
        print("Invoke protected suites as:\n"
              "    python common/run_gate.py            # whole manifest\n"
              "    python common/run_gate.py --suite <suite_id>\n")
        return 1

    print(f"GATE WIRING OK: {len(protected)} protected suites, "
          f"all invoked through common/run_gate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
