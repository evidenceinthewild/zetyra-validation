#!/usr/bin/env python3
"""
Prove the GSD validation gate can FAIL.

    python gsd/test_gate_sabotage.py

A gate is only evidence if it is capable of reporting failure. Every one of
today's misses was a gate that was working exactly as written and still said
"ALL VALIDATIONS PASSED":

  - Above-ceiling cases were skipped, so 226/226 quietly became 200/200 and
    46/46 became 32/32.
  - An R benchmark halted on an unexpected 422 and printed two blank lines,
    which in a terminal is indistinguishable from success.

So each scenario below deliberately breaks something the gate is supposed to
catch, and asserts the gate exits NONZERO. A scenario that passes means the
corresponding protection does not exist.

Scenarios 4 and 5 need a server that misbehaves on purpose, so a local mock
stands in for the API. The rest mutate the manifest or the environment.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAMMA = os.path.join(HERE, "test_gsd_hsd_gamma.py")
FIXTURES = os.path.join(HERE, "fixtures_gsd_hsd_gamma.json")

# ---------------------------------------------------------------------------
# Mock API. `mode` selects the misbehaviour under test.
# ---------------------------------------------------------------------------
MODE = {"name": "honest"}


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        k = body.get("k", 3)
        mode = MODE["name"]

        if k > 4:
            if mode == "accepts_k5":
                # Returns a plausible design for an uncertified look count.
                return self._json(200, {
                    "n_max": 600, "n_fixed": 500,
                    "efficacy_boundaries": [3.0] * k,
                    "futility_boundaries": [0.5] * k,
                    "information_fractions": [(i + 1) / k for i in range(k)],
                    "alpha_spent": [0.025] * k, "beta_spent": [0.1] * k,
                    "expected_n_h0": 400, "expected_n_h1": 450,
                    "resolved_spending": {},
                })
            if mode == "wrong_field":
                # Rejects, but blames the wrong input. A naive status check
                # passes while the k bound is unenforced.
                return self._json(422, {"detail": [{
                    "type": "less_than_equal", "loc": ["body", "alpha"],
                    "msg": "Input should be less than or equal to 4",
                    "ctx": {"le": 4}}]})
            return self._json(422, {"detail": [{
                "type": "less_than_equal", "loc": ["body", "k"],
                "msg": "Input should be less than or equal to 4",
                "ctx": {"le": 4}}]})

        return self._json(200, {
            "n_max": 483, "n_fixed": 467, "inflation_factor": 1.03,
            "expected_n_h0": 400, "expected_n_h1": 430,
            "efficacy_boundaries": [3.0] * k, "futility_boundaries": [0.5] * k,
            "information_fractions": [(i + 1) / k for i in range(k)],
            "alpha_spent": [0.025] * k, "beta_spent": [0.1] * k,
            "resolved_spending": {"efficacy_function": "HwangShihDecani",
                                  "efficacy_hsd_gamma": -4.0,
                                  "futility_function": "HwangShihDecani",
                                  "futility_hsd_gamma": -4.0,
                                  "futility_hsd_gamma_is_auto": True},
        })

    def _json(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start_mock():
    srv = HTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ---------------------------------------------------------------------------
def run(script, url, cwd=ROOT, env=None):
    """Run a suite; return (exit_code, stdout+stderr)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    # Invoked THROUGH the runner, because that is how CI invokes them. A
    # suite that dies before printing cannot detect its own silence; the
    # caller can, because it is still alive.
    runner = os.path.join(cwd, "common", "run_gate.py")
    p = subprocess.run([sys.executable, runner, script, url], cwd=cwd, env=e,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def scenario(name, expectation, fn):
    print(f"\n--- {name}")
    print(f"    expect: {expectation}")
    try:
        code, out = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    ERROR running scenario: {exc}")
        return False
    ok = code != 0
    tail = [l for l in out.strip().split("\n") if l.strip()][-1:] or ["<no output>"]
    print(f"    exit={code}  {'CAUGHT' if ok else 'NOT CAUGHT'}   {tail[0][:88]}")
    return ok


def with_mutated_fixture(mutate, url):
    """Copy the tree, mutate the fixture, run the gamma suite against `url`."""
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "results", "*.pyc"))
        path = os.path.join(dst, "gsd", "fixtures_gsd_hsd_gamma.json")
        cases = json.load(open(path))
        cases = mutate(cases)
        json.dump(cases, open(path, "w"))
        return run(os.path.join(dst, "gsd", "test_gsd_hsd_gamma.py"), url, cwd=dst)


def with_patched_suite(patch, url):
    """Copy the tree, textually patch the suite, run it."""
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "results", "*.pyc"))
        path = os.path.join(dst, "gsd", "test_gsd_hsd_gamma.py")
        src = open(path).read()
        open(path, "w").write(patch(src))
        return run(path, url, cwd=dst)


def main():
    srv, url = start_mock()
    outcomes = []

    # 1. A case vanishes from the manifest mid-run.
    outcomes.append(scenario(
        "1. remove one numerical case from the executed set",
        "gate fails: executed numerical ID set != manifest",
        lambda: with_patched_suite(
            lambda s: s.replace("    for case in numerical:",
                                "    for case in numerical[1:]:", 1), url)))

    # 2. Duplicated ID with another omitted -- the case a COUNT cannot catch.
    outcomes.append(scenario(
        "2. duplicate one case ID (and drop another, so totals match)",
        "gate fails: duplicate ID detected despite unchanged total",
        lambda: with_mutated_fixture(
            lambda cs: [dict(cs[0])] + cs[1:-1] + [dict(cs[0])], url)))

    # 3. An expected rejection is skipped rather than asserted.
    outcomes.append(scenario(
        "3. convert an expected rejection into a skip",
        "gate fails: rejection ID missing from run",
        lambda: with_patched_suite(
            lambda s: s.replace("    for case in rejections:",
                                "    for case in []:", 1), url)))

    # 4. API accepts an uncertified look count.
    def s4():
        MODE["name"] = "accepts_k5"
        try:
            return run(GAMMA, url)
        finally:
            MODE["name"] = "honest"
    outcomes.append(scenario(
        "4. API returns 200 for k=5",
        "gate fails: uncertified design was produced",
        s4))

    # 5. API rejects, but on the wrong field.
    def s5():
        MODE["name"] = "wrong_field"
        try:
            return run(GAMMA, url)
        finally:
            MODE["name"] = "honest"
    outcomes.append(scenario(
        "5. API returns 422 blaming the wrong field",
        "gate fails: refused for the wrong reason",
        s5))

    # 6. Suite produces no output at all.
    outcomes.append(scenario(
        "6. suite emits blank stdout",
        "gate fails: empty output must never read as success",
        lambda: with_patched_suite(
            lambda s: s.replace("def main():", "def main():\n    sys.exit(0)\n", 1)
            .replace("    sys.exit(0)\n", "    import os as _o; _o._exit(0)\n", 1),
            url)))

    # 7. R benchmark aborts before its summary (the real failure mode seen).
    def s7():
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "repo")
            shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "results", "*.pyc"))
            rp = os.path.join(dst, "gsd", "test_gsdesign_benchmark.R")
            src = open(rp).read()
            src = src.replace("for (config in test_configs) {",
                              'stop("simulated abort before summary")\nfor (config in test_configs) {', 1)
            open(rp, "w").write(src)
            p = subprocess.run(
                [sys.executable, os.path.join(dst, "common", "run_gate.py"),
                 rp, url + "/api/v1/validation"],
                cwd=dst, capture_output=True, text=True, timeout=900)
            return p.returncode, p.stdout + p.stderr
    outcomes.append(scenario(
        "7. R benchmark aborts before printing its summary",
        "nonzero exit: a halt must not read as success",
        s7))

    srv.shutdown()
    print("\n" + "=" * 78)
    caught = sum(outcomes)
    print(f"SABOTAGE MATRIX: {caught}/{len(outcomes)} scenarios caught")
    if caught == len(outcomes):
        print("The gate can fail. Its green result is evidence.")
        sys.exit(0)
    print("SOME SABOTAGE WENT UNDETECTED -- the gate reports success it "
          "cannot justify.")
    sys.exit(1)


if __name__ == "__main__":
    main()
