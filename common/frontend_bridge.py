"""
Bridge to the Zetyra frontend's TypeScript math modules.

Spawns a Node.js subprocess that imports the actual TypeScript source files
from the Zetyra frontend checkout and exchanges JSON with it over stdin/stdout.
This lets Python validation scripts exercise the exact code that ships to
users — not a hand-maintained mirror.

Requires Node.js 22+ for native .ts support (strip-types).

Typical usage:
    with FrontendBridge("chi_square", frontend_path) as bridge:
        result = bridge.call("chi_square_p_value", x=3.0, df=2)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_FRONTEND_PATH = Path(__file__).resolve().parent.parent.parent / "Zetyra" / "frontend"


class FrontendBridgeError(RuntimeError):
    pass


class FrontendBridge:
    """Newline-delimited JSON RPC over a persistent Node subprocess."""

    def __init__(self, driver_name: str, frontend_path: Path | str | None = None):
        self.driver_name = driver_name
        self.frontend_path = Path(frontend_path) if frontend_path else DEFAULT_FRONTEND_PATH
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        if shutil.which("node") is None:
            raise FrontendBridgeError(
                "Node.js is required to validate frontend math. Install Node 22+ "
                "and re-run, or set ZETYRA_SKIP_FRONTEND_BRIDGE=1 to skip these tests."
            )
        if not self.frontend_path.exists():
            raise FrontendBridgeError(
                f"Frontend path {self.frontend_path} not found. Pass --frontend-path "
                "or set ZETYRA_FRONTEND_PATH to the Zetyra/frontend checkout."
            )
        driver = Path(__file__).resolve().parent.parent / "free" / f"{self.driver_name}_driver.mjs"
        if not driver.exists():
            raise FrontendBridgeError(f"Driver not found: {driver}")

        loader = Path(__file__).resolve().parent.parent / "free" / "ts_loader.mjs"
        self.proc = subprocess.Popen(
            ["node", "--experimental-loader", str(loader), str(driver), str(self.frontend_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            # Silence MODULE_TYPELESS_PACKAGE_JSON and --experimental-loader
            # deprecation noise — we're intentionally importing .ts files from
            # a non-ESM package via a resolve hook.
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        return self

    def __exit__(self, *_):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    def call(self, op: str, **kwargs):
        if self.proc is None or self.proc.poll() is not None:
            raise FrontendBridgeError("bridge subprocess not running")
        req = json.dumps({"op": op, **kwargs})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise FrontendBridgeError(f"bridge died. stderr:\n{stderr}")
        resp = json.loads(line)
        if "error" in resp:
            raise FrontendBridgeError(f"{op} failed: {resp['error']}")
        return _restore_floats(resp["result"])


def _restore_floats(v):
    """Turn the driver's Infinity/-Infinity/NaN string sentinels back into floats."""
    if isinstance(v, str):
        if v == "Infinity":
            return float("inf")
        if v == "-Infinity":
            return float("-inf")
        if v == "NaN":
            return float("nan")
        return v
    if isinstance(v, list):
        return [_restore_floats(x) for x in v]
    if isinstance(v, dict):
        return {k: _restore_floats(x) for k, x in v.items()}
    return v


def resolve_frontend_path(argv_path: str | None) -> Path:
    """CLI-friendly resolver: explicit arg > env var > default peer checkout."""
    if argv_path:
        return Path(argv_path).expanduser().resolve()
    env = os.environ.get("ZETYRA_FRONTEND_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_FRONTEND_PATH
