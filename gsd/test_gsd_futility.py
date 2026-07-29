#!/usr/bin/env python3
"""
Validate Zetyra GSD futility boundaries against gsDesign test.type=4.

Why this file exists
--------------------
The existing gsDesign benchmark (test_gsdesign_benchmark.R) runs at
test.type=1, which validates EFFICACY boundaries only. Nothing compared a
futility bound to a reference, and a defect lived there undetected: futility
was derived by inverting the beta-spending function against a standard normal,
omitting the drift term. That put the bounds deep in the negative tail
(e.g. [-1.69, -1.43] where gsDesign gives [+0.377, +1.279]), so a design
essentially never stopped for futility, and the sample size was computed as
though futility stopping cost no power.

Futility is not a cosmetic output: it drives the expected sample size a sponsor
plans around, it is written into DMC charters, and it changes the maximum N by
roughly 22% for a standard 3-look design. It needs its own oracle.

Reference values
----------------
Generated with gsDesign 3.10.1:

    gsDesign(k=K, test.type=4, alpha=0.025, beta=1-power,
             sfu=sfLDOF, sfl=sfLDPocock)

test.type=4 is non-binding futility: efficacy bounds are derived ignoring
futility (so Type I error holds whether or not a futility stop is honoured),
while the sample size is inflated to recover the power that futility stopping
gives away.

Also checks the efficacy-only escape hatch (beta_spending_function="none"),
which must reproduce gsDesign test.type=1 inflation rather than silently
charging the futility premium.

Usage:
    python gsd/test_gsd_futility.py [base_url]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.gsd_gate import (
    CERTIFIED_MAX_K, Gate, check_rejection,
)
import pandas as pd

# Same tolerances as the backend fixture oracle
# (backend/tests/test_gsd.py::test_futility_matches_gsdesign_test_type_4), so a
# deployment that passes here agrees with gsDesign to the same degree the unit
# tests demand. Measured agreement is ~0.002 on bounds, ~0.0017 on the
# inflation ratio and ~0.0016 on expected N. Deliberately not loosened: a wide
# tolerance is exactly how the efficacy fixtures previously failed to notice an
# incorrect reference.
BOUNDARY_TOLERANCE = 0.003
RATIO_TOLERANCE = 0.0025
EN_TOLERANCE = 0.003

# gsDesign 3.10.1, test.type=4, alpha=0.025, sfu=sfLDOF, sfl=sfLDPocock.
# Emitted directly by R -- do not hand-transcribe these. Regenerate with:
#   Rscript -e 'library(gsDesign); x <- gsDesign(k=K, test.type=4, alpha=0.025,
#     beta=0.10, sfu=sfLDOF, sfl=sfLDPocock);
#     cat(x$upper$bound, x$lower$bound, x$n.I[K], x$en[1]/max(x$n.I))' 
GSDESIGN_T4 = [
    {
        "name": "k=2, 90% power",
        "k": 2, "alpha": 0.025, "power": 0.90, "effect_size": 0.3,
        "z_efficacy": [2.9626, 1.9686],
        "z_futility": [0.9231, 1.9686],
        "inflation_ratio": 1.15304,
        "en_h0_ratio": 0.5882,
    },
    {
        "name": "k=3, 90% power",
        "k": 3, "alpha": 0.025, "power": 0.90, "effect_size": 0.3,
        "z_efficacy": [3.7103, 2.5114, 1.9930],
        "z_futility": [0.3767, 1.2785, 1.9930],
        "inflation_ratio": 1.22232,
        "en_h0_ratio": 0.4783,
    },
    {
        "name": "k=4, 90% power",
        "k": 4, "alpha": 0.025, "power": 0.90, "effect_size": 0.3,
        "z_efficacy": [4.3326, 2.9631, 2.3590, 2.0141],
        "z_futility": [0.0182, 0.8292, 1.4537, 2.0141],
        "inflation_ratio": 1.26192,
        "en_h0_ratio": 0.4307,
    },
    {
        "name": "k=5, 90% power",
        "k": 5, "alpha": 0.025, "power": 0.90, "effect_size": 0.3,
        "z_efficacy": [4.8769, 3.3570, 2.6803, 2.2898, 2.0310],
        "z_futility": [-0.2428, 0.5024, 1.0766, 1.5580, 2.0310],
        "inflation_ratio": 1.28741,
        "en_h0_ratio": 0.4046,
    },
]


def _rows_for_case(client, case):
    """Compare one design against its gsDesign reference."""
    rows = []
    r = client.gsd(
        k=case["k"],
        alpha=case["alpha"],
        power=case["power"],
        effect_size=case["effect_size"],
        spending_function="OBrienFleming",
        beta_spending_function="Pocock",
        test_type="one_sided",
    )
    # /api/v1/validation/gsd returns a FLAT shape (n_max, efficacy_boundaries,
    # ...), not the calculator's nested one (max_sample_size,
    # boundaries.efficacy, ...). Read the flat fields.
    eff = r["efficacy_boundaries"]
    fut = r["futility_boundaries"]

    for i, expected in enumerate(case["z_efficacy"]):
        dev = abs(eff[i] - expected)
        rows.append({
            "test": f"[{case['name']}] efficacy look {i+1}",
            "zetyra": round(eff[i], 4), "gsdesign": expected,
            "deviation": round(dev, 5), "pass": dev < BOUNDARY_TOLERANCE,
        })

    for i, expected in enumerate(case["z_futility"]):
        got = fut[i]
        if got is None:
            rows.append({
                "test": f"[{case['name']}] futility look {i+1}",
                "zetyra": None, "gsdesign": expected,
                "deviation": None, "pass": False,
            })
            continue
        dev = abs(got - expected)
        rows.append({
            "test": f"[{case['name']}] futility look {i+1}",
            "zetyra": round(got, 4), "gsdesign": expected,
            "deviation": round(dev, 5), "pass": dev < BOUNDARY_TOLERANCE,
        })

    # Futility must sit strictly inside the continuation region at interims and
    # meet efficacy at the final look -- reaching the end without crossing IS
    # the futility outcome.
    ordered = all(
        fut[i] < eff[i] for i in range(len(eff) - 1) if fut[i] is not None
    )
    rows.append({
        "test": f"[{case['name']}] futility < efficacy at interims",
        "zetyra": str([round(f, 3) if f is not None else None for f in fut]),
        "gsdesign": "strict at interims", "deviation": None, "pass": ordered,
    })
    meet = fut[-1] is not None and abs(fut[-1] - eff[-1]) < 1e-3
    rows.append({
        "test": f"[{case['name']}] futility meets efficacy at final look",
        "zetyra": round(fut[-1], 4) if fut[-1] is not None else None,
        "gsdesign": round(eff[-1], 4), "deviation": None, "pass": meet,
    })

    # Inflation: futility stopping costs power, which must be paid for in N.
    ratio = r["n_max"] / r["n_fixed"]
    dev = abs(ratio - case["inflation_ratio"])
    rows.append({
        "test": f"[{case['name']}] max-N inflation ratio",
        "zetyra": round(ratio, 5), "gsdesign": case["inflation_ratio"],
        "deviation": round(dev, 5), "pass": dev < RATIO_TOLERANCE,
    })

    en0 = r["expected_n_h0"] / r["n_max"]
    dev = abs(en0 - case["en_h0_ratio"])
    rows.append({
        "test": f"[{case['name']}] E[N|H0] / max N",
        "zetyra": round(en0, 4), "gsdesign": case["en_h0_ratio"],
        "deviation": round(dev, 5), "pass": dev < EN_TOLERANCE,
    })
    return rows



def _rejection_rows_for_case(client, case, gate):
    """An above-ceiling case must be REFUSED, and refused on `k`.

    Emits exactly the labels this case used to emit numerically, so the
    assertion count is unchanged and only the meaning differs. Skipping it
    instead -- which is what the suite did first -- dropped the total from
    46 to 32 while still printing ALL VALIDATIONS PASSED.
    """
    name = case["name"]
    resp = client.gsd_raw(
        k=case["k"], alpha=case["alpha"], power=case["power"],
        effect_size=case["effect_size"],
        spending_function="OBrienFleming",
        beta_spending_function="Pocock",
        test_type="one_sided",
    )
    check_rejection(resp, name, gate, expected_field="k")
    ok = gate.results[-1].passed
    detail = gate.results[-1].detail

    body = {}
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        pass
    clean = ok and "n_max" not in body

    labels = [f"[{name}] efficacy look {i+1}" for i in range(len(case["z_efficacy"]))]
    labels += [f"[{name}] futility look {i+1}" for i in range(len(case["z_futility"]))]
    labels += [f"[{name}] futility < efficacy at interims",
               f"[{name}] futility meets efficacy at final look",
               f"[{name}] max-N inflation ratio",
               f"[{name}] E[N|H0] / max N"]

    rows = [{
        "test": labels[0] + "  [EXPECTED REJECTION]",
        "zetyra": f"HTTP {resp.status_code}",
        "gsdesign": (f"422 less_than_equal on k, le={CERTIFIED_MAX_K}"
                     if ok else detail[:60]),
        "deviation": None, "pass": ok,
    }]
    for label in labels[1:]:
        rows.append({
            "test": label + "  [NOT PRODUCED]",
            "zetyra": "absent",
            "gsdesign": "no numerical output for an uncertified design",
            "deviation": None, "pass": clean,
        })
    return rows


def _rows_for_efficacy_only(client):
    """beta_spending_function='none' must give a gsDesign test.type=1 design."""
    rows = []
    r = client.gsd(
        k=3, alpha=0.025, power=0.90, effect_size=0.3,
        spending_function="OBrienFleming",
        beta_spending_function="none",
        test_type="one_sided",
    )
    fut = r["futility_boundaries"]

    rows.append({
        "test": "[efficacy-only] no futility boundaries returned",
        "zetyra": str(fut), "gsdesign": "all null",
        "deviation": None, "pass": all(f is None for f in fut),
    })

    # gsDesign test.type=1, k=3, sfu=sfLDOF: n.I[3] = 1.01185.
    ratio = r["n_max"] / r["n_fixed"]
    dev = abs(ratio - 1.01185)
    rows.append({
        "test": "[efficacy-only] inflation matches test.type=1",
        "zetyra": round(ratio, 5), "gsdesign": 1.01185,
        "deviation": round(dev, 5), "pass": dev < 0.01,
    })
    return rows


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url) if base_url else get_client()

    numerical = [c for c in GSDESIGN_T4 if c["k"] <= CERTIFIED_MAX_K]
    rejections = [c for c in GSDESIGN_T4 if c["k"] > CERTIFIED_MAX_K]
    gate = Gate(
        suite="GSD futility (gsDesign test.type=4)",
        suite_id="gsd-futility",
        expected_numerical={c["name"] for c in numerical} | {"efficacy-only"},
        expected_rejection={c["name"] for c in rejections},
    )
    ids = [c["name"] for c in GSDESIGN_T4]
    if len(ids) != len(set(ids)):
        print(f"DUPLICATE CASE IDS: {sorted({i for i in ids if ids.count(i) > 1})}")
        sys.exit(1)

    print("=" * 78)
    print("GSD FUTILITY VALIDATION vs gsDesign 3.10.1 (test.type=4)")
    print(f"{len(numerical)} numerical designs + {len(rejections)} expected "
          f"rejections (ceiling k<={CERTIFIED_MAX_K})")
    print("=" * 78)

    results = []
    for case in numerical:
        rows = _rows_for_case(client, case)
        results.extend(rows)
        gate.record_numerical(case["name"], all(r["pass"] for r in rows),
                              "; ".join(r["test"] for r in rows if not r["pass"])[:200])
    for case in rejections:
        results.extend(_rejection_rows_for_case(client, case, gate))

    eff = _rows_for_efficacy_only(client)
    results.extend(eff)
    gate.record_numerical("efficacy-only", all(r["pass"] for r in eff),
                          "; ".join(r["test"] for r in eff if not r["pass"])[:200])

    df = pd.DataFrame(results)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "gsd_futility_validation.csv"), index=False)

    rej_rows = sum(1 for r in results
                   if "EXPECTED REJECTION" in r["test"] or "NOT PRODUCED" in r["test"])
    print(f"\n{int(df['pass'].sum())}/{len(df)} assertions passed "
          f"({len(df) - rej_rows} numerical, {rej_rows} rejection)")
    gate.finish()


if __name__ == "__main__":
    main()
