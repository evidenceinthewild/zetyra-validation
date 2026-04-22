#!/usr/bin/env python3
"""
Validate the Chi-Square Calculator math shipped in the frontend (Free Tier).

The Zetyra Chi-Square Calculator runs entirely client-side — there is no
backend endpoint. This script exercises the actual TypeScript module that
ships to users (via a Node.js subprocess bridge), NOT a hand-maintained
Python mirror. If page.tsx regresses, these tests will catch it.

The TS module under test is:
    frontend/src/lib/stats/chi_square.ts
imported directly by:
    frontend/src/app/calculators/chi-square/page.tsx

Requires Node.js 22+ (native TypeScript strip-types support).

This script validates:
   1. Chi-square survival function (p-value) vs scipy.stats.chi2.sf
   2. Chi-square critical value (inverse CDF) vs scipy.stats.chi2.ppf
   3. Pearson 2x2 with Yates correction vs scipy.stats.chi2_contingency
   4. Pearson r×c without correction vs scipy (3x3, 2x4, 3x5 tables)
   5. Cramér's V and φ vs hand-calculated reference
   6. McNemar classical chi-square statistic and p-value
   7. Fisher's exact 2x2 two-sided vs scipy.stats.fisher_exact
   8. Edge cases (Yates null, proportional r×c null, extreme deviation underflow)

References:
  * Yates (1934) "Contingency tables involving small numbers and the χ² test."
  * McNemar (1947) "Note on the sampling error of the difference between
    correlated proportions or percentages."
  * Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences."
  * Abramowitz & Stegun (1965) 7.1.26 (erf approx, df=1 fast path).
  * Press et al. (2007) Numerical Recipes §6.2 (Lentz continued fraction).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import chi2, fisher_exact

from common.frontend_bridge import FrontendBridge, resolve_frontend_path


# ─── p-value ──────────────────────────────────────────────────────────

def validate_p_value(bridge: FrontendBridge) -> pd.DataFrame:
    """Chi-square p-value matches scipy.stats.chi2.sf."""
    cases = [
        (1, 0.5), (1, 1.0), (1, 3.841), (1, 5.0), (1, 10.0), (1, 30.0),
        (2, 0.5), (2, 3.0), (2, 5.99), (2, 10.0),
        (5, 1.0), (5, 5.0), (5, 11.07), (5, 15.0),
        (10, 5.0), (10, 18.31), (10, 30.0),
        (20, 10.0), (20, 31.41), (20, 50.0),
    ]
    rows = []
    for df, x in cases:
        p_fe = bridge.call("chi_square_p_value", x=x, df=df)
        p_sp = float(chi2.sf(x, df))
        dev = abs(p_fe - p_sp)
        rows.append({
            "test": f"p-value df={df} x={x}",
            "frontend_p": round(p_fe, 10),
            "scipy_p": round(p_sp, 10),
            "deviation": dev,
            "pass": dev < 1.5e-6,
        })
    return pd.DataFrame(rows)


# ─── critical values ──────────────────────────────────────────────────

def validate_critical_value(bridge: FrontendBridge) -> pd.DataFrame:
    """Chi-square critical values match scipy.stats.chi2.ppf to ≤ 5e-4."""
    cases = [(0.001, 1), (0.01, 1), (0.05, 1), (0.10, 1),
             (0.01, 2), (0.05, 2), (0.10, 2),
             (0.01, 5), (0.05, 5), (0.10, 5),
             (0.05, 10), (0.05, 20)]
    rows = []
    for alpha, df in cases:
        cv_fe = bridge.call("chi_square_critical", alpha=alpha, df=df)
        cv_sp = float(chi2.ppf(1 - alpha, df))
        dev = abs(cv_fe - cv_sp)
        rows.append({
            "test": f"critical α={alpha} df={df}",
            "frontend_cv": round(cv_fe, 6),
            "scipy_cv": round(cv_sp, 6),
            "deviation": dev,
            "pass": dev < 5e-4,
        })
    return pd.DataFrame(rows)


# ─── Pearson 2x2 with Yates ───────────────────────────────────────────

def validate_pearson_2x2_yates(bridge: FrontendBridge) -> pd.DataFrame:
    """2x2 Pearson chi-square with Yates correction matches scipy."""
    cases = [
        [[50, 50], [30, 70]],
        [[90, 10], [10, 90]],
        [[5, 20], [20, 5]],
        [[100, 100], [95, 105]],
        [[8, 2], [1, 9]],
    ]
    rows = []
    for tbl in cases:
        r = bridge.call("pearson_chi_square", table=tbl)
        sp_chi, sp_p, _, _ = sp_stats.chi2_contingency(tbl, correction=True)
        chi_dev = abs(r["chiSq"] - sp_chi)
        p_dev = abs(r["pValue"] - sp_p)
        rows.append({
            "test": f"2x2 Yates {tbl}",
            "frontend_chi": round(r["chiSq"], 6),
            "scipy_chi": round(sp_chi, 6),
            "frontend_p": round(r["pValue"], 8),
            "scipy_p": round(sp_p, 8),
            "pass": chi_dev < 1e-6 and p_dev < 1e-5,
        })
    return pd.DataFrame(rows)


# ─── Pearson r×c ──────────────────────────────────────────────────────

def validate_pearson_rxc(bridge: FrontendBridge) -> pd.DataFrame:
    """r×c Pearson chi-square (no correction — only 2x2 uses Yates)."""
    cases = [
        [[10, 20, 30], [15, 25, 35], [20, 30, 40]],
        [[25, 30, 45, 10], [15, 25, 20, 40]],
        [[100, 50, 50, 50, 50], [50, 100, 50, 50, 50], [50, 50, 100, 50, 50]],
    ]
    rows = []
    for tbl in cases:
        r = bridge.call("pearson_chi_square", table=tbl)
        sp_chi, sp_p, sp_df, _ = sp_stats.chi2_contingency(tbl, correction=False)
        chi_dev = abs(r["chiSq"] - sp_chi)
        p_dev = abs(r["pValue"] - sp_p)
        rows.append({
            "test": f"{len(tbl)}x{len(tbl[0])} no-correction",
            "frontend_chi": round(r["chiSq"], 6),
            "scipy_chi": round(sp_chi, 6),
            "frontend_df": r["df"], "scipy_df": sp_df,
            "pass": chi_dev < 1e-6 and p_dev < 1e-5 and r["df"] == sp_df,
        })
    return pd.DataFrame(rows)


# ─── Cramér's V and φ ─────────────────────────────────────────────────

def validate_effect_sizes(bridge: FrontendBridge) -> pd.DataFrame:
    """Cramér's V and φ match hand calculation."""
    r = bridge.call("pearson_chi_square", table=[[90, 10], [10, 90]])
    phi_hand = math.sqrt(r["chiSq"] / r["N"])
    v_hand = phi_hand  # V = phi for 2x2
    rows = [
        {"test": "φ = √(χ²/N) for 2×2 [[90,10],[10,90]]",
         "frontend_phi": round(r["phi"], 6),
         "hand_phi": round(phi_hand, 6),
         "pass": abs(r["phi"] - phi_hand) < 1e-6},
        {"test": "Cramér's V = φ for 2×2",
         "frontend_V": round(r["cramersV"], 6),
         "hand_V": round(v_hand, 6),
         "pass": abs(r["cramersV"] - v_hand) < 1e-6},
    ]

    r2 = bridge.call("pearson_chi_square", table=[[10, 20, 30], [15, 25, 35], [20, 30, 40]])
    v_hand_3x3 = math.sqrt(r2["chiSq"] / (r2["N"] * 2))
    rows.append({
        "test": "Cramér's V = √(χ²/(N·(k-1))) for 3×3",
        "frontend_V": round(r2["cramersV"], 6),
        "hand_V": round(v_hand_3x3, 6),
        "pass": abs(r2["cramersV"] - v_hand_3x3) < 1e-6,
    })
    return pd.DataFrame(rows)


# ─── McNemar ──────────────────────────────────────────────────────────

def validate_mcnemar(bridge: FrontendBridge) -> pd.DataFrame:
    """McNemar classical: χ² = (b-c)²/(b+c), df=1."""
    cases = [(80, 12, 25, 83), (100, 10, 10, 100), (50, 30, 5, 15), (200, 4, 16, 180)]
    rows = []
    for a, b, c, d in cases:
        r = bridge.call("mcnemar_classical", a=a, b=b, c=c, d=d)
        ref_chi = (b - c) ** 2 / (b + c)
        ref_p = float(chi2.sf(ref_chi, 1))
        chi_dev = abs(r["chiSq"] - ref_chi)
        p_dev = abs(r["pValue"] - ref_p)
        rows.append({
            "test": f"McNemar ({a},{b},{c},{d})",
            "frontend_chi": round(r["chiSq"], 6),
            "ref_chi": round(ref_chi, 6),
            "frontend_p": round(r["pValue"], 8),
            "ref_p": round(ref_p, 8),
            "discordant": r["discordant"],
            "pass": chi_dev < 1e-10 and p_dev < 1.5e-6,
        })
    return pd.DataFrame(rows)


# ─── Fisher's exact ───────────────────────────────────────────────────

def validate_fisher_exact(bridge: FrontendBridge) -> pd.DataFrame:
    """Fisher's 2x2 two-sided p-value vs scipy.stats.fisher_exact."""
    cases = [(7, 3, 1, 9), (10, 0, 0, 10), (5, 5, 4, 6), (20, 5, 3, 22),
             (1, 9, 8, 2)]
    rows = []
    for a, b, c, d in cases:
        p_fe = bridge.call("fisher_exact_2x2", a=a, b=b, c=c, d=d)
        _, p_sp = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        dev = abs(p_fe - p_sp)
        rows.append({
            "test": f"Fisher 2x2 ({a},{b},{c},{d})",
            "frontend_p": round(p_fe, 8),
            "scipy_p": round(p_sp, 8),
            "deviation": dev,
            "pass": dev < 1e-8,
        })
    return pd.DataFrame(rows)


# ─── Edge cases ───────────────────────────────────────────────────────

def validate_edge_cases(bridge: FrontendBridge) -> pd.DataFrame:
    """Yates null case, r×c null, extreme-deviation underflow."""
    r_null = bridge.call("pearson_chi_square", table=[[25, 25], [25, 25]])
    r_null_3x3 = bridge.call("pearson_chi_square",
                             table=[[10, 20, 30], [10, 20, 30], [10, 20, 30]])
    r_extreme = bridge.call("pearson_chi_square",
                            table=[[1000, 10], [10, 1000]])
    rows = [
        {"test": "Balanced 2×2 [[25,25],[25,25]] -> Yates-adjusted χ²=0.04, p≈0.84",
         "chi_sq": round(r_null["chiSq"], 4),
         "p_value": round(r_null["pValue"], 4),
         "pass": abs(r_null["chiSq"] - 0.04) < 1e-10 and 0.80 < r_null["pValue"] < 0.90},
        {"test": "Proportional 3×3 (no correction) -> χ²=0, p=1",
         "chi_sq": round(r_null_3x3["chiSq"], 6),
         "p_value": round(r_null_3x3["pValue"], 6),
         "pass": r_null_3x3["chiSq"] < 1e-10 and r_null_3x3["pValue"] > 0.99},
        {"test": "Extreme deviation [[1000,10],[10,1000]] -> p underflows to ≈ 0",
         "chi_sq": round(r_extreme["chiSq"], 2),
         "p_value": r_extreme["pValue"],
         "pass": r_extreme["pValue"] < 1e-100},
    ]
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default=None,
                        help="unused; kept for CI symmetry with other scripts")
    parser.add_argument("--frontend-path", default=None,
                        help="path to Zetyra/frontend (env ZETYRA_FRONTEND_PATH also works)")
    args = parser.parse_args()
    _ = args.base_url

    frontend_path = resolve_frontend_path(args.frontend_path)

    suites = [
        ("1. Chi-square p-value (survival function)", validate_p_value),
        ("2. Chi-square critical value", validate_critical_value),
        ("3. Pearson 2x2 with Yates correction", validate_pearson_2x2_yates),
        ("4. Pearson r×c (no correction)", validate_pearson_rxc),
        ("5. Effect sizes (φ, Cramér's V)", validate_effect_sizes),
        ("6. McNemar classical", validate_mcnemar),
        ("7. Fisher's exact 2×2 two-sided", validate_fisher_exact),
        ("8. Edge cases", validate_edge_cases),
    ]

    print("=" * 70)
    print("CHI-SQUARE CALCULATOR VALIDATION (frontend TS module via Node bridge)")
    print(f"frontend: {frontend_path}")
    print("=" * 70)

    all_pass = True
    all_frames: list[pd.DataFrame] = []
    with FrontendBridge("chi_square", frontend_path) as bridge:
        for name, fn in suites:
            print(f"\n{name}")
            print("-" * 70)
            try:
                df = fn(bridge)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_pass = False
                continue
            print(df.to_string(index=False))
            df_out = df.copy()
            df_out.insert(0, "suite", name)
            all_frames.append(df_out)
            if not df["pass"].all():
                all_pass = False

    if all_frames:
        os.makedirs("results", exist_ok=True)
        all_results = pd.concat(all_frames, ignore_index=True, sort=False)
        all_results.to_csv("results/chi_square_validation.csv", index=False)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        return 0
    print("SOME VALIDATIONS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
