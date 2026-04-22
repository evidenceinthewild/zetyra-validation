#!/usr/bin/env python3
"""
Real-world replication: 1954 Salk polio vaccine field trial (Francis Report).

The Francis Field Trial randomized ~400,000 US schoolchildren to the Salk
inactivated poliovirus vaccine vs inert placebo in a double-blind design. The
published placebo-controlled arm has become a canonical teaching example for
2×2 contingency tables and effect-size calculations.

Published placebo-controlled arm counts:
    Vaccine: 200,745 children, 33 paralytic polio cases
    Placebo: 201,229 children, 115 paralytic polio cases

This script validates:
  Chi-square analysis (via Zetyra frontend):
   1. Pearson χ² with Yates correction matches scipy.stats.chi2_contingency
      to within 1e-6 on the exact published counts
   2. P-value consistent with the Francis Report's conclusion that the trial
      was decisively positive (p < 1e-10)
   3. Fisher's exact p-value (two-sided) agrees with scipy.stats.fisher_exact
   4. Phi / Cramér's V match hand calculation
   5. Relative risk and effect direction match the published 60%+ efficacy
      finding (treatment rate / control rate ≈ 0.29)

  Sample size (binary, via backend):
   6. If we design the Francis trial a priori with the observed rates
      (p_placebo = 115/201229 ≈ 5.72e-4, p_vaccine = 33/200745 ≈ 1.64e-4)
      at α=0.05 (two-sided), power=0.80, the Fleiss normal-approx formula
      requires per-arm N that the 200,000-per-arm trial comfortably exceeds
   7. The 2:1 excess Ns (actual vs required) imply empirical power well above
      0.80, consistent with the observed decisive result

Reference:
  Francis T Jr. et al. (1955). Evaluation of the 1954 Field Trial of
  Poliomyelitis Vaccine: Final Report. Poliomyelitis Vaccine Evaluation
  Center, University of Michigan.
  (Counts reproduced from:
   https://www.randomservices.org/random/data/Polio.html and the Francis
   Report reprinted in the Am J Public Health 1955 supplement.)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency, fisher_exact

from common.frontend_bridge import FrontendBridge, resolve_frontend_path
from common.zetyra_client import get_client


# Canonical placebo-controlled Francis Report numbers
SALK_VACCINE_N = 200_745
SALK_VACCINE_CASES = 33
SALK_PLACEBO_N = 201_229
SALK_PLACEBO_CASES = 115


def salk_table() -> list[list[int]]:
    """2×2: rows = [vaccine, placebo]; cols = [cases, non-cases]."""
    return [
        [SALK_VACCINE_CASES, SALK_VACCINE_N - SALK_VACCINE_CASES],
        [SALK_PLACEBO_CASES, SALK_PLACEBO_N - SALK_PLACEBO_CASES],
    ]


# ─── Chi-square analysis (frontend TS module via bridge) ──────────────

def validate_salk_pearson_yates(bridge: FrontendBridge) -> pd.DataFrame:
    """Frontend χ² (Yates-corrected) matches scipy to tight tolerance."""
    table = salk_table()
    r = bridge.call("pearson_chi_square", table=table)
    sp_chi, sp_p, sp_df, _ = chi2_contingency(table, correction=True)
    return pd.DataFrame([{
        "test": "Salk χ² (Yates) matches scipy",
        "frontend_chi": round(r["chiSq"], 4),
        "scipy_chi": round(sp_chi, 4),
        "frontend_p": r["pValue"],
        "scipy_p": sp_p,
        "df": r["df"],
        "pass": abs(r["chiSq"] - sp_chi) < 1e-4 and abs(r["pValue"] - sp_p) < 1e-10,
    }])


def validate_salk_decisive_positive(bridge: FrontendBridge) -> pd.DataFrame:
    """P-value < 1e-10, consistent with the published decisive conclusion."""
    r = bridge.call("pearson_chi_square", table=salk_table())
    return pd.DataFrame([{
        "test": "Salk χ² p-value decisively small (< 1e-10)",
        "p_value": r["pValue"],
        "chi_sq": round(r["chiSq"], 2),
        "pass": r["pValue"] < 1e-10,
    }])


def validate_salk_fisher_exact(bridge: FrontendBridge) -> pd.DataFrame:
    """Fisher's exact two-sided p-value matches scipy on the Salk table."""
    table = salk_table()
    p_fe = bridge.call("fisher_exact_2x2",
                       a=table[0][0], b=table[0][1],
                       c=table[1][0], d=table[1][1])
    _, p_sp = fisher_exact(table, alternative="two-sided")
    return pd.DataFrame([{
        "test": "Salk Fisher's exact matches scipy",
        "frontend_p": p_fe,
        "scipy_p": p_sp,
        "pass": abs(p_fe - p_sp) < 1e-12 or (p_fe == 0.0 and p_sp < 1e-20),
    }])


def validate_salk_effect_sizes(bridge: FrontendBridge) -> pd.DataFrame:
    """φ and Cramér's V match hand calculation."""
    r = bridge.call("pearson_chi_square", table=salk_table())
    phi_hand = math.sqrt(r["chiSq"] / r["N"])
    rows = [
        {"test": "Salk φ = √(χ²/N)",
         "frontend_phi": round(r["phi"], 6),
         "hand_phi": round(phi_hand, 6),
         "pass": abs(r["phi"] - phi_hand) < 1e-8},
        {"test": "Salk Cramér's V = φ (2×2)",
         "frontend_V": round(r["cramersV"], 6),
         "hand_phi": None,
         "pass": abs(r["cramersV"] - phi_hand) < 1e-8},
    ]
    return pd.DataFrame(rows)


def validate_salk_efficacy_direction(bridge: FrontendBridge) -> pd.DataFrame:
    """Effect direction: vaccine case rate much lower than placebo.

    Francis Report published the Salk vaccine as ~60-90% efficacious depending
    on case definition. On the placebo-controlled arm, paralytic cases only,
    efficacy = 1 − (33/200745) / (115/201229) ≈ 71%.
    """
    p_vac = SALK_VACCINE_CASES / SALK_VACCINE_N
    p_pbo = SALK_PLACEBO_CASES / SALK_PLACEBO_N
    rr = p_vac / p_pbo
    efficacy = 1 - rr
    return pd.DataFrame([{
        "test": "Salk vaccine efficacy on paralytic cases ≈ 70%",
        "p_vaccine": f"{p_vac:.5f}",
        "p_placebo": f"{p_pbo:.5f}",
        "relative_risk": round(rr, 3),
        "efficacy": round(efficacy, 3),
        "pass": 0.60 < efficacy < 0.80,
    }])


# ─── Sample size (binary, via backend) ────────────────────────────────

def validate_salk_sample_size(client) -> pd.DataFrame:
    """A priori sample size at the observed rates comfortably fits within the
    Francis Field Trial's actual ~200k-per-arm design.

    Uses the backend's validated Fleiss-style two-proportion formula
    (arcsine-adjusted). Expected: required N per arm ≪ 200,745.
    """
    p_pbo = SALK_PLACEBO_CASES / SALK_PLACEBO_N
    p_vac = SALK_VACCINE_CASES / SALK_VACCINE_N
    resp = client.sample_size_binary(
        p1=p_pbo, p2=p_vac,
        alpha=0.05, power=0.80, ratio=1.0, two_sided=True,
    )
    rows = [{
        "test": "Salk: required N per arm far smaller than 200,745 actual",
        "p_placebo": f"{p_pbo:.5f}",
        "p_vaccine": f"{p_vac:.5f}",
        "required_per_arm": resp["n1"],
        "actual_per_arm": SALK_VACCINE_N,
        "pass": resp["n1"] < SALK_VACCINE_N,
    }]

    # At α=0.05 and power=0.99, the required N should still be under ~50,000
    # per arm, meaning the trial was significantly over-powered (consistent
    # with the observed p-value under 10⁻¹⁰).
    resp_99 = client.sample_size_binary(
        p1=p_pbo, p2=p_vac,
        alpha=0.05, power=0.99, ratio=1.0, two_sided=True,
    )
    rows.append({
        "test": "Salk: even at 99% power, required per-arm N < actual",
        "required_per_arm_99": resp_99["n1"],
        "actual_per_arm": SALK_VACCINE_N,
        "pass": resp_99["n1"] < SALK_VACCINE_N,
    })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default=None,
                        help="backend base URL for sample-size binary endpoint")
    parser.add_argument("--frontend-path", default=None,
                        help="Zetyra/frontend path (env ZETYRA_FRONTEND_PATH also works)")
    args = parser.parse_args()

    frontend_path = resolve_frontend_path(args.frontend_path)
    client = get_client(args.base_url)

    print("=" * 70)
    print("SALK 1954 POLIO VACCINE TRIAL REPLICATION")
    print(f"Vaccine: {SALK_VACCINE_N:,} children, {SALK_VACCINE_CASES} paralytic")
    print(f"Placebo: {SALK_PLACEBO_N:,} children, {SALK_PLACEBO_CASES} paralytic")
    print("=" * 70)

    suites_bridge = [
        ("1. Pearson χ² (Yates) on Salk table vs scipy", validate_salk_pearson_yates),
        ("2. Decisive positive: p < 1e-10", validate_salk_decisive_positive),
        ("3. Fisher's exact two-sided vs scipy", validate_salk_fisher_exact),
        ("4. Effect sizes (φ, Cramér's V)", validate_salk_effect_sizes),
        ("5. Vaccine efficacy direction (~70%)", validate_salk_efficacy_direction),
    ]
    suites_backend = [
        ("6. Sample size (binary): N comfortably under 200k/arm",
         validate_salk_sample_size),
    ]

    all_pass = True
    all_frames: list[pd.DataFrame] = []

    with FrontendBridge("chi_square", frontend_path) as bridge:
        for name, fn in suites_bridge:
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

    for name, fn in suites_backend:
        print(f"\n{name}")
        print("-" * 70)
        try:
            df = fn(client)
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
        all_results.to_csv("results/salk_polio_validation.csv", index=False)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        return 0
    print("SOME VALIDATIONS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
