#!/usr/bin/env python3
"""
Validate Basket Trial Calculator Against I-SPY 2 Trial Design

I-SPY 2 (Investigation of Serial Studies to Predict Your Therapeutic
Response with Imaging And moLecular Analysis 2) is an adaptive platform
trial evaluating multiple experimental agents in biomarker-defined
breast cancer subtypes. Each biomarker signature is evaluated
independently — mapping naturally to our basket trial calculator with
method="independent".

Tests use published I-SPY 2 drug-subtype results to verify that:
1. Graduated drug-subtype combinations show high posterior exceedance
2. Futile combinations show low posterior exceedance
3. Multi-basket structures correctly reflect subtype heterogeneity
4. Threshold sensitivity behaves monotonically
5. Large-sample and null-scenario properties hold
6. Beta-Binomial conjugate posterior matches closed-form

Published I-SPY 2 results used:
- Veliparib+Carboplatin in TNBC: pCR 51% vs 26%, n=72+44 (Rugo et al. 2016)
- Neratinib in HER2+/HR-: pCR 56% vs 33%, n=115+78 (Park et al. 2016)
- Pembrolizumab in TNBC: pCR 60% vs 22%, n=29+60 (Nanda et al. 2020)
- Pembrolizumab in HR+/HER2-: pCR 30% vs 13%, n=40+90 (Nanda et al. 2020)

References:
- Barker AD et al. (2009) I-SPY 2: An Adaptive Breast Cancer Trial
  Design in the Setting of Neoadjuvant Chemotherapy. Clinical
  Pharmacology & Therapeutics 86(1):97-100.
- Park JW et al. (2016) Adaptive Randomization of Neratinib in Early
  Breast Cancer. NEJM 375(1):11-22.
- Rugo HS et al. (2016) Adaptive Randomization of Veliparib-Carboplatin
  Treatment in Breast Cancer. NEJM 375(1):23-34.
- Nanda R et al. (2020) Effect of Pembrolizumab Plus Neoadjuvant
  Chemotherapy on Pathologic Complete Response in Women With Early-Stage
  Breast Cancer. JAMA Oncology 6(5):676-684.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
import pandas as pd
from scipy import stats as sp_stats


# ─── Test functions ──────────────────────────────────────────────────

def validate_ispy2_graduations(client) -> pd.DataFrame:
    """Validate that graduated I-SPY 2 drug-subtype combos show high posterior exceedance."""
    results = []

    # Test 1: Veliparib+Carboplatin graduation in TNBC
    # Published: pCR 51% (exp) vs 26% (ctrl), n=72+44=116, graduated
    z = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[116, 116],
        null_rates=[0.26, 0.26],
        alternative_rates=[0.51, 0.51],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z["analytical_results"]
    prob = ar["per_basket"][0]["posterior_prob"]
    decision = ar["per_basket"][0]["decision"]
    results.append({
        "test": "Veliparib TNBC graduation: posterior_prob > 0.95",
        "posterior_prob": round(prob, 4),
        "decision": decision,
        "pass": prob > 0.95 and decision == "go",
    })

    # Test 2: Veliparib futility in non-TNBC (small effect)
    # Hypothetical non-TNBC subtype with minimal benefit: alt ~ null
    z_fut = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[80, 80],
        null_rates=[0.26, 0.26],
        alternative_rates=[0.28, 0.28],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_fut = z_fut["analytical_results"]
    prob_fut = ar_fut["per_basket"][0]["posterior_prob"]
    decision_fut = ar_fut["per_basket"][0]["decision"]
    results.append({
        "test": "Veliparib non-TNBC futility: no-go decision",
        "posterior_prob": round(prob_fut, 4),
        "decision": decision_fut,
        "pass": decision_fut == "no-go" and prob_fut < 0.95,
    })

    # Test 3: Pembrolizumab graduation in TNBC
    # Published: pCR 60% vs 22%, n=29+60=89, graduated
    z_pembro_tnbc = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[89, 89],
        null_rates=[0.22, 0.22],
        alternative_rates=[0.60, 0.60],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_pt = z_pembro_tnbc["analytical_results"]
    prob_pt = ar_pt["per_basket"][0]["posterior_prob"]
    results.append({
        "test": "Pembrolizumab TNBC graduation: posterior_prob > 0.95",
        "posterior_prob": round(prob_pt, 4),
        "decision": ar_pt["per_basket"][0]["decision"],
        "pass": prob_pt > 0.95 and ar_pt["per_basket"][0]["decision"] == "go",
    })

    # Test 4: Pembrolizumab graduation in HR+/HER2-
    # Published: pCR 30% vs 13%, n=40+90=130, graduated
    z_pembro_hr = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[130, 130],
        null_rates=[0.13, 0.13],
        alternative_rates=[0.30, 0.30],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_ph = z_pembro_hr["analytical_results"]
    prob_ph = ar_ph["per_basket"][0]["posterior_prob"]
    results.append({
        "test": "Pembrolizumab HR+/HER2- graduation: posterior_prob > 0.95",
        "posterior_prob": round(prob_ph, 4),
        "decision": ar_ph["per_basket"][0]["decision"],
        "pass": prob_ph > 0.95 and ar_ph["per_basket"][0]["decision"] == "go",
    })

    # Test 5: Neratinib graduation in HER2+/HR-
    # Published: pCR 56% vs 33%, n=115+78=193, graduated
    z_nera = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[193, 193],
        null_rates=[0.33, 0.33],
        alternative_rates=[0.56, 0.56],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_n = z_nera["analytical_results"]
    prob_n = ar_n["per_basket"][0]["posterior_prob"]
    results.append({
        "test": "Neratinib HER2+/HR- graduation: posterior_prob > 0.95",
        "posterior_prob": round(prob_n, 4),
        "decision": ar_n["per_basket"][0]["decision"],
        "pass": prob_n > 0.95 and ar_n["per_basket"][0]["decision"] == "go",
    })

    return pd.DataFrame(results)


def validate_multi_basket(client) -> pd.DataFrame:
    """Validate multi-basket structure mimicking 4 I-SPY 2 signatures."""
    results = []

    # Test 6: 4-basket design with heterogeneous I-SPY 2 subtypes
    # Basket 0: TNBC (Veliparib) — active
    # Basket 1: HER2+/HR- (Neratinib) — active
    # Basket 2: HR+/HER2- (Pembrolizumab) — active
    # Basket 3: Hypothetical low-response subtype — null
    z_multi = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[116, 193, 130, 80],
        null_rates=[0.26, 0.33, 0.13, 0.20],
        alternative_rates=[0.51, 0.56, 0.30, 0.22],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z_multi["analytical_results"]

    # Verify correct basket count
    results.append({
        "test": "Multi-basket: 4 baskets returned",
        "n_baskets": ar["n_baskets"],
        "per_basket_len": len(ar["per_basket"]),
        "pass": ar["n_baskets"] == 4 and len(ar["per_basket"]) == 4,
    })

    # Active baskets (0-2) should be go, null basket (3) should be no-go
    decisions = [b["decision"] for b in ar["per_basket"]]
    active_go = all(d == "go" for d in decisions[:3])
    null_nogo = decisions[3] == "no-go"
    results.append({
        "test": "Multi-basket: active go, null no-go",
        "decisions": decisions,
        "pass": active_go and null_nogo,
    })

    return pd.DataFrame(results)


def validate_threshold_sensitivity(client) -> pd.DataFrame:
    """Validate that higher decision threshold produces fewer go decisions."""
    results = []

    # Test 7: Threshold sensitivity with borderline effects
    # Use a moderate effect that is borderline at high thresholds
    common = dict(
        method="independent",
        n_baskets=4,
        n_per_basket=[50, 50, 50, 50],
        null_rates=[0.20, 0.20, 0.20, 0.20],
        alternative_rates=[0.35, 0.35, 0.35, 0.35],
        prior_alpha=1.0,
        prior_beta=1.0,
    )

    z_low = client.basket(decision_threshold=0.90, **common)
    z_high = client.basket(decision_threshold=0.99, **common)

    n_go_low = z_low["analytical_results"]["n_go_decisions"]
    n_go_high = z_high["analytical_results"]["n_go_decisions"]

    results.append({
        "test": "Threshold sensitivity: 0.99 vs 0.90 fewer go",
        "n_go_at_090": n_go_low,
        "n_go_at_099": n_go_high,
        "pass": n_go_high <= n_go_low,
    })

    return pd.DataFrame(results)


def validate_large_sample(client) -> pd.DataFrame:
    """Validate large-sample convergence with I-SPY 2 effect sizes."""
    results = []

    # Test 8: With very large N and I-SPY 2 effect sizes, all active baskets -> go
    z_large = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[1000, 1000, 1000, 1000],
        null_rates=[0.26, 0.33, 0.13, 0.22],
        alternative_rates=[0.51, 0.56, 0.30, 0.60],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z_large["analytical_results"]
    all_go = all(b["decision"] == "go" for b in ar["per_basket"])
    probs = [b["posterior_prob"] for b in ar["per_basket"]]
    results.append({
        "test": "Large N: all I-SPY 2 active baskets -> go",
        "probs": [round(p, 4) for p in probs],
        "n_go": ar["n_go_decisions"],
        "pass": all_go and ar["n_go_decisions"] == 4,
    })

    return pd.DataFrame(results)


def validate_null_scenario(client) -> pd.DataFrame:
    """Validate that null scenario (no effect) produces no go decisions."""
    results = []

    # Test 9: alternative_rates = null_rates -> no go at threshold 0.95
    z_null = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[120, 120, 120, 120],
        null_rates=[0.26, 0.33, 0.13, 0.22],
        alternative_rates=[0.26, 0.33, 0.13, 0.22],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z_null["analytical_results"]
    results.append({
        "test": "Null scenario: no effect -> no go decisions",
        "n_go": ar["n_go_decisions"],
        "decisions": [b["decision"] for b in ar["per_basket"]],
        "pass": ar["n_go_decisions"] == 0,
    })

    return pd.DataFrame(results)


def validate_beta_binomial_reference(client) -> pd.DataFrame:
    """Verify Beta-Binomial conjugate posterior against closed-form with I-SPY 2 data."""
    results = []

    # Test 10: Beta(1,1) prior + Veliparib TNBC data
    # 51% of 72 experimental patients = ~37 successes in 72 trials
    # Under Beta(1,1) prior: posterior mean = (1+s)/(2+n) = (1+37)/(2+72) = 38/74
    n_trial = 72
    s = round(0.51 * n_trial)  # 37 successes
    expected_mean = (1 + s) / (2 + n_trial)

    z = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[n_trial, n_trial],
        null_rates=[0.26, 0.26],
        alternative_rates=[0.51, 0.51],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z["analytical_results"]
    pm = ar["per_basket"][0]["posterior_mean"]

    # Also verify posterior exceedance against scipy Beta CDF
    # P(p > null | data) = 1 - Beta_CDF(null; alpha_post, beta_post)
    alpha_post = 1 + s
    beta_post = 1 + n_trial - s
    expected_exceedance = 1 - sp_stats.beta.cdf(0.26, alpha_post, beta_post)

    prob = ar["per_basket"][0]["posterior_prob"]

    results.append({
        "test": "Beta-Binomial: Veliparib TNBC posterior mean",
        "expected_mean": round(expected_mean, 4),
        "api_mean": round(pm, 4),
        "pass": abs(pm - expected_mean) < 0.02,
    })

    results.append({
        "test": "Beta-Binomial: Veliparib TNBC exceedance vs scipy",
        "expected_exceedance": round(expected_exceedance, 4),
        "api_exceedance": round(prob, 4),
        "pass": abs(prob - expected_exceedance) < 0.02,
    })

    return pd.DataFrame(results)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("I-SPY 2 BASKET TRIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    sections = [
        ("1. I-SPY 2 Graduation Decisions (Tests 1-5)", validate_ispy2_graduations),
        ("2. Multi-Basket Structure (Test 6)", validate_multi_basket),
        ("3. Threshold Sensitivity (Test 7)", validate_threshold_sensitivity),
        ("4. Large Sample Convergence (Test 8)", validate_large_sample),
        ("5. Null Scenario (Test 9)", validate_null_scenario),
        ("6. Beta-Binomial Reference (Test 10)", validate_beta_binomial_reference),
    ]

    for header, fn in sections:
        print(f"\n{header}")
        print("-" * 70)
        df = fn(client)
        print(df.to_string(index=False))
        all_frames.append(df)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/ispy2_basket_validation.csv", index=False)

    all_pass = all(df["pass"].all() for df in all_frames)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
