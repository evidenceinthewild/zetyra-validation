#!/usr/bin/env python3
"""
Validate Basket Trial Calculator

Tests the three analysis methods (Independent, BHM, EXNEX) across
analytical, simulation, and input-validation scenarios:

1-5.   Independent analysis: per-basket posteriors, all-null, all-active,
       threshold sensitivity, n_baskets contract
6-9.   BHM: shrinkage toward grand mean, heterogeneous detection, tau
       estimate, shrinkage property
10-12. EXNEX: per-basket posteriors, high/low exchangeability weight
       convergence
13-16. Simulation: power for active baskets, Type I error bound, BHM
       per_basket_power, FWER under complete null
17-19. Input validation: n_baskets < 2, mismatched arrays, bad threshold
20-21. Reference checks: Beta-Binomial conjugate, large-n exceedance

References:
- Berry SM et al. (2013) Bayesian Hierarchical Models for Basket Trials.
  Clinical Trials 10(5):720-734.
- Neuenschwander B et al. (2016) Robust exchangeability designs for early
  phase clinical trials with multiple strata. Pharmaceutical Statistics
  15(2):123-134.
- FDA (2022) Master Protocols: Efficient Clinical Trial Design Strategies.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import mc_rate_upper_bound
import pandas as pd


# --------------------------------------------------------------------------
# Independent Analysis (Tests 1-5)
# --------------------------------------------------------------------------

def validate_independent(client) -> pd.DataFrame:
    """Validate independent (no borrowing) basket analysis."""
    results = []

    # Test 1: Two baskets — verify posterior exceedance probabilities returned
    z = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[30, 30],
        null_rates=[0.15, 0.15],
        alternative_rates=[0.40, 0.40],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z["analytical_results"]
    pb = ar["per_basket"]
    has_probs = all("posterior_prob" in b for b in pb)
    probs_valid = all(0.0 <= b["posterior_prob"] <= 1.0 for b in pb)
    results.append({
        "test": "Independent: two baskets have posterior_prob",
        "n_baskets_returned": len(pb),
        "pass": has_probs and probs_valid and len(pb) == 2,
    })

    # Test 2: All-null scenario — no basket should have high go probability
    z_null = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[30, 30, 30, 30],
        null_rates=[0.20, 0.20, 0.20, 0.20],
        alternative_rates=[0.20, 0.20, 0.20, 0.20],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_null = z_null["analytical_results"]
    n_go_null = ar_null["n_go_decisions"]
    results.append({
        "test": "Independent: all-null -> no go decisions",
        "n_go": n_go_null,
        "pass": n_go_null == 0,
    })

    # Test 3: All-active scenario — all baskets should have high exceedance
    z_active = client.basket(
        method="independent",
        n_baskets=3,
        n_per_basket=[60, 60, 60],
        null_rates=[0.10, 0.10, 0.10],
        alternative_rates=[0.50, 0.50, 0.50],
        decision_threshold=0.90,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_active = z_active["analytical_results"]
    all_go = all(b["decision"] == "go" for b in ar_active["per_basket"])
    results.append({
        "test": "Independent: all-active -> all go",
        "n_go": ar_active["n_go_decisions"],
        "pass": all_go and ar_active["n_go_decisions"] == 3,
    })

    # Test 4: Higher threshold -> fewer go decisions
    z_low_thresh = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[25, 25, 25, 25],
        null_rates=[0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.35, 0.35, 0.35],
        decision_threshold=0.80,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    z_high_thresh = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[25, 25, 25, 25],
        null_rates=[0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.35, 0.35, 0.35],
        decision_threshold=0.99,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    n_go_low = z_low_thresh["analytical_results"]["n_go_decisions"]
    n_go_high = z_high_thresh["analytical_results"]["n_go_decisions"]
    results.append({
        "test": "Independent: higher threshold -> fewer go",
        "n_go_low_thresh": n_go_low,
        "n_go_high_thresh": n_go_high,
        "pass": n_go_high <= n_go_low,
    })

    # Test 5: n_baskets matches requested
    for n_req in [2, 5, 7]:
        z_n = client.basket(
            method="independent",
            n_baskets=n_req,
            n_per_basket=[20] * n_req,
            null_rates=[0.15] * n_req,
            alternative_rates=[0.40] * n_req,
            decision_threshold=0.95,
        )
        n_ret = z_n["analytical_results"]["n_baskets"]
        n_pb = len(z_n["analytical_results"]["per_basket"])
        results.append({
            "test": f"Independent: n_baskets={n_req} matches",
            "n_returned": n_ret,
            "n_per_basket_len": n_pb,
            "pass": n_ret == n_req and n_pb == n_req,
        })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# BHM Analysis (Tests 6-9)
# --------------------------------------------------------------------------

def validate_bhm(client) -> pd.DataFrame:
    """Validate Bayesian Hierarchical Model (Berry et al. 2013)."""
    results = []

    # Test 6: Homogeneous rates — BHM shrinkage should pull posteriors together
    z_bhm = client.basket(
        method="bhm",
        n_baskets=5,
        n_per_basket=[30, 30, 30, 30, 30],
        null_rates=[0.15, 0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.40, 0.40, 0.40, 0.40, 0.40],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_bhm = z_bhm["analytical_results"]
    means = [b["posterior_mean"] for b in ar_bhm["per_basket"]]
    spread = max(means) - min(means)
    results.append({
        "test": "BHM: homogeneous rates -> tight posteriors",
        "spread": round(spread, 4),
        "pass": spread < 0.10,  # Homogeneous baskets -> posteriors close together
    })

    # Test 7: Heterogeneous rates — BHM should still detect the strong basket
    z_het = client.basket(
        method="bhm",
        n_baskets=4,
        n_per_basket=[40, 40, 40, 40],
        null_rates=[0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.15, 0.15, 0.15, 0.55],
        decision_threshold=0.90,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar_het = z_het["analytical_results"]
    strong_basket = ar_het["per_basket"][3]
    results.append({
        "test": "BHM: heterogeneous -> strong basket detected",
        "strong_prob": strong_basket["posterior_prob"],
        "strong_decision": strong_basket["decision"],
        "pass": strong_basket["decision"] == "go",
    })

    # Test 8: BHM tau estimate present in response
    het_info = ar_bhm.get("heterogeneity")
    has_tau = het_info is not None and "tau" in het_info
    results.append({
        "test": "BHM: tau estimate present",
        "has_heterogeneity": het_info is not None,
        "has_tau": has_tau,
        "pass": has_tau,
    })

    # Test 9: BHM posterior means should be between independent estimate and
    #         grand mean (shrinkage property)
    z_indep = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[40, 40, 40, 40],
        null_rates=[0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.20, 0.30, 0.40, 0.50],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    z_bhm2 = client.basket(
        method="bhm",
        n_baskets=4,
        n_per_basket=[40, 40, 40, 40],
        null_rates=[0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.20, 0.30, 0.40, 0.50],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    indep_means = [b["posterior_mean"] for b in z_indep["analytical_results"]["per_basket"]]
    bhm_means = [b["posterior_mean"] for b in z_bhm2["analytical_results"]["per_basket"]]
    grand_mean = sum(indep_means) / len(indep_means)

    # Each BHM mean should be pulled toward the grand mean relative to independent
    shrinkage_ok = True
    for i in range(4):
        ind = indep_means[i]
        bhm = bhm_means[i]
        # BHM mean should be between independent and grand mean (or very close)
        lo = min(ind, grand_mean) - 0.05
        hi = max(ind, grand_mean) + 0.05
        if not (lo <= bhm <= hi):
            shrinkage_ok = False
    results.append({
        "test": "BHM: shrinkage toward grand mean",
        "indep_means": [round(m, 3) for m in indep_means],
        "bhm_means": [round(m, 3) for m in bhm_means],
        "grand_mean": round(grand_mean, 3),
        "pass": shrinkage_ok,
    })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# EXNEX Analysis (Tests 10-12)
# --------------------------------------------------------------------------

def validate_exnex(client) -> pd.DataFrame:
    """Validate EXNEX (Neuenschwander et al. 2016)."""
    results = []

    common_kwargs = dict(
        n_baskets=4,
        n_per_basket=[35, 35, 35, 35],
        null_rates=[0.15, 0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.35, 0.35, 0.35],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )

    # Test 10: EXNEX returns per-basket posteriors
    z_exnex = client.basket(method="exnex", w_ex=[0.5, 0.5, 0.5, 0.5], **common_kwargs)
    ar = z_exnex["analytical_results"]
    pb = ar["per_basket"]
    has_probs = all("posterior_prob" in b for b in pb)
    results.append({
        "test": "EXNEX: returns per-basket posteriors",
        "n_baskets": len(pb),
        "pass": has_probs and len(pb) == 4,
    })

    # Test 11: High exchangeability weight -> results closer to BHM
    z_bhm = client.basket(method="bhm", **common_kwargs)
    z_high_w = client.basket(method="exnex", w_ex=[0.95, 0.95, 0.95, 0.95], **common_kwargs)

    bhm_probs = [b["posterior_prob"] for b in z_bhm["analytical_results"]["per_basket"]]
    high_w_probs = [b["posterior_prob"] for b in z_high_w["analytical_results"]["per_basket"]]
    dist_to_bhm = sum(abs(a - b) for a, b in zip(high_w_probs, bhm_probs))
    results.append({
        "test": "EXNEX: high w_ex -> close to BHM",
        "dist_to_bhm": round(dist_to_bhm, 4),
        "pass": dist_to_bhm < 0.5,  # Should be very close
    })

    # Test 12: Low exchangeability weight -> results closer to independent
    z_indep = client.basket(method="independent", **common_kwargs)
    z_low_w = client.basket(method="exnex", w_ex=[0.05, 0.05, 0.05, 0.05], **common_kwargs)

    indep_probs = [b["posterior_prob"] for b in z_indep["analytical_results"]["per_basket"]]
    low_w_probs = [b["posterior_prob"] for b in z_low_w["analytical_results"]["per_basket"]]
    dist_to_indep = sum(abs(a - b) for a, b in zip(low_w_probs, indep_probs))
    results.append({
        "test": "EXNEX: low w_ex -> close to independent",
        "dist_to_indep": round(dist_to_indep, 4),
        "pass": dist_to_indep < 0.5,
    })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Simulation Tests (Tests 13-16)
# --------------------------------------------------------------------------

def validate_simulation(client) -> pd.DataFrame:
    """Validate Monte Carlo simulation operating characteristics."""
    results = []

    SIM_KWARGS = dict(simulate=True, n_simulations=2000, simulation_seed=42)

    # Test 13: Independent simulation — power > 0 for active baskets
    z_sim = client.basket(
        method="independent",
        n_baskets=3,
        n_per_basket=[30, 30, 30],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.45, 0.45, 0.15],  # First two active, third null
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
        **SIM_KWARGS,
    )
    sim = z_sim["simulation"]
    per_power = sim["estimates"]["per_basket_power"]
    active_power = [p for p in per_power[:2] if p is not None]
    results.append({
        "test": "Simulation: power > 0 for active baskets",
        "per_basket_power": per_power,
        "pass": all(p > 0 for p in active_power),
    })

    # Test 14: Type I error <= 0.10 for null baskets
    per_t1 = sim["estimates"]["per_basket_type1_error"]
    n_sims = sim["n_simulations"]
    null_basket_t1 = per_t1[2]  # Third basket is null
    ub = mc_rate_upper_bound(null_basket_t1, n_sims, confidence=0.99)
    results.append({
        "test": "Simulation: null basket type I error <= 0.10",
        "null_basket_t1": null_basket_t1,
        "upper_bound_99": round(ub, 4),
        "pass": ub <= 0.10,
    })

    # Test 15: BHM simulation works and returns per_basket_power
    z_bhm_sim = client.basket(
        method="bhm",
        n_baskets=3,
        n_per_basket=[30, 30, 30],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.45, 0.45, 0.15],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
        **SIM_KWARGS,
    )
    bhm_sim = z_bhm_sim["simulation"]
    has_power = "per_basket_power" in bhm_sim["estimates"]
    results.append({
        "test": "Simulation: BHM returns per_basket_power",
        "has_per_basket_power": has_power,
        "pass": has_power and bhm_sim["estimates"]["per_basket_power"] is not None,
    })

    # Test 16: FWER under complete null is reported
    # Note: independent analysis does NOT control FWER — it's per-basket only.
    z_fwer = client.basket(
        method="independent",
        n_baskets=4,
        n_per_basket=[30, 30, 30, 30],
        null_rates=[0.20, 0.20, 0.20, 0.20],
        alternative_rates=[0.20, 0.20, 0.20, 0.20],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
        **SIM_KWARGS,
    )
    fwer = z_fwer["simulation"]["estimates"]["fwer"]
    results.append({
        "test": "Simulation: FWER reported (independent, no multiplicity)",
        "fwer": fwer,
        "pass": fwer is not None and 0.0 <= fwer <= 1.0,
    })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Input Validation (Tests 17-19)
# --------------------------------------------------------------------------

def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return 400/422."""
    results = []

    # Test 17: n_baskets < 2 rejected
    resp = client.basket_raw(
        method="independent",
        n_baskets=1,
        n_per_basket=[30],
        null_rates=[0.15],
        alternative_rates=[0.40],
        decision_threshold=0.95,
    )
    results.append({
        "test": "Guard: n_baskets < 2 rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Test 18: Mismatched array lengths rejected
    resp = client.basket_raw(
        method="independent",
        n_baskets=3,
        n_per_basket=[30, 30],  # Only 2, but n_baskets=3
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.40, 0.40, 0.40],
        decision_threshold=0.95,
    )
    results.append({
        "test": "Guard: mismatched array lengths rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Test 19: decision_threshold outside (0.5, 1) rejected
    resp = client.basket_raw(
        method="independent",
        n_baskets=3,
        n_per_basket=[30, 30, 30],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.40, 0.40, 0.40],
        decision_threshold=0.3,  # Below 0.5 lower bound
    )
    results.append({
        "test": "Guard: decision_threshold outside (0.5,1) rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Reference Checks (Tests 20-21)
# --------------------------------------------------------------------------

def validate_reference(client) -> pd.DataFrame:
    """Verify analytical results against known Beta-Binomial conjugate formulas."""
    results = []

    # Test 20: Beta-Binomial conjugate check
    # With Beta(1,1) prior and s successes in n trials,
    # posterior is Beta(1+s, 1+n-s), posterior mean = (1+s)/(2+n)
    n_trial = 40
    true_rate = 0.35
    s = round(true_rate * n_trial)  # 14 successes
    expected_mean = (1 + s) / (2 + n_trial)

    z = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[n_trial, n_trial],
        null_rates=[0.15, 0.15],
        alternative_rates=[true_rate, true_rate],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    ar = z["analytical_results"]
    pm = ar["per_basket"][0]["posterior_mean"]
    results.append({
        "test": "Reference: Beta-Binomial conjugate mean",
        "expected_mean": round(expected_mean, 4),
        "api_mean": pm,
        "pass": abs(pm - expected_mean) < 0.01,
    })

    # Test 21: With large n and clear effect, independent posterior exceedance -> 1.0
    z_large = client.basket(
        method="independent",
        n_baskets=2,
        n_per_basket=[200, 200],
        null_rates=[0.10, 0.10],
        alternative_rates=[0.50, 0.50],
        decision_threshold=0.95,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    probs = [b["posterior_prob"] for b in z_large["analytical_results"]["per_basket"]]
    results.append({
        "test": "Reference: large n + clear effect -> exceedance ~1.0",
        "probs": probs,
        "pass": all(p > 0.999 for p in probs),
    })

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BASKET TRIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    sections = [
        ("1. Independent Analysis", validate_independent),
        ("2. BHM Analysis", validate_bhm),
        ("3. EXNEX Analysis", validate_exnex),
        ("4. Simulation", validate_simulation),
        ("5. Input Guards", validate_input_guards),
        ("6. Reference Checks", validate_reference),
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
    all_results.to_csv("results/basket_validation.csv", index=False)

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
