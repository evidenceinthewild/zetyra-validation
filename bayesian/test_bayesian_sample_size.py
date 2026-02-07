#!/usr/bin/env python3
"""
Validate Bayesian Sample Size Calculator (Single-Arm Binary)

Tests:
1. Analytical posterior: Beta(α+r, β+n-r) — exact
2. MC sample size search: find_sample_size() — binomial CI assertions
3. Seed reproducibility
4. Input guards (422/400 for invalid inputs)
5. Boundary-condition scenarios
6. Invariants: higher threshold → larger n, larger effect → smaller n
7. Schema contracts

References:
- Berry et al. (2010) "Bayesian Adaptive Methods for Clinical Trials"
- REBYOTA PUNCH CD2 (Phase 2b): 25/45 responders, two-dose arm (FDA BLA 125739)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import mc_rate_upper_bound, mc_rate_lower_bound, assert_schema
import pandas as pd

POSTERIOR_TOLERANCE = 0.001  # Exact for conjugate update


# ─── Reference implementations ────────────────────────────────────────

def reference_posterior(prior_alpha, prior_beta, events, n):
    """Beta-Binomial conjugate update."""
    return prior_alpha + events, prior_beta + (n - events)


# ─── Test functions ───────────────────────────────────────────────────

def validate_analytical_posterior(client) -> pd.DataFrame:
    """Validate analytical posterior via the endpoint's posterior_at_alt fields."""
    scenarios = [
        {
            "name": "Beta(1,1) uninformative, null=0.10, alt=0.25",
            "prior_alpha": 1.0, "prior_beta": 1.0,
            "null_rate": 0.10, "alt_rate": 0.25,
        },
        {
            "name": "Beta(2,8) informative, null=0.10, alt=0.25",
            "prior_alpha": 2.0, "prior_beta": 8.0,
            "null_rate": 0.10, "alt_rate": 0.25,
        },
        {
            "name": "REBYOTA prior, null=0.45, alt=0.65",
            "prior_alpha": 13.5, "prior_beta": 11.0,
            "null_rate": 0.45, "alt_rate": 0.65,
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_sample_size_single_arm(
            prior_alpha=s["prior_alpha"],
            prior_beta=s["prior_beta"],
            null_rate=s["null_rate"],
            alternative_rate=s["alt_rate"],
            decision_threshold=0.95,
            n_simulations=500,  # Low sims just to get posterior check
            n_min=20, n_max=30, n_step=10,
        )

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_sample_size_single_arm")

        rec_n = zetyra["recommended_n"]
        expected_events = round(s["alt_rate"] * rec_n)

        ref_alpha, ref_beta = reference_posterior(
            s["prior_alpha"], s["prior_beta"], expected_events, rec_n
        )

        alpha_ok = abs(zetyra["posterior_at_alt_alpha"] - ref_alpha) < POSTERIOR_TOLERANCE
        beta_ok = abs(zetyra["posterior_at_alt_beta"] - ref_beta) < POSTERIOR_TOLERANCE

        passed = alpha_ok and beta_ok and len(schema_errors) == 0
        results.append({
            "test": f"Posterior: {s['name']}",
            "rec_n": rec_n,
            "events": expected_events,
            "zetyra_post_a": zetyra["posterior_at_alt_alpha"],
            "ref_post_a": ref_alpha,
            "zetyra_post_b": zetyra["posterior_at_alt_beta"],
            "ref_post_b": ref_beta,
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_mc_sample_size(client) -> pd.DataFrame:
    """Validate MC sample size search with binomial CI assertions."""
    scenarios = [
        {
            "name": "Berry Phase II: null=0.10, alt=0.25",
            "prior_alpha": 1.0, "prior_beta": 1.0,
            "null_rate": 0.10, "alt_rate": 0.25,
            "threshold": 0.95, "power": 0.80,
            "type1": 0.05,
            "n_min": 10, "n_max": 120, "n_step": 5,
            "n_sims": 5000,
            "expected_n_range": (15, 120),
        },
        {
            "name": "REBYOTA-inspired: null=0.45, alt=0.65, informative prior",
            "prior_alpha": 13.5, "prior_beta": 11.0,
            "null_rate": 0.45, "alt_rate": 0.65,
            "threshold": 0.975, "power": 0.80,
            "type1": 0.05,
            "n_min": 10, "n_max": 200, "n_step": 10,
            "n_sims": 5000,
            "expected_n_range": (10, 150),
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_sample_size_single_arm(
            prior_alpha=s["prior_alpha"],
            prior_beta=s["prior_beta"],
            null_rate=s["null_rate"],
            alternative_rate=s["alt_rate"],
            decision_threshold=s["threshold"],
            target_power=s["power"],
            target_type1_error=s["type1"],
            n_simulations=s["n_sims"],
            n_min=s["n_min"],
            n_max=s["n_max"],
            n_step=s["n_step"],
        )

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_sample_size_single_arm")

        n_in_range = s["expected_n_range"][0] <= zetyra["recommended_n"] <= s["expected_n_range"][1]

        # Binomial CI assertions instead of fixed tolerances
        type1_ub = mc_rate_upper_bound(zetyra["type1_error"], s["n_sims"])
        power_lb = mc_rate_lower_bound(zetyra["power"], s["n_sims"])
        type1_ok = type1_ub <= s["type1"] + 0.02  # CI upper bound within margin
        power_ok = power_lb >= s["power"] - 0.05   # CI lower bound within margin

        passed = n_in_range and type1_ok and power_ok and zetyra["constraints_met"] and len(schema_errors) == 0
        results.append({
            "test": f"MC: {s['name']}",
            "rec_n": zetyra["recommended_n"],
            "n_range": f"{s['expected_n_range']}",
            "type1": round(zetyra["type1_error"], 4),
            "type1_ub": round(type1_ub, 4),
            "power": round(zetyra["power"], 4),
            "power_lb": round(power_lb, 4),
            "constraints_met": zetyra["constraints_met"],
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_reproducibility(client) -> pd.DataFrame:
    """Two calls with same seed must return identical results."""
    results = []
    common_args = dict(
        prior_alpha=1.0, prior_beta=1.0,
        null_rate=0.10, alternative_rate=0.25,
        decision_threshold=0.95,
        target_power=0.80, target_type1_error=0.05,
        n_simulations=2000,
        n_min=20, n_max=80, n_step=10,
    )

    r1 = client.bayesian_sample_size_single_arm(**common_args, seed=12345)
    r2 = client.bayesian_sample_size_single_arm(**common_args, seed=12345)

    same_n = r1["recommended_n"] == r2["recommended_n"]
    same_type1 = r1["type1_error"] == r2["type1_error"]
    same_power = r1["power"] == r2["power"]

    results.append({
        "test": "Reproducibility: same seed → identical results",
        "n1": r1["recommended_n"], "n2": r2["recommended_n"],
        "type1_match": same_type1, "power_match": same_power,
        "pass": same_n and same_type1 and same_power,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate that invalid inputs return 400/422."""
    results = []

    # Each payload includes all required fields; only the field under test is invalid
    valid_base = {
        "prior_alpha": 1.0, "prior_beta": 1.0,
        "null_rate": 0.1, "alternative_rate": 0.3,
        "decision_threshold": 0.95, "n_simulations": 500,
        "n_min": 20, "n_max": 100, "n_step": 20,
    }

    guards = [
        {
            "name": "prior_alpha < 0",
            "field": "prior_alpha",
            "data": {**valid_base, "prior_alpha": -1},
        },
        {
            "name": "null_rate ≥ 1",
            "field": "null_rate",
            "data": {**valid_base, "null_rate": 1.0},
        },
        {
            "name": "null_rate = 0",
            "field": "null_rate",
            "data": {**valid_base, "null_rate": 0.0},
        },
        {
            "name": "n_simulations too low",
            "field": "n_simulations",
            "data": {**valid_base, "n_simulations": 10},
        },
    ]

    for g in guards:
        resp = client._post_raw("/bayesian/sample-size-single-arm", g["data"])
        status_ok = resp.status_code in (400, 422)
        field_mentioned = g["field"] in resp.text
        results.append({
            "test": f"Guard: {g['name']}",
            "status_code": resp.status_code,
            "field_in_body": field_mentioned,
            "pass": status_ok and field_mentioned,
        })

    return pd.DataFrame(results)


def validate_boundary_cases(client) -> pd.DataFrame:
    """Boundary-condition scenarios."""
    results = []

    # Very strong prior (ESS=200) should dominate likelihood
    strong = client.bayesian_sample_size_single_arm(
        prior_alpha=100.0, prior_beta=100.0,
        null_rate=0.30, alternative_rate=0.60,
        decision_threshold=0.95,
        n_simulations=1000,
        n_min=10, n_max=50, n_step=10,
    )
    schema_ok = len(assert_schema(strong, "bayesian_sample_size_single_arm")) == 0
    results.append({
        "test": "Boundary: very strong prior (ESS=200)",
        "rec_n": strong["recommended_n"],
        "pass": schema_ok and strong["recommended_n"] >= 10,
    })

    # Very weak prior (near-zero ESS)
    weak = client.bayesian_sample_size_single_arm(
        prior_alpha=0.01, prior_beta=0.01,
        null_rate=0.10, alternative_rate=0.30,
        decision_threshold=0.95,
        n_simulations=1000,
        n_min=10, n_max=100, n_step=10,
    )
    schema_ok = len(assert_schema(weak, "bayesian_sample_size_single_arm")) == 0
    results.append({
        "test": "Boundary: very weak prior (ESS≈0.02)",
        "rec_n": weak["recommended_n"],
        "pass": schema_ok and weak["recommended_n"] >= 10,
    })

    return pd.DataFrame(results)


def validate_invariants(client) -> pd.DataFrame:
    """Invariant/property tests."""
    results = []

    base = dict(
        prior_alpha=1.0, prior_beta=1.0,
        null_rate=0.10, target_type1_error=0.05,
        n_simulations=3000,
        n_min=10, n_max=200, n_step=5,
        seed=99,
    )

    # Invariant 1: Higher target power → larger n (or equal)
    low_power = client.bayesian_sample_size_single_arm(
        **base, alternative_rate=0.30, decision_threshold=0.95,
        target_power=0.70,
    )
    high_power = client.bayesian_sample_size_single_arm(
        **base, alternative_rate=0.30, decision_threshold=0.95,
        target_power=0.90,
    )
    schema_lp = assert_schema(low_power, "bayesian_sample_size_single_arm")
    schema_hp = assert_schema(high_power, "bayesian_sample_size_single_arm")
    results.append({
        "test": "Invariant: higher target power → larger n",
        "n_low_power": low_power["recommended_n"],
        "n_high_power": high_power["recommended_n"],
        "pass": high_power["recommended_n"] >= low_power["recommended_n"] and len(schema_lp) == 0 and len(schema_hp) == 0,
    })

    # Invariant 2: Larger effect (alt further from null) → smaller n
    small_effect = client.bayesian_sample_size_single_arm(
        **base, alternative_rate=0.20, decision_threshold=0.95,
        target_power=0.80,
    )
    large_effect = client.bayesian_sample_size_single_arm(
        **base, alternative_rate=0.40, decision_threshold=0.95,
        target_power=0.80,
    )
    schema_se = assert_schema(small_effect, "bayesian_sample_size_single_arm")
    schema_le = assert_schema(large_effect, "bayesian_sample_size_single_arm")
    results.append({
        "test": "Invariant: larger effect → smaller n",
        "n_small_effect": small_effect["recommended_n"],
        "n_large_effect": large_effect["recommended_n"],
        "pass": large_effect["recommended_n"] <= small_effect["recommended_n"] and len(schema_se) == 0 and len(schema_le) == 0,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN SAMPLE SIZE (SINGLE-ARM) VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Analytical Posterior Check")
    print("-" * 70)
    post_results = validate_analytical_posterior(client)
    print(post_results.to_string(index=False))
    all_frames.append(post_results)

    print("\n2. MC Sample Size Search (Binomial CI)")
    print("-" * 70)
    mc_results = validate_mc_sample_size(client)
    print(mc_results.to_string(index=False))
    all_frames.append(mc_results)

    print("\n3. Seed Reproducibility")
    print("-" * 70)
    repro_results = validate_reproducibility(client)
    print(repro_results.to_string(index=False))
    all_frames.append(repro_results)

    print("\n4. Input Guards")
    print("-" * 70)
    guard_results = validate_input_guards(client)
    print(guard_results.to_string(index=False))
    all_frames.append(guard_results)

    print("\n5. Boundary Cases")
    print("-" * 70)
    boundary_results = validate_boundary_cases(client)
    print(boundary_results.to_string(index=False))
    all_frames.append(boundary_results)

    print("\n6. Invariants")
    print("-" * 70)
    inv_results = validate_invariants(client)
    print(inv_results.to_string(index=False))
    all_frames.append(inv_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_sample_size_validation.csv", index=False)

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
