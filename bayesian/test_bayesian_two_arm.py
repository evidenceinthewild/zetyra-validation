#!/usr/bin/env python3
"""
Validate Bayesian Two-Arm Design Calculator (Binary + Continuous)

Tests (Binary):
1. MC null control: Type I error under equal rates (binomial CI assertions)
2. MC power: Power under treatment effect (binomial CI assertions)
3. Directional checks: larger effect → smaller n, lower threshold → smaller n
4. Seed reproducibility
5. Input guards (422/400 for invalid inputs)
6. Symmetry: same seed + same rates → identical results
7. Schema contracts

Tests (Continuous — Normal difference):
8. Analytical posterior: Normal-Normal conjugate on δ
9. MC sample size: type I error + power within binomial CI bounds
10. Vague-prior convergence: Bayesian n → frequentist two-sample z-test n
11. Invariants: larger effect → smaller n, larger variance → larger n
12. Input guards (422 for missing continuous fields)

References:
- Berry et al. (2010) "Bayesian Adaptive Methods for Clinical Trials"
- Spiegelhalter et al. (2004) "Bayesian Approaches to Clinical Trials and Health-Care Evaluation"
- REBYOTA PUNCH CD3 (Phase 3): 126/177 treatment, 53/85 placebo (FDA BLA 125739)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import mc_rate_upper_bound, mc_rate_lower_bound, assert_schema
import pandas as pd


def validate_two_arm(client) -> pd.DataFrame:
    """Validate two-arm MC simulations with binomial CI assertions."""
    scenarios = [
        {
            "name": "Superiority: ctrl=0.30, treat=0.50, flat priors",
            "control_rate": 0.30,
            "treatment_rate": 0.50,
            "treat_prior_a": 1.0, "treat_prior_b": 1.0,
            "ctrl_prior_a": 1.0, "ctrl_prior_b": 1.0,
            "threshold": 0.95,
            "target_power": 0.80,
            "target_type1": 0.05,
            "n_sims": 2000,
            "n_min": 20, "n_max": 200, "n_step": 20,
            "checks": {
                "type1_ub_max": 0.08,  # CI upper bound for type I
                "power_lb_min": 0.70,  # CI lower bound for power
                "n_range": (20, 180),
            },
        },
        {
            "name": "PUNCH CD3 rates: ctrl=0.624, treat=0.712, flat priors",
            "control_rate": 0.624,
            "treatment_rate": 0.712,
            "treat_prior_a": 1.0, "treat_prior_b": 1.0,
            "ctrl_prior_a": 1.0, "ctrl_prior_b": 1.0,
            "threshold": 0.95,
            "target_power": 0.80,
            "target_type1": 0.05,
            "n_sims": 2000,
            "n_min": 50, "n_max": 500, "n_step": 25,
            "checks": {
                "type1_ub_max": 0.08,
                "power_lb_min": 0.70,
                "n_range": (50, 500),
            },
        },
        {
            "name": "Large effect: ctrl=0.20, treat=0.50, flat priors",
            "control_rate": 0.20,
            "treatment_rate": 0.50,
            "treat_prior_a": 1.0, "treat_prior_b": 1.0,
            "ctrl_prior_a": 1.0, "ctrl_prior_b": 1.0,
            "threshold": 0.95,
            "target_power": 0.80,
            "target_type1": 0.05,
            "n_sims": 2000,
            "n_min": 20, "n_max": 120, "n_step": 10,
            "checks": {
                "type1_ub_max": 0.08,
                "power_lb_min": 0.70,
                "n_range": (20, 100),
            },
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_two_arm(
            treatment_prior_alpha=s["treat_prior_a"],
            treatment_prior_beta=s["treat_prior_b"],
            control_prior_alpha=s["ctrl_prior_a"],
            control_prior_beta=s["ctrl_prior_b"],
            control_rate=s["control_rate"],
            treatment_rate=s["treatment_rate"],
            decision_threshold=s["threshold"],
            target_power=s["target_power"],
            target_type1_error=s["target_type1"],
            n_simulations=s["n_sims"],
            n_min=s["n_min"],
            n_max=s["n_max"],
            n_step=s["n_step"],
        )

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_two_arm")

        checks = s["checks"]
        n_ok = checks["n_range"][0] <= zetyra["recommended_n_per_arm"] <= checks["n_range"][1]

        # Binomial CI assertions
        type1_ub = mc_rate_upper_bound(zetyra["type1_error"], s["n_sims"])
        power_lb = mc_rate_lower_bound(zetyra["power"], s["n_sims"])
        type1_ok = type1_ub <= checks["type1_ub_max"]
        power_ok = power_lb >= checks["power_lb_min"]

        passed = type1_ok and power_ok and n_ok and zetyra["constraints_met"] and len(schema_errors) == 0
        results.append({
            "test": s["name"],
            "rec_n_per_arm": zetyra["recommended_n_per_arm"],
            "n_total": zetyra["n_total"],
            "type1": round(zetyra["type1_error"], 4),
            "type1_ub": round(type1_ub, 4),
            "power": round(zetyra["power"], 4),
            "power_lb": round(power_lb, 4),
            "constraints_met": zetyra["constraints_met"],
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Validate directional properties of two-arm design."""
    results = []

    # Property: Larger effect → smaller n
    small_effect = client.bayesian_two_arm(
        control_rate=0.30, treatment_rate=0.40,
        decision_threshold=0.95, target_power=0.80, target_type1_error=0.05,
        n_simulations=2000, n_min=20, n_max=500, n_step=20,
    )
    large_effect = client.bayesian_two_arm(
        control_rate=0.30, treatment_rate=0.60,
        decision_threshold=0.95, target_power=0.80, target_type1_error=0.05,
        n_simulations=2000, n_min=20, n_max=500, n_step=20,
    )
    schema_se = assert_schema(small_effect, "bayesian_two_arm")
    schema_le = assert_schema(large_effect, "bayesian_two_arm")

    larger_effect_smaller_n = large_effect["recommended_n_per_arm"] <= small_effect["recommended_n_per_arm"]
    results.append({
        "test": "Property: larger effect → smaller n",
        "small_effect_n": small_effect["recommended_n_per_arm"],
        "large_effect_n": large_effect["recommended_n_per_arm"],
        "pass": larger_effect_smaller_n
                and small_effect["constraints_met"] and large_effect["constraints_met"]
                and len(schema_se) == 0 and len(schema_le) == 0,
    })

    # Property: Higher decision threshold → larger n (stricter evidence bar)
    # Use moderate thresholds (0.95 vs 0.975) that both produce feasible designs
    low_threshold = client.bayesian_two_arm(
        control_rate=0.30, treatment_rate=0.45,
        decision_threshold=0.95, target_power=0.80, target_type1_error=0.05,
        n_simulations=3000, n_min=50, n_max=800, n_step=25,
        seed=77,
    )
    high_threshold = client.bayesian_two_arm(
        control_rate=0.30, treatment_rate=0.45,
        decision_threshold=0.975, target_power=0.80, target_type1_error=0.05,
        n_simulations=3000, n_min=50, n_max=800, n_step=25,
        seed=77,
    )
    schema_lt = assert_schema(low_threshold, "bayesian_two_arm")
    schema_ht = assert_schema(high_threshold, "bayesian_two_arm")
    higher_needs_more = high_threshold["recommended_n_per_arm"] >= low_threshold["recommended_n_per_arm"]
    results.append({
        "test": "Property: higher threshold → larger n",
        "low_thresh_n": low_threshold["recommended_n_per_arm"],
        "high_thresh_n": high_threshold["recommended_n_per_arm"],
        "pass": higher_needs_more
                and low_threshold["constraints_met"] and high_threshold["constraints_met"]
                and len(schema_lt) == 0 and len(schema_ht) == 0,
    })

    return pd.DataFrame(results)


def validate_reproducibility(client) -> pd.DataFrame:
    """Two calls with same seed must return identical results."""
    results = []
    common_args = dict(
        control_rate=0.30, treatment_rate=0.50,
        decision_threshold=0.95,
        target_power=0.80, target_type1_error=0.05,
        n_simulations=2000,
        n_min=20, n_max=200, n_step=20,
    )

    r1 = client.bayesian_two_arm(**common_args, seed=12345)
    r2 = client.bayesian_two_arm(**common_args, seed=12345)

    same_n = r1["recommended_n_per_arm"] == r2["recommended_n_per_arm"]
    same_type1 = r1["type1_error"] == r2["type1_error"]
    same_power = r1["power"] == r2["power"]

    results.append({
        "test": "Reproducibility: same seed → identical results",
        "n1": r1["recommended_n_per_arm"], "n2": r2["recommended_n_per_arm"],
        "type1_match": same_type1, "power_match": same_power,
        "pass": same_n and same_type1 and same_power,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate that invalid inputs return 400/422."""
    results = []

    # Each payload includes all required fields; only the field under test is invalid
    valid_base = {
        "control_rate": 0.3, "treatment_rate": 0.5,
        "decision_threshold": 0.95, "n_simulations": 500,
        "n_min": 20, "n_max": 100, "n_step": 20,
    }

    guards = [
        {
            "name": "control_rate ≤ 0",
            "field": "control_rate",
            "data": {**valid_base, "control_rate": 0.0},
        },
        {
            "name": "treatment_rate ≥ 1",
            "field": "treatment_rate",
            "data": {**valid_base, "treatment_rate": 1.0},
        },
        {
            "name": "negative prior_alpha",
            "field": "treatment_prior_alpha",
            "data": {**valid_base, "treatment_prior_alpha": -1},
        },
        {
            "name": "n_simulations too low",
            "field": "n_simulations",
            "data": {**valid_base, "n_simulations": 10},
        },
    ]

    for g in guards:
        resp = client._post_raw("/bayesian/two-arm", g["data"])
        status_ok = resp.status_code in (400, 422)
        field_mentioned = g["field"] in resp.text
        results.append({
            "test": f"Guard: {g['name']}",
            "status_code": resp.status_code,
            "field_in_body": field_mentioned,
            "pass": status_ok and field_mentioned,
        })

    return pd.DataFrame(results)


def validate_symmetry(client) -> pd.DataFrame:
    """Swapping treatment/control labels under null should give same type I error."""
    results = []

    # Test 1: Under null (equal rates), swapping labels is identity — same seed must match
    r1 = client.bayesian_two_arm(
        control_rate=0.40, treatment_rate=0.40,
        decision_threshold=0.95,
        n_simulations=2000, n_min=20, n_max=200, n_step=20,
        seed=42,
    )
    r2 = client.bayesian_two_arm(
        control_rate=0.40, treatment_rate=0.40,
        decision_threshold=0.95,
        n_simulations=2000, n_min=20, n_max=200, n_step=20,
        seed=42,
    )
    schema_r1 = assert_schema(r1, "bayesian_two_arm")
    schema_r2 = assert_schema(r2, "bayesian_two_arm")
    results.append({
        "test": "Symmetry: null, same seed → identical results",
        "n1": r1["recommended_n_per_arm"], "n2": r2["recommended_n_per_arm"],
        "type1_1": round(r1["type1_error"], 4),
        "type1_2": round(r2["type1_error"], 4),
        "pass": (r1["recommended_n_per_arm"] == r2["recommended_n_per_arm"]
                 and r1["type1_error"] == r2["type1_error"]
                 and len(schema_r1) == 0 and len(schema_r2) == 0),
    })

    # Test 2: Mirror symmetry around 0.5 — rates (0.30, 0.50) and (0.50, 0.70)
    # have the same Δ=0.20 and Var(p)=Var(1-p), so n should be similar.
    forward = client.bayesian_two_arm(
        control_rate=0.30, treatment_rate=0.50,
        decision_threshold=0.95, target_power=0.80, target_type1_error=0.05,
        n_simulations=3000, n_min=20, n_max=300, n_step=20,
        seed=99,
    )
    mirrored = client.bayesian_two_arm(
        control_rate=0.50, treatment_rate=0.70,
        decision_threshold=0.95, target_power=0.80, target_type1_error=0.05,
        n_simulations=3000, n_min=20, n_max=300, n_step=20,
        seed=99,
    )
    # Same Δ and symmetric variance around 0.5 — n should be close
    schema_fwd = assert_schema(forward, "bayesian_two_arm")
    schema_mir = assert_schema(mirrored, "bayesian_two_arm")
    n_close = abs(forward["recommended_n_per_arm"] - mirrored["recommended_n_per_arm"]) <= 40
    results.append({
        "test": "Symmetry: mirror around 0.5 → similar n",
        "forward_n": forward["recommended_n_per_arm"],
        "mirrored_n": mirrored["recommended_n_per_arm"],
        "pass": n_close and len(schema_fwd) == 0 and len(schema_mir) == 0,
    })

    # Test 3: Same Δ=0.20 at asymmetric base rates (0.10, 0.30) —
    # binomial variance differs more, so allow wider tolerance
    asymmetric = client.bayesian_two_arm(
        control_rate=0.10, treatment_rate=0.30,
        decision_threshold=0.95, target_power=0.80, target_type1_error=0.05,
        n_simulations=3000, n_min=20, n_max=300, n_step=20,
        seed=99,
    )
    schema_asym = assert_schema(asymmetric, "bayesian_two_arm")
    n_close_asym = abs(forward["recommended_n_per_arm"] - asymmetric["recommended_n_per_arm"]) <= 60
    results.append({
        "test": "Symmetry: same Δ, different base rates → similar n",
        "forward_n": forward["recommended_n_per_arm"],
        "asymmetric_n": asymmetric["recommended_n_per_arm"],
        "pass": n_close_asym and len(schema_fwd) == 0 and len(schema_asym) == 0,
    })

    return pd.DataFrame(results)


# ─── Continuous reference implementations ─────────────────────────────

def reference_posterior_continuous(prior_mean, prior_variance, observed_diff, data_var_delta):
    """Normal-Normal conjugate on δ: returns (posterior_mean, posterior_variance)."""
    post_var = 1.0 / (1.0 / prior_variance + 1.0 / data_var_delta)
    post_mean = post_var * (prior_mean / prior_variance + observed_diff / data_var_delta)
    return post_mean, post_var


def frequentist_two_sample_z_n(alpha, power_target, data_variance, delta):
    """Frequentist n_per_arm for two-sample z-test: n = 2(z_α + z_β)² σ² / δ²."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power_target)
    return 2 * (z_alpha + z_beta) ** 2 * data_variance / delta ** 2


# ─── Continuous test functions ────────────────────────────────────────

def validate_continuous_two_arm(client) -> pd.DataFrame:
    """Validate two-arm continuous MC simulations with binomial CI assertions."""
    scenarios = [
        {
            "name": "Moderate: δ=0.5, σ²=1, flat prior",
            "prior_effect_mean": 0.0, "prior_effect_variance": 10.0,
            "data_variance": 1.0, "alternative_difference": 0.5,
            "threshold": 0.95,
            "target_power": 0.80, "target_type1": 0.05,
            "n_sims": 3000,
            "n_min": 10, "n_max": 200, "n_step": 10,
            "checks": {
                "type1_ub_max": 0.08,
                "power_lb_min": 0.70,
                "n_range": (10, 150),
            },
        },
        {
            "name": "Small effect: δ=0.3, σ²=2, vague prior",
            "prior_effect_mean": 0.0, "prior_effect_variance": 10.0,
            "data_variance": 2.0, "alternative_difference": 0.3,
            "threshold": 0.95,
            "target_power": 0.80, "target_type1": 0.05,
            "n_sims": 3000,
            "n_min": 20, "n_max": 500, "n_step": 20,
            "checks": {
                "type1_ub_max": 0.08,
                "power_lb_min": 0.70,
                "n_range": (30, 400),
            },
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_two_arm(
            endpoint_type="continuous",
            prior_effect_mean=s["prior_effect_mean"],
            prior_effect_variance=s["prior_effect_variance"],
            data_variance=s["data_variance"],
            alternative_difference=s["alternative_difference"],
            decision_threshold=s["threshold"],
            target_power=s["target_power"],
            target_type1_error=s["target_type1"],
            n_simulations=s["n_sims"],
            n_min=s["n_min"],
            n_max=s["n_max"],
            n_step=s["n_step"],
        )

        schema_errors = assert_schema(zetyra, "bayesian_two_arm_continuous")

        checks = s["checks"]
        n_ok = checks["n_range"][0] <= zetyra["recommended_n_per_arm"] <= checks["n_range"][1]

        type1_ub = mc_rate_upper_bound(zetyra["type1_error"], s["n_sims"])
        power_lb = mc_rate_lower_bound(zetyra["power"], s["n_sims"])
        type1_ok = type1_ub <= checks["type1_ub_max"]
        power_ok = power_lb >= checks["power_lb_min"]

        passed = type1_ok and power_ok and n_ok and zetyra["constraints_met"] and len(schema_errors) == 0
        results.append({
            "test": f"Continuous: {s['name']}",
            "rec_n_per_arm": zetyra["recommended_n_per_arm"],
            "n_total": zetyra["n_total"],
            "type1": round(zetyra["type1_error"], 4),
            "type1_ub": round(type1_ub, 4),
            "power": round(zetyra["power"], 4),
            "power_lb": round(power_lb, 4),
            "constraints_met": zetyra["constraints_met"],
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_continuous_analytical_posterior(client) -> pd.DataFrame:
    """Validate Normal-Normal conjugate posterior for two-arm (on δ)."""
    TOLERANCE = 0.001

    scenarios = [
        {
            "name": "N(0,10) prior, σ²=1, δ=0.5",
            "prior_effect_mean": 0.0, "prior_effect_variance": 10.0,
            "data_variance": 1.0, "alternative_difference": 0.5,
        },
        {
            "name": "N(0.1, 0.5) informative, σ²=2, δ=0.3",
            "prior_effect_mean": 0.1, "prior_effect_variance": 0.5,
            "data_variance": 2.0, "alternative_difference": 0.3,
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_two_arm(
            endpoint_type="continuous",
            prior_effect_mean=s["prior_effect_mean"],
            prior_effect_variance=s["prior_effect_variance"],
            data_variance=s["data_variance"],
            alternative_difference=s["alternative_difference"],
            decision_threshold=0.95,
            n_simulations=500,
            n_min=10, n_max=50, n_step=10,
        )

        schema_errors = assert_schema(zetyra, "bayesian_two_arm_continuous")

        rec_n = zetyra["recommended_n_per_arm"]
        # data_var_delta = σ² (1/n_t + 1/n_c) = σ² × 2/n for 1:1 allocation
        data_var_delta = s["data_variance"] * (1.0 / rec_n + 1.0 / rec_n)
        ref_mean, ref_var = reference_posterior_continuous(
            s["prior_effect_mean"], s["prior_effect_variance"],
            s["alternative_difference"], data_var_delta,
        )

        mean_ok = abs(zetyra["posterior_at_alt_mean"] - ref_mean) < TOLERANCE
        var_ok = abs(zetyra["posterior_at_alt_variance"] - ref_var) < TOLERANCE

        passed = mean_ok and var_ok and len(schema_errors) == 0
        results.append({
            "test": f"Continuous Posterior: {s['name']}",
            "rec_n": rec_n,
            "zetyra_mean": zetyra["posterior_at_alt_mean"],
            "ref_mean": round(ref_mean, 6),
            "zetyra_var": zetyra["posterior_at_alt_variance"],
            "ref_var": round(ref_var, 6),
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_continuous_vague_prior_convergence(client) -> pd.DataFrame:
    """With vague prior, Bayesian n_per_arm ≈ frequentist two-sample z-test n."""
    results = []

    scenarios = [
        {"delta": 0.5, "sigma2": 1.0, "label": "δ=0.5, σ²=1"},
        {"delta": 0.3, "sigma2": 2.0, "label": "δ=0.3, σ²=2"},
    ]

    for s in scenarios:
        zetyra = client.bayesian_two_arm(
            endpoint_type="continuous",
            prior_effect_mean=0.0, prior_effect_variance=100.0,
            data_variance=s["sigma2"],
            alternative_difference=s["delta"],
            decision_threshold=0.95,
            target_power=0.80, target_type1_error=0.05,
            n_simulations=5000,
            n_min=5, n_max=500, n_step=5,
        )

        freq_n = frequentist_two_sample_z_n(0.05, 0.80, s["sigma2"], s["delta"])
        ratio = zetyra["recommended_n_per_arm"] / freq_n
        close_enough = 0.5 <= ratio <= 2.0

        results.append({
            "test": f"Vague prior → freq: {s['label']}",
            "bayesian_n_per_arm": zetyra["recommended_n_per_arm"],
            "frequentist_n_per_arm": round(freq_n, 1),
            "ratio": round(ratio, 3),
            "pass": close_enough,
        })

    return pd.DataFrame(results)


def validate_continuous_properties(client) -> pd.DataFrame:
    """Invariant/property tests for continuous two-arm endpoint."""
    results = []

    base = dict(
        endpoint_type="continuous",
        prior_effect_mean=0.0, prior_effect_variance=10.0,
        data_variance=1.0,
        decision_threshold=0.95,
        target_power=0.80, target_type1_error=0.05,
        n_simulations=3000,
        n_min=5, n_max=300, n_step=5,
        seed=42,
    )

    # Property 1: Larger effect → smaller n
    small_eff = client.bayesian_two_arm(**base, alternative_difference=0.3)
    large_eff = client.bayesian_two_arm(**base, alternative_difference=0.6)
    schema_se = assert_schema(small_eff, "bayesian_two_arm_continuous")
    schema_le = assert_schema(large_eff, "bayesian_two_arm_continuous")
    results.append({
        "test": "Continuous property: larger effect → smaller n",
        "n_small_effect": small_eff["recommended_n_per_arm"],
        "n_large_effect": large_eff["recommended_n_per_arm"],
        "pass": large_eff["recommended_n_per_arm"] <= small_eff["recommended_n_per_arm"]
                and len(schema_se) == 0 and len(schema_le) == 0,
    })

    # Property 2: Larger data variance → larger n
    low_var = client.bayesian_two_arm(
        **{**base, "data_variance": 1.0}, alternative_difference=0.4,
    )
    high_var = client.bayesian_two_arm(
        **{**base, "data_variance": 4.0}, alternative_difference=0.4,
    )
    schema_lv = assert_schema(low_var, "bayesian_two_arm_continuous")
    schema_hv = assert_schema(high_var, "bayesian_two_arm_continuous")
    results.append({
        "test": "Continuous property: larger variance → larger n",
        "n_low_var": low_var["recommended_n_per_arm"],
        "n_high_var": high_var["recommended_n_per_arm"],
        "pass": high_var["recommended_n_per_arm"] >= low_var["recommended_n_per_arm"]
                and len(schema_lv) == 0 and len(schema_hv) == 0,
    })

    return pd.DataFrame(results)


def validate_continuous_input_guards(client) -> pd.DataFrame:
    """Validate that missing/invalid continuous fields return 422."""
    results = []

    guards = [
        {
            "name": "continuous missing prior_effect_variance",
            "data": {
                "endpoint_type": "continuous",
                "prior_effect_mean": 0.0, "data_variance": 1.0,
                "alternative_difference": 0.5,
                "n_simulations": 500, "n_min": 10, "n_max": 50, "n_step": 10,
            },
            "field": "prior_effect_variance",
        },
        {
            "name": "continuous missing data_variance",
            "data": {
                "endpoint_type": "continuous",
                "prior_effect_mean": 0.0, "prior_effect_variance": 1.0,
                "alternative_difference": 0.5,
                "n_simulations": 500, "n_min": 10, "n_max": 50, "n_step": 10,
            },
            "field": "data_variance",
        },
        {
            "name": "continuous missing alternative_difference",
            "data": {
                "endpoint_type": "continuous",
                "prior_effect_mean": 0.0, "prior_effect_variance": 1.0,
                "data_variance": 1.0,
                "n_simulations": 500, "n_min": 10, "n_max": 50, "n_step": 10,
            },
            "field": "alternative_difference",
        },
    ]

    for g in guards:
        resp = client._post_raw("/bayesian/two-arm", g["data"])
        status_ok = resp.status_code in (400, 422)
        field_mentioned = g["field"] in resp.text
        results.append({
            "test": f"Guard: {g['name']}",
            "status_code": resp.status_code,
            "field_in_body": field_mentioned,
            "pass": status_ok and field_mentioned,
        })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN TWO-ARM VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Two-Arm MC Validation (Binomial CI)")
    print("-" * 70)
    mc_results = validate_two_arm(client)
    print(mc_results.to_string(index=False))
    all_frames.append(mc_results)

    print("\n2. Directional Properties")
    print("-" * 70)
    prop_results = validate_properties(client)
    print(prop_results.to_string(index=False))
    all_frames.append(prop_results)

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

    print("\n5. Symmetry (Null Hypothesis)")
    print("-" * 70)
    sym_results = validate_symmetry(client)
    print(sym_results.to_string(index=False))
    all_frames.append(sym_results)

    print("\n" + "=" * 70)
    print("CONTINUOUS ENDPOINT TESTS")
    print("=" * 70)

    print("\n6. Continuous Two-Arm MC Validation")
    print("-" * 70)
    cont_mc = validate_continuous_two_arm(client)
    print(cont_mc.to_string(index=False))
    all_frames.append(cont_mc)

    print("\n7. Continuous Analytical Posterior")
    print("-" * 70)
    cont_post = validate_continuous_analytical_posterior(client)
    print(cont_post.to_string(index=False))
    all_frames.append(cont_post)

    print("\n8. Continuous Vague-Prior → Frequentist Convergence")
    print("-" * 70)
    cont_freq = validate_continuous_vague_prior_convergence(client)
    print(cont_freq.to_string(index=False))
    all_frames.append(cont_freq)

    print("\n9. Continuous Properties")
    print("-" * 70)
    cont_prop = validate_continuous_properties(client)
    print(cont_prop.to_string(index=False))
    all_frames.append(cont_prop)

    print("\n10. Continuous Input Guards")
    print("-" * 70)
    cont_guards = validate_continuous_input_guards(client)
    print(cont_guards.to_string(index=False))
    all_frames.append(cont_guards)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_two_arm_validation.csv", index=False)

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
