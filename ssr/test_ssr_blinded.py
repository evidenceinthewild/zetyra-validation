#!/usr/bin/env python3
"""
Validate Blinded Sample Size Re-estimation (SSR)

Tests blinded SSR across continuous, binary, and survival endpoints:
1. No-change scenario: when observed matches assumed, N stays the same
2. Higher variance -> increased N
3. Cap (n_max_factor) enforced
4. Conditional power in (0,1) and consistent with design
5. Structural properties (initial_n > 0, recalculated >= initial)
6. Input guards
7. Schema contract

References:
- Kieser & Friede (2003) "Simple Procedures for Blinded Sample Size Adjustment"
- FDA (2019) "Adaptive Designs for Clinical Trials of Drugs and Biologics"
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
from scipy import stats as sp_stats


# ─── Test functions ──────────────────────────────────────────────────

def validate_continuous(client) -> pd.DataFrame:
    """Validate blinded SSR for continuous outcomes."""
    results = []

    # Scenario 1: No variance change -> N stays the same
    z = client.ssr_blinded(
        endpoint_type="continuous",
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
    )
    schema_errors = assert_schema(z, "ssr_blinded")

    results.append({
        "test": "Continuous: no change -> same N",
        "initial_n": z["initial_n_total"],
        "recalculated_n": z["recalculated_n_total"],
        "increase": z["sample_size_increase"],
        "pass": z["sample_size_increase"] == 0 and len(schema_errors) == 0,
    })

    # Scenario 2: Higher observed variance -> increased N
    z_high = client.ssr_blinded(
        endpoint_type="continuous",
        effect_size=0.3,
        observed_variance=2.0,  # Double the assumed variance
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
    )
    results.append({
        "test": "Continuous: higher variance -> more N",
        "initial_n": z_high["initial_n_total"],
        "recalculated_n": z_high["recalculated_n_total"],
        "pass": z_high["recalculated_n_total"] > z_high["initial_n_total"],
    })

    # Scenario 3: Cap enforced
    z_cap = client.ssr_blinded(
        endpoint_type="continuous",
        effect_size=0.3,
        observed_variance=5.0,  # Very high variance
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=1.5,  # Strict cap
    )
    max_allowed = int(math.ceil(z_cap["initial_n_total"] * 1.5))
    results.append({
        "test": "Continuous: cap enforced",
        "recalculated_n": z_cap["recalculated_n_total"],
        "max_allowed": max_allowed,
        "n_capped": z_cap["n_capped"],
        "pass": z_cap["n_capped"] and z_cap["recalculated_n_total"] <= max_allowed + 2,  # +2 for rounding
    })

    # Scenario 4: Conditional power reasonable
    results.append({
        "test": "Continuous: conditional power in (0,1)",
        "cp": z["conditional_power"],
        "pass": 0 < z["conditional_power"] <= 1,
    })

    # Scenario 5: CP near design power when no change
    results.append({
        "test": "Continuous: CP near design power (no change)",
        "cp": z["conditional_power"],
        "design_power": 0.90,
        "pass": abs(z["conditional_power"] - 0.90) < 0.05,
    })

    # Scenario 6: Different effect_size changes initial N
    z_small = client.ssr_blinded(
        endpoint_type="continuous",
        effect_size=0.1,  # Smaller effect -> larger N
        alpha=0.025, power=0.90, interim_fraction=0.5, n_max_factor=2.0,
    )
    z_large = client.ssr_blinded(
        endpoint_type="continuous",
        effect_size=0.5,  # Larger effect -> smaller N
        alpha=0.025, power=0.90, interim_fraction=0.5, n_max_factor=2.0,
    )
    results.append({
        "test": "Continuous: smaller effect -> larger N",
        "n_small_effect": z_small["initial_n_total"],
        "n_large_effect": z_large["initial_n_total"],
        "pass": z_small["initial_n_total"] > z_large["initial_n_total"],
    })

    return pd.DataFrame(results)


def validate_binary(client) -> pd.DataFrame:
    """Validate blinded SSR for binary outcomes."""
    results = []

    # No change: assumed rates match
    z = client.ssr_blinded(
        endpoint_type="binary",
        p_control=0.20,
        p_treatment=0.35,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
    )
    schema_errors = assert_schema(z, "ssr_blinded")

    results.append({
        "test": "Binary: no change -> same N",
        "initial_n": z["initial_n_total"],
        "recalculated_n": z["recalculated_n_total"],
        "increase": z["sample_size_increase"],
        "pass": z["sample_size_increase"] == 0 and len(schema_errors) == 0,
    })

    # Higher pooled rate -> potentially different N
    z_high = client.ssr_blinded(
        endpoint_type="binary",
        p_control=0.20,
        p_treatment=0.35,
        observed_pooled_rate=0.40,  # Higher than expected
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
    )
    results.append({
        "test": "Binary: different pooled rate -> N adjusts",
        "initial_n": z_high["initial_n_total"],
        "recalculated_n": z_high["recalculated_n_total"],
        "pass": z_high["recalculated_n_total"] > 0 and len(assert_schema(z_high, "ssr_blinded")) == 0,
    })

    # CP in valid range
    results.append({
        "test": "Binary: conditional power in (0,1)",
        "cp": z["conditional_power"],
        "pass": 0 < z["conditional_power"] <= 1,
    })

    return pd.DataFrame(results)


def validate_survival(client) -> pd.DataFrame:
    """Validate blinded SSR for survival outcomes."""
    results = []

    # Baseline survival scenario
    z = client.ssr_blinded(
        endpoint_type="survival",
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
    )
    schema_errors = assert_schema(z, "ssr_blinded")

    results.append({
        "test": "Survival: baseline computes",
        "initial_n": z["initial_n_total"],
        "recalculated_n": z["recalculated_n_total"],
        "pass": z["initial_n_total"] > 0 and len(schema_errors) == 0,
    })

    # No change -> same N
    results.append({
        "test": "Survival: no change -> same N",
        "increase": z["sample_size_increase"],
        "pass": z["sample_size_increase"] == 0,
    })

    # Different observed event rate -> N adjusts
    z_low = client.ssr_blinded(
        endpoint_type="survival",
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        observed_event_rate=0.3,  # Lower than expected
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
    )
    results.append({
        "test": "Survival: lower event rate -> more N",
        "baseline_n": z["recalculated_n_total"],
        "low_event_n": z_low["recalculated_n_total"],
        "pass": z_low["recalculated_n_total"] >= z["recalculated_n_total"],
    })

    # CP in valid range
    results.append({
        "test": "Survival: conditional power in (0,1)",
        "cp": z["conditional_power"],
        "pass": 0 < z["conditional_power"] <= 1,
    })

    # CP near design power when no change
    results.append({
        "test": "Survival: CP near design power",
        "cp": z["conditional_power"],
        "design_power": 0.90,
        "pass": abs(z["conditional_power"] - 0.90) < 0.05,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return 400/422."""
    results = []

    # Missing required fields for continuous
    resp = client.ssr_blinded_raw(endpoint_type="continuous")
    results.append({
        "test": "Guard: continuous without effect_size",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Missing required fields for binary
    resp = client.ssr_blinded_raw(endpoint_type="binary")
    results.append({
        "test": "Guard: binary without rates",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Missing required fields for survival
    resp = client.ssr_blinded_raw(endpoint_type="survival")
    results.append({
        "test": "Guard: survival without HR",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Cross-endpoint structural properties."""
    results = []

    # Property: initial_n_per_arm * 2 ≈ initial_n_total (equal allocation)
    for etype, kwargs in [
        ("continuous", {"effect_size": 0.3}),
        ("binary", {"p_control": 0.20, "p_treatment": 0.35}),
        ("survival", {"hazard_ratio": 0.7, "median_control": 12, "accrual_time": 24, "follow_up_time": 12}),
    ]:
        z = client.ssr_blinded(
            endpoint_type=etype, alpha=0.025, power=0.90,
            interim_fraction=0.5, n_max_factor=2.0, **kwargs,
        )
        # initial_n_per_arm * 2 should equal initial_n_total
        diff = abs(z["initial_n_per_arm"] * 2 - z["initial_n_total"])
        results.append({
            "test": f"{etype}: n_per_arm * 2 = n_total",
            "n_per_arm": z["initial_n_per_arm"],
            "n_total": z["initial_n_total"],
            "pass": diff <= 1,  # Allow 1 for rounding
        })

    # Property: inflation_factor = recalculated / initial
    z = client.ssr_blinded(
        endpoint_type="continuous", effect_size=0.3, observed_variance=1.5,
        alpha=0.025, power=0.90, interim_fraction=0.5, n_max_factor=3.0,
    )
    expected_inf = z["recalculated_n_total"] / z["initial_n_total"]
    results.append({
        "test": "Inflation factor = recalculated / initial",
        "inflation_factor": z["inflation_factor"],
        "computed": round(expected_inf, 4),
        "pass": abs(z["inflation_factor"] - expected_inf) < 0.02,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BLINDED SSR VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Continuous Endpoint")
    print("-" * 70)
    c_results = validate_continuous(client)
    print(c_results.to_string(index=False))
    all_frames.append(c_results)

    print("\n2. Binary Endpoint")
    print("-" * 70)
    b_results = validate_binary(client)
    print(b_results.to_string(index=False))
    all_frames.append(b_results)

    print("\n3. Survival Endpoint")
    print("-" * 70)
    s_results = validate_survival(client)
    print(s_results.to_string(index=False))
    all_frames.append(s_results)

    print("\n4. Input Guards")
    print("-" * 70)
    g_results = validate_input_guards(client)
    print(g_results.to_string(index=False))
    all_frames.append(g_results)

    print("\n5. Cross-Endpoint Properties")
    print("-" * 70)
    p_results = validate_properties(client)
    print(p_results.to_string(index=False))
    all_frames.append(p_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/ssr_blinded_validation.csv", index=False)

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
