#!/usr/bin/env python3
"""
Validate Pocock-Simon Minimization Calculator

Tests covariate-adaptive randomization across analytical and simulation tiers:
1. Analytical: expected imbalance under pure randomization, factor summaries
2. Simulation: minimization vs pure random comparison, per-factor balance
3. Input guards: invalid factors, prevalences, imbalance functions
4. Structural properties: p_randomization extremes, multi-factor scaling

References:
- Pocock & Simon (1975) — Sequential treatment assignment with balancing
  of prognostic factors in controlled clinical trial
- Taves (1974) — Minimization: a new method of assigning patients
- Scott et al. (2002) — Simulation study comparing minimization with
  simple and stratified randomization
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd


# ─── Shared factor definitions ───────────────────────────────────────

TWO_FACTORS = [
    {"name": "Age group", "levels": ["<65", ">=65"], "prevalences": [0.6, 0.4], "weight": 1.0},
    {"name": "Sex", "levels": ["Male", "Female"], "prevalences": [0.5, 0.5], "weight": 1.0},
]

THREE_FACTORS = TWO_FACTORS + [
    {"name": "Stage", "levels": ["I", "II", "III"], "prevalences": [0.3, 0.4, 0.3], "weight": 1.5},
]


# ─── Test functions ──────────────────────────────────────────────────

def validate_analytical(client) -> pd.DataFrame:
    """Validate analytical (Tier 1) minimization results."""
    results = []

    # Test 1: Two-arm, 2 factors — response has analytical_results
    z = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        imbalance_function="range", simulate=False,
    )
    has_analytical = "analytical_results" in z and z["analytical_results"] is not None
    results.append({
        "test": "Analytical: two-arm 2-factor returns analytical_results",
        "has_analytical": has_analytical,
        "pass": has_analytical,
    })

    # Test 2: Analytical returns pure_random_expected_imbalance per factor
    ar = z.get("analytical_results", {})
    pri = ar.get("pure_random_expected_imbalance", {})
    has_factor_keys = "factor_0" in pri and "factor_1" in pri
    results.append({
        "test": "Analytical: pure_random_expected_imbalance per factor",
        "factor_keys": list(pri.keys()),
        "pass": has_factor_keys,
    })

    # Test 3: factor_summary with correct names
    fs = ar.get("factor_summary", [])
    names = [f["name"] for f in fs] if fs else []
    results.append({
        "test": "Analytical: factor_summary has correct names",
        "names": names,
        "pass": names == ["Age group", "Sex"],
    })

    # Test 4: Three-arm design — correct arm count
    z3 = client.minimization(
        n_arms=3, n_total=300, factors=TWO_FACTORS,
        imbalance_function="range", simulate=False,
    )
    ar3 = z3.get("analytical_results", {})
    results.append({
        "test": "Analytical: three-arm returns n_arms=3",
        "n_arms": ar3.get("n_arms"),
        "pass": ar3.get("n_arms") == 3,
    })

    # Test 5: Variance imbalance function echoed correctly
    zv = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        imbalance_function="variance", simulate=False,
    )
    arv = zv.get("analytical_results", {})
    pri_v = arv.get("pure_random_expected_imbalance", {})
    # Check that factor_0 mentions variance
    f0 = pri_v.get("factor_0", {})
    has_variance_label = f0.get("imbalance_function") == "variance"
    results.append({
        "test": "Analytical: variance imbalance_function echoed",
        "imbalance_function": f0.get("imbalance_function"),
        "pass": has_variance_label,
    })

    # Test 6: Factor weights accepted (different weights)
    weighted_factors = [
        {"name": "Age", "levels": ["<65", ">=65"], "prevalences": [0.6, 0.4], "weight": 2.0},
        {"name": "Sex", "levels": ["M", "F"], "prevalences": [0.5, 0.5], "weight": 0.5},
    ]
    zw = client.minimization(
        n_arms=2, n_total=200, factors=weighted_factors,
        imbalance_function="range", simulate=False,
    )
    arw = zw.get("analytical_results", {})
    fsw = arw.get("factor_summary", [])
    weights = [f.get("weight") for f in fsw] if fsw else []
    results.append({
        "test": "Analytical: different factor weights accepted",
        "weights": weights,
        "pass": len(weights) == 2 and weights[0] == 2.0 and weights[1] == 0.5,
    })

    return pd.DataFrame(results)


def validate_simulation(client) -> pd.DataFrame:
    """Validate simulation (Tier 2) minimization results."""
    results = []

    # Test 7: Minimization produces lower imbalance than pure random
    z = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        p_randomization=0.75, imbalance_function="range",
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    sim = z.get("simulation", {})
    est = sim.get("estimates", {})
    overall = est.get("overall_weighted_imbalance", {})
    min_imb = overall.get("minimization", 999)
    pure_imb = overall.get("pure_random", 0)
    results.append({
        "test": "Simulation: minimization < pure random imbalance",
        "min_imbalance": min_imb,
        "pure_imbalance": pure_imb,
        "pass": min_imb < pure_imb,
    })

    # Test 8: Higher p_randomization (0.90) produces lower imbalance than lower (0.60)
    z_high = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        p_randomization=0.90, imbalance_function="range",
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    z_low = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        p_randomization=0.60, imbalance_function="range",
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    high_imb = z_high.get("simulation", {}).get("estimates", {}).get(
        "overall_weighted_imbalance", {}
    ).get("minimization", 999)
    low_imb = z_low.get("simulation", {}).get("estimates", {}).get(
        "overall_weighted_imbalance", {}
    ).get("minimization", 0)
    results.append({
        "test": "Simulation: p=0.90 lower imbalance than p=0.60",
        "imbalance_p90": high_imb,
        "imbalance_p60": low_imb,
        "pass": high_imb < low_imb,
    })

    # Test 9: Simulation returns per-factor imbalance data
    fb = est.get("factor_balance", {})
    has_age = "Age group" in fb
    has_sex = "Sex" in fb
    age_data = fb.get("Age group", {})
    has_keys = all(
        k in age_data
        for k in ["minimization_mean_imbalance", "pure_random_mean_imbalance", "reduction_percent"]
    )
    results.append({
        "test": "Simulation: per-factor imbalance data present",
        "factors_found": list(fb.keys()),
        "pass": has_age and has_sex and has_keys,
    })

    # Test 10: Three-arm simulation works
    z3 = client.minimization(
        n_arms=3, n_total=300, factors=TWO_FACTORS,
        p_randomization=0.75, imbalance_function="range",
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    sim3 = z3.get("simulation", {})
    est3 = sim3.get("estimates", {})
    arm_counts = est3.get("arm_counts_minimization", {})
    results.append({
        "test": "Simulation: three-arm works (3 arm counts)",
        "n_arm_keys": len(arm_counts),
        "pass": len(arm_counts) == 3,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return 400/422."""
    results = []

    # Test 11: Factor with only 1 level rejected
    resp = client.minimization_raw(
        n_arms=2, n_total=200,
        factors=[
            {"name": "Bad", "levels": ["Only"], "prevalences": [1.0], "weight": 1.0},
        ],
        imbalance_function="range",
    )
    results.append({
        "test": "Guard: factor with 1 level rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Test 12: Factor prevalences not summing to ~1.0 rejected
    resp = client.minimization_raw(
        n_arms=2, n_total=200,
        factors=[
            {"name": "Bad", "levels": ["A", "B"], "prevalences": [0.3, 0.3], "weight": 1.0},
        ],
        imbalance_function="range",
    )
    results.append({
        "test": "Guard: prevalences not summing to 1.0 rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Test 13: Too many factors (>10) rejected
    many_factors = [
        {"name": f"F{i}", "levels": ["A", "B"], "prevalences": [0.5, 0.5], "weight": 1.0}
        for i in range(11)
    ]
    resp = client.minimization_raw(
        n_arms=2, n_total=200, factors=many_factors, imbalance_function="range",
    )
    results.append({
        "test": "Guard: >10 factors rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Test 14: Invalid imbalance function rejected
    resp = client.minimization_raw(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        imbalance_function="bogus",
    )
    results.append({
        "test": "Guard: invalid imbalance_function rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    return pd.DataFrame(results)


def validate_structural(client) -> pd.DataFrame:
    """Validate structural properties of minimization."""
    results = []

    # Test 15: p_randomization=1.0 is pure deterministic (very low imbalance)
    z_det = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        p_randomization=1.0, imbalance_function="range",
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    det_imb = z_det.get("simulation", {}).get("estimates", {}).get(
        "overall_weighted_imbalance", {}
    ).get("minimization", 999)
    det_pure = z_det.get("simulation", {}).get("estimates", {}).get(
        "overall_weighted_imbalance", {}
    ).get("pure_random", 0)
    # Deterministic minimization should have very low imbalance
    results.append({
        "test": "Structural: p=1.0 deterministic -> very low imbalance",
        "det_imbalance": det_imb,
        "pure_imbalance": det_pure,
        "pass": det_imb < det_pure * 0.5,  # At least 50% reduction
    })

    # Test 16: p_randomization=0.5 approaches random
    z_half = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        p_randomization=0.50, imbalance_function="range",
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    half_imb = z_half.get("simulation", {}).get("estimates", {}).get(
        "overall_weighted_imbalance", {}
    ).get("minimization", 0)
    half_pure = z_half.get("simulation", {}).get("estimates", {}).get(
        "overall_weighted_imbalance", {}
    ).get("pure_random", 999)
    # p=0.5 for 2-arm is essentially random — imbalance should be close to pure random
    # Allow within 30% of pure random imbalance
    ratio = half_imb / half_pure if half_pure > 0 else 999
    results.append({
        "test": "Structural: p=0.5 approaches random (ratio near 1)",
        "half_imbalance": half_imb,
        "pure_imbalance": half_pure,
        "ratio": round(ratio, 3),
        "pass": ratio > 0.7,  # Not much better than random
    })

    # Test 17: More factors -> generally more imbalance to manage
    z_2f = client.minimization(
        n_arms=2, n_total=200, factors=TWO_FACTORS,
        imbalance_function="range", simulate=False,
    )
    z_3f = client.minimization(
        n_arms=2, n_total=200, factors=THREE_FACTORS,
        imbalance_function="range", simulate=False,
    )
    ar_2f = z_2f.get("analytical_results", {})
    ar_3f = z_3f.get("analytical_results", {})
    n_factors_2 = ar_2f.get("n_factors", 0)
    n_factors_3 = ar_3f.get("n_factors", 0)
    # 3-factor design should have more factor keys in pure_random_expected_imbalance
    pri_3f = ar_3f.get("pure_random_expected_imbalance", {})
    results.append({
        "test": "Structural: more factors -> more imbalance entries",
        "n_factors_2": n_factors_2,
        "n_factors_3": n_factors_3,
        "n_imbalance_keys_3f": len(pri_3f),
        "pass": n_factors_3 == 3 and len(pri_3f) == 3,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("MINIMIZATION (POCOCK-SIMON) VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Analytical Tests")
    print("-" * 70)
    a_results = validate_analytical(client)
    print(a_results.to_string(index=False))
    all_frames.append(a_results)

    print("\n2. Simulation Tests")
    print("-" * 70)
    s_results = validate_simulation(client)
    print(s_results.to_string(index=False))
    all_frames.append(s_results)

    print("\n3. Input Guards")
    print("-" * 70)
    g_results = validate_input_guards(client)
    print(g_results.to_string(index=False))
    all_frames.append(g_results)

    print("\n4. Structural Properties")
    print("-" * 70)
    p_results = validate_structural(client)
    print(p_results.to_string(index=False))
    all_frames.append(p_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/minimization_validation.csv", index=False)

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
