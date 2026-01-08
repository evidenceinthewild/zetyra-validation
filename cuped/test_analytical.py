#!/usr/bin/env python3
"""
Validate Zetyra CUPED Calculator against analytical formulas

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces
variance using the formula: Var_adjusted = Var * (1 - ρ²)

Reference:
- Deng et al. (2013) "Improving the Sensitivity of Online Controlled
  Experiments by Utilizing Pre-Experiment Data" (WSDM)
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.zetyra_client import get_client
import pandas as pd
import numpy as np
from scipy import stats

TOLERANCE = 0.01  # 1% relative difference


def reference_cuped(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    correlation: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Calculate CUPED using analytical formulas."""
    delta = baseline_mean * mde
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n_original = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

    r_squared = correlation ** 2
    variance_reduction_factor = 1 - r_squared
    adjusted_std = baseline_std * np.sqrt(variance_reduction_factor)
    n_adjusted = 2 * ((z_alpha + z_beta) * adjusted_std / delta) ** 2

    return {
        "n_original": int(np.ceil(n_original)),
        "n_adjusted": int(np.ceil(n_adjusted)),
        "variance_reduction_factor": variance_reduction_factor,
    }


def validate_cuped_numerical(client, scenarios: list) -> pd.DataFrame:
    """Validate CUPED against reference formulas."""
    results = []

    for scenario in scenarios:
        zetyra = client.cuped(**scenario)
        reference = reference_cuped(**scenario)

        n_orig_dev = abs(zetyra["n_original"] - reference["n_original"]) / reference["n_original"]
        n_adj_dev = abs(zetyra["n_adjusted"] - reference["n_adjusted"]) / max(reference["n_adjusted"], 1)
        vrf_dev = abs(zetyra["variance_reduction_factor"] - reference["variance_reduction_factor"])

        results.append({
            "correlation": scenario["correlation"],
            "zetyra_n_adj": zetyra["n_adjusted"],
            "ref_n_adj": reference["n_adjusted"],
            "zetyra_vrf": round(zetyra["variance_reduction_factor"], 4),
            "ref_vrf": round(reference["variance_reduction_factor"], 4),
            "pass": n_orig_dev <= TOLERANCE and n_adj_dev <= TOLERANCE and vrf_dev <= 0.001,
        })

    return pd.DataFrame(results)


def validate_cuped_properties(client) -> pd.DataFrame:
    """Validate mathematical properties of CUPED."""
    results = []
    base_params = {
        "baseline_mean": 100,
        "baseline_std": 20,
        "mde": 0.05,
        "alpha": 0.05,
        "power": 0.80,
    }

    # Property 1: Zero correlation → no reduction
    result = client.cuped(**base_params, correlation=0.0)
    prop1_pass = result["n_original"] == result["n_adjusted"]
    results.append({
        "property": "Zero correlation → no reduction",
        "expected": "n_original == n_adjusted",
        "actual": f"{result['n_original']} == {result['n_adjusted']}",
        "pass": prop1_pass,
    })

    # Property 2: VRF = 1 - ρ²
    for rho in [0.3, 0.5, 0.7, 0.9]:
        result = client.cuped(**base_params, correlation=rho)
        expected_vrf = 1 - rho ** 2
        actual_vrf = result["variance_reduction_factor"]
        results.append({
            "property": f"VRF = 1 - ρ² (ρ={rho})",
            "expected": f"{expected_vrf:.4f}",
            "actual": f"{actual_vrf:.4f}",
            "pass": abs(actual_vrf - expected_vrf) < 0.001,
        })

    # Property 3: Symmetry
    pos_result = client.cuped(**base_params, correlation=0.7)
    neg_result = client.cuped(**base_params, correlation=-0.7)
    results.append({
        "property": "Symmetry: |ρ| determines reduction",
        "expected": f"n(ρ=0.7) == n(ρ=-0.7)",
        "actual": f"{pos_result['n_adjusted']} == {neg_result['n_adjusted']}",
        "pass": pos_result["n_adjusted"] == neg_result["n_adjusted"],
    })

    # Property 4: Higher correlation → smaller sample size
    rho_05 = client.cuped(**base_params, correlation=0.5)
    rho_08 = client.cuped(**base_params, correlation=0.8)
    results.append({
        "property": "Higher ρ → smaller n",
        "expected": f"n(ρ=0.8) < n(ρ=0.5)",
        "actual": f"{rho_08['n_adjusted']} < {rho_05['n_adjusted']}",
        "pass": rho_08["n_adjusted"] < rho_05["n_adjusted"],
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("CUPED ANALYTICAL VALIDATION")
    print("=" * 70)

    scenarios = [
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.0},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.3},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.5},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.7},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.9},
    ]

    print("\nNumerical Validation")
    print("-" * 70)
    numerical_results = validate_cuped_numerical(client, scenarios)
    print(numerical_results.to_string(index=False))

    print("\nMathematical Properties")
    print("-" * 70)
    property_results = validate_cuped_properties(client)
    print(property_results.to_string(index=False))

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat([numerical_results, property_results], ignore_index=True)
    all_results.to_csv("results/cuped_validation_results.csv", index=False)

    all_pass = numerical_results["pass"].all() and property_results["pass"].all()

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
