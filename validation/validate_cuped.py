"""
Validate Zetyra CUPED Calculator

Validates CUPED (Controlled-experiment Using Pre-Experiment Data)
variance reduction calculations against analytical formulas.

Reference:
- Deng et al. (2013) "Improving the Sensitivity of Online Controlled Experiments
  by Utilizing Pre-Experiment Data"
"""

import pandas as pd
import numpy as np
from scipy import stats
from zetyra_client import get_client

# Tolerance for validation
TOLERANCE = 0.01  # 1% relative difference


def reference_cuped(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    correlation: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate CUPED variance reduction using analytical formulas.

    Key formulas:
    - Variance reduction factor: (1 - ρ²)
    - Adjusted variance: σ² × (1 - ρ²)
    - Sample size reduction: proportional to variance reduction
    """
    # Effect size (absolute)
    delta = baseline_mean * mde

    # Z-values
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Original sample size (per arm)
    n_original = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

    # Variance reduction factor
    r_squared = correlation ** 2
    variance_reduction_factor = 1 - r_squared

    # Adjusted variance and sample size
    adjusted_std = baseline_std * np.sqrt(variance_reduction_factor)
    n_adjusted = 2 * ((z_alpha + z_beta) * adjusted_std / delta) ** 2

    # Variance reduction percentage
    variance_reduction_pct = (1 - variance_reduction_factor) * 100

    return {
        "n_original": int(np.ceil(n_original)),
        "n_adjusted": int(np.ceil(n_adjusted)),
        "variance_reduction_factor": variance_reduction_factor,
        "variance_reduction_pct": variance_reduction_pct,
        "r_squared": r_squared,
    }


def validate_cuped(client, scenarios: list) -> pd.DataFrame:
    """Validate CUPED against reference formulas."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.cuped(**scenario)

        # Get reference result
        reference = reference_cuped(**scenario)

        # Calculate deviations
        n_orig_dev = abs(zetyra["n_original"] - reference["n_original"]) / reference["n_original"]
        n_adj_dev = abs(zetyra["n_adjusted"] - reference["n_adjusted"]) / max(reference["n_adjusted"], 1)
        vrf_dev = abs(zetyra["variance_reduction_factor"] - reference["variance_reduction_factor"])

        results.append({
            "correlation": scenario["correlation"],
            "zetyra_n_orig": zetyra["n_original"],
            "ref_n_orig": reference["n_original"],
            "zetyra_n_adj": zetyra["n_adjusted"],
            "ref_n_adj": reference["n_adjusted"],
            "zetyra_vrf": round(zetyra["variance_reduction_factor"], 4),
            "ref_vrf": round(reference["variance_reduction_factor"], 4),
            "n_deviation_pct": round(n_adj_dev * 100, 2),
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

    # Property 2: Variance reduction factor = 1 - ρ²
    for rho in [0.3, 0.5, 0.7, 0.9]:
        result = client.cuped(**base_params, correlation=rho)
        expected_vrf = 1 - rho ** 2
        actual_vrf = result["variance_reduction_factor"]
        prop_pass = abs(actual_vrf - expected_vrf) < 0.001
        results.append({
            "property": f"VRF = 1 - ρ² (ρ={rho})",
            "expected": f"{expected_vrf:.4f}",
            "actual": f"{actual_vrf:.4f}",
            "pass": prop_pass,
        })

    # Property 3: Symmetry - negative correlation same reduction as positive
    pos_result = client.cuped(**base_params, correlation=0.7)
    neg_result = client.cuped(**base_params, correlation=-0.7)
    prop3_pass = pos_result["n_adjusted"] == neg_result["n_adjusted"]
    results.append({
        "property": "Symmetry: |ρ| determines reduction",
        "expected": f"n(ρ=0.7) == n(ρ=-0.7)",
        "actual": f"{pos_result['n_adjusted']} == {neg_result['n_adjusted']}",
        "pass": prop3_pass,
    })

    # Property 4: Higher correlation → smaller sample size
    rho_05 = client.cuped(**base_params, correlation=0.5)
    rho_08 = client.cuped(**base_params, correlation=0.8)
    prop4_pass = rho_08["n_adjusted"] < rho_05["n_adjusted"]
    results.append({
        "property": "Higher ρ → smaller n",
        "expected": f"n(ρ=0.8) < n(ρ=0.5)",
        "actual": f"{rho_08['n_adjusted']} < {rho_05['n_adjusted']}",
        "pass": prop4_pass,
    })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full CUPED validation."""
    client = get_client(base_url)

    # Test scenarios across correlation range
    scenarios = [
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.0},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.3},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.5},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.7},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": 0.9},
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "correlation": -0.6},
        {"baseline_mean": 50, "baseline_std": 15, "mde": 0.10, "correlation": 0.6},
        {"baseline_mean": 200, "baseline_std": 40, "mde": 0.03, "correlation": 0.8, "alpha": 0.01, "power": 0.90},
    ]

    numerical_results = validate_cuped(client, scenarios)
    property_results = validate_cuped_properties(client)

    return {
        "numerical": numerical_results,
        "properties": property_results,
        "all_pass": numerical_results["pass"].all() and property_results["pass"].all(),
    }


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA CUPED VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nNumerical Validation (vs Analytical Formulas)")
    print("-" * 70)
    print(results["numerical"].to_string(index=False))

    print("\nMathematical Properties")
    print("-" * 70)
    print(results["properties"].to_string(index=False))

    print("\n" + "=" * 70)
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
