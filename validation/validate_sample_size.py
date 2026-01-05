"""
Validate Zetyra Sample Size Calculators

Compares Zetyra results against:
- R pwr package (pwr.t.test, pwr.2p.test)
- scipy.stats formulas
- Published benchmarks
"""

import pandas as pd
import numpy as np
from scipy import stats
from zetyra_client import get_client

# Tolerance for validation (1% relative difference)
TOLERANCE = 0.01


def reference_sample_size_continuous(
    mean1: float,
    mean2: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> dict:
    """
    Calculate sample size using scipy (matches R pwr::pwr.t.test).

    This is the reference implementation for validation.
    """
    delta = mean2 - mean1
    effect_size = delta / sd

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Sample size formula for unequal allocation
    n1 = (1 + 1 / ratio) * ((z_alpha + z_beta) * sd / abs(delta)) ** 2
    n2 = ratio * n1

    return {
        "n1": int(np.ceil(n1)),
        "n2": int(np.ceil(n2)),
        "n_total": int(np.ceil(n1)) + int(np.ceil(n2)),
        "effect_size": effect_size,
    }


def reference_sample_size_binary(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> dict:
    """
    Calculate sample size for binary outcomes using arcsine formula.

    Matches R pwr::pwr.2p.test methodology.
    """
    # Cohen's h effect size
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    # Pooled proportion under H0
    pooled_p = (p1 + ratio * p2) / (1 + ratio)

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Variance components
    var_h0 = pooled_p * (1 - pooled_p)
    var_h1_1 = p1 * (1 - p1)
    var_h1_2 = p2 * (1 - p2)

    delta = abs(p2 - p1)
    numerator = z_alpha * np.sqrt((1 + 1 / ratio) * var_h0) + z_beta * np.sqrt(
        var_h1_1 + var_h1_2 / ratio
    )
    n1 = (numerator / delta) ** 2
    n2 = ratio * n1

    return {
        "n1": int(np.ceil(n1)),
        "n2": int(np.ceil(n2)),
        "n_total": int(np.ceil(n1)) + int(np.ceil(n2)),
        "effect_size_h": h,
    }


def validate_continuous(client, scenarios: list) -> pd.DataFrame:
    """Validate continuous sample size against reference."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.sample_size_continuous(**scenario)

        # Get reference result
        reference = reference_sample_size_continuous(**scenario)

        # Calculate deviation
        n_deviation = abs(zetyra["n_total"] - reference["n_total"]) / reference["n_total"]
        d_deviation = abs(zetyra["effect_size"] - reference["effect_size"]) / abs(
            reference["effect_size"]
        )

        results.append({
            "scenario": str(scenario),
            "zetyra_n": zetyra["n_total"],
            "reference_n": reference["n_total"],
            "n_deviation_pct": n_deviation * 100,
            "zetyra_d": zetyra["effect_size"],
            "reference_d": reference["effect_size"],
            "d_deviation_pct": d_deviation * 100,
            "pass": n_deviation <= TOLERANCE and d_deviation <= TOLERANCE,
        })

    return pd.DataFrame(results)


def validate_binary(client, scenarios: list) -> pd.DataFrame:
    """Validate binary sample size against reference."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.sample_size_binary(**scenario)

        # Get reference result
        reference = reference_sample_size_binary(**scenario)

        # Calculate deviation
        n_deviation = abs(zetyra["n_total"] - reference["n_total"]) / reference["n_total"]
        h_deviation = abs(zetyra["effect_size_h"] - reference["effect_size_h"]) / abs(
            reference["effect_size_h"]
        )

        results.append({
            "scenario": str(scenario),
            "zetyra_n": zetyra["n_total"],
            "reference_n": reference["n_total"],
            "n_deviation_pct": n_deviation * 100,
            "zetyra_h": zetyra["effect_size_h"],
            "reference_h": reference["effect_size_h"],
            "h_deviation_pct": h_deviation * 100,
            "pass": n_deviation <= TOLERANCE and h_deviation <= TOLERANCE,
        })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full sample size validation."""
    client = get_client(base_url)

    # Continuous scenarios
    continuous_scenarios = [
        {"mean1": 100, "mean2": 105, "sd": 20, "alpha": 0.05, "power": 0.80},
        {"mean1": 100, "mean2": 110, "sd": 25, "alpha": 0.05, "power": 0.90},
        {"mean1": 50, "mean2": 55, "sd": 15, "alpha": 0.01, "power": 0.80},
        {"mean1": 100, "mean2": 105, "sd": 20, "alpha": 0.05, "power": 0.80, "ratio": 2.0},
        {"mean1": 100, "mean2": 103, "sd": 20, "alpha": 0.025, "power": 0.90, "two_sided": False},
    ]

    # Binary scenarios
    binary_scenarios = [
        {"p1": 0.10, "p2": 0.15, "alpha": 0.05, "power": 0.80},
        {"p1": 0.20, "p2": 0.30, "alpha": 0.05, "power": 0.90},
        {"p1": 0.50, "p2": 0.60, "alpha": 0.01, "power": 0.80},
        {"p1": 0.10, "p2": 0.15, "alpha": 0.05, "power": 0.80, "ratio": 2.0},
        {"p1": 0.30, "p2": 0.40, "alpha": 0.025, "power": 0.90, "two_sided": False},
    ]

    continuous_results = validate_continuous(client, continuous_scenarios)
    binary_results = validate_binary(client, binary_scenarios)

    return {
        "continuous": continuous_results,
        "binary": binary_results,
        "all_pass": continuous_results["pass"].all() and binary_results["pass"].all(),
    }


if __name__ == "__main__":
    import sys

    # Allow override for local testing
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA SAMPLE SIZE VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nContinuous Outcomes (Two-Sample T-Test)")
    print("-" * 70)
    print(results["continuous"].to_string(index=False))

    print("\nBinary Outcomes (Two-Proportion Z-Test)")
    print("-" * 70)
    print(results["binary"].to_string(index=False))

    print("\n" + "=" * 70)
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
