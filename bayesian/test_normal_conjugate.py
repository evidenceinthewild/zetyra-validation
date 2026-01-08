#!/usr/bin/env python3
"""
Validate Normal-Normal conjugate posterior calculations

References:
- Spiegelhalter et al. (1986) "Monitoring clinical trials"
- Gelman et al. (2013) "Bayesian Data Analysis"
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.zetyra_client import get_client
import pandas as pd
import numpy as np
from scipy import stats

TOLERANCE = 0.01  # 1% tolerance


def reference_posterior_continuous(
    prior_mean: float,
    prior_var: float,
    interim_effect: float,
    interim_var: float,
) -> dict:
    """
    Calculate posterior using Normal-Normal conjugate formula.

    Prior: θ ~ N(μ₀, σ₀²)
    Likelihood: x | θ ~ N(θ, σ²/n)
    Posterior: θ | x ~ N(μ₁, σ₁²)
    """
    prior_precision = 1 / prior_var
    data_precision = 1 / interim_var

    posterior_precision = prior_precision + data_precision
    posterior_var = 1 / posterior_precision
    posterior_mean = posterior_var * (prior_mean * prior_precision + interim_effect * data_precision)

    z = stats.norm.ppf(0.975)
    ci_lower = posterior_mean - z * np.sqrt(posterior_var)
    ci_upper = posterior_mean + z * np.sqrt(posterior_var)

    return {
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def validate_posterior_continuous(client, scenarios: list) -> pd.DataFrame:
    """Validate continuous posterior calculations."""
    results = []

    for scenario in scenarios:
        zetyra = client.bayesian_continuous(**scenario)
        reference = reference_posterior_continuous(
            prior_mean=scenario["prior_mean"],
            prior_var=scenario["prior_var"],
            interim_effect=scenario["interim_effect"],
            interim_var=scenario["interim_var"],
        )

        mean_dev = abs(zetyra["posterior_mean"] - reference["posterior_mean"])
        var_dev = abs(zetyra["posterior_var"] - reference["posterior_var"])

        results.append({
            "scenario": f"prior=({scenario['prior_mean']}, {scenario['prior_var']})",
            "zetyra_mean": round(zetyra["posterior_mean"], 4),
            "ref_mean": round(reference["posterior_mean"], 4),
            "zetyra_var": round(zetyra["posterior_var"], 4),
            "ref_var": round(reference["posterior_var"], 4),
            "pass": mean_dev < 0.01 and var_dev < 0.01,
        })

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Validate predictive power properties."""
    results = []

    # Property 1: Strong effect → high PP
    strong = client.bayesian_continuous(
        prior_mean=0.0, prior_var=0.5,
        interim_effect=0.8, interim_var=0.05,
        interim_n=200, final_n=300, success_threshold=0.95,
    )
    results.append({
        "property": "Strong effect → high PP",
        "expected": "PP > 0.7",
        "actual": f"PP = {strong['predictive_probability']:.3f}",
        "pass": strong["predictive_probability"] > 0.7,
    })

    # Property 2: Null effect → low PP
    null = client.bayesian_continuous(
        prior_mean=0.0, prior_var=0.5,
        interim_effect=0.0, interim_var=0.1,
        interim_n=100, final_n=200, success_threshold=0.95,
    )
    results.append({
        "property": "Null effect → low PP",
        "expected": "PP < 0.3",
        "actual": f"PP = {null['predictive_probability']:.3f}",
        "pass": null["predictive_probability"] < 0.3,
    })

    # Property 3: Optimistic prior → higher PP
    optimistic = client.bayesian_continuous(
        prior_mean=0.3, prior_var=0.5,
        interim_effect=0.3, interim_var=0.1,
        interim_n=100, final_n=200,
    )
    skeptical = client.bayesian_continuous(
        prior_mean=0.0, prior_var=0.5,
        interim_effect=0.3, interim_var=0.1,
        interim_n=100, final_n=200,
    )
    results.append({
        "property": "Optimistic prior → higher PP",
        "expected": "PP(prior=0.3) ≥ PP(prior=0.0)",
        "actual": f"{optimistic['predictive_probability']:.3f} ≥ {skeptical['predictive_probability']:.3f}",
        "pass": optimistic["predictive_probability"] >= skeptical["predictive_probability"],
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("NORMAL-NORMAL CONJUGATE VALIDATION")
    print("=" * 70)

    scenarios = [
        {"prior_mean": 0.0, "prior_var": 1.0, "interim_effect": 0.3, "interim_var": 0.1, "interim_n": 100, "final_n": 200},
        {"prior_mean": 0.5, "prior_var": 0.5, "interim_effect": 0.4, "interim_var": 0.05, "interim_n": 150, "final_n": 200},
        {"prior_mean": 0.0, "prior_var": 2.0, "interim_effect": 0.6, "interim_var": 0.2, "interim_n": 50, "final_n": 100},
    ]

    print("\nPosterior Validation")
    print("-" * 70)
    posterior_results = validate_posterior_continuous(client, scenarios)
    print(posterior_results.to_string(index=False))

    print("\nPredictive Properties")
    print("-" * 70)
    property_results = validate_properties(client)
    print(property_results.to_string(index=False))

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat([posterior_results, property_results], ignore_index=True)
    all_results.to_csv("results/normal_conjugate_validation.csv", index=False)

    all_pass = posterior_results["pass"].all() and property_results["pass"].all()

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
