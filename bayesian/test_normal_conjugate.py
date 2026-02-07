#!/usr/bin/env python3
"""
Validate Normal-Normal conjugate posterior calculations

Tests:
1. Exact posterior validation (conjugate formula)
2. Predictive power directional properties
3. Schema contracts
4. Boundary cases

References:
- Spiegelhalter et al. (2004) "Bayesian Approaches to Clinical Trials and Health-Care Evaluation"
- Gelman et al. (2013) "Bayesian Data Analysis"
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
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

    Prior: theta ~ N(mu_0, sigma_0^2)
    Likelihood: x | theta ~ N(theta, sigma^2/n)
    Posterior: theta | x ~ N(mu_1, sigma_1^2)
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

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_continuous")

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
            "pass": mean_dev < TOLERANCE and var_dev < TOLERANCE and len(schema_errors) == 0,
        })

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Validate predictive power properties."""
    results = []

    # Property 1: Strong effect -> high PP
    strong = client.bayesian_continuous(
        prior_mean=0.0, prior_var=0.5,
        interim_effect=0.8, interim_var=0.05,
        interim_n=200, final_n=300, success_threshold=0.95,
    )
    schema_errors = assert_schema(strong, "bayesian_continuous")
    results.append({
        "property": "Strong effect -> high PP",
        "expected": "PP > 0.7",
        "actual": f"PP = {strong['predictive_probability']:.3f}",
        "pass": strong["predictive_probability"] > 0.7 and len(schema_errors) == 0,
    })

    # Property 2: Null effect -> low PP
    null = client.bayesian_continuous(
        prior_mean=0.0, prior_var=0.5,
        interim_effect=0.0, interim_var=0.1,
        interim_n=100, final_n=200, success_threshold=0.95,
    )
    schema_errors_null = assert_schema(null, "bayesian_continuous")
    results.append({
        "property": "Null effect -> low PP",
        "expected": "PP < 0.3",
        "actual": f"PP = {null['predictive_probability']:.3f}",
        "pass": null["predictive_probability"] < 0.3 and len(schema_errors_null) == 0,
    })

    # Property 3: Optimistic prior -> higher PP
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
    schema_errors_opt = assert_schema(optimistic, "bayesian_continuous")
    schema_errors_skep = assert_schema(skeptical, "bayesian_continuous")
    results.append({
        "property": "Optimistic prior -> higher PP",
        "expected": "PP(prior=0.3) >= PP(prior=0.0)",
        "actual": f"{optimistic['predictive_probability']:.3f} >= {skeptical['predictive_probability']:.3f}",
        "pass": optimistic["predictive_probability"] >= skeptical["predictive_probability"] and len(schema_errors_opt) == 0 and len(schema_errors_skep) == 0,
    })

    return pd.DataFrame(results)


def validate_boundary_cases(client) -> pd.DataFrame:
    """Boundary-condition scenarios."""
    results = []

    # Very strong prior (low variance)
    z = client.bayesian_continuous(
        prior_mean=0.0, prior_var=0.01,
        interim_effect=0.5, interim_var=0.1,
        interim_n=100, final_n=200,
    )
    schema_ok = len(assert_schema(z, "bayesian_continuous")) == 0
    # Strong prior centered at 0 should pull posterior toward 0
    results.append({
        "test": "Boundary: very strong prior (var=0.01)",
        "posterior_mean": round(z["posterior_mean"], 4),
        "pass": schema_ok and abs(z["posterior_mean"]) < 0.1,
    })

    # Very vague prior (high variance)
    z = client.bayesian_continuous(
        prior_mean=0.0, prior_var=100.0,
        interim_effect=0.5, interim_var=0.1,
        interim_n=100, final_n=200,
    )
    schema_ok = len(assert_schema(z, "bayesian_continuous")) == 0
    # Vague prior -> posterior dominated by data
    results.append({
        "test": "Boundary: very vague prior (var=100)",
        "posterior_mean": round(z["posterior_mean"], 4),
        "pass": schema_ok and abs(z["posterior_mean"] - 0.5) < 0.05,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("NORMAL-NORMAL CONJUGATE VALIDATION")
    print("=" * 70)

    all_frames = []

    scenarios = [
        {"prior_mean": 0.0, "prior_var": 1.0, "interim_effect": 0.3, "interim_var": 0.1, "interim_n": 100, "final_n": 200},
        {"prior_mean": 0.5, "prior_var": 0.5, "interim_effect": 0.4, "interim_var": 0.05, "interim_n": 150, "final_n": 200},
        {"prior_mean": 0.0, "prior_var": 2.0, "interim_effect": 0.6, "interim_var": 0.2, "interim_n": 50, "final_n": 100},
    ]

    print("\nPosterior Validation")
    print("-" * 70)
    posterior_results = validate_posterior_continuous(client, scenarios)
    print(posterior_results.to_string(index=False))
    all_frames.append(posterior_results)

    print("\nPredictive Properties")
    print("-" * 70)
    property_results = validate_properties(client)
    print(property_results.to_string(index=False))
    all_frames.append(property_results)

    print("\nBoundary Cases")
    print("-" * 70)
    boundary_results = validate_boundary_cases(client)
    print(boundary_results.to_string(index=False))
    all_frames.append(boundary_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/normal_conjugate_validation.csv", index=False)

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
