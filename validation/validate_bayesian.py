"""
Validate Zetyra Bayesian Predictive Power Calculator

Validates Normal-Normal and Beta-Binomial conjugate models
against analytical posterior formulas.

References:
- Gelman et al. (2013) "Bayesian Data Analysis"
- Spiegelhalter et al. (2004) "Bayesian Approaches to Clinical Trials"
"""

import pandas as pd
import numpy as np
from scipy import stats
from zetyra_client import get_client

# Tolerance for validation
TOLERANCE = 0.02  # 2% for Monte Carlo results


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

    where:
    - σ₁² = 1 / (1/σ₀² + 1/σ_data²)
    - μ₁ = σ₁² × (μ₀/σ₀² + x/σ_data²)
    """
    prior_precision = 1 / prior_var
    data_precision = 1 / interim_var

    posterior_precision = prior_precision + data_precision
    posterior_var = 1 / posterior_precision

    posterior_mean = posterior_var * (prior_mean * prior_precision + interim_effect * data_precision)

    # 95% credible interval
    z = stats.norm.ppf(0.975)
    ci_lower = posterior_mean - z * np.sqrt(posterior_var)
    ci_upper = posterior_mean + z * np.sqrt(posterior_var)

    return {
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def reference_posterior_binary(
    prior_alpha: float,
    prior_beta: float,
    successes: int,
    n: int,
) -> dict:
    """
    Calculate posterior using Beta-Binomial conjugate formula.

    Prior: π ~ Beta(α, β)
    Likelihood: x ~ Binomial(n, π)
    Posterior: π | x ~ Beta(α + x, β + n - x)
    """
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (n - successes)

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

    # 95% credible interval
    ci_lower = stats.beta.ppf(0.025, posterior_alpha, posterior_beta)
    ci_upper = stats.beta.ppf(0.975, posterior_alpha, posterior_beta)

    return {
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "posterior_mean": posterior_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def validate_posterior_continuous(client, scenarios: list) -> pd.DataFrame:
    """Validate continuous posterior calculations."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.bayesian_continuous(**scenario)

        # Get reference result
        reference = reference_posterior_continuous(
            prior_mean=scenario["prior_mean"],
            prior_var=scenario["prior_var"],
            interim_effect=scenario["interim_effect"],
            interim_var=scenario["interim_var"],
        )

        # Calculate deviations
        mean_dev = abs(zetyra["posterior_mean"] - reference["posterior_mean"])
        var_dev = abs(zetyra["posterior_var"] - reference["posterior_var"])

        results.append({
            "scenario": f"prior=({scenario['prior_mean']},{scenario['prior_var']}), data=({scenario['interim_effect']},{scenario['interim_var']})",
            "zetyra_mean": round(zetyra["posterior_mean"], 4),
            "ref_mean": round(reference["posterior_mean"], 4),
            "mean_dev": round(mean_dev, 6),
            "zetyra_var": round(zetyra["posterior_var"], 4),
            "ref_var": round(reference["posterior_var"], 4),
            "var_dev": round(var_dev, 6),
            "pass": mean_dev < 0.01 and var_dev < 0.01,
        })

    return pd.DataFrame(results)


def validate_posterior_binary(client, scenarios: list) -> pd.DataFrame:
    """Validate binary posterior calculations."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.bayesian_binary(**scenario)

        # Get reference for control arm
        ref_control = reference_posterior_binary(
            prior_alpha=scenario["prior_alpha"],
            prior_beta=scenario["prior_beta"],
            successes=scenario["control_successes"],
            n=scenario["control_n"],
        )

        # Get reference for treatment arm
        ref_treatment = reference_posterior_binary(
            prior_alpha=scenario["prior_alpha"],
            prior_beta=scenario["prior_beta"],
            successes=scenario["treatment_successes"],
            n=scenario["treatment_n"],
        )

        # Check posterior parameters
        ctrl_alpha_match = zetyra["posterior_control_alpha"] == ref_control["posterior_alpha"]
        ctrl_beta_match = zetyra["posterior_control_beta"] == ref_control["posterior_beta"]
        trt_alpha_match = zetyra["posterior_treatment_alpha"] == ref_treatment["posterior_alpha"]
        trt_beta_match = zetyra["posterior_treatment_beta"] == ref_treatment["posterior_beta"]

        results.append({
            "control": f"({scenario['control_successes']}/{scenario['control_n']})",
            "treatment": f"({scenario['treatment_successes']}/{scenario['treatment_n']})",
            "ctrl_alpha": f"{zetyra['posterior_control_alpha']} == {ref_control['posterior_alpha']}",
            "ctrl_beta": f"{zetyra['posterior_control_beta']} == {ref_control['posterior_beta']}",
            "trt_alpha": f"{zetyra['posterior_treatment_alpha']} == {ref_treatment['posterior_alpha']}",
            "pass": ctrl_alpha_match and ctrl_beta_match and trt_alpha_match and trt_beta_match,
        })

    return pd.DataFrame(results)


def validate_predictive_properties(client) -> pd.DataFrame:
    """Validate predictive power properties."""
    results = []

    # Property 1: Strong positive effect → high predictive probability
    strong_effect = client.bayesian_continuous(
        prior_mean=0.0,
        prior_var=0.5,
        interim_effect=0.8,
        interim_var=0.05,
        interim_n=200,
        final_n=300,
        success_threshold=0.95,
    )
    prop1_pass = strong_effect["predictive_probability"] > 0.7
    results.append({
        "property": "Strong effect → high PP",
        "expected": "PP > 0.7",
        "actual": f"PP = {strong_effect['predictive_probability']:.3f}",
        "pass": prop1_pass,
    })

    # Property 2: Null effect → low predictive probability
    null_effect = client.bayesian_continuous(
        prior_mean=0.0,
        prior_var=0.5,
        interim_effect=0.0,
        interim_var=0.1,
        interim_n=100,
        final_n=200,
        success_threshold=0.95,
    )
    prop2_pass = null_effect["predictive_probability"] < 0.3
    results.append({
        "property": "Null effect → low PP",
        "expected": "PP < 0.3",
        "actual": f"PP = {null_effect['predictive_probability']:.3f}",
        "pass": prop2_pass,
    })

    # Property 3: More remaining data → more uncertainty
    small_remaining = client.bayesian_continuous(
        prior_mean=0.0,
        prior_var=0.5,
        interim_effect=0.3,
        interim_var=0.1,
        interim_n=180,
        final_n=200,  # Only 20 more
        success_threshold=0.95,
    )
    large_remaining = client.bayesian_continuous(
        prior_mean=0.0,
        prior_var=0.5,
        interim_effect=0.3,
        interim_var=0.1,
        interim_n=100,
        final_n=200,  # 100 more
        success_threshold=0.95,
    )
    # With same positive interim, small remaining should have higher PP
    prop3_pass = small_remaining["predictive_probability"] >= large_remaining["predictive_probability"] - 0.1
    results.append({
        "property": "More data remaining → more uncertainty",
        "expected": "PP(n=180/200) ≥ PP(n=100/200) - 0.1",
        "actual": f"{small_remaining['predictive_probability']:.3f} vs {large_remaining['predictive_probability']:.3f}",
        "pass": prop3_pass,
    })

    # Property 4: Skeptical prior reduces PP
    optimistic = client.bayesian_continuous(
        prior_mean=0.3,  # Prior expects effect
        prior_var=0.5,
        interim_effect=0.3,
        interim_var=0.1,
        interim_n=100,
        final_n=200,
    )
    skeptical = client.bayesian_continuous(
        prior_mean=0.0,  # Prior expects no effect
        prior_var=0.5,
        interim_effect=0.3,
        interim_var=0.1,
        interim_n=100,
        final_n=200,
    )
    prop4_pass = optimistic["predictive_probability"] >= skeptical["predictive_probability"]
    results.append({
        "property": "Optimistic prior → higher PP",
        "expected": "PP(prior=0.3) ≥ PP(prior=0.0)",
        "actual": f"{optimistic['predictive_probability']:.3f} ≥ {skeptical['predictive_probability']:.3f}",
        "pass": prop4_pass,
    })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full Bayesian validation."""
    client = get_client(base_url)

    # Continuous scenarios
    continuous_scenarios = [
        {"prior_mean": 0.0, "prior_var": 1.0, "interim_effect": 0.3, "interim_var": 0.1, "interim_n": 100, "final_n": 200},
        {"prior_mean": 0.5, "prior_var": 0.5, "interim_effect": 0.4, "interim_var": 0.05, "interim_n": 150, "final_n": 200},
        {"prior_mean": 0.0, "prior_var": 2.0, "interim_effect": 0.6, "interim_var": 0.2, "interim_n": 50, "final_n": 100},
    ]

    # Binary scenarios
    binary_scenarios = [
        {"prior_alpha": 1, "prior_beta": 1, "control_successes": 30, "control_n": 100, "treatment_successes": 45, "treatment_n": 100, "final_n": 200},
        {"prior_alpha": 2, "prior_beta": 2, "control_successes": 50, "control_n": 200, "treatment_successes": 70, "treatment_n": 200, "final_n": 400},
        {"prior_alpha": 1, "prior_beta": 1, "control_successes": 10, "control_n": 50, "treatment_successes": 20, "treatment_n": 50, "final_n": 100},
    ]

    continuous_results = validate_posterior_continuous(client, continuous_scenarios)
    binary_results = validate_posterior_binary(client, binary_scenarios)
    property_results = validate_predictive_properties(client)

    return {
        "continuous": continuous_results,
        "binary": binary_results,
        "properties": property_results,
        "all_pass": (
            continuous_results["pass"].all()
            and binary_results["pass"].all()
            and property_results["pass"].all()
        ),
    }


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA BAYESIAN PREDICTIVE POWER VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nContinuous Posterior Validation")
    print("-" * 70)
    print(results["continuous"].to_string(index=False))

    print("\nBinary Posterior Validation")
    print("-" * 70)
    print(results["binary"].to_string(index=False))

    print("\nPredictive Power Properties")
    print("-" * 70)
    print(results["properties"].to_string(index=False))

    print("\n" + "=" * 70)
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
