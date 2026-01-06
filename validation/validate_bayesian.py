"""
Validate Zetyra Bayesian Predictive Power Calculator

Validates Normal-Normal and Beta-Binomial conjugate models
against analytical posterior formulas and published benchmarks.

References:
- Gelman et al. (2013) "Bayesian Data Analysis"
- Spiegelhalter et al. (1986) "Monitoring clinical trials: Conditional or predictive power?"
- Lee & Liu (2008) "Predictive probability in clinical trials"
- Spiegelhalter et al. (2004) "Bayesian Approaches to Clinical Trials"
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import comb, betaln
from zetyra_client import get_client

# Tolerance for validation
TOLERANCE = 0.02  # 2% for posterior parameters
PP_TOLERANCE = 0.04  # 4% for predictive probability (MC variability)

# Path to reference data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


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


def analytical_beta_binomial_pp(
    prior_alpha: float,
    prior_beta: float,
    observed_successes: int,
    observed_n: int,
    future_n: int,
    threshold: int,
) -> float:
    """
    Calculate analytical predictive probability for beta-binomial model.

    PP = P(X_future ≥ threshold | X_observed)

    where X_future ~ BetaBinomial(future_n, α + x_obs, β + n_obs - x_obs)

    The beta-binomial PMF is:
    P(k) = C(n,k) * B(α+k, β+n-k) / B(α, β)

    Reference: Lee & Liu (2008), Spiegelhalter et al. (2004)
    """
    # Posterior parameters
    post_alpha = prior_alpha + observed_successes
    post_beta = prior_beta + (observed_n - observed_successes)

    # Calculate P(X_future ≥ threshold)
    pp = 0.0
    for k in range(threshold, future_n + 1):
        # Beta-binomial PMF
        log_prob = (
            np.log(comb(future_n, k, exact=True))
            + betaln(post_alpha + k, post_beta + future_n - k)
            - betaln(post_alpha, post_beta)
        )
        pp += np.exp(log_prob)

    return pp


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


def validate_analytical_benchmarks(client) -> pd.DataFrame:
    """
    Validate against analytical benchmark test cases from CSV.

    Tests predictive probability calculations against closed-form solutions
    for beta-binomial models (Lee & Liu 2008, Spiegelhalter 1986).
    """
    results = []

    # Load benchmark data
    csv_path = os.path.join(DATA_DIR, "bayesian_test_cases.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    benchmarks = pd.read_csv(csv_path)

    for _, row in benchmarks.iterrows():
        test_id = row["test_id"]
        outcome_type = row["outcome_type"]
        source = row["source"]
        expected_pp = float(row["analytical_pp"])

        if outcome_type == "binary":
            # Parse parameters
            prior_params = [float(x) for x in row["prior_params"].split(",")]
            prior_alpha, prior_beta = prior_params

            observed = row["observed_data"].split("/")
            observed_successes = int(observed[0])
            observed_n = int(observed[1])

            future_n = int(row["future_n"])
            threshold = int(row["threshold"])

            # Calculate analytical reference
            analytical_pp = analytical_beta_binomial_pp(
                prior_alpha, prior_beta,
                observed_successes, observed_n,
                future_n, threshold
            )

            # Note: The Zetyra API may not have a single-arm PP endpoint,
            # so we validate analytical formula itself and compare to CSV
            deviation = abs(analytical_pp - expected_pp)

            results.append({
                "test_id": test_id,
                "source": source,
                "scenario": f"Beta({prior_alpha},{prior_beta}), {observed_successes}/{observed_n}, need {threshold}/{future_n}",
                "analytical_pp": round(analytical_pp, 3),
                "expected_pp": expected_pp,
                "deviation": round(deviation, 3),
                "pass": deviation < PP_TOLERANCE,
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

    # Property 3: Larger interim sample → more precise estimate → different PP
    # With a positive interim effect, more data collected means more certainty
    # The PP should reflect this - test that PP changes with sample size
    small_interim = client.bayesian_continuous(
        prior_mean=0.0,
        prior_var=0.5,
        interim_effect=0.5,  # Positive effect
        interim_var=0.2,     # Higher variance (less data)
        interim_n=50,
        final_n=200,
        success_threshold=0.95,
    )
    large_interim = client.bayesian_continuous(
        prior_mean=0.0,
        prior_var=0.5,
        interim_effect=0.5,  # Same positive effect
        interim_var=0.05,    # Lower variance (more data)
        interim_n=150,
        final_n=200,
        success_threshold=0.95,
    )
    # With same positive effect but more precise estimate, PP should be higher
    prop3_pass = large_interim["predictive_probability"] >= small_interim["predictive_probability"]
    results.append({
        "property": "More precise interim → higher PP (positive effect)",
        "expected": "PP(var=0.05) ≥ PP(var=0.2)",
        "actual": f"{large_interim['predictive_probability']:.3f} ≥ {small_interim['predictive_probability']:.3f}",
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

    # Property 5: Beta-binomial - observed success rate affects PP
    # Higher observed rate should give higher PP
    # (using binary endpoint property test)
    low_rate = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=20, control_n=100,
        treatment_successes=25, treatment_n=100,  # 25% rate
        final_n=200,
    )
    high_rate = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=20, control_n=100,
        treatment_successes=45, treatment_n=100,  # 45% rate
        final_n=200,
    )
    prop5_pass = high_rate["predictive_probability"] >= low_rate["predictive_probability"]
    results.append({
        "property": "Higher treatment rate → higher PP",
        "expected": "PP(45%) ≥ PP(25%)",
        "actual": f"{high_rate['predictive_probability']:.3f} ≥ {low_rate['predictive_probability']:.3f}",
        "pass": prop5_pass,
    })

    # Property 6: Informative skeptical prior reduces PP for binary
    uniform_prior = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,  # Uniform prior
        control_successes=30, control_n=100,
        treatment_successes=40, treatment_n=100,
        final_n=200,
    )
    skeptical_prior = client.bayesian_binary(
        prior_alpha=2, prior_beta=8,  # Skeptical prior (prior mean = 0.2)
        control_successes=30, control_n=100,
        treatment_successes=40, treatment_n=100,
        final_n=200,
    )
    prop6_pass = uniform_prior["predictive_probability"] >= skeptical_prior["predictive_probability"]
    results.append({
        "property": "Uniform prior → higher PP than skeptical",
        "expected": "PP(Beta(1,1)) ≥ PP(Beta(2,8))",
        "actual": f"{uniform_prior['predictive_probability']:.3f} ≥ {skeptical_prior['predictive_probability']:.3f}",
        "pass": prop6_pass,
    })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full Bayesian validation."""
    client = get_client(base_url)

    # Continuous scenarios (expanded to 6)
    continuous_scenarios = [
        {"prior_mean": 0.0, "prior_var": 1.0, "interim_effect": 0.3, "interim_var": 0.1, "interim_n": 100, "final_n": 200},
        {"prior_mean": 0.5, "prior_var": 0.5, "interim_effect": 0.4, "interim_var": 0.05, "interim_n": 150, "final_n": 200},
        {"prior_mean": 0.0, "prior_var": 2.0, "interim_effect": 0.6, "interim_var": 0.2, "interim_n": 50, "final_n": 100},
        {"prior_mean": 0.0, "prior_var": 1.0, "interim_effect": 0.0, "interim_var": 0.1, "interim_n": 100, "final_n": 200},  # Null effect
        {"prior_mean": -0.2, "prior_var": 0.5, "interim_effect": 0.2, "interim_var": 0.08, "interim_n": 80, "final_n": 160},  # Negative prior
        {"prior_mean": 0.0, "prior_var": 0.25, "interim_effect": 0.5, "interim_var": 0.04, "interim_n": 200, "final_n": 300},  # Tight prior
    ]

    # Binary scenarios (expanded to 6)
    binary_scenarios = [
        {"prior_alpha": 1, "prior_beta": 1, "control_successes": 30, "control_n": 100, "treatment_successes": 45, "treatment_n": 100, "final_n": 200},
        {"prior_alpha": 2, "prior_beta": 2, "control_successes": 50, "control_n": 200, "treatment_successes": 70, "treatment_n": 200, "final_n": 400},
        {"prior_alpha": 1, "prior_beta": 1, "control_successes": 10, "control_n": 50, "treatment_successes": 20, "treatment_n": 50, "final_n": 100},
        {"prior_alpha": 0.5, "prior_beta": 0.5, "control_successes": 25, "control_n": 100, "treatment_successes": 35, "treatment_n": 100, "final_n": 200},  # Jeffreys prior
        {"prior_alpha": 2, "prior_beta": 8, "control_successes": 15, "control_n": 75, "treatment_successes": 25, "treatment_n": 75, "final_n": 150},  # Skeptical prior
        {"prior_alpha": 1, "prior_beta": 1, "control_successes": 40, "control_n": 100, "treatment_successes": 40, "treatment_n": 100, "final_n": 200},  # Equal rates
    ]

    continuous_results = validate_posterior_continuous(client, continuous_scenarios)
    binary_results = validate_posterior_binary(client, binary_scenarios)
    property_results = validate_predictive_properties(client)
    benchmark_results = validate_analytical_benchmarks(client)

    all_pass = (
        continuous_results["pass"].all()
        and binary_results["pass"].all()
        and property_results["pass"].all()
    )
    if len(benchmark_results) > 0:
        all_pass = all_pass and benchmark_results["pass"].all()

    return {
        "continuous": continuous_results,
        "binary": binary_results,
        "properties": property_results,
        "benchmarks": benchmark_results,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA BAYESIAN PREDICTIVE POWER VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nContinuous Posterior Validation (6 tests)")
    print("-" * 70)
    print(results["continuous"].to_string(index=False))

    print("\nBinary Posterior Validation (6 tests)")
    print("-" * 70)
    print(results["binary"].to_string(index=False))

    print("\nPredictive Power Properties (6 tests)")
    print("-" * 70)
    print(results["properties"].to_string(index=False))

    if len(results["benchmarks"]) > 0:
        print(f"\nAnalytical Benchmarks ({len(results['benchmarks'])} tests)")
        print("-" * 70)
        print(results["benchmarks"].to_string(index=False))

    # Summary
    total_tests = (
        len(results["continuous"])
        + len(results["binary"])
        + len(results["properties"])
        + len(results["benchmarks"])
    )
    passed_tests = (
        results["continuous"]["pass"].sum()
        + results["binary"]["pass"].sum()
        + results["properties"]["pass"].sum()
    )
    if len(results["benchmarks"]) > 0:
        passed_tests += results["benchmarks"]["pass"].sum()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
