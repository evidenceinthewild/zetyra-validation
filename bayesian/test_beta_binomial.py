#!/usr/bin/env python3
"""
Validate Beta-Binomial conjugate posterior and predictive probability

Tests:
1. Exact posterior validation (conjugate formula)
2. Predictive probability directional properties
3. Schema contracts
4. Boundary-condition scenarios

References:
- Lee & Liu (2008) "Predictive probability in clinical trials"
- Gelman et al. (2013) "Bayesian Data Analysis"
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd


def reference_posterior_binary(
    prior_alpha: float,
    prior_beta: float,
    successes: int,
    n: int,
) -> dict:
    """Calculate posterior using Beta-Binomial conjugate formula."""
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (n - successes)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

    return {
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "posterior_mean": posterior_mean,
    }


def validate_posterior_binary(client, scenarios: list) -> pd.DataFrame:
    """Validate binary posterior calculations."""
    results = []

    for scenario in scenarios:
        zetyra = client.bayesian_binary(**scenario)

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_binary")

        ref_control = reference_posterior_binary(
            prior_alpha=scenario["prior_alpha"],
            prior_beta=scenario["prior_beta"],
            successes=scenario["control_successes"],
            n=scenario["control_n"],
        )
        ref_treatment = reference_posterior_binary(
            prior_alpha=scenario["prior_alpha"],
            prior_beta=scenario["prior_beta"],
            successes=scenario["treatment_successes"],
            n=scenario["treatment_n"],
        )

        ctrl_alpha_match = zetyra["posterior_control_alpha"] == ref_control["posterior_alpha"]
        ctrl_beta_match = zetyra["posterior_control_beta"] == ref_control["posterior_beta"]
        trt_alpha_match = zetyra["posterior_treatment_alpha"] == ref_treatment["posterior_alpha"]
        trt_beta_match = zetyra["posterior_treatment_beta"] == ref_treatment["posterior_beta"]

        results.append({
            "scenario": f"C={scenario['control_successes']}/{scenario['control_n']}, T={scenario['treatment_successes']}/{scenario['treatment_n']}",
            "control_posterior": f"Beta({zetyra['posterior_control_alpha']}, {zetyra['posterior_control_beta']})",
            "treatment_posterior": f"Beta({zetyra['posterior_treatment_alpha']}, {zetyra['posterior_treatment_beta']})",
            "pass": ctrl_alpha_match and ctrl_beta_match and trt_alpha_match and trt_beta_match and len(schema_errors) == 0,
        })

    return pd.DataFrame(results)


def validate_pp_properties(client) -> pd.DataFrame:
    """Validate predictive probability directional properties via API."""
    results = []

    # Property 1: Strong treatment effect -> high PP
    strong = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=25, control_n=100,
        treatment_successes=50, treatment_n=100,
        final_n=200,
    )
    schema_errors = assert_schema(strong, "bayesian_binary")
    results.append({
        "property": "Strong effect (25% vs 50%) -> high PP",
        "pp": round(strong["predictive_probability"], 3),
        "pass": strong["predictive_probability"] > 0.7 and len(schema_errors) == 0,
    })

    # Property 2: No treatment effect -> low PP
    null = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=30, control_n=100,
        treatment_successes=30, treatment_n=100,
        final_n=200,
    )
    schema_errors_null = assert_schema(null, "bayesian_binary")
    results.append({
        "property": "No effect (30% vs 30%) -> low PP",
        "pp": round(null["predictive_probability"], 3),
        "pass": null["predictive_probability"] < 0.3 and len(schema_errors_null) == 0,
    })

    # Property 3: More interim data with effect -> higher PP
    early = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=10, control_n=30,
        treatment_successes=18, treatment_n=30,
        final_n=200,
    )
    late = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=30, control_n=100,
        treatment_successes=50, treatment_n=100,
        final_n=200,
    )
    schema_errors_early = assert_schema(early, "bayesian_binary")
    schema_errors_late = assert_schema(late, "bayesian_binary")
    results.append({
        "property": "More data with effect -> higher PP",
        "pp_early": round(early["predictive_probability"], 3),
        "pp_late": round(late["predictive_probability"], 3),
        "pass": late["predictive_probability"] >= early["predictive_probability"] and len(schema_errors_early) == 0 and len(schema_errors_late) == 0,
    })

    return pd.DataFrame(results)


def validate_boundary_cases(client) -> pd.DataFrame:
    """Boundary-condition scenarios for beta-binomial."""
    results = []

    # Jeffreys prior (0.5, 0.5)
    z = client.bayesian_binary(
        prior_alpha=0.5, prior_beta=0.5,
        control_successes=20, control_n=50,
        treatment_successes=30, treatment_n=50,
        final_n=100,
    )
    schema_ok = len(assert_schema(z, "bayesian_binary")) == 0
    results.append({
        "test": "Boundary: Jeffreys prior (0.5, 0.5)",
        "pp": round(z["predictive_probability"], 3),
        "pass": schema_ok and 0 <= z["predictive_probability"] <= 1,
    })

    # Strong informative prior
    z = client.bayesian_binary(
        prior_alpha=50, prior_beta=50,
        control_successes=20, control_n=50,
        treatment_successes=30, treatment_n=50,
        final_n=100,
    )
    schema_ok = len(assert_schema(z, "bayesian_binary")) == 0
    results.append({
        "test": "Boundary: strong prior (50, 50)",
        "pp": round(z["predictive_probability"], 3),
        "pass": schema_ok and 0 <= z["predictive_probability"] <= 1,
    })

    # Zero successes in control
    z = client.bayesian_binary(
        prior_alpha=1, prior_beta=1,
        control_successes=0, control_n=20,
        treatment_successes=10, treatment_n=20,
        final_n=50,
    )
    schema_ok = len(assert_schema(z, "bayesian_binary")) == 0
    results.append({
        "test": "Boundary: zero control successes",
        "pp": round(z["predictive_probability"], 3),
        "pass": schema_ok and z["predictive_probability"] > 0.5,  # Strong effect
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BETA-BINOMIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    scenarios = [
        {"prior_alpha": 1, "prior_beta": 1, "control_successes": 30, "control_n": 100, "treatment_successes": 45, "treatment_n": 100, "final_n": 200},
        {"prior_alpha": 0.5, "prior_beta": 0.5, "control_successes": 25, "control_n": 100, "treatment_successes": 35, "treatment_n": 100, "final_n": 200},
        {"prior_alpha": 2, "prior_beta": 8, "control_successes": 15, "control_n": 75, "treatment_successes": 25, "treatment_n": 75, "final_n": 150},
    ]

    print("\n1. Posterior Validation")
    print("-" * 70)
    posterior_results = validate_posterior_binary(client, scenarios)
    print(posterior_results.to_string(index=False))
    all_frames.append(posterior_results)

    print("\n2. Predictive Probability Properties")
    print("-" * 70)
    pp_results = validate_pp_properties(client)
    print(pp_results.to_string(index=False))
    all_frames.append(pp_results)

    print("\n3. Boundary Cases")
    print("-" * 70)
    boundary_results = validate_boundary_cases(client)
    print(boundary_results.to_string(index=False))
    all_frames.append(boundary_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_validation_results.csv", index=False)

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
