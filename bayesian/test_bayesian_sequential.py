#!/usr/bin/env python3
"""
Validate Bayesian Sequential Monitoring Boundaries (Continuous)

Tests analytical z-score boundaries from Zhou & Ji (2024):
c_k = Phi^-1(gamma) * sqrt(1 + sigma^2/(n_k * nu^2)) - mu * sqrt(sigma^2) / (sqrt(n_k) * nu^2)

Additional tests:
1. Structural properties (monotonicity, futility < efficacy, convergence)
2. Input guards (422/400 for invalid inputs)
3. Boundary cases (single look, many looks, large n)
4. Invariants: increasing threshold -> higher boundaries
5. Schema contracts

References:
- Zhou, T., & Ji, Y. (2024) "On Bayesian Sequential Clinical Trial Designs"
  NEJSDS, 2(1), 136-151.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

BOUNDARY_TOLERANCE = 0.01  # Tight for analytical formula


# ─── Reference implementation ─────────────────────────────────────────

def reference_boundary_continuous(
    prior_mean, prior_variance, data_variance, n_k, threshold
):
    """
    Zhou & Ji (2024) boundary formula:
    c_k = Phi^-1(gamma) * sqrt(1 + sigma^2/(n_k * nu^2)) - mu * sqrt(sigma^2) / (sqrt(n_k) * nu^2)
    """
    c = (
        sp_stats.norm.ppf(threshold) * np.sqrt(1 + data_variance / (n_k * prior_variance))
        - prior_mean * np.sqrt(data_variance) / (np.sqrt(n_k) * prior_variance)
    )
    return round(c, 4)


# ─── Test functions ───────────────────────────────────────────────────

def validate_analytical_boundaries(client) -> pd.DataFrame:
    """Validate analytical boundary formula."""
    scenarios = [
        {
            "name": "Zhou & Ji (2024) example",
            "prior_mean": 0.0, "prior_variance": 1.0, "data_variance": 1.0,
            "n_per_look": [30, 60, 90],
            "efficacy_threshold": 0.975, "futility_threshold": 0.10,
        },
        {
            "name": "Informative prior",
            "prior_mean": 0.3, "prior_variance": 0.5, "data_variance": 1.0,
            "n_per_look": [50, 100],
            "efficacy_threshold": 0.95, "futility_threshold": 0.10,
        },
        {
            "name": "Vague prior (nu^2=100)",
            "prior_mean": 0.0, "prior_variance": 100.0, "data_variance": 1.0,
            "n_per_look": [25, 50, 75, 100],
            "efficacy_threshold": 0.975, "futility_threshold": None,
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_sequential(
            endpoint_type="continuous",
            n_per_look=s["n_per_look"],
            prior_mean=s["prior_mean"],
            prior_variance=s["prior_variance"],
            data_variance=s["data_variance"],
            efficacy_threshold=s["efficacy_threshold"],
            futility_threshold=s["futility_threshold"],
        )

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_sequential")

        for i, n_k in enumerate(s["n_per_look"]):
            ref_eff = reference_boundary_continuous(
                s["prior_mean"], s["prior_variance"], s["data_variance"],
                n_k, s["efficacy_threshold"],
            )
            zetyra_eff = zetyra["efficacy_boundaries"][i]

            eff_ok = abs(zetyra_eff - ref_eff) < BOUNDARY_TOLERANCE

            row = {
                "test": f"{s['name']}, look {i+1} (n={n_k})",
                "zetyra_eff": zetyra_eff,
                "ref_eff": ref_eff,
                "eff_deviation": round(abs(zetyra_eff - ref_eff), 4),
                "pass": eff_ok and len(schema_errors) == 0,
            }

            # Check futility if applicable
            if s["futility_threshold"] is not None:
                ref_fut = reference_boundary_continuous(
                    s["prior_mean"], s["prior_variance"], s["data_variance"],
                    n_k, s["futility_threshold"],
                )
                zetyra_fut = zetyra["futility_boundaries"][i]
                fut_ok = abs(zetyra_fut - ref_fut) < BOUNDARY_TOLERANCE
                row["zetyra_fut"] = zetyra_fut
                row["ref_fut"] = ref_fut
                row["pass"] = eff_ok and fut_ok and len(schema_errors) == 0

            results.append(row)

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Validate structural properties of boundaries."""
    results = []

    # Property 1: Efficacy boundaries monotonically decrease with n
    z = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[30, 60, 90, 120],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.975, futility_threshold=0.10,
    )
    schema_errors = assert_schema(z, "bayesian_sequential")
    eff = z["efficacy_boundaries"]
    monotone = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "Efficacy boundaries monotonically decrease",
        "boundaries": str([round(b, 3) for b in eff]),
        "pass": monotone and len(schema_errors) == 0,
    })

    # Property 2: Futility boundaries < efficacy boundaries at every look
    fut = z["futility_boundaries"]
    fut_less = all(
        f is not None and f < e
        for f, e in zip(fut, eff)
    )
    results.append({
        "test": "Futility < efficacy at each look",
        "boundaries": f"eff={[round(b, 3) for b in eff]}, fut={[round(b, 3) if b is not None else None for b in fut]}",
        "pass": fut_less and len(schema_errors) == 0,
    })

    # Property 3: Vague prior -> boundaries converge to Phi^-1(gamma) ~ 1.96
    vague = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[25, 50, 75, 100],
        prior_mean=0.0, prior_variance=100.0, data_variance=1.0,
        efficacy_threshold=0.975, futility_threshold=None,
    )
    schema_errors_vague = assert_schema(vague, "bayesian_sequential")
    z_critical = sp_stats.norm.ppf(0.975)  # 1.96
    converges_to_freq = all(
        abs(b - z_critical) < 0.05 for b in vague["efficacy_boundaries"]
    )
    results.append({
        "test": f"Vague prior converges to z={z_critical:.2f}",
        "boundaries": str([round(b, 3) for b in vague["efficacy_boundaries"]]),
        "pass": converges_to_freq and len(schema_errors_vague) == 0,
    })

    # Property 4: Informative prior shifts boundaries
    null_prior = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[50, 100],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.95,
    )
    positive_prior = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[50, 100],
        prior_mean=0.5, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.95,
    )
    schema_errors_null = assert_schema(null_prior, "bayesian_sequential")
    schema_errors_pos = assert_schema(positive_prior, "bayesian_sequential")
    prior_lowers = all(
        p < n for p, n in zip(
            positive_prior["efficacy_boundaries"],
            null_prior["efficacy_boundaries"]
        )
    )
    results.append({
        "test": "Positive prior mean -> lower boundaries",
        "null_prior": str([round(b, 3) for b in null_prior["efficacy_boundaries"]]),
        "positive_prior": str([round(b, 3) for b in positive_prior["efficacy_boundaries"]]),
        "pass": prior_lowers and len(schema_errors_null) == 0 and len(schema_errors_pos) == 0,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate that invalid inputs return 400/422."""
    results = []

    # Each payload includes all required fields; only the field under test is invalid
    valid_base = {
        "endpoint_type": "continuous", "n_per_look": [50, 100],
        "prior_mean": 0.0, "prior_variance": 1.0, "data_variance": 1.0,
        "efficacy_threshold": 0.975,
    }

    guards = [
        {
            "name": "prior_variance = 0",
            "field": "prior_variance",
            "data": {**valid_base, "prior_variance": 0},
        },
        {
            "name": "data_variance = 0",
            "field": "data_variance",
            "data": {**valid_base, "data_variance": 0},
        },
        {
            "name": "invalid endpoint_type",
            "field": "endpoint_type",
            "data": {**valid_base, "endpoint_type": "invalid"},
        },
    ]

    for g in guards:
        resp = client._post_raw("/bayesian/sequential", g["data"])
        status_ok = resp.status_code in (400, 422)
        field_mentioned = g["field"] in resp.text
        results.append({
            "test": f"Guard: {g['name']}",
            "status_code": resp.status_code,
            "field_in_body": field_mentioned,
            "pass": status_ok and field_mentioned,
        })

    return pd.DataFrame(results)


def validate_boundary_cases(client) -> pd.DataFrame:
    """Boundary-condition scenarios."""
    results = []

    # Single look
    z = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[100],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.975,
    )
    schema_single = assert_schema(z, "bayesian_sequential")
    ref = reference_boundary_continuous(0.0, 1.0, 1.0, 100, 0.975)
    results.append({
        "test": "Boundary: single look",
        "zetyra_eff": z["efficacy_boundaries"][0],
        "ref_eff": ref,
        "pass": abs(z["efficacy_boundaries"][0] - ref) < BOUNDARY_TOLERANCE and len(schema_single) == 0,
    })

    # Many looks (8)
    z = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[20, 40, 60, 80, 100, 120, 140, 160],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.975, futility_threshold=0.10,
    )
    schema_many = assert_schema(z, "bayesian_sequential")
    # Verify all boundaries computed and monotonicity holds
    eff = z["efficacy_boundaries"]
    monotone = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "Boundary: 8 looks, monotonicity",
        "n_boundaries": len(eff),
        "pass": len(eff) == 8 and monotone and len(schema_many) == 0,
    })

    # Large n
    z = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[1000, 2000, 3000],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.975,
    )
    schema_large = assert_schema(z, "bayesian_sequential")
    # With large n and finite prior variance, boundaries should be close to z_critical
    z_crit = sp_stats.norm.ppf(0.975)
    close_to_freq = all(abs(b - z_crit) < 0.1 for b in z["efficacy_boundaries"])
    results.append({
        "test": "Boundary: large n -> frequentist",
        "boundaries": str([round(b, 3) for b in z["efficacy_boundaries"]]),
        "pass": close_to_freq and len(schema_large) == 0,
    })

    return pd.DataFrame(results)


def validate_invariants(client) -> pd.DataFrame:
    """Additional invariant tests."""
    results = []

    # Invariant: Increasing threshold -> higher boundaries
    low = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[50, 100],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.90,
    )
    high = client.bayesian_sequential(
        endpoint_type="continuous",
        n_per_look=[50, 100],
        prior_mean=0.0, prior_variance=1.0, data_variance=1.0,
        efficacy_threshold=0.99,
    )
    schema_low = assert_schema(low, "bayesian_sequential")
    schema_high = assert_schema(high, "bayesian_sequential")
    higher_thresh = all(
        h > l for h, l in zip(high["efficacy_boundaries"], low["efficacy_boundaries"])
    )
    results.append({
        "test": "Invariant: higher threshold -> higher boundaries",
        "low": str([round(b, 3) for b in low["efficacy_boundaries"]]),
        "high": str([round(b, 3) for b in high["efficacy_boundaries"]]),
        "pass": higher_thresh and len(schema_low) == 0 and len(schema_high) == 0,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN SEQUENTIAL MONITORING VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Analytical Boundary Formula")
    print("-" * 70)
    boundary_results = validate_analytical_boundaries(client)
    print(boundary_results.to_string(index=False))
    all_frames.append(boundary_results)

    print("\n2. Structural Properties")
    print("-" * 70)
    prop_results = validate_properties(client)
    print(prop_results.to_string(index=False))
    all_frames.append(prop_results)

    print("\n3. Input Guards")
    print("-" * 70)
    guard_results = validate_input_guards(client)
    print(guard_results.to_string(index=False))
    all_frames.append(guard_results)

    print("\n4. Boundary Cases")
    print("-" * 70)
    bc_results = validate_boundary_cases(client)
    print(bc_results.to_string(index=False))
    all_frames.append(bc_results)

    print("\n5. Invariants")
    print("-" * 70)
    inv_results = validate_invariants(client)
    print(inv_results.to_string(index=False))
    all_frames.append(inv_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_sequential_validation.csv", index=False)

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
