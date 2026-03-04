#!/usr/bin/env python3
"""
Bayesian Sequential Survival: Zhou & Ji Boundary Cross-Validation

Extends the structural tests in test_bayesian_sequential_survival.py with
published cross-validation. The survival mapping uses:
  data_variance = 4  (Schoenfeld: Var(log HR) = 4/d)
  n_k = events_k / 2

We verify that boundaries from the /bayesian/sequential/survival endpoint
exactly match the Zhou & Ji (2024) boundary formula applied with these
mapped parameters. Additionally, we verify Type I error control via
multivariate normal MC integration (same approach as test_zhou_ji_table3.py).

References:
- Zhou, T., & Ji, Y. (2024) Table 3, Stopping_boundaries.R companion code
- Schoenfeld (1983) Var(log HR) = 4/d
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


BOUNDARY_TOLERANCE = 0.02  # Tight — these are deterministic


# ─── Reference implementations ────────────────────────────────────────

def bound_bayes_pp_survival(prior_mean, prior_variance, events_per_look, gamma):
    """
    Zhou & Ji boundary formula, survival-mapped.

    Survival mapping: data_variance=4, n_k = events_k / 2

    c_k = Φ⁻¹(γ) * √(1 + data_variance / (n_k * ν²)) - μ * √(data_variance) / (√(n_k) * ν²)
    """
    data_variance = 4.0
    events = np.array(events_per_look, dtype=float)
    n_k = events / 2  # Schoenfeld mapping

    c = (
        sp_stats.norm.ppf(gamma) * np.sqrt(1 + data_variance / (n_k * prior_variance))
        - prior_mean * np.sqrt(data_variance) / (np.sqrt(n_k) * prior_variance)
    )
    return c


def calc_type_i_error_survival(boundaries, events_per_look):
    """
    Type I error via MC multivariate normal integration.

    Under H0 (true log(HR) = 0), the z-statistics at interim looks
    have correlation ρ(i,j) = √(n_i/n_j) = √(d_i/d_j) for i < j.
    """
    K = len(boundaries)
    d = np.array(events_per_look, dtype=float)

    # Build correlation matrix from event counts
    corr = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            corr[i, j] = corr[j, i] = np.sqrt(d[i] / d[j])

    # MC integration
    rng = np.random.default_rng(42)
    n_samples = 500_000
    samples = rng.multivariate_normal(np.zeros(K), corr, size=n_samples)
    prob_no_reject = np.mean(np.all(samples < np.array(boundaries), axis=1))
    return 1 - prob_no_reject


# ─── Test functions ───────────────────────────────────────────────────

def validate_survival_boundary_formula(client) -> pd.DataFrame:
    """Verify survival boundaries match Zhou & Ji formula with mapped params."""
    results = []

    scenarios = [
        {
            "name": "Standard (d=[100,200,300], γ=0.975)",
            "events": [100, 200, 300],
            "efficacy_threshold": 0.975,
            "prior_variance": 1.0,
        },
        {
            "name": "Many looks (d=[50,100,150,200,250], γ=0.95)",
            "events": [50, 100, 150, 200, 250],
            "efficacy_threshold": 0.95,
            "prior_variance": 1.0,
        },
        {
            "name": "Large events (d=[200,400,600,800], γ=0.975)",
            "events": [200, 400, 600, 800],
            "efficacy_threshold": 0.975,
            "prior_variance": 1.0,
        },
        {
            "name": "Front-loaded (d=[200,250,300], γ=0.95)",
            "events": [200, 250, 300],
            "efficacy_threshold": 0.95,
            "prior_variance": 1.0,
        },
    ]

    for s in scenarios:
        # Call Zetyra survival API
        zetyra = client.bayesian_sequential_survival(
            endpoint_type="survival",
            n_per_look=s["events"],
            hazard_ratio=0.7,  # metadata only
            efficacy_threshold=s["efficacy_threshold"],
            futility_threshold=0.10,
        )

        schema_errors = assert_schema(zetyra, "bayesian_sequential")

        # Reference boundaries
        ref_boundaries = bound_bayes_pp_survival(
            0.0, s["prior_variance"], s["events"], s["efficacy_threshold"]
        )

        for i, d_k in enumerate(s["events"]):
            ref_val = ref_boundaries[i]
            zetyra_val = zetyra["efficacy_boundaries"][i]
            dev = abs(zetyra_val - ref_val)

            results.append({
                "test": f"{s['name']}, look {i+1} (d={d_k})",
                "reference": round(ref_val, 4),
                "zetyra": round(zetyra_val, 4),
                "deviation": round(dev, 4),
                "pass": dev < BOUNDARY_TOLERANCE and len(schema_errors) == 0,
            })

    return pd.DataFrame(results)


def validate_survival_type_i_error(client) -> pd.DataFrame:
    """Verify Type I error control using Zetyra API-returned boundaries.

    Uses the boundaries from the Zetyra API (not local reference) to compute
    Type I error via MC multivariate normal integration. Also checks the
    Type I rate is reasonable for the design's gamma threshold.
    """
    results = []

    configs = [
        {
            "name": "Standard 3-look (γ=0.975, vague prior)",
            "events": [100, 200, 300],
            "efficacy_threshold": 0.975,
            "prior_variance": 1.0,
            # γ=0.975 → single-look ≈ 0.025, 3 looks inflates to ~0.05.
            # Observed: 0.0485. Band catches both inflation and over-conservatism.
            "type_i_lower": 0.03,
            "type_i_upper": 0.06,
        },
        {
            "name": "5-look strict (γ=0.99, vague prior)",
            "events": [50, 100, 150, 200, 250],
            "efficacy_threshold": 0.99,
            "prior_variance": 1.0,
            # γ=0.99 → single-look ≈ 0.01, 5 looks inflates to ~0.025.
            # Observed: 0.0250. Band catches both inflation and over-conservatism.
            "type_i_lower": 0.015,
            "type_i_upper": 0.04,
        },
    ]

    for cfg in configs:
        # Get boundaries from Zetyra API
        zetyra = client.bayesian_sequential_survival(
            endpoint_type="survival",
            n_per_look=cfg["events"],
            hazard_ratio=0.7,
            efficacy_threshold=cfg["efficacy_threshold"],
            futility_threshold=0.10,
        )
        api_boundaries = zetyra["efficacy_boundaries"]

        # MC Type I error using API boundaries
        type_i = calc_type_i_error_survival(api_boundaries, cfg["events"])

        in_band = cfg["type_i_lower"] <= type_i <= cfg["type_i_upper"]
        results.append({
            "test": f"Type I (API boundaries): {cfg['name']}",
            "boundaries": str([round(b, 3) for b in api_boundaries]),
            "type_i_mc": round(type_i, 4),
            "band": f"[{cfg['type_i_lower']}, {cfg['type_i_upper']}]",
            "pass": in_band,
        })

    return pd.DataFrame(results)


def validate_vague_prior_convergence(client) -> pd.DataFrame:
    """
    With vague prior (ν² → ∞), survival boundaries should converge to
    Φ⁻¹(γ), the fixed-sample z-threshold.

    This is the survival analog of the continuous vague-prior test in
    test_zhou_ji_table3.py.
    """
    results = []

    gamma = 0.975
    z_gamma = sp_stats.norm.ppf(gamma)  # ≈ 1.96

    # Large events + truly vague prior (ν² = 1e6): boundary → z_gamma
    large_events = [500, 1000, 1500, 2000]
    zetyra = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=large_events,
        hazard_ratio=0.7,
        prior_variance=1e6,
        efficacy_threshold=gamma,
        futility_threshold=0.10,
    )

    for i, d_k in enumerate(large_events):
        boundary = zetyra["efficacy_boundaries"][i]
        dev = abs(boundary - z_gamma)
        results.append({
            "test": f"Vague prior convergence (d={d_k}): → Φ⁻¹({gamma})",
            "boundary": round(boundary, 4),
            "z_gamma": round(z_gamma, 4),
            "deviation": round(dev, 4),
            "pass": dev < 0.05,
        })

    return pd.DataFrame(results)


def validate_survival_futility_boundaries(client) -> pd.DataFrame:
    """Verify futility boundaries also match the formula with mapped params."""
    results = []

    events = [100, 200, 300]
    fut_threshold = 0.10

    zetyra = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=events,
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=fut_threshold,
    )

    # Futility boundaries use the same formula with γ = futility_threshold
    ref_fut = bound_bayes_pp_survival(0.0, 1.0, events, fut_threshold)

    for i, d_k in enumerate(events):
        ref_val = ref_fut[i]
        zetyra_val = zetyra["futility_boundaries"][i]
        dev = abs(zetyra_val - ref_val)

        results.append({
            "test": f"Futility look {i+1} (d={d_k})",
            "reference": round(ref_val, 4),
            "zetyra": round(zetyra_val, 4),
            "deviation": round(dev, 4),
            "pass": dev < BOUNDARY_TOLERANCE,
        })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN SEQUENTIAL SURVIVAL: ZHOU & JI CROSS-VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Survival Boundary Formula (5 scenarios)")
    print("-" * 70)
    bf = validate_survival_boundary_formula(client)
    print(bf.to_string(index=False))
    all_frames.append(bf)

    print("\n2. Type I Error (MC Multivariate Normal)")
    print("-" * 70)
    ti = validate_survival_type_i_error(client)
    print(ti.to_string(index=False))
    all_frames.append(ti)

    print("\n3. Vague Prior Convergence (boundary → Φ⁻¹(γ))")
    print("-" * 70)
    vc = validate_vague_prior_convergence(client)
    print(vc.to_string(index=False))
    all_frames.append(vc)

    print("\n4. Futility Boundary Formula")
    print("-" * 70)
    fb = validate_survival_futility_boundaries(client)
    print(fb.to_string(index=False))
    all_frames.append(fb)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_sequential_survival_benchmark.csv", index=False)

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
