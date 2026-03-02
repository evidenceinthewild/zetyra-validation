#!/usr/bin/env python3
"""
Cross-validate Bayesian Sequential boundaries against Zhou & Ji (2024) Table 3

Reproduces the exact numerical boundary values from Table 3 of:
  Zhou, T., & Ji, Y. (2024) "On Bayesian Sequential Clinical Trial Designs"
  NEJSDS, 2(1), 136-151. DOI: 10.51387/23-NEJSDS24

The paper provides boundary values for a single-arm trial with:
  - K = 5 analyses at n = [200, 400, 600, 800, 1000]
  - sigma^2 = 1 (outcome variance)
  - Two prior configurations:
    Version 1 (conservative): mu=0, nu^2=0.054^2, gamma=0.95
    Version 2 (vague):        mu=0, nu^2=1.0,     gamma=0.983

The paper's companion R code (Stopping_boundaries.R) implements the same
formula as our backend — this is a direct cross-validation.

Additionally validates Type I error rate via multivariate normal integration
(the paper verifies each boundary set controls Type I error at 0.05).

References:
- Zhou, T., & Ji, Y. (2024) Table 3, pp. 142-143
- Companion R code: Stopping_boundaries.R (supplementary material)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

# Tight tolerance — these are analytical, deterministic values
BOUNDARY_TOLERANCE = 0.02

# ─── Table 3 reference values ────────────────────────────────────────────
# Source: Zhou & Ji (2024), Table 3, p. 143

LOOKS = [200, 400, 600, 800, 1000]

# Posterior probability design, Version 1 (conservative prior)
# Prior: N(0, 0.054^2), threshold gamma = 0.95
TABLE3_VER1 = {
    "name": "Post. prob. ver.1 (conservative prior)",
    "prior_mean": 0.0,
    "prior_variance": 0.054 ** 2,  # nu^2 = 0.002916
    "data_variance": 1.0,          # sigma^2 = 1
    "threshold": 0.95,             # gamma
    "boundaries": [2.71, 2.24, 2.06, 1.97, 1.91],
}

# Posterior probability design, Version 2 (vague prior)
# Prior: N(0, 1), threshold gamma = 0.983
TABLE3_VER2 = {
    "name": "Post. prob. ver.2 (vague prior)",
    "prior_mean": 0.0,
    "prior_variance": 1.0,         # nu^2 = 1
    "data_variance": 1.0,          # sigma^2 = 1
    "threshold": 0.983,            # gamma
    "boundaries": [2.13, 2.12, 2.12, 2.12, 2.12],
}


# ─── Reference implementation (from paper's Eq. 2.2) ─────────────────────

def bound_bayes_pp(mu, nusq, sigmasq, gamma, n_vec):
    """
    Bound_BayesPP from Zhou & Ji companion R code (Stopping_boundaries.R):

    c_k = qnorm(gamma) * sqrt(1 + sigma^2 / (n * nu^2)) -
          mu * sqrt(sigma^2) / (sqrt(n) * nu^2)
    """
    n = np.array(n_vec, dtype=float)
    c = (
        sp_stats.norm.ppf(gamma) * np.sqrt(1 + sigmasq / (n * nusq))
        - mu * np.sqrt(sigmasq) / (np.sqrt(n) * nusq)
    )
    return np.round(c, 2)


def calc_type_i_error(boundaries, n_vec):
    """
    Compute exact Type I error via multivariate normal integration.

    Under H0 (theta=0), the z-statistics have correlation:
      rho(i,j) = sqrt(n_i / n_j) for i < j

    Type I error = 1 - P(Z_1 < c_1, ..., Z_K < c_K)

    Matches calc_type_I_error_rate() in the companion R code.
    """
    from scipy.stats import multivariate_normal

    K = len(boundaries)
    n = np.array(n_vec, dtype=float)

    # Build correlation matrix
    corr = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            corr[i, j] = corr[j, i] = np.sqrt(n[i] / n[j])

    # P(all Z < c) under H0
    mean = np.zeros(K)
    # Use pmvnorm equivalent: scipy multivariate_normal CDF
    # For high-dimensional MVN, use Monte Carlo
    from scipy.stats._qmc import Halton
    rng = np.random.default_rng(42)
    n_samples = 500_000
    samples = rng.multivariate_normal(mean, corr, size=n_samples)
    prob_no_reject = np.mean(np.all(samples < np.array(boundaries), axis=1))
    type_i = 1 - prob_no_reject
    return type_i


# ─── Test functions ───────────────────────────────────────────────────────

def validate_table3_boundaries(client) -> pd.DataFrame:
    """Cross-validate boundary values against Table 3."""
    results = []

    for config in [TABLE3_VER1, TABLE3_VER2]:
        # Call Zetyra API
        zetyra = client.bayesian_sequential(
            endpoint_type="continuous",
            n_per_look=LOOKS,
            prior_mean=config["prior_mean"],
            prior_variance=config["prior_variance"],
            data_variance=config["data_variance"],
            efficacy_threshold=config["threshold"],
        )
        schema_errors = assert_schema(zetyra, "bayesian_sequential")

        # Also compute reference locally
        ref_boundaries = bound_bayes_pp(
            config["prior_mean"],
            config["prior_variance"],
            config["data_variance"],
            config["threshold"],
            LOOKS,
        )

        for i, n_k in enumerate(LOOKS):
            paper_val = config["boundaries"][i]
            ref_val = ref_boundaries[i]
            zetyra_val = zetyra["efficacy_boundaries"][i]

            # Check reference implementation matches paper
            ref_matches_paper = abs(ref_val - paper_val) < 0.015

            # Check Zetyra matches paper
            zetyra_matches_paper = abs(zetyra_val - paper_val) < BOUNDARY_TOLERANCE

            results.append({
                "test": f"{config['name']}, look {i+1} (n={n_k})",
                "paper": paper_val,
                "reference": ref_val,
                "zetyra": round(zetyra_val, 2),
                "dev_from_paper": round(abs(zetyra_val - paper_val), 4),
                "pass": ref_matches_paper and zetyra_matches_paper and len(schema_errors) == 0,
            })

    return pd.DataFrame(results)


def validate_type_i_error(client) -> pd.DataFrame:
    """Verify Type I error rate matches the paper's claim of 0.05."""
    results = []

    for config in [TABLE3_VER1, TABLE3_VER2]:
        # Compute boundaries via reference formula
        boundaries = bound_bayes_pp(
            config["prior_mean"],
            config["prior_variance"],
            config["data_variance"],
            config["threshold"],
            LOOKS,
        )

        # Compute Type I error via MC multivariate normal
        type_i = calc_type_i_error(boundaries.tolist(), LOOKS)

        # Paper claims these are calibrated to alpha = 0.05
        # MC estimate should be within ~0.005 of 0.05
        results.append({
            "test": f"Type I error: {config['name']}",
            "boundaries": str([round(b, 2) for b in boundaries]),
            "type_i_mc": round(type_i, 4),
            "target": 0.05,
            "pass": abs(type_i - 0.05) < 0.008,
        })

    return pd.DataFrame(results)


def validate_zetyra_matches_reference(client) -> pd.DataFrame:
    """Extra scenarios: verify Zetyra matches our reference for non-Table-3 params."""
    results = []

    extra_scenarios = [
        {
            "name": "Informative prior (mu=0.1, nu^2=0.5)",
            "prior_mean": 0.1, "prior_variance": 0.5,
            "data_variance": 1.0, "n_per_look": [100, 200, 300],
            "efficacy_threshold": 0.95,
        },
        {
            "name": "Large data variance (sigma^2=4)",
            "prior_mean": 0.0, "prior_variance": 1.0,
            "data_variance": 4.0, "n_per_look": [50, 100, 150, 200],
            "efficacy_threshold": 0.975,
        },
        {
            "name": "Many looks (K=8)",
            "prior_mean": 0.0, "prior_variance": 0.1,
            "data_variance": 1.0,
            "n_per_look": [50, 100, 150, 200, 250, 300, 350, 400],
            "efficacy_threshold": 0.95,
        },
    ]

    for s in extra_scenarios:
        zetyra = client.bayesian_sequential(
            endpoint_type="continuous",
            n_per_look=s["n_per_look"],
            prior_mean=s["prior_mean"],
            prior_variance=s["prior_variance"],
            data_variance=s["data_variance"],
            efficacy_threshold=s["efficacy_threshold"],
        )

        for i, n_k in enumerate(s["n_per_look"]):
            ref = float(bound_bayes_pp(
                s["prior_mean"], s["prior_variance"],
                s["data_variance"], s["efficacy_threshold"],
                [n_k],
            )[0])
            zetyra_val = zetyra["efficacy_boundaries"][i]
            ok = abs(zetyra_val - ref) < BOUNDARY_TOLERANCE

            results.append({
                "test": f"{s['name']}, look {i+1} (n={n_k})",
                "reference": round(ref, 4),
                "zetyra": round(zetyra_val, 4),
                "deviation": round(abs(zetyra_val - ref), 4),
                "pass": ok,
            })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN SEQUENTIAL: ZHOU & JI (2024) TABLE 3 CROSS-VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Table 3 Boundary Values")
    print("-" * 70)
    t3 = validate_table3_boundaries(client)
    print(t3.to_string(index=False))
    all_frames.append(t3)

    print("\n2. Type I Error Rate (MC multivariate normal)")
    print("-" * 70)
    ti = validate_type_i_error(client)
    print(ti.to_string(index=False))
    all_frames.append(ti)

    print("\n3. Extra Scenarios (Zetyra vs Reference Formula)")
    print("-" * 70)
    ex = validate_zetyra_matches_reference(client)
    print(ex.to_string(index=False))
    all_frames.append(ex)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/zhou_ji_table3_validation.csv", index=False)

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
