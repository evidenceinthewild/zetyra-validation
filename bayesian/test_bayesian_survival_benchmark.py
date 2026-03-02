#!/usr/bin/env python3
"""
Bayesian Survival PP: Published Benchmark & Textbook Validation

Extends the analytical tests in test_bayesian_survival.py with:
1. Textbook worked examples (Spiegelhalter et al., 2004 framework)
2. Cross-validation with manual predictive probability calculation
3. Frequentist convergence: vague prior + many events → PP ≈ conditional power
4. Known-outcome scenarios: extreme HR with many events → PP ≈ 0 or ≈ 1

The core math (Normal-Normal conjugate on log HR):
  Prior: log(HR) ~ N(μ₀, σ₀²)
  Data: log(HR_hat) ~ N(log(HR_true), 4/d)  [Schoenfeld]
  Posterior: Normal-Normal conjugate update
  PP = P(final z > z_α | interim data, posterior)

References:
- Spiegelhalter, Abrams, Myles (2004) "Bayesian Approaches to Clinical Trials"
- Schoenfeld (1983) "Sample-Size Formula for the Proportional-Hazards Regression Model"
- Dmitrienko, Wang (2006) "Bayesian predictive approach to interim monitoring"
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


POSTERIOR_TOL = 0.001
PP_TOL = 0.03  # MC tolerance


# ─── Reference implementations ────────────────────────────────────────

def reference_posterior(observed_hr, interim_events, prior_mean=0.0, prior_var=1.0):
    """Normal-Normal conjugate posterior on -log(HR) scale."""
    neg_log_hr = -math.log(observed_hr)
    neg_prior_mean = -prior_mean
    data_var = 4.0 / interim_events

    post_prec = 1 / prior_var + 1 / data_var
    post_var = 1 / post_prec
    post_mean_neg = post_var * (neg_prior_mean / prior_var + neg_log_hr / data_var)
    post_mean = -post_mean_neg  # Back to log(HR) scale

    return post_mean, post_var


def reference_predictive_probability(
    observed_hr, interim_events, total_events,
    prior_mean=0.0, prior_var=1.0, success_threshold=0.95, n_mc=200000, seed=42
):
    """
    Monte Carlo predictive probability of trial success.

    1. Draw θ from posterior
    2. Simulate remaining data: x_future ~ N(θ, 4/d_remaining) on -log(HR) scale
    3. Combine interim + future for final z-statistic
    4. Success if P(θ > 0 | all data) ≥ success_threshold
       equivalently: final z > Φ⁻¹(threshold) adjusted for posterior
    """
    rng = np.random.default_rng(seed)

    neg_log_hr = -math.log(observed_hr)
    neg_prior_mean = -prior_mean
    data_var_interim = 4.0 / interim_events

    # Posterior on -log(HR) scale
    post_prec = 1 / prior_var + 1 / data_var_interim
    post_var = 1 / post_prec
    post_mean_neg = post_var * (neg_prior_mean / prior_var + neg_log_hr / data_var_interim)

    # Remaining events
    d_remaining = total_events - interim_events
    data_var_remaining = 4.0 / d_remaining

    successes = 0
    for _ in range(n_mc):
        # Draw true -log(HR) from posterior
        theta = rng.normal(post_mean_neg, math.sqrt(post_var))

        # Simulate future data
        x_future = rng.normal(theta, math.sqrt(data_var_remaining))

        # Combined estimate: precision-weighted
        combined_prec = 1 / data_var_interim + 1 / data_var_remaining
        combined_mean = (neg_log_hr / data_var_interim + x_future / data_var_remaining) / combined_prec
        combined_var = 1 / combined_prec

        # Final posterior: combine combined data with prior
        final_prec = 1 / prior_var + combined_prec
        final_var = 1 / final_prec
        final_mean = final_var * (neg_prior_mean / prior_var + combined_mean * combined_prec)

        # Success: P(θ > 0 | all data) = Φ(final_mean / √final_var)
        prob_positive = sp_stats.norm.cdf(final_mean / math.sqrt(final_var))
        if prob_positive >= success_threshold:
            successes += 1

    return successes / n_mc


# ─── Test functions ───────────────────────────────────────────────────

def validate_textbook_posteriors(client) -> pd.DataFrame:
    """
    Textbook-style worked examples for Bayesian survival posterior.

    Based on the Spiegelhalter et al. (2004) framework:
    'For a trial with observed HR = h based on d events, with
    prior log(HR) ~ N(μ₀, σ₀²), the posterior mean and variance
    are obtained by standard conjugate update.'
    """
    results = []

    examples = [
        {
            "name": "Textbook 1: Standard trial (HR=0.70, d=200, vague prior)",
            "observed_hr": 0.70, "interim_events": 200, "total_planned_events": 400,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 1.0,
        },
        {
            "name": "Textbook 2: Near-complete (HR=0.75, d=280/300, vague prior)",
            "observed_hr": 0.75, "interim_events": 280, "total_planned_events": 300,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 1.0,
        },
        {
            "name": "Textbook 3: Strong prior toward no effect (HR=0.65, d=100)",
            "observed_hr": 0.65, "interim_events": 100, "total_planned_events": 250,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 0.1,
        },
        {
            "name": "Textbook 4: Optimistic prior (μ=-0.36, HR=0.80, d=150)",
            "observed_hr": 0.80, "interim_events": 150, "total_planned_events": 300,
            "prior_log_hr_mean": -0.36, "prior_log_hr_variance": 0.25,
        },
        {
            "name": "Textbook 5: Harmful treatment (HR=1.10, d=200, vague prior)",
            "observed_hr": 1.10, "interim_events": 200, "total_planned_events": 400,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 1.0,
        },
    ]

    for ex in examples:
        zetyra = client.bayesian_survival(
            observed_hr=ex["observed_hr"],
            interim_events=ex["interim_events"],
            total_planned_events=ex["total_planned_events"],
            prior_log_hr_mean=ex["prior_log_hr_mean"],
            prior_log_hr_variance=ex["prior_log_hr_variance"],
            success_threshold=0.95,
            n_simulations=10000,
        )

        schema_errors = assert_schema(zetyra, "bayesian_survival")

        ref_mean, ref_var = reference_posterior(
            ex["observed_hr"], ex["interim_events"],
            ex["prior_log_hr_mean"], ex["prior_log_hr_variance"],
        )

        mean_ok = abs(zetyra["posterior_log_hr_mean"] - ref_mean) < POSTERIOR_TOL
        var_ok = abs(zetyra["posterior_log_hr_variance"] - ref_var) < POSTERIOR_TOL
        hr_ok = abs(zetyra["hr_posterior_mean"] - math.exp(ref_mean)) < 0.01

        results.append({
            "test": f"{ex['name']}: posterior mean",
            "zetyra": round(zetyra["posterior_log_hr_mean"], 6),
            "reference": round(ref_mean, 6),
            "pass": mean_ok and len(schema_errors) == 0,
        })
        results.append({
            "test": f"{ex['name']}: posterior variance",
            "zetyra": round(zetyra["posterior_log_hr_variance"], 6),
            "reference": round(ref_var, 6),
            "pass": var_ok,
        })
        results.append({
            "test": f"{ex['name']}: HR posterior",
            "zetyra": round(zetyra["hr_posterior_mean"], 4),
            "reference": round(math.exp(ref_mean), 4),
            "pass": hr_ok,
        })

    return pd.DataFrame(results)


def validate_predictive_probability_reference(client) -> pd.DataFrame:
    """
    Cross-validate Zetyra PP against independent MC reference implementation.

    This is the strongest validation: our reference_predictive_probability()
    implements the same math independently from the Zetyra backend.
    """
    results = []

    scenarios = [
        {
            "name": "Strong effect, halfway (HR=0.65, d=150/300)",
            "observed_hr": 0.65, "interim_events": 150, "total_planned_events": 300,
        },
        {
            "name": "Moderate effect, early (HR=0.75, d=100/350)",
            "observed_hr": 0.75, "interim_events": 100, "total_planned_events": 350,
        },
        {
            "name": "Weak effect, late (HR=0.90, d=250/300)",
            "observed_hr": 0.90, "interim_events": 250, "total_planned_events": 300,
        },
    ]

    for s in scenarios:
        zetyra = client.bayesian_survival(
            observed_hr=s["observed_hr"],
            interim_events=s["interim_events"],
            total_planned_events=s["total_planned_events"],
            prior_log_hr_mean=0.0, prior_log_hr_variance=1.0,
            success_threshold=0.95, n_simulations=50000,
        )

        ref_pp = reference_predictive_probability(
            s["observed_hr"], s["interim_events"], s["total_planned_events"],
            prior_mean=0.0, prior_var=1.0, success_threshold=0.95,
            n_mc=200000, seed=99,
        )

        dev = abs(zetyra["predictive_probability"] - ref_pp)

        results.append({
            "test": f"PP: {s['name']}",
            "zetyra_pp": round(zetyra["predictive_probability"], 4),
            "reference_pp": round(ref_pp, 4),
            "deviation": round(dev, 4),
            "pass": dev < PP_TOL,
        })

    return pd.DataFrame(results)


def validate_frequentist_convergence(client) -> pd.DataFrame:
    """
    With vague prior, Bayesian PP converges to frequentist conditional power.

    CP = Φ(z_interim × √R - z_α × √(R - 1))
    where R = d_total / d_interim, z_interim = -log(HR_hat) / √(4/d)
    """
    results = []

    scenarios = [
        {"observed_hr": 0.70, "interim_events": 200, "total_planned_events": 400},
        {"observed_hr": 0.80, "interim_events": 150, "total_planned_events": 300},
        {"observed_hr": 0.60, "interim_events": 100, "total_planned_events": 200},
    ]

    for s in scenarios:
        # Zetyra with very vague prior
        zetyra = client.bayesian_survival(
            observed_hr=s["observed_hr"],
            interim_events=s["interim_events"],
            total_planned_events=s["total_planned_events"],
            prior_log_hr_mean=0.0, prior_log_hr_variance=100.0,
            success_threshold=0.975,  # One-sided α=0.025
            n_simulations=50000,
        )

        # Frequentist conditional power
        z_interim = -math.log(s["observed_hr"]) / math.sqrt(4.0 / s["interim_events"])
        R = s["total_planned_events"] / s["interim_events"]
        z_alpha = sp_stats.norm.ppf(0.975)
        cp = sp_stats.norm.cdf(z_interim * math.sqrt(R) - z_alpha * math.sqrt(R - 1))

        # With very vague prior, PP ≈ CP (not exact because Bayesian uses
        # posterior predictive, but should be close)
        dev = abs(zetyra["predictive_probability"] - cp)

        results.append({
            "test": f"Freq. convergence: HR={s['observed_hr']}, d={s['interim_events']}/{s['total_planned_events']}",
            "bayesian_pp": round(zetyra["predictive_probability"], 4),
            "freq_cp": round(cp, 4),
            "deviation": round(dev, 4),
            "pass": dev < 0.10,  # Allow reasonable deviation
        })

    return pd.DataFrame(results)


def validate_known_outcomes(client) -> pd.DataFrame:
    """
    Edge cases where the answer is known a priori.

    1. Overwhelming efficacy: HR=0.3, d=400/500 → PP ≈ 1.0
    2. Clearly harmful: HR=1.5, d=200/300 → PP ≈ 0.0
    3. Nearly complete + strong effect: HR=0.5, d=290/300 → PP close to 1.0
    """
    results = []

    # Case 1: Overwhelming efficacy
    z1 = client.bayesian_survival(
        observed_hr=0.3, interim_events=400, total_planned_events=500,
        success_threshold=0.95, n_simulations=50000,
    )
    results.append({
        "test": "Overwhelming efficacy: HR=0.3, d=400/500 → PP ≈ 1.0",
        "pp": z1["predictive_probability"],
        "recommendation": z1["recommendation"],
        "pass": z1["predictive_probability"] > 0.99 and z1["recommendation"] == "stop_for_efficacy",
    })

    # Case 2: Clearly harmful
    z2 = client.bayesian_survival(
        observed_hr=1.5, interim_events=200, total_planned_events=300,
        success_threshold=0.95, n_simulations=50000,
    )
    results.append({
        "test": "Clearly harmful: HR=1.5, d=200/300 → PP ≈ 0.0",
        "pp": z2["predictive_probability"],
        "recommendation": z2["recommendation"],
        "pass": z2["predictive_probability"] < 0.01,
    })

    # Case 3: Nearly complete with strong effect
    z3 = client.bayesian_survival(
        observed_hr=0.5, interim_events=290, total_planned_events=300,
        success_threshold=0.95, n_simulations=50000,
    )
    results.append({
        "test": "Near-complete + strong: HR=0.5, d=290/300 → PP > 0.95",
        "pp": z3["predictive_probability"],
        "pass": z3["predictive_probability"] > 0.95,
    })

    # Case 4: Just started, weak signal
    z4 = client.bayesian_survival(
        observed_hr=0.90, interim_events=20, total_planned_events=400,
        success_threshold=0.95, n_simulations=50000,
    )
    results.append({
        "test": "Early + weak: HR=0.90, d=20/400 → PP < 0.50",
        "pp": z4["predictive_probability"],
        "pass": z4["predictive_probability"] < 0.50,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN SURVIVAL PP: PUBLISHED BENCHMARK VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Textbook Posterior Verification (5 examples)")
    print("-" * 70)
    tp = validate_textbook_posteriors(client)
    print(tp.to_string(index=False))
    all_frames.append(tp)

    print("\n2. Predictive Probability Cross-Validation (MC reference)")
    print("-" * 70)
    pp = validate_predictive_probability_reference(client)
    print(pp.to_string(index=False))
    all_frames.append(pp)

    print("\n3. Frequentist Convergence (vague prior → CP)")
    print("-" * 70)
    fc = validate_frequentist_convergence(client)
    print(fc.to_string(index=False))
    all_frames.append(fc)

    print("\n4. Known-Outcome Scenarios")
    print("-" * 70)
    ko = validate_known_outcomes(client)
    print(ko.to_string(index=False))
    all_frames.append(ko)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_survival_benchmark.csv", index=False)

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
