#!/usr/bin/env python3
"""
Validate Bayesian Predictive Power for Survival/TTE Outcomes

Tests the log(HR) Normal-Normal conjugate framework:
1. Analytical posterior: posterior mean, variance, CI match closed-form
2. Predictive probability: MC estimate consistent with analytical bounds
3. Monotonicity: more events -> higher predictive power (for same HR)
4. Prior sensitivity: informative prior shifts posterior
5. Input guards
6. Schema contract

Math:
  Prior: log(HR) ~ N(mu_0, sigma_0^2)
  Likelihood: -log(HR_hat) ~ N(-log(HR_true), 4/d)  [Schoenfeld]
  Posterior: Normal-Normal conjugate update on -log(HR) scale
  Success: P(HR < 1 | data) = P(-log(HR) > 0 | data) >= threshold

References:
- Schoenfeld (1983) Var(log HR) = 4/d
- Spiegelhalter et al. (2004) Bayesian Approaches to Clinical Trials
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema, mc_rate_within
import pandas as pd
from scipy import stats as sp_stats


POSTERIOR_TOLERANCE = 0.001  # Tight for deterministic conjugate math
PP_TOLERANCE = 0.03  # MC tolerance for predictive probability


# ─── Reference implementation ────────────────────────────────────────

def reference_posterior(observed_hr, interim_events, prior_log_hr_mean=0.0, prior_log_hr_variance=1.0):
    """
    Compute analytical posterior for log(HR) using Normal-Normal conjugate.

    Returns (posterior_log_hr_mean, posterior_log_hr_variance, ci_lower, ci_upper).
    """
    neg_log_hr = -math.log(observed_hr)
    neg_prior_mean = -prior_log_hr_mean
    interim_var = 4.0 / interim_events

    post_prec = 1 / prior_log_hr_variance + 1 / interim_var
    post_var = 1 / post_prec
    post_mean_neg = post_var * (neg_prior_mean / prior_log_hr_variance + neg_log_hr / interim_var)

    # Convert back to log(HR) scale
    post_mean = -post_mean_neg

    # 95% CI on log(HR), then convert to HR scale
    z = sp_stats.norm.ppf(0.975)
    log_ci_lo = -(post_mean_neg + z * math.sqrt(post_var))
    log_ci_hi = -(post_mean_neg - z * math.sqrt(post_var))

    # HR-scale CI: exp(log HR CI)
    hr_ci_lo = math.exp(log_ci_lo)
    hr_ci_hi = math.exp(log_ci_hi)

    return post_mean, post_var, hr_ci_lo, hr_ci_hi


# ─── Test functions ──────────────────────────────────────────────────

def validate_analytical_posterior(client) -> pd.DataFrame:
    """Validate posterior matches closed-form Normal-Normal conjugate."""
    results = []

    scenarios = [
        {
            "name": "Standard (HR=0.7, d=150)",
            "observed_hr": 0.7, "interim_events": 150, "total_planned_events": 300,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 1.0,
        },
        {
            "name": "Weak effect (HR=0.9, d=200)",
            "observed_hr": 0.9, "interim_events": 200, "total_planned_events": 400,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 1.0,
        },
        {
            "name": "Strong prior (mu=-0.3, var=0.25)",
            "observed_hr": 0.75, "interim_events": 100, "total_planned_events": 250,
            "prior_log_hr_mean": -0.3, "prior_log_hr_variance": 0.25,
        },
        {
            "name": "Many events (d=500)",
            "observed_hr": 0.8, "interim_events": 500, "total_planned_events": 600,
            "prior_log_hr_mean": 0.0, "prior_log_hr_variance": 1.0,
        },
    ]

    for s in scenarios:
        zetyra = client.bayesian_survival(
            observed_hr=s["observed_hr"],
            interim_events=s["interim_events"],
            total_planned_events=s["total_planned_events"],
            prior_log_hr_mean=s["prior_log_hr_mean"],
            prior_log_hr_variance=s["prior_log_hr_variance"],
            success_threshold=0.95,
            n_simulations=10000,
        )

        schema_errors = assert_schema(zetyra, "bayesian_survival")

        ref_mean, ref_var, ref_ci_lo, ref_ci_hi = reference_posterior(
            s["observed_hr"], s["interim_events"],
            s["prior_log_hr_mean"], s["prior_log_hr_variance"],
        )

        mean_ok = abs(zetyra["posterior_log_hr_mean"] - ref_mean) < POSTERIOR_TOLERANCE
        var_ok = abs(zetyra["posterior_log_hr_variance"] - ref_var) < POSTERIOR_TOLERANCE
        # CI is now on HR scale (exp of log-HR CI)
        ci_lo_ok = abs(zetyra["credible_interval_lower"] - ref_ci_lo) < 0.01
        ci_hi_ok = abs(zetyra["credible_interval_upper"] - ref_ci_hi) < 0.01

        # HR posterior mean = exp(posterior_log_hr_mean)
        hr_ok = abs(zetyra["hr_posterior_mean"] - math.exp(ref_mean)) < 0.01

        results.append({
            "test": f"{s['name']}: posterior mean",
            "zetyra": round(zetyra["posterior_log_hr_mean"], 6),
            "reference": round(ref_mean, 6),
            "pass": mean_ok and len(schema_errors) == 0,
        })
        results.append({
            "test": f"{s['name']}: posterior variance",
            "zetyra": round(zetyra["posterior_log_hr_variance"], 6),
            "reference": round(ref_var, 6),
            "pass": var_ok,
        })
        results.append({
            "test": f"{s['name']}: HR CI lower",
            "zetyra": round(zetyra["credible_interval_lower"], 4),
            "reference": round(ref_ci_lo, 4),
            "pass": ci_lo_ok,
        })
        results.append({
            "test": f"{s['name']}: HR CI upper",
            "zetyra": round(zetyra["credible_interval_upper"], 4),
            "reference": round(ref_ci_hi, 4),
            "pass": ci_hi_ok,
        })
        results.append({
            "test": f"{s['name']}: HR posterior = exp(log HR)",
            "zetyra_hr": round(zetyra["hr_posterior_mean"], 4),
            "ref_hr": round(math.exp(ref_mean), 4),
            "pass": hr_ok,
        })

    return pd.DataFrame(results)


def validate_predictive_probability(client) -> pd.DataFrame:
    """Validate MC predictive probability is reasonable."""
    results = []

    # Strong signal: HR=0.6, d=200/300 -> should be high PP
    z_strong = client.bayesian_survival(
        observed_hr=0.6, interim_events=200, total_planned_events=300,
        prior_log_hr_mean=0.0, prior_log_hr_variance=1.0,
        success_threshold=0.95, n_simulations=50000,
    )
    results.append({
        "test": "Strong signal (HR=0.6, d=200): PP > 0.90",
        "pp": z_strong["predictive_probability"],
        "recommendation": z_strong["recommendation"],
        "pass": z_strong["predictive_probability"] > 0.90,
    })

    # Weak signal: HR=0.95, d=50/300 -> should be low PP
    z_weak = client.bayesian_survival(
        observed_hr=0.95, interim_events=50, total_planned_events=300,
        prior_log_hr_mean=0.0, prior_log_hr_variance=1.0,
        success_threshold=0.95, n_simulations=50000,
    )
    results.append({
        "test": "Weak signal (HR=0.95, d=50): PP < 0.30",
        "pp": z_weak["predictive_probability"],
        "recommendation": z_weak["recommendation"],
        "pass": z_weak["predictive_probability"] < 0.30,
    })

    # Recommendation consistency
    results.append({
        "test": "Strong signal -> stop_for_efficacy",
        "recommendation": z_strong["recommendation"],
        "pass": z_strong["recommendation"] == "stop_for_efficacy",
    })
    results.append({
        "test": "Weak signal -> not stop_for_efficacy",
        "recommendation": z_weak["recommendation"],
        "pass": z_weak["recommendation"] != "stop_for_efficacy",
    })

    return pd.DataFrame(results)


def validate_monotonicity(client) -> pd.DataFrame:
    """More events with same HR -> higher predictive power."""
    results = []

    event_counts = [50, 100, 200, 300]
    pps = []
    for d in event_counts:
        z = client.bayesian_survival(
            observed_hr=0.7, interim_events=d, total_planned_events=400,
            prior_log_hr_mean=0.0, prior_log_hr_variance=1.0,
            success_threshold=0.95, n_simulations=20000,
        )
        pps.append(z["predictive_probability"])

    mono = all(pps[i] <= pps[i + 1] for i in range(len(pps) - 1))
    results.append({
        "test": "PP increases with more events (same HR)",
        "events": str(event_counts),
        "pps": str([round(p, 4) for p in pps]),
        "pass": mono,
    })

    # Stronger HR -> higher PP (same events)
    hrs = [0.9, 0.8, 0.7, 0.6]
    pps_hr = []
    for hr in hrs:
        z = client.bayesian_survival(
            observed_hr=hr, interim_events=150, total_planned_events=300,
            prior_log_hr_mean=0.0, prior_log_hr_variance=1.0,
            success_threshold=0.95, n_simulations=20000,
        )
        pps_hr.append(z["predictive_probability"])

    mono_hr = all(pps_hr[i] <= pps_hr[i + 1] for i in range(len(pps_hr) - 1))
    results.append({
        "test": "PP increases with stronger HR (same events)",
        "hrs": str(hrs),
        "pps": str([round(p, 4) for p in pps_hr]),
        "pass": mono_hr,
    })

    return pd.DataFrame(results)


def validate_prior_sensitivity(client) -> pd.DataFrame:
    """Informative prior shifts posterior toward prior."""
    results = []

    # Vague prior: posterior dominated by data
    z_vague = client.bayesian_survival(
        observed_hr=0.7, interim_events=100, total_planned_events=300,
        prior_log_hr_mean=0.0, prior_log_hr_variance=100.0,
        success_threshold=0.95, n_simulations=10000,
    )

    # Skeptical prior centered at HR=1 (log(1)=0) with tight variance
    z_skeptical = client.bayesian_survival(
        observed_hr=0.7, interim_events=100, total_planned_events=300,
        prior_log_hr_mean=0.0, prior_log_hr_variance=0.1,
        success_threshold=0.95, n_simulations=10000,
    )

    # Optimistic prior centered at HR=0.6 (log(0.6)=-0.51) with tight variance
    z_optimistic = client.bayesian_survival(
        observed_hr=0.7, interim_events=100, total_planned_events=300,
        prior_log_hr_mean=-0.51, prior_log_hr_variance=0.1,
        success_threshold=0.95, n_simulations=10000,
    )

    # Vague prior: posterior mean ≈ data log(HR)
    data_log_hr = math.log(0.7)
    results.append({
        "test": "Vague prior: posterior ≈ data",
        "posterior": round(z_vague["posterior_log_hr_mean"], 4),
        "data_log_hr": round(data_log_hr, 4),
        "pass": abs(z_vague["posterior_log_hr_mean"] - data_log_hr) < 0.02,
    })

    # Skeptical prior pulls posterior toward 0 (HR=1)
    results.append({
        "test": "Skeptical prior: posterior > data log(HR)",
        "posterior_skeptical": round(z_skeptical["posterior_log_hr_mean"], 4),
        "data_log_hr": round(data_log_hr, 4),
        "pass": z_skeptical["posterior_log_hr_mean"] > data_log_hr,
    })

    # Optimistic prior pushes posterior further from 0
    results.append({
        "test": "Optimistic prior: posterior < skeptical posterior",
        "posterior_optimistic": round(z_optimistic["posterior_log_hr_mean"], 4),
        "posterior_skeptical": round(z_skeptical["posterior_log_hr_mean"], 4),
        "pass": z_optimistic["posterior_log_hr_mean"] < z_skeptical["posterior_log_hr_mean"],
    })

    # Skeptical prior -> lower PP than vague
    results.append({
        "test": "Skeptical prior -> lower PP than vague",
        "pp_skeptical": round(z_skeptical["predictive_probability"], 4),
        "pp_vague": round(z_vague["predictive_probability"], 4),
        "pass": z_skeptical["predictive_probability"] < z_vague["predictive_probability"],
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return 400/422."""
    results = []

    # HR=0 should fail (must be positive)
    resp = client.bayesian_survival_raw(
        observed_hr=0.0, interim_events=150, total_planned_events=300,
    )
    results.append({
        "test": "Guard: HR=0",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # HR >= 1 is now allowed (treatment may appear harmful)
    resp = client.bayesian_survival_raw(
        observed_hr=1.2, interim_events=150, total_planned_events=300,
    )
    results.append({
        "test": "Accept: HR=1.2 (harmful treatment)",
        "status_code": resp.status_code,
        "pass": resp.status_code == 200,
    })

    # More interim events than planned
    resp = client.bayesian_survival_raw(
        observed_hr=0.7, interim_events=400, total_planned_events=300,
    )
    results.append({
        "test": "Guard: interim > planned events",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Zero events
    resp = client.bayesian_survival_raw(
        observed_hr=0.7, interim_events=0, total_planned_events=300,
    )
    results.append({
        "test": "Guard: zero interim events",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    return pd.DataFrame(results)


def validate_ci_ordering(client) -> pd.DataFrame:
    """HR CI lower < HR posterior mean < HR CI upper (all on HR scale)."""
    results = []

    scenarios = [
        {"observed_hr": 0.7, "interim_events": 150, "total_planned_events": 300},
        {"observed_hr": 0.8, "interim_events": 80, "total_planned_events": 200},
        {"observed_hr": 0.6, "interim_events": 250, "total_planned_events": 400},
    ]

    for s in scenarios:
        z = client.bayesian_survival(**s, success_threshold=0.95, n_simulations=10000)
        lo = z["credible_interval_lower"]
        hr_mean = z["hr_posterior_mean"]
        hi = z["credible_interval_upper"]

        results.append({
            "test": f"HR={s['observed_hr']}, d={s['interim_events']}: HR CI ordering",
            "ci_lo": round(lo, 4),
            "hr_mean": round(hr_mean, 4),
            "ci_hi": round(hi, 4),
            "pass": lo < hr_mean < hi,
        })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN PREDICTIVE POWER (SURVIVAL) VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Analytical Posterior")
    print("-" * 70)
    ap_results = validate_analytical_posterior(client)
    print(ap_results.to_string(index=False))
    all_frames.append(ap_results)

    print("\n2. Predictive Probability")
    print("-" * 70)
    pp_results = validate_predictive_probability(client)
    print(pp_results.to_string(index=False))
    all_frames.append(pp_results)

    print("\n3. Monotonicity")
    print("-" * 70)
    m_results = validate_monotonicity(client)
    print(m_results.to_string(index=False))
    all_frames.append(m_results)

    print("\n4. Prior Sensitivity")
    print("-" * 70)
    ps_results = validate_prior_sensitivity(client)
    print(ps_results.to_string(index=False))
    all_frames.append(ps_results)

    print("\n5. Input Guards")
    print("-" * 70)
    ig_results = validate_input_guards(client)
    print(ig_results.to_string(index=False))
    all_frames.append(ig_results)

    print("\n6. CI Ordering")
    print("-" * 70)
    ci_results = validate_ci_ordering(client)
    print(ci_results.to_string(index=False))
    all_frames.append(ci_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_survival_validation.csv", index=False)

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
