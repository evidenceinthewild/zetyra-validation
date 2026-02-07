#!/usr/bin/env python3
"""
Validate Bayesian Prior Elicitation Calculator

Tests three elicitation methods:
1. ESS-based: α = mean × ESS, β = (1-mean) × ESS
2. Historical: α = 1 + δ×events, β = 1 + δ×(n-events)
3. Quantile matching: numerical optimization (deterministic)
4. Input guards (422/400 for invalid inputs)
5. Boundary-condition scenarios
6. Schema contracts
7. Oracle quantile-matching comparison via scipy.optimize

References:
- Berry et al. (2010) "Bayesian Adaptive Methods for Clinical Trials"
- Morita, Thall & Muller (2008) "Determining the Effective Sample Size of a Parametric Prior"
- REBYOTA PUNCH CD2 (Phase 2b): 25/45 responders, two-dose arm (FDA BLA 125739)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PARAM_TOLERANCE = 0.01  # Exact for analytical
QUANTILE_TOLERANCE = 0.02  # Looser for numerical optimization


# ─── Reference implementations ────────────────────────────────────────

def reference_ess_prior(mean, ess):
    """ESS-based: α = mean × ESS, β = (1-mean) × ESS."""
    alpha = mean * ess
    beta = (1 - mean) * ess
    return {"alpha": alpha, "beta": beta}


def reference_historical_prior(n_events, n_total, discount):
    """Power prior: α = 1 + δ×events, β = 1 + δ×(n-events)."""
    alpha = 1 + discount * n_events
    beta = 1 + discount * (n_total - n_events)
    return {"alpha": alpha, "beta": beta}


def reference_beta_summary(alpha, beta):
    """Beta distribution summary statistics."""
    mean = alpha / (alpha + beta)
    ess = alpha + beta
    variance = (alpha * beta) / (ess ** 2 * (ess + 1))
    mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else None
    return {"mean": mean, "variance": variance, "mode": mode, "ess": ess}


def reference_quantile_matching(quantiles, values):
    """Fit Beta(α,β) to minimize squared quantile deviations via scipy."""
    def loss(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e12
        fitted = [stats.beta.ppf(q, a, b) for q in quantiles]
        return sum((f - v) ** 2 for f, v in zip(fitted, values))

    result = minimize(loss, x0=[2.0, 2.0], method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-12})
    return {"alpha": result.x[0], "beta": result.x[1]}


# ─── Test functions ───────────────────────────────────────────────────

def validate_ess_based(client) -> pd.DataFrame:
    """Validate ESS-based prior elicitation."""
    scenarios = [
        {"name": "Weakly informative (ESS=2)", "mean": 0.30, "ess": 2},
        {"name": "Berry et al. (2010)", "mean": 0.25, "ess": 10},
        {"name": "Vague prior", "mean": 0.50, "ess": 2},
        {"name": "Moderate informative", "mean": 0.15, "ess": 20},
    ]

    results = []
    for s in scenarios:
        zetyra = client.prior_elicitation(method="ess_based", mean=s["mean"], ess=s["ess"])
        ref = reference_ess_prior(s["mean"], s["ess"])
        ref_summary = reference_beta_summary(ref["alpha"], ref["beta"])

        schema_errors = assert_schema(zetyra, "prior_elicitation")

        alpha_ok = abs(zetyra["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
        beta_ok = abs(zetyra["beta"] - ref["beta"]) < PARAM_TOLERANCE
        mean_ok = abs(zetyra["mean"] - ref_summary["mean"]) < PARAM_TOLERANCE
        ess_ok = abs(zetyra["ess"] - ref_summary["ess"]) < PARAM_TOLERANCE

        passed = alpha_ok and beta_ok and mean_ok and ess_ok and len(schema_errors) == 0
        results.append({
            "test": f"ESS: {s['name']}",
            "zetyra_alpha": zetyra["alpha"],
            "ref_alpha": ref["alpha"],
            "zetyra_beta": zetyra["beta"],
            "ref_beta": ref["beta"],
            "zetyra_ess": zetyra["ess"],
            "ref_ess": ref_summary["ess"],
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_historical(client) -> pd.DataFrame:
    """Validate historical (power prior) elicitation using REBYOTA data."""
    scenarios = [
        {"name": "REBYOTA CD2 δ=0.5", "n_events": 25, "n_total": 45, "discount": 0.5},
        {"name": "REBYOTA CD2 δ=1.0", "n_events": 25, "n_total": 45, "discount": 1.0},
        {"name": "REBYOTA CD2 δ=0.1", "n_events": 25, "n_total": 45, "discount": 0.1},
        {"name": "REBYOTA CD2 δ=0.0", "n_events": 25, "n_total": 45, "discount": 0.0},
    ]

    results = []
    for s in scenarios:
        zetyra = client.prior_elicitation(
            method="historical",
            n_events=s["n_events"],
            n_total=s["n_total"],
            discount_factor=s["discount"],
        )
        ref = reference_historical_prior(s["n_events"], s["n_total"], s["discount"])
        ref_summary = reference_beta_summary(ref["alpha"], ref["beta"])

        schema_errors = assert_schema(zetyra, "prior_elicitation")

        alpha_ok = abs(zetyra["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
        beta_ok = abs(zetyra["beta"] - ref["beta"]) < PARAM_TOLERANCE
        mean_ok = abs(zetyra["mean"] - ref_summary["mean"]) < PARAM_TOLERANCE

        passed = alpha_ok and beta_ok and mean_ok and len(schema_errors) == 0
        results.append({
            "test": f"Historical: {s['name']}",
            "zetyra_alpha": zetyra["alpha"],
            "ref_alpha": ref["alpha"],
            "zetyra_beta": zetyra["beta"],
            "ref_beta": ref["beta"],
            "zetyra_mean": round(zetyra["mean"], 4),
            "ref_mean": round(ref_summary["mean"], 4),
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_quantile_matching(client) -> pd.DataFrame:
    """Validate quantile matching (numerical optimization)."""
    scenarios = [
        {
            "name": "Berry: median≈0.25, 90%CI=[0.10,0.40]",
            "quantiles": [0.05, 0.50, 0.95],
            "values": [0.10, 0.25, 0.40],
        },
        {
            "name": "Tight prior: median≈0.50, 90%CI=[0.40,0.60]",
            "quantiles": [0.05, 0.50, 0.95],
            "values": [0.40, 0.50, 0.60],
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.prior_elicitation(
            method="quantile_matching",
            quantiles=s["quantiles"],
            quantile_values=s["values"],
        )

        schema_errors = assert_schema(zetyra, "prior_elicitation")

        # Check that fitted quantiles are close to targets
        all_ok = len(schema_errors) == 0
        returned_quantiles = zetyra.get("quantiles", {}) or {}
        for q, target_v in zip(s["quantiles"], s["values"]):
            q_key = f"q{int(q * 100):02d}"
            if q_key in returned_quantiles:
                fitted_v = returned_quantiles[q_key]
            else:
                # Compute from returned alpha, beta
                fitted_v = stats.beta.ppf(q, zetyra["alpha"], zetyra["beta"])

            if abs(fitted_v - target_v) > QUANTILE_TOLERANCE:
                all_ok = False

        results.append({
            "test": f"Quantile: {s['name']}",
            "zetyra_alpha": round(zetyra["alpha"], 3),
            "zetyra_beta": round(zetyra["beta"], 3),
            "zetyra_mean": round(zetyra["mean"], 4),
            "pass": all_ok,
        })

    return pd.DataFrame(results)


def validate_quantile_oracle(client) -> pd.DataFrame:
    """Compare API quantile matching against scipy.optimize reference."""
    scenarios = [
        {
            "name": "Berry median≈0.25",
            "quantiles": [0.05, 0.50, 0.95],
            "values": [0.10, 0.25, 0.40],
        },
        {
            "name": "Tight median≈0.50",
            "quantiles": [0.05, 0.50, 0.95],
            "values": [0.40, 0.50, 0.60],
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.prior_elicitation(
            method="quantile_matching",
            quantiles=s["quantiles"],
            quantile_values=s["values"],
        )
        ref = reference_quantile_matching(s["quantiles"], s["values"])

        # Schema check
        schema_errors = assert_schema(zetyra, "prior_elicitation")

        # Both should produce similar α,β (within 0.5 since optimization landscape is flat)
        alpha_close = abs(zetyra["alpha"] - ref["alpha"]) < 0.5
        beta_close = abs(zetyra["beta"] - ref["beta"]) < 0.5

        # More importantly: fitted quantiles from both should be close to targets
        z_q = [stats.beta.ppf(q, zetyra["alpha"], zetyra["beta"]) for q in s["quantiles"]]
        r_q = [stats.beta.ppf(q, ref["alpha"], ref["beta"]) for q in s["quantiles"]]
        q_close = all(abs(z - r) < QUANTILE_TOLERANCE for z, r in zip(z_q, r_q))

        results.append({
            "test": f"Oracle: {s['name']}",
            "zetyra_alpha": round(zetyra["alpha"], 3),
            "ref_alpha": round(ref["alpha"], 3),
            "zetyra_beta": round(zetyra["beta"], 3),
            "ref_beta": round(ref["beta"], 3),
            "pass": ((alpha_close and beta_close) or q_close) and len(schema_errors) == 0,
        })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate that invalid inputs return 400/422."""
    results = []

    guards = [
        {
            "name": "invalid method",
            "field": "method",
            "data": {"method": "invalid_method", "mean": 0.3, "ess": 10},
        },
        {
            "name": "ess = 0 (must be > 0)",
            "field": "ess",
            "data": {"method": "ess_based", "mean": 0.3, "ess": 0},
        },
        {
            "name": "mean > 1",
            "field": "mean",
            "data": {"method": "ess_based", "mean": 1.5, "ess": 10},
        },
        {
            "name": "discount_factor > 1",
            "field": "discount_factor",
            "data": {"method": "historical", "n_events": 10, "n_total": 20, "discount_factor": 1.5},
        },
    ]

    for g in guards:
        resp = client._post_raw("/bayesian/prior-elicitation", g["data"])
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

    # ESS = 1 (minimum)
    z = client.prior_elicitation(method="ess_based", mean=0.50, ess=1)
    ref = reference_ess_prior(0.50, 1)
    ref_s = reference_beta_summary(ref["alpha"], ref["beta"])
    schema_errors = assert_schema(z, "prior_elicitation")
    results.append({
        "test": "Boundary: ESS=1 minimum",
        "zetyra_alpha": z["alpha"], "ref_alpha": ref["alpha"],
        "zetyra_beta": z["beta"], "ref_beta": ref["beta"],
        "pass": (abs(z["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
                 and abs(z["beta"] - ref["beta"]) < PARAM_TOLERANCE
                 and abs(z["mean"] - ref_s["mean"]) < PARAM_TOLERANCE
                 and abs(z["ess"] - ref_s["ess"]) < PARAM_TOLERANCE
                 and len(schema_errors) == 0),
    })

    # ESS = 1000 (very strong)
    z = client.prior_elicitation(method="ess_based", mean=0.30, ess=1000)
    ref = reference_ess_prior(0.30, 1000)
    ref_s = reference_beta_summary(ref["alpha"], ref["beta"])
    schema_errors = assert_schema(z, "prior_elicitation")
    results.append({
        "test": "Boundary: ESS=1000 very strong",
        "zetyra_alpha": z["alpha"], "ref_alpha": ref["alpha"],
        "zetyra_beta": z["beta"], "ref_beta": ref["beta"],
        "pass": (abs(z["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
                 and abs(z["beta"] - ref["beta"]) < PARAM_TOLERANCE
                 and abs(z["mean"] - ref_s["mean"]) < PARAM_TOLERANCE
                 and abs(z["ess"] - ref_s["ess"]) < PARAM_TOLERANCE
                 and len(schema_errors) == 0),
    })

    # Near-zero mean
    z = client.prior_elicitation(method="ess_based", mean=0.001, ess=10)
    ref = reference_ess_prior(0.001, 10)
    ref_s = reference_beta_summary(ref["alpha"], ref["beta"])
    schema_errors = assert_schema(z, "prior_elicitation")
    results.append({
        "test": "Boundary: mean≈0",
        "zetyra_alpha": z["alpha"], "ref_alpha": ref["alpha"],
        "zetyra_beta": z["beta"], "ref_beta": ref["beta"],
        "pass": (abs(z["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
                 and abs(z["beta"] - ref["beta"]) < PARAM_TOLERANCE
                 and abs(z["mean"] - ref_s["mean"]) < PARAM_TOLERANCE
                 and abs(z["ess"] - ref_s["ess"]) < PARAM_TOLERANCE
                 and len(schema_errors) == 0),
    })

    # Near-one mean
    z = client.prior_elicitation(method="ess_based", mean=0.999, ess=10)
    ref = reference_ess_prior(0.999, 10)
    ref_s = reference_beta_summary(ref["alpha"], ref["beta"])
    schema_errors = assert_schema(z, "prior_elicitation")
    results.append({
        "test": "Boundary: mean≈1",
        "zetyra_alpha": z["alpha"], "ref_alpha": ref["alpha"],
        "zetyra_beta": z["beta"], "ref_beta": ref["beta"],
        "pass": (abs(z["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
                 and abs(z["beta"] - ref["beta"]) < PARAM_TOLERANCE
                 and abs(z["mean"] - ref_s["mean"]) < PARAM_TOLERANCE
                 and abs(z["ess"] - ref_s["ess"]) < PARAM_TOLERANCE
                 and len(schema_errors) == 0),
    })

    # Historical: zero events
    z = client.prior_elicitation(method="historical", n_events=0, n_total=50, discount_factor=0.5)
    ref = reference_historical_prior(0, 50, 0.5)
    ref_s = reference_beta_summary(ref["alpha"], ref["beta"])
    schema_errors = assert_schema(z, "prior_elicitation")
    results.append({
        "test": "Boundary: zero events",
        "zetyra_alpha": z["alpha"], "ref_alpha": ref["alpha"],
        "zetyra_beta": z["beta"], "ref_beta": ref["beta"],
        "pass": (abs(z["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
                 and abs(z["beta"] - ref["beta"]) < PARAM_TOLERANCE
                 and abs(z["mean"] - ref_s["mean"]) < PARAM_TOLERANCE
                 and abs(z["ess"] - ref_s["ess"]) < PARAM_TOLERANCE
                 and len(schema_errors) == 0),
    })

    # Historical: all events
    z = client.prior_elicitation(method="historical", n_events=50, n_total=50, discount_factor=0.5)
    ref = reference_historical_prior(50, 50, 0.5)
    ref_s = reference_beta_summary(ref["alpha"], ref["beta"])
    schema_errors = assert_schema(z, "prior_elicitation")
    results.append({
        "test": "Boundary: all events",
        "zetyra_alpha": z["alpha"], "ref_alpha": ref["alpha"],
        "zetyra_beta": z["beta"], "ref_beta": ref["beta"],
        "pass": (abs(z["alpha"] - ref["alpha"]) < PARAM_TOLERANCE
                 and abs(z["beta"] - ref["beta"]) < PARAM_TOLERANCE
                 and abs(z["mean"] - ref_s["mean"]) < PARAM_TOLERANCE
                 and abs(z["ess"] - ref_s["ess"]) < PARAM_TOLERANCE
                 and len(schema_errors) == 0),
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("PRIOR ELICITATION VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. ESS-Based Priors")
    print("-" * 70)
    ess_results = validate_ess_based(client)
    print(ess_results.to_string(index=False))
    all_frames.append(ess_results)

    print("\n2. Historical Priors (REBYOTA)")
    print("-" * 70)
    hist_results = validate_historical(client)
    print(hist_results.to_string(index=False))
    all_frames.append(hist_results)

    print("\n3. Quantile Matching")
    print("-" * 70)
    qm_results = validate_quantile_matching(client)
    print(qm_results.to_string(index=False))
    all_frames.append(qm_results)

    print("\n4. Quantile Oracle (scipy.optimize)")
    print("-" * 70)
    oracle_results = validate_quantile_oracle(client)
    print(oracle_results.to_string(index=False))
    all_frames.append(oracle_results)

    print("\n5. Input Guards")
    print("-" * 70)
    guard_results = validate_input_guards(client)
    print(guard_results.to_string(index=False))
    all_frames.append(guard_results)

    print("\n6. Boundary Cases")
    print("-" * 70)
    boundary_results = validate_boundary_cases(client)
    print(boundary_results.to_string(index=False))
    all_frames.append(boundary_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/prior_elicitation_validation.csv", index=False)

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
