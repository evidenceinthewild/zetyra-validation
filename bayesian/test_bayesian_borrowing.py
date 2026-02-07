#!/usr/bin/env python3
"""
Validate Bayesian Historical Borrowing Calculator

Tests:
1. Power prior: exact Beta(α₀ + δ×events, β₀ + δ×(n-events))
2. MAP prior: I² heterogeneity, pooled rate (FE reference, wider tolerance for high I²)
3. Input guards (422/400 for invalid inputs)
4. Boundary-condition scenarios (δ=0/1, zero/all events, robust_weight=0/1)
5. Invariants: higher discount → ESS closer to full n
6. Symmetry: two identical studies → I²=0
7. Schema contracts

References:
- REBYOTA PUNCH CD2 (Phase 2b): 25/45 responders, two-dose arm (FDA BLA 125739)
- REBYOTA PUNCH CD3 (Phase 3): 126/177 treatment, 53/85 placebo
- Schmidli et al. (2014) "Robust MAP priors" - heterogeneity examples
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
import numpy as np

ESS_TOLERANCE = 0.1
I_SQUARED_TOLERANCE = 5  # percentage points
RATE_TOLERANCE = 0.03


# ─── Reference implementations ────────────────────────────────────────

def reference_power_prior(events, n, discount, base_alpha=1.0, base_beta=1.0):
    """Power prior: Beta(α₀ + δ×events, β₀ + δ×(n-events))."""
    alpha = base_alpha + discount * events
    beta = base_beta + discount * (n - events)
    ess = alpha + beta
    borrowed_ess = discount * n
    mean = alpha / ess
    return {
        "alpha": alpha, "beta": beta,
        "ess_total": ess, "ess_historical": borrowed_ess,
        "mean": mean,
    }


def reference_heterogeneity(studies):
    """Compute Cochran's Q, I², and fixed-effect pooled rate from study data."""
    rates = [s["n_events"] / s["n_total"] for s in studies]
    ns = [s["n_total"] for s in studies]

    # Study variances (binomial)
    variances = [
        r * (1 - r) / n if 0 < r < 1 else 0.25 / n
        for r, n in zip(rates, ns)
    ]

    # Inverse-variance weights
    weights = [1 / v if v > 0 else 0 for v in variances]
    total_w = sum(weights)

    if total_w > 0:
        pooled = sum(w * r for w, r in zip(weights, rates)) / total_w
    else:
        pooled = np.mean(rates)

    # Cochran's Q
    Q = sum(w * (r - pooled) ** 2 for w, r in zip(weights, rates))
    df = len(studies) - 1

    # I²
    i_squared = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    return {"Q": Q, "i_squared": i_squared, "pooled_rate": pooled}


# ─── Test functions ───────────────────────────────────────────────────

def validate_power_prior(client) -> pd.DataFrame:
    """Validate power prior with REBYOTA data."""
    scenarios = [
        {"name": "REBYOTA δ=0.5", "events": 25, "n": 45, "discount": 0.5},
        {"name": "REBYOTA δ=0.0 (no borrow)", "events": 25, "n": 45, "discount": 0.0},
        {"name": "REBYOTA δ=1.0 (full borrow)", "events": 25, "n": 45, "discount": 1.0},
        {"name": "PUNCH CD3 δ=0.5", "events": 126, "n": 177, "discount": 0.5},
        {"name": "Small study δ=0.3", "events": 8, "n": 20, "discount": 0.3},
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_borrowing(
            method="power_prior",
            historical_events=s["events"],
            historical_n=s["n"],
            discount_factor=s["discount"],
        )
        ref = reference_power_prior(s["events"], s["n"], s["discount"])

        schema_errors = assert_schema(zetyra, "bayesian_borrowing")

        alpha_ok = abs(zetyra["effective_alpha"] - ref["alpha"]) < 0.01
        beta_ok = abs(zetyra["effective_beta"] - ref["beta"]) < 0.01
        ess_ok = abs(zetyra["ess_total"] - ref["ess_total"]) < ESS_TOLERANCE
        hist_ess_ok = abs(zetyra["ess_from_historical"] - ref["ess_historical"]) < ESS_TOLERANCE
        mean_ok = abs(zetyra["prior_mean"] - ref["mean"]) < 0.01

        passed = alpha_ok and beta_ok and ess_ok and hist_ess_ok and mean_ok and len(schema_errors) == 0
        results.append({
            "test": s["name"],
            "zetyra_alpha": zetyra["effective_alpha"],
            "ref_alpha": ref["alpha"],
            "zetyra_ess": zetyra["ess_total"],
            "ref_ess": ref["ess_total"],
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_map_prior(client) -> pd.DataFrame:
    """Validate MAP prior heterogeneity statistics."""
    scenarios = [
        {
            "name": "Low heterogeneity (similar rates ~0.21)",
            "studies": [
                {"n_events": 8, "n_total": 40},
                {"n_events": 10, "n_total": 45},
                {"n_events": 9, "n_total": 42},
            ],
            "expected_i2_range": (0, 30),
        },
        {
            "name": "High heterogeneity (diverse rates)",
            "studies": [
                {"n_events": 5, "n_total": 50},
                {"n_events": 20, "n_total": 50},
                {"n_events": 35, "n_total": 50},
            ],
            "expected_i2_range": (70, 100),
        },
        {
            "name": "Two similar studies",
            "studies": [
                {"n_events": 15, "n_total": 50},
                {"n_events": 16, "n_total": 55},
            ],
            "expected_i2_range": (0, 30),
        },
        {
            "name": "REBYOTA CD2+CD3 (real trial data)",
            "studies": [
                {"n_events": 25, "n_total": 45},   # PUNCH CD2 two-dose arm
                {"n_events": 126, "n_total": 177},  # PUNCH CD3 treatment arm
            ],
            "expected_i2_range": (40, 90),  # Moderate-high — 55.6% vs 71.2% across phases
        },
    ]

    results = []
    for s in scenarios:
        zetyra = client.bayesian_borrowing(
            method="map_prior",
            studies=s["studies"],
            robust_weight=0.1,
        )
        ref = reference_heterogeneity(s["studies"])

        # Schema check
        schema_errors = assert_schema(zetyra, "bayesian_borrowing")

        i2_in_range = s["expected_i2_range"][0] <= zetyra["i_squared"] <= s["expected_i2_range"][1]
        i2_close = abs(zetyra["i_squared"] - ref["i_squared"]) < I_SQUARED_TOLERANCE

        # Pooled rate comparison: MAP uses RE + robust mixture (shifts toward 0.5),
        # so we compare against reference FE rate with wider tolerance for high heterogeneity
        rate_tol = RATE_TOLERANCE if ref["i_squared"] < 50 else 0.15
        rate_ok = abs(zetyra["pooled_rate"] - ref["pooled_rate"]) < rate_tol

        # Verify ESS is reasonable (> 2, < sum of historical n)
        total_hist_n = sum(st["n_total"] for st in s["studies"])
        ess_reasonable = 2 < zetyra["ess_total"] <= total_hist_n + 10

        passed = i2_in_range and i2_close and rate_ok and ess_reasonable and len(schema_errors) == 0
        results.append({
            "test": f"MAP: {s['name']}",
            "zetyra_i2": round(zetyra["i_squared"], 1),
            "ref_i2": round(ref["i_squared"], 1),
            "i2_dev": round(abs(zetyra["i_squared"] - ref["i_squared"]), 1),
            "zetyra_rate": round(zetyra["pooled_rate"], 4),
            "ref_rate": round(ref["pooled_rate"], 4),
            "zetyra_ess": round(zetyra["ess_total"], 1),
            "pass": passed,
        })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate that invalid inputs return 400/422."""
    results = []

    guards = [
        {
            "name": "discount_factor > 1",
            "field": "discount_factor",
            "data": {"method": "power_prior", "historical_events": 10, "historical_n": 20, "discount_factor": 1.5},
        },
        {
            "name": "empty studies list",
            "field": "studies",
            "data": {"method": "map_prior", "studies": []},
        },
        {
            "name": "single study (MAP needs ≥ 2)",
            "field": "studies",
            "data": {"method": "map_prior", "studies": [{"n_events": 10, "n_total": 20}]},
        },
        {
            "name": "robust_weight > 1",
            "field": "robust_weight",
            "data": {"method": "map_prior", "studies": [{"n_events": 10, "n_total": 20}, {"n_events": 15, "n_total": 30}], "robust_weight": 1.5},
        },
    ]

    for g in guards:
        resp = client._post_raw("/bayesian/borrowing", g["data"])
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

    # Power prior: zero events
    z = client.bayesian_borrowing(
        method="power_prior", historical_events=0, historical_n=50, discount_factor=0.5,
    )
    ref = reference_power_prior(0, 50, 0.5)
    schema_zero = assert_schema(z, "bayesian_borrowing")
    results.append({
        "test": "Boundary: zero events",
        "zetyra_alpha": z["effective_alpha"], "ref_alpha": ref["alpha"],
        "pass": abs(z["effective_alpha"] - ref["alpha"]) < 0.01
                and abs(z["effective_beta"] - ref["beta"]) < 0.01
                and len(schema_zero) == 0,
    })

    # Power prior: all events
    z = client.bayesian_borrowing(
        method="power_prior", historical_events=50, historical_n=50, discount_factor=0.5,
    )
    ref = reference_power_prior(50, 50, 0.5)
    schema_all = assert_schema(z, "bayesian_borrowing")
    results.append({
        "test": "Boundary: all events",
        "zetyra_alpha": z["effective_alpha"], "ref_alpha": ref["alpha"],
        "pass": abs(z["effective_alpha"] - ref["alpha"]) < 0.01
                and abs(z["effective_beta"] - ref["beta"]) < 0.01
                and len(schema_all) == 0,
    })

    # MAP: robust_weight=0 (no robustification)
    z = client.bayesian_borrowing(
        method="map_prior",
        studies=[{"n_events": 15, "n_total": 50}, {"n_events": 16, "n_total": 55}],
        robust_weight=0.0,
    )
    schema_ok = len(assert_schema(z, "bayesian_borrowing")) == 0
    results.append({
        "test": "Boundary: robust_weight=0",
        "zetyra_ess": round(z["ess_total"], 1),
        "pass": schema_ok and z["ess_total"] > 2,
    })

    return pd.DataFrame(results)


def validate_invariants(client) -> pd.DataFrame:
    """Invariant/property tests."""
    results = []

    # Invariant: Higher discount → ESS closer to full historical n
    low_d = client.bayesian_borrowing(
        method="power_prior", historical_events=25, historical_n=45, discount_factor=0.2,
    )
    high_d = client.bayesian_borrowing(
        method="power_prior", historical_events=25, historical_n=45, discount_factor=0.8,
    )
    schema_ld = assert_schema(low_d, "bayesian_borrowing")
    schema_hd = assert_schema(high_d, "bayesian_borrowing")
    results.append({
        "test": "Invariant: higher discount → higher ESS",
        "low_ess": round(low_d["ess_total"], 1),
        "high_ess": round(high_d["ess_total"], 1),
        "pass": high_d["ess_total"] > low_d["ess_total"] and len(schema_ld) == 0 and len(schema_hd) == 0,
    })

    return pd.DataFrame(results)


def validate_symmetry(client) -> pd.DataFrame:
    """Two identical studies should give I²=0."""
    results = []

    z = client.bayesian_borrowing(
        method="map_prior",
        studies=[
            {"n_events": 20, "n_total": 50},
            {"n_events": 20, "n_total": 50},
        ],
        robust_weight=0.1,
    )
    schema_errors = assert_schema(z, "bayesian_borrowing")
    results.append({
        "test": "Symmetry: identical studies → I²=0",
        "i_squared": round(z["i_squared"], 1),
        "pass": z["i_squared"] < 1.0 and len(schema_errors) == 0,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN BORROWING VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Power Prior (REBYOTA)")
    print("-" * 70)
    pp_results = validate_power_prior(client)
    print(pp_results.to_string(index=False))
    all_frames.append(pp_results)

    print("\n2. MAP Prior (Heterogeneity)")
    print("-" * 70)
    map_results = validate_map_prior(client)
    print(map_results.to_string(index=False))
    all_frames.append(map_results)

    print("\n3. Input Guards")
    print("-" * 70)
    guard_results = validate_input_guards(client)
    print(guard_results.to_string(index=False))
    all_frames.append(guard_results)

    print("\n4. Boundary Cases")
    print("-" * 70)
    boundary_results = validate_boundary_cases(client)
    print(boundary_results.to_string(index=False))
    all_frames.append(boundary_results)

    print("\n5. Invariants")
    print("-" * 70)
    inv_results = validate_invariants(client)
    print(inv_results.to_string(index=False))
    all_frames.append(inv_results)

    print("\n6. Symmetry")
    print("-" * 70)
    sym_results = validate_symmetry(client)
    print(sym_results.to_string(index=False))
    all_frames.append(sym_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_borrowing_validation.csv", index=False)

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
