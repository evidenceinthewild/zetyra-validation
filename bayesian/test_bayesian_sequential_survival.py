#!/usr/bin/env python3
"""
Validate Bayesian Sequential Monitoring Boundaries (Survival/TTE)

Tests survival mapping: log(HR_hat) ~ N(log(HR_true), 4/d)
Uses data_variance=4, n_k = events_k / 2 for the Normal-Normal framework.

Tests:
1. Survival boundaries match continuous boundaries with mapped parameters
2. Structural properties (monotonicity, futility < efficacy)
3. Stronger HR -> lower boundaries (easier to stop)
4. Schema contract

References:
- Zhou, T., & Ji, Y. (2024) "On Bayesian Sequential Clinical Trial Designs"
- Schoenfeld (1983) variance formula: Var(log HR) = 4/d
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

BOUNDARY_TOLERANCE = 0.05


# ─── Reference implementation ────────────────────────────────────────

def reference_boundary_continuous(prior_mean, prior_variance, data_variance, n_k, threshold):
    """Zhou & Ji (2024) boundary formula mapped from continuous."""
    c = (
        sp_stats.norm.ppf(threshold) * np.sqrt(1 + data_variance / (n_k * prior_variance))
        - prior_mean * np.sqrt(data_variance) / (np.sqrt(n_k) * prior_variance)
    )
    return round(c, 4)


# ─── Test functions ──────────────────────────────────────────────────

def validate_survival_boundaries(client) -> pd.DataFrame:
    """Validate survival boundaries match continuous with mapped params."""
    results = []

    # Survival mapping: data_variance=4, n_k = events_k / 2
    events_per_look = [100, 200, 300]

    # Get survival boundaries from the API
    zetyra_surv = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=events_per_look,
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )

    schema_errors = assert_schema(zetyra_surv, "bayesian_sequential")

    # Compute reference boundaries using continuous formula with mapped params
    # prior_mean=0, prior_variance=1.0, data_variance=4.0, n_k = events/2
    for i, d_k in enumerate(events_per_look):
        n_k = d_k / 2  # Survival mapping
        ref_eff = reference_boundary_continuous(0.0, 1.0, 4.0, n_k, 0.975)
        ref_fut = reference_boundary_continuous(0.0, 1.0, 4.0, n_k, 0.10)

        zetyra_eff = zetyra_surv["efficacy_boundaries"][i]
        zetyra_fut = zetyra_surv["futility_boundaries"][i]

        eff_ok = abs(zetyra_eff - ref_eff) < BOUNDARY_TOLERANCE
        fut_ok = abs(zetyra_fut - ref_fut) < BOUNDARY_TOLERANCE

        results.append({
            "test": f"Look {i+1} (d={d_k}): efficacy",
            "zetyra": round(zetyra_eff, 4),
            "reference": ref_eff,
            "deviation": round(abs(zetyra_eff - ref_eff), 4),
            "pass": eff_ok and len(schema_errors) == 0,
        })
        results.append({
            "test": f"Look {i+1} (d={d_k}): futility",
            "zetyra": round(zetyra_fut, 4),
            "reference": ref_fut,
            "deviation": round(abs(zetyra_fut - ref_fut), 4),
            "pass": fut_ok and len(schema_errors) == 0,
        })

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Validate structural properties of survival boundaries."""
    results = []

    # Property 1: Efficacy boundaries decrease with more events
    z = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=[50, 100, 150, 200],
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    eff = z["efficacy_boundaries"]
    monotone = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "Efficacy boundaries decrease with events",
        "boundaries": str([round(b, 3) for b in eff]),
        "pass": monotone,
    })

    # Property 2: Futility < efficacy at every look
    fut = z["futility_boundaries"]
    fut_less = all(
        f is not None and f < e
        for f, e in zip(fut, eff)
    )
    results.append({
        "test": "Futility < efficacy at each look",
        "pass": fut_less,
    })

    # Property 3: endpoint_type in response is "survival"
    results.append({
        "test": "endpoint_type is 'survival'",
        "endpoint_type": z["endpoint_type"],
        "pass": z["endpoint_type"] == "survival",
    })

    # Property 4: n_looks matches events_per_look length
    results.append({
        "test": "n_looks = len(events_per_look)",
        "n_looks": z["n_looks"],
        "expected": 4,
        "pass": z["n_looks"] == 4,
    })

    return pd.DataFrame(results)


def validate_hr_monotonicity(client) -> pd.DataFrame:
    """Stronger HR should not increase boundaries (same events)."""
    results = []

    events_per_look = [100, 200, 300]

    # With vague prior (prior_variance=1), boundaries mainly depend on data_variance & n_k
    # Since the API uses the run() method which maps internally,
    # boundaries should be similar regardless of HR (boundaries are in z-score space)
    z1 = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=events_per_look,
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    z2 = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=events_per_look,
        hazard_ratio=0.5,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )

    # With uninformative prior, boundaries should be nearly identical
    # (prior_mean=0 means HR doesn't affect boundary computation, only OC)
    for i in range(len(events_per_look)):
        diff = abs(z1["efficacy_boundaries"][i] - z2["efficacy_boundaries"][i])
        results.append({
            "test": f"Look {i+1}: HR=0.7 vs HR=0.5 boundary diff",
            "hr07": round(z1["efficacy_boundaries"][i], 4),
            "hr05": round(z2["efficacy_boundaries"][i], 4),
            "diff": round(diff, 4),
            "pass": diff < 0.1,  # Should be very close
        })

    return pd.DataFrame(results)


def validate_different_event_schedules(client) -> pd.DataFrame:
    """More events per look -> lower boundaries (more information)."""
    results = []

    # Fewer events
    z_few = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=[50, 100],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
    )
    # More events
    z_many = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=[200, 400],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
    )

    # With more events, boundaries should be lower (closer to freq limit)
    for i in range(2):
        results.append({
            "test": f"Look {i+1}: more events -> lower boundary",
            "few_events_boundary": round(z_few["efficacy_boundaries"][i], 4),
            "many_events_boundary": round(z_many["efficacy_boundaries"][i], 4),
            "pass": z_many["efficacy_boundaries"][i] <= z_few["efficacy_boundaries"][i],
        })

    return pd.DataFrame(results)


def validate_nonuniform_schedule(client) -> pd.DataFrame:
    """Non-uniform event schedules produce boundaries matching reference."""
    results = []

    # Non-uniform: heavy early, light later
    front_loaded = [200, 250, 300]  # Big first look, small increments later
    z_front = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=front_loaded,
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )

    # Uniform with same total
    uniform = [100, 200, 300]
    z_uniform = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=uniform,
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )

    # Front-loaded schedule should have LOWER first boundary (more info at look 1)
    # because 200 events at look 1 vs 100 events at look 1
    results.append({
        "test": "Non-uniform: front-loaded has lower first boundary",
        "front_b1": round(z_front["efficacy_boundaries"][0], 4),
        "uniform_b1": round(z_uniform["efficacy_boundaries"][0], 4),
        "pass": z_front["efficacy_boundaries"][0] < z_uniform["efficacy_boundaries"][0],
    })

    # Last look has same events (300) -> boundaries should match
    results.append({
        "test": "Non-uniform: same final events -> same final boundary",
        "front_bk": round(z_front["efficacy_boundaries"][-1], 4),
        "uniform_bk": round(z_uniform["efficacy_boundaries"][-1], 4),
        "pass": abs(z_front["efficacy_boundaries"][-1] - z_uniform["efficacy_boundaries"][-1]) < BOUNDARY_TOLERANCE,
    })

    # Verify against analytical reference for the non-uniform schedule
    for i, d_k in enumerate(front_loaded):
        n_k = d_k / 2  # Survival mapping
        ref_eff = reference_boundary_continuous(0.0, 1.0, 4.0, n_k, 0.975)
        diff = abs(z_front["efficacy_boundaries"][i] - ref_eff)
        results.append({
            "test": f"Non-uniform look {i+1} (d={d_k}): matches reference",
            "zetyra": round(z_front["efficacy_boundaries"][i], 4),
            "reference": ref_eff,
            "diff": round(diff, 4),
            "pass": diff < BOUNDARY_TOLERANCE,
        })

    return pd.DataFrame(results)


def validate_odd_event_counts(client) -> pd.DataFrame:
    """Odd cumulative events should not be truncated by integer division."""
    results = []

    # Odd event counts: 101, 201, 301
    odd_events = [101, 201, 301]
    z_odd = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=odd_events,
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )

    # Even event counts: 100, 200, 300
    even_events = [100, 200, 300]
    z_even = client.bayesian_sequential_survival(
        endpoint_type="survival",
        n_per_look=even_events,
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )

    # Boundaries for n_k = d/2 should differ from n_k = d//2 when d is odd
    # With d=101: d/2=50.5 vs d//2=50. Boundary should be slightly lower for 50.5.
    for i in range(len(odd_events)):
        d_odd = odd_events[i]
        d_even = even_events[i]
        n_k_exact = d_odd / 2  # 50.5, 100.5, 150.5
        ref_eff = reference_boundary_continuous(0.0, 1.0, 4.0, n_k_exact, 0.975)
        diff = abs(z_odd["efficacy_boundaries"][i] - ref_eff)

        results.append({
            "test": f"Odd d={d_odd}: matches d/2 reference (not d//2)",
            "zetyra": round(z_odd["efficacy_boundaries"][i], 4),
            "reference": ref_eff,
            "diff": round(diff, 4),
            "pass": diff < BOUNDARY_TOLERANCE,
        })

    # Odd boundaries should be slightly lower than even (more info at 50.5 vs 50)
    results.append({
        "test": "Odd events -> slightly lower boundaries than even",
        "odd_b1": round(z_odd["efficacy_boundaries"][0], 4),
        "even_b1": round(z_even["efficacy_boundaries"][0], 4),
        "pass": z_odd["efficacy_boundaries"][0] < z_even["efficacy_boundaries"][0],
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate malformed n_per_look schedules are rejected."""
    results = []

    # Single look (survival requires at least 2 for sequential monitoring)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
    )
    results.append({
        "test": "Guard: single look rejected (survival)",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Non-increasing schedule
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[200, 200, 300],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
    )
    results.append({
        "test": "Guard: non-increasing rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Decreasing schedule
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[300, 200, 100],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
    )
    results.append({
        "test": "Guard: decreasing rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Zero events
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[0, 100, 200],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
    )
    results.append({
        "test": "Guard: zero events rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Unequal allocation rejected (Schoenfeld 4/d assumes 1:1)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        allocation_ratio=2.0,
    )
    results.append({
        "test": "Guard: unequal allocation rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # futility_threshold=0 rejected (produces -inf boundary)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=0.0,
    )
    results.append({
        "test": "Guard: futility_threshold=0 rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # futility_threshold >= efficacy_threshold rejected (inverts stopping)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        hazard_ratio=0.7,
        efficacy_threshold=0.975,
        futility_threshold=0.98,
    )
    results.append({
        "test": "Guard: futility >= efficacy rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # hazard_ratio omitted is accepted (optional metadata)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    results.append({
        "test": "Accept: hazard_ratio omitted",
        "status_code": resp.status_code,
        "pass": resp.status_code == 200,
    })

    # hazard_ratio >= 1 accepted as metadata (no longer used in computation)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        hazard_ratio=1.2,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    results.append({
        "test": "Accept: hazard_ratio >= 1 (metadata)",
        "status_code": resp.status_code,
        "pass": resp.status_code == 200,
    })

    # All metadata fields accept arbitrary values (not used in computation)
    resp = client.bayesian_sequential_survival_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        hazard_ratio=-0.5,
        median_control=-10.0,
        accrual_time=-1.0,
        follow_up_time=-1.0,
        dropout_rate=2.0,
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    results.append({
        "test": "Accept: arbitrary metadata values",
        "status_code": resp.status_code,
        "pass": resp.status_code == 200,
    })

    # Route infers endpoint_type — omitting it (defaults to "continuous") should still work
    resp = client.bayesian_sequential_survival_raw(
        n_per_look=[100, 200, 300],
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    results.append({
        "test": "Accept: endpoint_type omitted (route infers survival)",
        "status_code": resp.status_code,
        "pass": resp.status_code == 200 and resp.json().get("endpoint_type") == "survival",
    })

    # Generic /sequential route redirects survival requests to dedicated route
    resp = client.bayesian_sequential_raw(
        endpoint_type="survival",
        n_per_look=[100, 200, 300],
        efficacy_threshold=0.975,
        futility_threshold=0.10,
    )
    results.append({
        "test": "Guard: generic route redirects survival requests",
        "status_code": resp.status_code,
        "pass": resp.status_code == 400 and "survival" in resp.json().get("detail", "").lower(),
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("BAYESIAN SEQUENTIAL SURVIVAL VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Survival Boundary Formula")
    print("-" * 70)
    b_results = validate_survival_boundaries(client)
    print(b_results.to_string(index=False))
    all_frames.append(b_results)

    print("\n2. Structural Properties")
    print("-" * 70)
    p_results = validate_properties(client)
    print(p_results.to_string(index=False))
    all_frames.append(p_results)

    print("\n3. HR Independence (Uninformative Prior)")
    print("-" * 70)
    h_results = validate_hr_monotonicity(client)
    print(h_results.to_string(index=False))
    all_frames.append(h_results)

    print("\n4. Event Schedule Effects")
    print("-" * 70)
    e_results = validate_different_event_schedules(client)
    print(e_results.to_string(index=False))
    all_frames.append(e_results)

    print("\n5. Non-Uniform Event Schedules")
    print("-" * 70)
    n_results = validate_nonuniform_schedule(client)
    print(n_results.to_string(index=False))
    all_frames.append(n_results)

    print("\n6. Odd Event Counts")
    print("-" * 70)
    o_results = validate_odd_event_counts(client)
    print(o_results.to_string(index=False))
    all_frames.append(o_results)

    print("\n7. Input Guards")
    print("-" * 70)
    g_results = validate_input_guards(client)
    print(g_results.to_string(index=False))
    all_frames.append(g_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/bayesian_sequential_survival_validation.csv", index=False)

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
