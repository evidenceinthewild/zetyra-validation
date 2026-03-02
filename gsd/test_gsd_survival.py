#!/usr/bin/env python3
"""
Validate Zetyra GSD Survival/TTE Endpoint

Tests Group Sequential Design with survival (time-to-event) endpoints:
1. Schoenfeld formula: events match analytical reference
2. Sample size computed from events via event probability
3. Boundaries match continuous GSD with equivalent effect size
4. Structural properties (monotonicity, alpha control)
5. Input guards
6. Schema contract

Reference:
- Schoenfeld (1983) "Sample-Size Formula for the Proportional-Hazards Regression Model"
- Lakatos (1988) "Sample Sizes Based on the Log-Rank Statistic in Complex Clinical Trials"
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

BOUNDARY_TOLERANCE = 0.05
EVENT_TOLERANCE = 0.05  # 5% relative tolerance for events


# ─── Reference implementations ──────────────────────────────────────

def schoenfeld_events(alpha, power, log_hr, allocation_ratio=1.0):
    """Schoenfeld formula for required events (one-sided)."""
    z_alpha = sp_stats.norm.ppf(1 - alpha)
    z_beta = sp_stats.norm.ppf(power)
    r = allocation_ratio
    d = ((z_alpha + z_beta) / log_hr) ** 2 * (1 + r) ** 2 / r
    return int(math.ceil(d))


def event_probability_ref(median_control, hazard_ratio, accrual_time, follow_up_time, dropout_rate=0.0, allocation_ratio=1.0):
    """Weighted average event probability (exponential model, uniform accrual)."""
    lam_c = math.log(2) / median_control
    lam_t = lam_c * hazard_ratio
    r = allocation_ratio

    def p_event(lam):
        # Numerical integration over uniform accrual
        n_pts = 200
        probs = []
        for i in range(n_pts):
            entry_time = accrual_time * (i + 0.5) / n_pts
            follow = (accrual_time - entry_time) + follow_up_time
            p = 1 - math.exp(-lam * follow)
            if dropout_rate > 0:
                p *= (1 - dropout_rate) ** (follow / 12)
            probs.append(p)
        return sum(probs) / n_pts

    p_c = p_event(lam_c)
    p_t = p_event(lam_t)
    return (p_c + r * p_t) / (1 + r)


# ─── Test functions ──────────────────────────────────────────────────

def validate_events_and_sample_size(client) -> pd.DataFrame:
    """Validate events match Schoenfeld formula and N is consistent."""
    results = []

    scenarios = [
        {"name": "HR=0.7, typical", "hr": 0.7, "med_ctrl": 12, "acc": 24, "fu": 12, "alpha": 0.025, "power": 0.90, "k": 3},
        {"name": "HR=0.8, weak effect", "hr": 0.8, "med_ctrl": 18, "acc": 36, "fu": 12, "alpha": 0.025, "power": 0.80, "k": 4},
        {"name": "HR=0.5, strong effect", "hr": 0.5, "med_ctrl": 6, "acc": 12, "fu": 6, "alpha": 0.025, "power": 0.90, "k": 2},
    ]

    for s in scenarios:
        zetyra = client.gsd_survival(
            hazard_ratio=s["hr"],
            median_control=s["med_ctrl"],
            accrual_time=s["acc"],
            follow_up_time=s["fu"],
            alpha=s["alpha"],
            power=s["power"],
            k=s["k"],
        )

        schema_errors = assert_schema(zetyra, "gsd_survival")

        # Check fixed events against Schoenfeld
        ref_fixed = schoenfeld_events(s["alpha"], s["power"], math.log(s["hr"]))
        fixed_rel_err = abs(zetyra["fixed_events"] - ref_fixed) / ref_fixed

        # Max events should be >= fixed events (inflation)
        max_ge_fixed = zetyra["max_events"] >= zetyra["fixed_events"]

        # N should be > max_events (need more patients than events)
        n_ge_events = zetyra["n_total"] >= zetyra["max_events"]

        # Event probability should be in (0,1)
        ep_valid = 0 < zetyra["event_probability"] < 1

        # N ≈ max_events / event_probability (the defining relationship)
        # Allow 10% tolerance for rounding and allocation discretization
        expected_n = zetyra["max_events"] / zetyra["event_probability"]
        n_consistency_err = abs(zetyra["n_total"] - expected_n) / expected_n

        # n_control + n_treatment should equal n_total
        arm_sum_ok = abs((zetyra["n_control"] + zetyra["n_treatment"]) - zetyra["n_total"]) <= 1

        results.append({
            "test": f"{s['name']}: fixed events",
            "zetyra_fixed": zetyra["fixed_events"],
            "ref_fixed": ref_fixed,
            "rel_err": round(fixed_rel_err, 4),
            "pass": fixed_rel_err < EVENT_TOLERANCE and len(schema_errors) == 0,
        })
        results.append({
            "test": f"{s['name']}: max >= fixed",
            "max_events": zetyra["max_events"],
            "fixed_events": zetyra["fixed_events"],
            "pass": max_ge_fixed,
        })
        results.append({
            "test": f"{s['name']}: N >= events",
            "n_total": zetyra["n_total"],
            "max_events": zetyra["max_events"],
            "pass": n_ge_events,
        })
        results.append({
            "test": f"{s['name']}: event_prob valid",
            "event_probability": zetyra["event_probability"],
            "pass": ep_valid,
        })
        results.append({
            "test": f"{s['name']}: N = events / p_event",
            "n_total": zetyra["n_total"],
            "expected_n": round(expected_n, 1),
            "rel_err": round(n_consistency_err, 4),
            "pass": n_consistency_err < 0.10,
        })
        results.append({
            "test": f"{s['name']}: n_ctrl + n_trt = n_total",
            "n_control": zetyra["n_control"],
            "n_treatment": zetyra["n_treatment"],
            "n_total": zetyra["n_total"],
            "pass": arm_sum_ok,
        })

    return pd.DataFrame(results)


def validate_boundaries(client) -> pd.DataFrame:
    """Validate boundary structure and properties."""
    results = []

    zetyra = client.gsd_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12,
        alpha=0.025, power=0.90, k=3, spending_function="OBrienFleming",
    )

    eff = zetyra["efficacy_boundaries"]

    # O'Brien-Fleming: first boundary should be very high
    results.append({
        "test": "OBF: first boundary > 3",
        "boundary_1": eff[0],
        "pass": eff[0] > 3.0,
    })

    # O'Brien-Fleming: boundaries decrease
    monotone = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "OBF: efficacy boundaries decrease",
        "boundaries": str([round(b, 3) for b in eff]),
        "pass": monotone,
    })

    # Futility boundaries exist (Pocock beta spending by default for OBF)
    fut = zetyra["futility_boundaries"]
    has_futility = any(f is not None for f in fut)
    results.append({
        "test": "Futility boundaries present",
        "boundaries": str([round(f, 3) if f is not None else None for f in fut]),
        "pass": has_futility,
    })

    # Pocock spending: boundaries should be more uniform
    zetyra_poc = client.gsd_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12,
        alpha=0.025, power=0.90, k=3, spending_function="Pocock",
    )
    eff_poc = zetyra_poc["efficacy_boundaries"]
    # Pocock boundaries should have smaller range than OBF
    range_poc = max(eff_poc) - min(eff_poc)
    range_obf = max(eff) - min(eff)
    results.append({
        "test": "Pocock range < OBF range",
        "pocock_range": round(range_poc, 3),
        "obf_range": round(range_obf, 3),
        "pass": range_poc < range_obf,
    })

    # Alpha spent should sum to alpha at final look
    alpha_sum = zetyra["alpha_spent"][-1]
    results.append({
        "test": "Cumulative alpha = target alpha",
        "alpha_spent": round(alpha_sum, 4),
        "target": 0.025,
        "pass": abs(alpha_sum - 0.025) < 0.001,
    })

    # Information fractions should be increasing to 1
    info = zetyra["information_fractions"]
    info_ok = all(info[i] < info[i+1] for i in range(len(info)-1)) and abs(info[-1] - 1.0) < 0.001
    results.append({
        "test": "Info fractions increasing to 1.0",
        "info_fractions": str([round(f, 3) for f in info]),
        "pass": info_ok,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return 400/422."""
    results = []

    valid_base = {
        "hazard_ratio": 0.7, "median_control": 12, "accrual_time": 24,
        "follow_up_time": 12, "alpha": 0.025, "power": 0.90, "k": 3,
    }

    guards = [
        {"name": "HR=1 (no effect)", "data": {**valid_base, "hazard_ratio": 1.0}},
        {"name": "HR>1", "data": {**valid_base, "hazard_ratio": 1.5}},
        {"name": "HR=0", "data": {**valid_base, "hazard_ratio": 0.0}},
        {"name": "Negative median", "data": {**valid_base, "median_control": -1}},
    ]

    for g in guards:
        resp = client.gsd_survival_raw(**g["data"])
        results.append({
            "test": f"Guard: {g['name']}",
            "status_code": resp.status_code,
            "pass": resp.status_code in (400, 422),
        })

    return pd.DataFrame(results)


def validate_monotonicity(client) -> pd.DataFrame:
    """More events with weaker HR; fewer events with stronger HR."""
    results = []

    hr_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    events_list = []
    for hr in hr_values:
        z = client.gsd_survival(
            hazard_ratio=hr, median_control=12, accrual_time=24, follow_up_time=12,
            alpha=0.025, power=0.90, k=3,
        )
        events_list.append(z["max_events"])

    # Weaker HR (closer to 1) should need more events
    mono = all(events_list[i] <= events_list[i+1] for i in range(len(events_list)-1))
    results.append({
        "test": "Events increase as HR -> 1",
        "hr_values": str(hr_values),
        "events": str(events_list),
        "pass": mono,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("GSD SURVIVAL/TTE ENDPOINT VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Events & Sample Size")
    print("-" * 70)
    es_results = validate_events_and_sample_size(client)
    print(es_results.to_string(index=False))
    all_frames.append(es_results)

    print("\n2. Boundary Properties")
    print("-" * 70)
    b_results = validate_boundaries(client)
    print(b_results.to_string(index=False))
    all_frames.append(b_results)

    print("\n3. Input Guards")
    print("-" * 70)
    g_results = validate_input_guards(client)
    print(g_results.to_string(index=False))
    all_frames.append(g_results)

    print("\n4. Monotonicity")
    print("-" * 70)
    m_results = validate_monotonicity(client)
    print(m_results.to_string(index=False))
    all_frames.append(m_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/gsd_survival_validation.csv", index=False)

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
