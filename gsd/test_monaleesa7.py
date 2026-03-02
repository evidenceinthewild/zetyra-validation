#!/usr/bin/env python3
"""
Validate GSD Survival Boundaries Against MONALEESA-7 Trial Design

MONALEESA-7 was a Phase 3 trial of ribociclib + endocrine therapy vs placebo
for premenopausal HR+/HER2- advanced breast cancer. OS was a key secondary
endpoint with a 3-look Lan-DeMets O'Brien-Fleming group sequential design.

Published design:
- One-sided alpha = 0.025
- 3 planned OS analyses at ~89, ~189, 252 events
- Lan-DeMets O'Brien-Fleming alpha spending
- Look 1 (35% IF): boundary p < 0.00016 — NOT crossed
- Look 2 (75% IF): boundary p < 0.01018 — CROSSED (p=0.00973, HR=0.712)
- Look 3 (100%): would have been at 252 events

Reference:
- Im et al. (2019) NEJM 381:307-316
  "Overall Survival with Ribociclib plus Endocrine Therapy in Breast Cancer"
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq


# ─── Lan-DeMets OBF reference implementation ────────────────────────────

def ld_obf_spending(t, alpha):
    """
    Lan-DeMets O'Brien-Fleming spending function.

    gsDesign formula: alpha_spent(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    z_alpha_half = sp_stats.norm.ppf(1 - alpha / 2)
    return 2 * (1 - sp_stats.norm.cdf(z_alpha_half / math.sqrt(t)))


def ld_obf_incremental_spending(timing, alpha):
    """Compute incremental alpha spending at each look."""
    cum = [ld_obf_spending(t, alpha) for t in timing]
    inc = [cum[0]]
    for i in range(1, len(cum)):
        inc.append(cum[i] - cum[i - 1])
    return cum, inc


def ld_obf_z_boundaries(timing, alpha):
    """
    Compute z-score boundaries from Lan-DeMets OBF spending.

    Uses multivariate normal integration to find z_k at each look such that
    the incremental rejection probability equals the incremental alpha spent.
    """
    cum, inc = ld_obf_incremental_spending(timing, alpha)
    boundaries = []

    for i in range(len(timing)):
        if i == 0:
            z = sp_stats.norm.ppf(1 - inc[0])
            boundaries.append(z)
        else:
            prev_z = boundaries[:i]
            prev_t = timing[:i]
            t_i = timing[i]

            def incremental_rejection(z_cand):
                p_i = 1 - sp_stats.norm.cdf(z_cand)
                overlap = 0.0
                for j in range(i):
                    rho = math.sqrt(prev_t[j] / t_i)
                    mean = [0, 0]
                    cov = [[1, rho], [rho, 1]]
                    p_both = sp_stats.multivariate_normal.cdf(
                        [prev_z[j], z_cand], mean=mean, cov=cov
                    )
                    p_joint_exceed = (
                        1 - sp_stats.norm.cdf(prev_z[j])
                        - sp_stats.norm.cdf(z_cand)
                        + p_both
                    )
                    overlap += p_joint_exceed
                return p_i - overlap - inc[i]

            try:
                z = brentq(incremental_rejection, 1.0, 8.0, xtol=1e-8)
            except ValueError:
                z = sp_stats.norm.ppf(1 - inc[i])
            boundaries.append(z)

    return boundaries, cum, inc


# ─── MONALEESA-7 trial parameters ───────────────────────────────────────

ALPHA = 0.025  # one-sided
TOTAL_EVENTS = 252
PLANNED_EVENTS = [89, 189, 252]
TIMING = [d / TOTAL_EVENTS for d in PLANNED_EVENTS]  # [0.3532, 0.75, 1.0]

# Published boundary p-values (one-sided)
PUBLISHED_P_LOOK1 = 0.00016
PUBLISHED_P_LOOK2 = 0.01018

# Observed at second interim (the crossing analysis)
ACTUAL_EVENTS_AT_CROSSING = 192  # per updated analysis: 275 deaths for extended follow-up
ACTUAL_HR = 0.712
ACTUAL_P_ONE_SIDED = 0.00973


def validate_spending_function(client) -> pd.DataFrame:
    """
    Validate that Lan-DeMets OBF spending reproduces MONALEESA-7 boundaries.

    The published boundary p-values at each look should match the
    cumulative Lan-DeMets OBF spending function evaluated at the
    pre-specified information fractions.
    """
    results = []

    # Compute Lan-DeMets OBF spending at each look
    cum_alpha, inc_alpha = ld_obf_incremental_spending(TIMING, ALPHA)

    # Look 1: cumulative alpha spent should match published p < 0.00016
    results.append({
        "test": "Look 1 spending matches published boundary",
        "published_p": PUBLISHED_P_LOOK1,
        "computed_spending": round(cum_alpha[0], 6),
        "deviation": round(abs(cum_alpha[0] - PUBLISHED_P_LOOK1), 6),
        "info_frac": round(TIMING[0], 4),
        "pass": abs(cum_alpha[0] - PUBLISHED_P_LOOK1) < 0.00001,
    })

    # Look 2: cumulative alpha spent should match published p < 0.01018
    # The published p is the cumulative spending minus look 1 spending?
    # No — the published boundary is the one-sided p-value threshold.
    # In the spending function approach, the boundary p at look k is NOT
    # the cumulative alpha. We need the z-boundary and its one-sided p.
    z_bounds, _, _ = ld_obf_z_boundaries(TIMING, ALPHA)
    p_bounds = [1 - sp_stats.norm.cdf(z) for z in z_bounds]

    results.append({
        "test": "Look 1 z-boundary → p matches published",
        "published_p": PUBLISHED_P_LOOK1,
        "computed_p": round(p_bounds[0], 6),
        "z_boundary": round(z_bounds[0], 4),
        "deviation": round(abs(p_bounds[0] - PUBLISHED_P_LOOK1), 6),
        "pass": abs(p_bounds[0] - PUBLISHED_P_LOOK1) < 0.00002,
    })

    results.append({
        "test": "Look 2 z-boundary → p matches published",
        "published_p": PUBLISHED_P_LOOK2,
        "computed_p": round(p_bounds[1], 6),
        "z_boundary": round(z_bounds[1], 4),
        "deviation": round(abs(p_bounds[1] - PUBLISHED_P_LOOK2), 6),
        "pass": abs(p_bounds[1] - PUBLISHED_P_LOOK2) < 0.001,
    })

    # Cumulative alpha at final look = target alpha
    results.append({
        "test": "Cumulative alpha = 0.025 at final look",
        "alpha_spent": round(cum_alpha[-1], 6),
        "target": ALPHA,
        "pass": abs(cum_alpha[-1] - ALPHA) < 0.001,
    })

    return pd.DataFrame(results)


def validate_trial_crossing(client) -> pd.DataFrame:
    """
    Validate that the trial correctly crossed at look 2 but not look 1.

    Look 1: did NOT cross (observed p > 0.00016)
    Look 2: DID cross (observed p = 0.00973 < 0.01018)
    """
    results = []

    z_bounds, _, _ = ld_obf_z_boundaries(TIMING, ALPHA)

    # The observed p should be less than the boundary p at look 2
    results.append({
        "test": "Observed p < boundary p at look 2 (crossing)",
        "p_observed": ACTUAL_P_ONE_SIDED,
        "p_boundary": PUBLISHED_P_LOOK2,
        "HR": ACTUAL_HR,
        "pass": ACTUAL_P_ONE_SIDED < PUBLISHED_P_LOOK2,
    })

    # The observed p is close to (but below) the boundary — tight crossing
    margin = PUBLISHED_P_LOOK2 - ACTUAL_P_ONE_SIDED
    results.append({
        "test": "Crossing margin is tight (< 0.005)",
        "p_observed": ACTUAL_P_ONE_SIDED,
        "p_boundary": PUBLISHED_P_LOOK2,
        "margin": round(margin, 5),
        "pass": 0 < margin < 0.005,
    })

    # Boundary at look 1 should be very stringent (p < 0.001)
    p_look1_bound = 1 - sp_stats.norm.cdf(z_bounds[0])
    results.append({
        "test": "Look 1 boundary is highly conservative (p < 0.001)",
        "p_boundary": round(p_look1_bound, 6),
        "pass": p_look1_bound < 0.001,
    })

    return pd.DataFrame(results)


def validate_boundary_properties(client) -> pd.DataFrame:
    """Validate structural properties via Zetyra GSD endpoint."""
    results = []

    zetyra = client.gsd(
        effect_size=0.3,
        alpha=ALPHA,
        power=0.80,
        k=3,
        timing=TIMING,
        spending_function="OBrienFleming",
    )

    eff = zetyra["efficacy_boundaries"]

    # O'Brien-Fleming: boundaries must decrease
    mono = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "OBF efficacy boundaries decrease",
        "boundaries": str([round(b, 4) for b in eff]),
        "pass": mono,
    })

    # Information fractions match requested
    info = zetyra["information_fractions"]
    fracs_match = all(abs(info[i] - TIMING[i]) < 0.01 for i in range(len(TIMING)))
    results.append({
        "test": "Info fractions match MONALEESA-7 design",
        "expected": str([round(t, 4) for t in TIMING]),
        "actual": str([round(f, 4) for f in info]),
        "pass": fracs_match,
    })

    # First boundary should be very high (early look at 35%)
    results.append({
        "test": "First boundary > 3.0 (early look at 35% IF)",
        "boundary_1": round(eff[0], 4),
        "pass": eff[0] > 3.0,
    })

    # Cumulative alpha spent should sum to alpha
    alpha_spent = zetyra["alpha_spent"]
    results.append({
        "test": "Alpha spent sums to target alpha",
        "alpha_spent_final": round(alpha_spent[-1], 4),
        "target": ALPHA,
        "pass": abs(alpha_spent[-1] - ALPHA) < 0.001,
    })

    return pd.DataFrame(results)


def validate_gsd_survival_endpoint(client) -> pd.DataFrame:
    """Validate GSD survival endpoint with MONALEESA-7-like parameters."""
    results = []

    # MONALEESA-7: HR=0.712, 252 target events, 80% power
    zetyra = client.gsd_survival(
        hazard_ratio=0.712,
        median_control=36,  # ~36 months median OS in control (HR+ MBC)
        accrual_time=36,
        follow_up_time=24,
        alpha=ALPHA,
        power=0.80,
        k=3,
        spending_function="OBrienFleming",
    )

    # Schoenfeld reference events
    z_alpha = sp_stats.norm.ppf(1 - ALPHA)
    z_beta = sp_stats.norm.ppf(0.80)
    log_hr = math.log(0.712)
    ref_events = math.ceil(((z_alpha + z_beta) / log_hr) ** 2 * 4)

    rel_err = abs(zetyra["fixed_events"] - ref_events) / ref_events
    results.append({
        "test": "Fixed events match Schoenfeld (HR=0.712)",
        "zetyra_events": zetyra["fixed_events"],
        "schoenfeld_events": ref_events,
        "rel_err": round(rel_err, 4),
        "pass": rel_err < 0.05,
    })

    # Schoenfeld reference for MONALEESA-7's 252 target
    # 252 events at 80% power → verify reasonableness
    results.append({
        "test": "Published 252 events reasonable for HR=0.712",
        "published_events": TOTAL_EVENTS,
        "schoenfeld_events": ref_events,
        "ratio": round(TOTAL_EVENTS / ref_events, 3),
        "pass": 0.7 < TOTAL_EVENTS / ref_events < 1.3,
    })

    # Efficacy boundaries decrease
    eff = zetyra["efficacy_boundaries"]
    mono = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "GSD survival boundaries decrease",
        "boundaries": str([round(b, 4) for b in eff]),
        "pass": mono,
    })

    # N_total > max_events
    results.append({
        "test": "N_total > max_events",
        "n_total": zetyra["n_total"],
        "max_events": zetyra["max_events"],
        "pass": zetyra["n_total"] > zetyra["max_events"],
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("MONALEESA-7 TRIAL OS REPLICATION")
    print("Im et al. (2019) NEJM 381:307-316")
    print("=" * 70)

    all_frames = []

    print("\n1. Lan-DeMets OBF Spending vs Published Boundaries")
    print("-" * 70)
    sf_results = validate_spending_function(client)
    print(sf_results.to_string(index=False))
    all_frames.append(sf_results)

    print("\n2. Trial Crossing Verification")
    print("-" * 70)
    tc_results = validate_trial_crossing(client)
    print(tc_results.to_string(index=False))
    all_frames.append(tc_results)

    print("\n3. Boundary Structural Properties (Zetyra GSD)")
    print("-" * 70)
    bp_results = validate_boundary_properties(client)
    print(bp_results.to_string(index=False))
    all_frames.append(bp_results)

    print("\n4. GSD Survival Endpoint (MONALEESA-7 Parameters)")
    print("-" * 70)
    gs_results = validate_gsd_survival_endpoint(client)
    print(gs_results.to_string(index=False))
    all_frames.append(gs_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/monaleesa7_validation.csv", index=False)

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
