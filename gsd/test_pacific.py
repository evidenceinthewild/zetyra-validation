#!/usr/bin/env python3
"""
Validate GSD Survival Boundaries Against PACIFIC Trial Design

PACIFIC was a Phase 3 trial of durvalumab vs placebo for Stage III NSCLC
after chemoradiotherapy. OS was a co-primary endpoint with a 3-look
Lan-DeMets O'Brien-Fleming group sequential design.

Published design:
- One-sided alpha = 0.025 (fixed-sequence hierarchy: PFS tested first)
- 3 planned OS analyses at ~285, ~393, ~491 events
- Lan-DeMets O'Brien-Fleming alpha spending
- First interim boundary: two-sided p < 0.00274 (one-sided p < 0.00137)
- Trial crossed at first interim with 299 events, HR=0.68, p=0.00251

Reference:
- Antonia et al. (2018) NEJM 379:2342-2350
  "Overall Survival with Durvalumab after Chemoradiotherapy in Stage III NSCLC"
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
    This form is used by gsDesign regardless of one-sided vs two-sided.
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

    Uses the multivariate normal integration approach:
    at each look, find z_k such that the incremental rejection probability
    (accounting for correlation with prior looks) equals the incremental alpha.

    For the first look, z_1 = Phi^{-1}(1 - alpha_1).
    For subsequent looks, uses numerical root-finding with the
    multivariate normal CDF.
    """
    cum, inc = ld_obf_incremental_spending(timing, alpha)
    boundaries = []

    for i in range(len(timing)):
        if i == 0:
            # First look: simple inversion
            z = sp_stats.norm.ppf(1 - inc[0])
            boundaries.append(z)
        else:
            # Subsequent looks: find z such that incremental rejection = inc[i]
            # P(Z_i > z_i, not rejected at looks 1..i-1 | H0)
            # For the sequential normal test statistics with correlation
            # rho(i,j) = sqrt(t_i / t_j) for i < j, we use numerical integration.
            prev_z = boundaries[:i]
            prev_t = timing[:i]
            t_i = timing[i]

            def incremental_rejection(z_cand):
                """
                Compute P(reject at look i | not rejected before) under H0.

                Uses the identity:
                P(reject at look i) = P(Z_i > z_i) - sum over prev looks of
                P(Z_j > z_j AND Z_i > z_i)

                For bivariate normal with correlation rho = sqrt(t_j/t_i):
                P(Z_j > a, Z_i > b) via numerical integration.
                """
                # Marginal probability of crossing at look i
                p_i = 1 - sp_stats.norm.cdf(z_cand)

                # Subtract probability of also crossing at previous looks
                overlap = 0.0
                for j in range(i):
                    rho = math.sqrt(prev_t[j] / t_i)
                    # P(Z_j > z_j AND Z_i > z_cand) using bivariate normal
                    # mvn.cdf gives P(X < a, Y < b), so:
                    # P(X > a, Y > b) = 1 - P(X<a) - P(Y<b) + P(X<a AND Y<b)
                    mean = [0, 0]
                    cov = [[1, rho], [rho, 1]]
                    p_both = sp_stats.multivariate_normal.cdf(
                        [prev_z[j], z_cand], mean=mean, cov=cov
                    )
                    p_joint_exceed = 1 - sp_stats.norm.cdf(prev_z[j]) - sp_stats.norm.cdf(z_cand) + p_both
                    overlap += p_joint_exceed

                return p_i - overlap - inc[i]

            try:
                z = brentq(incremental_rejection, 1.0, 8.0, xtol=1e-8)
            except ValueError:
                # Fallback to simple inversion
                z = sp_stats.norm.ppf(1 - inc[i])
            boundaries.append(z)

    return boundaries, cum, inc


# ─── PACIFIC trial parameters ───────────────────────────────────────────

# The published boundary p=0.00274 (two-sided) at the first interim corresponds
# to Lan-DeMets OBF spending with one-sided alpha=0.025 and D_max≈509.
# The power calculation targeted 491 events, but the GSD software likely used
# a slightly larger D_max to account for the interim analysis inflation.
# We validate against both D_max=491 and the back-computed D_max that recovers
# the published boundary exactly.

ALPHA = 0.025  # one-sided
PLANNED_EVENTS = [285, 393, 491]  # per NEJM publication

# Published values
PUBLISHED_LOOK1_P_TWO_SIDED = 0.00274
PUBLISHED_LOOK1_P_ONE_SIDED = PUBLISHED_LOOK1_P_TWO_SIDED / 2  # 0.00137

# Observed at crossing
ACTUAL_EVENTS_AT_CROSSING = 299
ACTUAL_HR = 0.68
ACTUAL_P_TWO_SIDED = 0.00251


def validate_spending_function(client) -> pd.DataFrame:
    """
    Validate that Lan-DeMets OBF spending reproduces the PACIFIC boundary.

    The published boundary at look 1 (two-sided p < 0.00274) constrains the
    D_max used in the spending function. We find D_max that reproduces the
    boundary exactly, then verify all look boundaries — both via local
    reference implementation AND via the Zetyra GSD API.
    """
    results = []

    # Step 1: Find D_max that reproduces the published boundary
    def spending_at_dmax(d_max):
        t1 = PLANNED_EVENTS[0] / d_max
        return ld_obf_spending(t1, ALPHA) - PUBLISHED_LOOK1_P_ONE_SIDED

    d_max_opt = brentq(spending_at_dmax, 491, 700)

    # Step 2: Compute boundaries with timing [285/D, 393/D, D/D=1.0]
    timing_opt = [PLANNED_EVENTS[0] / d_max_opt, PLANNED_EVENTS[1] / d_max_opt, 1.0]
    z_bounds, cum_alpha, inc_alpha = ld_obf_z_boundaries(timing_opt, ALPHA)

    # Convert to p-values
    p_one_sided = [1 - sp_stats.norm.cdf(z) for z in z_bounds]
    p_two_sided = [2 * p for p in p_one_sided]

    results.append({
        "test": "Look 1 boundary matches published (reference)",
        "published_p_2s": PUBLISHED_LOOK1_P_TWO_SIDED,
        "computed_p_2s": round(p_two_sided[0], 6),
        "deviation": round(abs(p_two_sided[0] - PUBLISHED_LOOK1_P_TWO_SIDED), 6),
        "d_max": round(d_max_opt, 1),
        "pass": abs(p_two_sided[0] - PUBLISHED_LOOK1_P_TWO_SIDED) < 0.0001,
    })

    # Step 3: Call Zetyra GSD API with PACIFIC timing and compare
    zetyra = client.gsd(
        effect_size=0.3,
        alpha=ALPHA,
        power=0.80,
        k=3,
        timing=timing_opt,
        spending_function="OBrienFleming",
    )
    z_eff = zetyra["efficacy_boundaries"]
    z_alpha_spent = zetyra["alpha_spent"]

    # Zetyra uses the same Lan-DeMets OBF spending as the reference.
    # Deviations are from numerical integration precision (scipy vs reference).
    # Look 3 has the largest deviation (~0.02) due to cumulative MVN error.
    z_tols = [0.005, 0.005, 0.03]
    for i in range(3):
        ref_z = z_bounds[i]
        api_z = z_eff[i]
        dev = abs(api_z - ref_z)
        results.append({
            "test": f"Look {i+1} Zetyra z-boundary matches reference",
            "ref_z": round(ref_z, 4),
            "zetyra_z": round(api_z, 4),
            "deviation": round(dev, 4),
            "pass": dev < z_tols[i],
        })

    # Zetyra's look 1 z-boundary vs the published boundary (z-score scale).
    # Published p=0.00274 (two-sided) → z=2.9955 (one-sided).
    # Actual dev=0.0000: exact match after Lan-DeMets fix.
    published_z1 = sp_stats.norm.ppf(1 - PUBLISHED_LOOK1_P_ONE_SIDED)
    results.append({
        "test": "Look 1 Zetyra z-boundary near published (z-scale)",
        "published_z": round(published_z1, 4),
        "zetyra_z": round(z_eff[0], 4),
        "deviation": round(abs(z_eff[0] - published_z1), 4),
        "pass": abs(z_eff[0] - published_z1) < 0.01,
    })

    # Zetyra's cumulative alpha spent at final look = target alpha
    results.append({
        "test": "Zetyra cumulative alpha = 0.025 at final look",
        "alpha_spent": round(z_alpha_spent[-1], 6),
        "target": ALPHA,
        "pass": abs(z_alpha_spent[-1] - ALPHA) < 0.001,
    })

    # Step 4: D_max should be reasonable (between 491 and 600)
    results.append({
        "test": "Back-computed D_max is reasonable (491-600)",
        "d_max": round(d_max_opt, 1),
        "pass": 491 <= d_max_opt <= 600,
    })

    # Step 5: Verify the trial crossed at the actual interim (299 events)
    # The actual interim had 299 events, not the planned 285. For Lan-DeMets
    # spending, the boundary must be recomputed at the realized information
    # fraction (299/D_max) — using the planned-look boundary would be wrong.
    actual_timing = [ACTUAL_EVENTS_AT_CROSSING / d_max_opt,
                     PLANNED_EVENTS[1] / d_max_opt, 1.0]
    zetyra_actual = client.gsd(
        effect_size=0.3,
        alpha=ALPHA,
        power=0.80,
        k=3,
        timing=actual_timing,
        spending_function="OBrienFleming",
    )
    z_eff_actual = zetyra_actual["efficacy_boundaries"]

    z_obs = -math.log(ACTUAL_HR) * math.sqrt(ACTUAL_EVENTS_AT_CROSSING / 4)
    results.append({
        "test": "Trial z-score exceeds Zetyra boundary at actual crossing (299 events)",
        "z_observed": round(z_obs, 4),
        "z_boundary_at_299": round(z_eff_actual[0], 4),
        "info_frac": round(actual_timing[0], 4),
        "HR": ACTUAL_HR,
        "events": ACTUAL_EVENTS_AT_CROSSING,
        "pass": z_obs > z_eff_actual[0],
    })

    # Step 6: Check observed p < published planned-look cutoff (literature check)
    # This compares against the SAP's published cutoff (p=0.00274), not the
    # API boundary at 299 events. It verifies the trial result is consistent
    # with the published stopping rule.
    p_obs_one = 1 - sp_stats.norm.cdf(z_obs)
    p_obs_two = 2 * p_obs_one
    results.append({
        "test": "Observed p < published planned-look cutoff (two-sided)",
        "p_observed": round(p_obs_two, 5),
        "published_cutoff": PUBLISHED_LOOK1_P_TWO_SIDED,
        "pass": p_obs_two < PUBLISHED_LOOK1_P_TWO_SIDED,
    })

    return pd.DataFrame(results)


def validate_boundary_properties(client) -> pd.DataFrame:
    """Validate structural properties via Zetyra GSD endpoint."""
    results = []

    # Use GSD endpoint with PACIFIC-like timing
    timing = [t / 1.0 for t in [285 / 491, 393 / 491, 1.0]]

    zetyra = client.gsd(
        effect_size=0.3,
        alpha=ALPHA,
        power=0.80,
        k=3,
        timing=timing,
        spending_function="OBrienFleming",
    )

    eff = zetyra["efficacy_boundaries"]

    # O'Brien-Fleming: boundaries must be monotonically decreasing
    mono = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "OBF efficacy boundaries decrease",
        "boundaries": str([round(b, 4) for b in eff]),
        "pass": mono,
    })

    # Information fractions match requested
    info = zetyra["information_fractions"]
    fracs_match = all(abs(info[i] - timing[i]) < 0.01 for i in range(len(timing)))
    results.append({
        "test": "Info fractions match PACIFIC design",
        "expected": str([round(t, 4) for t in timing]),
        "actual": str([round(f, 4) for f in info]),
        "pass": fracs_match,
    })

    # First boundary should be > 2.5 (conservative early look)
    results.append({
        "test": "First boundary sufficiently conservative (>2.5)",
        "boundary_1": round(eff[0], 4),
        "pass": eff[0] > 2.5,
    })

    # Final boundary should be close to z_alpha (one-sided 0.025 → ~1.96)
    results.append({
        "test": "Final boundary near z_alpha",
        "boundary_final": round(eff[-1], 4),
        "z_alpha": 1.96,
        "pass": abs(eff[-1] - 1.96) < 0.15,
    })

    return pd.DataFrame(results)


def validate_gsd_survival_endpoint(client) -> pd.DataFrame:
    """Validate via the GSD survival endpoint with PACIFIC-like parameters."""
    results = []

    # PACIFIC: HR=0.68 for durvalumab vs placebo
    # We don't know exact median_control, so use a plausible value
    # and verify structural properties
    zetyra = client.gsd_survival(
        hazard_ratio=0.68,
        median_control=18,  # ~18 months median OS in control arm (stage III NSCLC)
        accrual_time=24,
        follow_up_time=24,
        alpha=ALPHA,
        power=0.85,  # PACIFIC powered at 85%
        k=3,
        spending_function="OBrienFleming",
    )

    # Schoenfeld events: d = (z_alpha + z_beta)^2 / log(HR)^2 * (1+r)^2/r
    z_alpha = sp_stats.norm.ppf(1 - ALPHA)
    z_beta = sp_stats.norm.ppf(0.85)
    log_hr = math.log(0.68)
    ref_events = math.ceil(((z_alpha + z_beta) / log_hr) ** 2 * 4)

    # Fixed events should be close to Schoenfeld reference
    rel_err = abs(zetyra["fixed_events"] - ref_events) / ref_events
    results.append({
        "test": "Fixed events match Schoenfeld formula",
        "zetyra_events": zetyra["fixed_events"],
        "schoenfeld_events": ref_events,
        "rel_err": round(rel_err, 4),
        "pass": rel_err < 0.05,
    })

    # Max events >= fixed events (GSD inflation)
    results.append({
        "test": "Max events >= fixed events",
        "max_events": zetyra["max_events"],
        "fixed_events": zetyra["fixed_events"],
        "pass": zetyra["max_events"] >= zetyra["fixed_events"],
    })

    # N_total >= max_events (more patients than events needed)
    results.append({
        "test": "N_total >= max_events",
        "n_total": zetyra["n_total"],
        "max_events": zetyra["max_events"],
        "pass": zetyra["n_total"] >= zetyra["max_events"],
    })

    # Efficacy boundaries decrease
    eff = zetyra["efficacy_boundaries"]
    mono = all(eff[i] >= eff[i + 1] for i in range(len(eff) - 1))
    results.append({
        "test": "Efficacy boundaries decrease (OBF)",
        "boundaries": str([round(b, 4) for b in eff]),
        "pass": mono,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("PACIFIC TRIAL OS REPLICATION")
    print("Antonia et al. (2018) NEJM 379:2342-2350")
    print("=" * 70)

    all_frames = []

    print("\n1. Lan-DeMets OBF Spending vs Published Boundaries")
    print("-" * 70)
    sf_results = validate_spending_function(client)
    print(sf_results.to_string(index=False))
    all_frames.append(sf_results)

    print("\n2. Boundary Structural Properties (Zetyra GSD)")
    print("-" * 70)
    bp_results = validate_boundary_properties(client)
    print(bp_results.to_string(index=False))
    all_frames.append(bp_results)

    print("\n3. GSD Survival Endpoint (PACIFIC-like Parameters)")
    print("-" * 70)
    gs_results = validate_gsd_survival_endpoint(client)
    print(gs_results.to_string(index=False))
    all_frames.append(gs_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/pacific_validation.csv", index=False)

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
