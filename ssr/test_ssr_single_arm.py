#!/usr/bin/env python3
"""
Validate Single-Arm SSR Calculator (Bayesian + Conditional Power)

The single-arm SSR calculator supports Phase II oncology trials with a binary
endpoint (ORR) compared against a fixed historical control rate p0. It supports
two stopping-rule modes:

  * Bayesian — posterior probability + predictive probability of success (PPoS)
  * Conditional power — Mehta-Pocock promising-zone framework

The calculator decouples two thresholds in Bayesian mode:
  * gamma_efficacy — the bar at the interim look for early stopping
  * gamma_final   — the bar at the final analysis (default 1 - alpha)

This script validates:
  1. Analytical posterior probability matches Beta-Binomial conjugate update
     (1 - Beta.cdf(p0, alpha + r, beta + n - r))
  2. Initial sample size matches the one-sample binomial normal approximation
  3. Promising-zone classification follows Mehta-Pocock conventions
  4. N_max cap is enforced (factor and absolute)
  5. Simulation reproducibility (same seed -> same OC table)
  6. Type I error <= alpha (Clopper-Pearson upper bound)
  7. Power monotonicity across the OC table true-rate scenarios
  8. gamma_final separation: raising gamma_efficacy alone does not depress
     final-look power (the bug fix this calculator was built around)
  9. Input guards (422/400 for invalid inputs)
 10. Schema contract — required output fields present

References:
  * Lee & Liu (2008) "A predictive probability design for phase II cancer
    clinical trials." Clinical Trials, 5(2), 93-106.
  * Mehta & Pocock (2011) "Adaptive increase in sample size when interim
    results are promising." Statistics in Medicine, 30(28), 3267-3284.
  * FDA (2019) "Adaptive Designs for Clinical Trials of Drugs and Biologics."
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import (
    mc_rate_upper_bound,
    mc_rate_lower_bound,
    assert_schema,
)
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

POSTERIOR_TOLERANCE = 0.005   # tight for conjugate update
INITIAL_N_TOLERANCE = 1       # ±1 patient on rounding for normal approx
SIM_TOLERANCE = 0.005         # exact match expected with same seed


# ─── Reference implementations ────────────────────────────────────────

def reference_posterior_prob(prior_alpha, prior_beta, events, n, p0):
    """P(p > p0 | data) under Beta-Binomial conjugate update."""
    a = prior_alpha + events
    b = prior_beta + (n - events)
    return float(1.0 - sp_stats.beta.cdf(p0, a, b))


def reference_initial_n_binary(alpha, power, p0, p1):
    """One-sample binomial normal approximation."""
    z_a = sp_stats.norm.ppf(1 - alpha)
    z_b = sp_stats.norm.ppf(power)
    num = z_a * np.sqrt(p0 * (1 - p0)) + z_b * np.sqrt(p1 * (1 - p1))
    n = (num / (p1 - p0)) ** 2
    return int(np.ceil(n))


# ─── Test functions ───────────────────────────────────────────────────

def validate_initial_n(client) -> pd.DataFrame:
    """Initial N matches the one-sample binomial normal approximation."""
    scenarios = [
        {"p0": 0.20, "p1": 0.40, "alpha": 0.025, "power": 0.80},
        {"p0": 0.10, "p1": 0.30, "alpha": 0.05,  "power": 0.80},
        {"p0": 0.30, "p1": 0.50, "alpha": 0.025, "power": 0.90},
        {"p0": 0.15, "p1": 0.35, "alpha": 0.025, "power": 0.80},
    ]
    rows = []
    for s in scenarios:
        resp = client.ssr_single_arm(
            endpoint_type="binary", ssr_method="bayesian",
            p0=s["p0"], p1=s["p1"], alpha=s["alpha"], power=s["power"],
            interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
            gamma_efficacy=0.99, delta_futility=0.10, simulate=False,
        )
        zetyra_n = resp["analytical_results"]["initial_n"]
        ref_n = reference_initial_n_binary(s["alpha"], s["power"], s["p0"], s["p1"])
        deviation = abs(zetyra_n - ref_n)
        rows.append({
            "test": f"Initial N: p0={s['p0']}, p1={s['p1']}, α={s['alpha']}, 1-β={s['power']}",
            "zetyra_n": zetyra_n,
            "ref_n": ref_n,
            "deviation": deviation,
            "pass": deviation <= INITIAL_N_TOLERANCE,
        })
    return pd.DataFrame(rows)


def validate_posterior_probability(client) -> pd.DataFrame:
    """Posterior P(p > p0 | data) at the interim matches Beta CDF formula."""
    scenarios = [
        # interim assumes events ≈ round(p1 * interim_n)
        {"name": "Jeffreys, p0=0.20, p1=0.40, n_int=18",
         "p0": 0.20, "p1": 0.40, "interim_fraction": 0.5,
         "prior_alpha": 0.5, "prior_beta": 0.5,
         "interim_n": 18, "events": round(0.40 * 18)},
        {"name": "Flat, p0=0.30, p1=0.50, n_int=20",
         "p0": 0.30, "p1": 0.50, "interim_fraction": 0.5,
         "prior_alpha": 1.0, "prior_beta": 1.0,
         "interim_n": 20, "events": round(0.50 * 20)},
        {"name": "Custom Beta(2,8), p0=0.10, p1=0.30, n_int=15",
         "p0": 0.10, "p1": 0.30, "interim_fraction": 0.5,
         "prior_alpha": 2.0, "prior_beta": 8.0,
         "interim_n": 15, "events": round(0.30 * 15)},
    ]
    rows = []
    for s in scenarios:
        resp = client.ssr_single_arm(
            endpoint_type="binary", ssr_method="bayesian",
            p0=s["p0"], p1=s["p1"], alpha=0.025, power=0.80,
            interim_n=s["interim_n"],
            prior_alpha=s["prior_alpha"], prior_beta=s["prior_beta"],
            gamma_efficacy=0.99, delta_futility=0.10, simulate=False,
        )
        ar = resp["analytical_results"]
        zetyra_pp = ar["posterior_probability"]
        ref_pp = reference_posterior_prob(
            s["prior_alpha"], s["prior_beta"], s["events"], s["interim_n"], s["p0"],
        )
        deviation = abs(zetyra_pp - ref_pp)
        rows.append({
            "test": s["name"],
            "zetyra_pp": round(zetyra_pp, 4),
            "ref_pp": round(ref_pp, 4),
            "deviation": round(deviation, 4),
            "pass": deviation <= POSTERIOR_TOLERANCE,
        })
    return pd.DataFrame(rows)


def validate_promising_zone(client) -> pd.DataFrame:
    """
    CP-mode zone classification follows Mehta-Pocock thresholds (4 zones:
    favorable, promising, unfavorable, futility).

    The analytical interim assumes events = round(p1 * interim_n), pinning
    the planned CP near the design power. We exercise all four zones by
    SLIDING the zone thresholds around that fixed planned CP — the cleanest
    way to test the classifier's contract without confounding the design.
    """
    base = dict(
        endpoint_type="binary", ssr_method="conditional_power",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80, interim_fraction=0.5,
        n_max_factor=1.5, simulate=False,
    )

    # Probe planned CP once so we can place thresholds correctly.
    probe = client.ssr_single_arm(
        **base,
        cp_futility=0.10, cp_promising_lower=0.30, cp_promising_upper=0.50,
    )
    cp_planned = probe["analytical_results"]["conditional_power_planned"]

    # Place all four zones around cp_planned. Each scenario puts cp_planned
    # in exactly one zone:
    eps = 0.02
    zones_under_test = [
        ("favorable: upper < cp_planned",
         dict(cp_futility=0.10, cp_promising_lower=0.20,
              cp_promising_upper=max(0.51, cp_planned - eps)),
         "favorable"),
        ("promising: lower < cp_planned < upper",
         dict(cp_futility=0.10, cp_promising_lower=max(0.20, cp_planned - 0.10),
              cp_promising_upper=min(0.99, cp_planned + 0.10)),
         "promising"),
        ("unfavorable: futility < cp_planned < lower",
         dict(cp_futility=max(0.05, cp_planned - 0.10),
              cp_promising_lower=min(0.99, cp_planned + 0.05),
              cp_promising_upper=min(0.999, cp_planned + 0.10)),
         "unfavorable"),
        ("futility: cp_planned < futility",
         dict(cp_futility=min(0.99, cp_planned + 0.05),
              cp_promising_lower=min(0.995, cp_planned + 0.10),
              cp_promising_upper=min(0.999, cp_planned + 0.15)),
         "futility"),
    ]
    rows = []
    for label, thresholds, expected_zone in zones_under_test:
        params = {**base, **thresholds}
        resp = client.ssr_single_arm(**params)
        ar = resp["analytical_results"]
        rows.append({
            "test": label,
            "zone": ar["zone"],
            "expected": expected_zone,
            "cp_planned": round(ar["conditional_power_planned"], 3),
            "pass": ar["zone"] == expected_zone,
        })
    return pd.DataFrame(rows)


def validate_n_cap(client) -> pd.DataFrame:
    """N_max cap is enforced in both factor and absolute forms."""
    rows = []
    # Factor cap: a healthy effect with a tight 1.5x cap. Initial_N=36 -> cap=54.
    resp_factor = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="conditional_power",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80, interim_fraction=0.5,
        cp_futility=0.10, cp_promising_lower=0.20, cp_promising_upper=0.50,
        n_max_factor=1.5, simulate=False,
    )
    ar = resp_factor["analytical_results"]
    initial_n = ar["initial_n"]
    cap_factor = int(np.ceil(initial_n * 1.5))
    rows.append({
        "test": "Factor cap: N_recalc <= ceil(1.5 * initial_N)",
        "initial_n": initial_n,
        "recalc_n": ar["recalculated_n"],
        "n_max_used": ar["n_max_used"],
        "cap": cap_factor,
        "pass": ar["recalculated_n"] <= cap_factor and ar["n_max_used"] == cap_factor,
    })

    # Absolute cap: choose a cap STRICTLY GREATER than initial_N (the
    # realistic use case). The calculator silently floors n_max_used to
    # initial_N when the absolute cap is below it, so we exercise the
    # documented "tighter than the factor cap" path here.
    # initial_N for this design is 36; n_max_factor=5.0 would allow up to
    # 180; absolute=60 should bind tighter.
    resp_abs = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="conditional_power",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80, interim_fraction=0.5,
        cp_futility=0.10, cp_promising_lower=0.20, cp_promising_upper=0.50,
        n_max_factor=5.0, n_max_absolute=60, simulate=False,
    )
    ar2 = resp_abs["analytical_results"]
    rows.append({
        "test": "Absolute cap (60) overrides factor cap (5.0 * initial_N)",
        "initial_n": ar2["initial_n"],
        "recalc_n": ar2["recalculated_n"],
        "n_max_used": ar2["n_max_used"],
        "cap": 60,
        "pass": ar2["recalculated_n"] <= 60 and ar2["n_max_used"] == 60,
    })
    return pd.DataFrame(rows)


def validate_seed_reproducibility(client) -> pd.DataFrame:
    """Same seed -> bit-identical OC table across all numeric columns."""
    common = dict(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, delta_futility=0.10,
        simulate=True, n_simulations=2000, simulation_seed=12345,
    )
    r1 = client.ssr_single_arm(**common)
    r2 = client.ssr_single_arm(**common)
    oc1 = r1["simulation"]["estimates"]["oc_table"]
    oc2 = r2["simulation"]["estimates"]["oc_table"]

    # Every numeric column in every row must match exactly.
    numeric_cols = [
        "power", "type1_error", "expected_n",
        "pr_efficacy_stop", "pr_futility_stop", "pr_n_hits_cap",
        "n_p10", "n_p50", "n_p90",
    ]
    rows = []
    for col in numeric_cols:
        max_dev = 0.0
        all_present = True
        for a, b in zip(oc1, oc2):
            va, vb = a.get(col), b.get(col)
            if va is None or vb is None:
                # type1_error is None for non-null rows; skip those.
                if va is None and vb is None:
                    continue
                all_present = False
                break
            max_dev = max(max_dev, abs(float(va) - float(vb)))
        rows.append({
            "test": f"Seed reproducibility: {col}",
            "max_deviation": max_dev,
            "pass": all_present and max_dev <= SIM_TOLERANCE,
        })
    # Also check top-level summary metrics.
    for k in ("type1_error", "power"):
        v1 = r1["simulation"].get(k)
        v2 = r2["simulation"].get(k)
        rows.append({
            "test": f"Seed reproducibility: simulation.{k}",
            "max_deviation": abs((v1 or 0) - (v2 or 0)),
            "pass": v1 == v2,
        })
    return pd.DataFrame(rows)


def validate_type1_error(client) -> pd.DataFrame:
    """
    Type I error at a calibrated configuration must respect the nominal α.

    Important: this calculator does NOT auto-calibrate γ_efficacy to α —
    the user owns the threshold choice. We test calibration where the
    user has chosen thresholds that should bound T1E, and separately
    REPORT (without failing) the T1E at the default γ_eff=0.99 / γ_final=0.975
    so reviewers can see the operating-characteristic trade-off.
    """
    n_sims = 5000
    alpha = 0.025
    rows = []

    # Calibrated configuration — γ_eff and γ_final both raised so T1E lies under α.
    calibrated = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=alpha, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.995, gamma_final=0.99, delta_futility=0.10,
        simulate=True, n_simulations=n_sims, simulation_seed=2026,
    )
    null_row = next(r for r in calibrated["simulation"]["estimates"]["oc_table"]
                    if r["true_rate"] == 0.20)
    t1e_cal = null_row["type1_error"]
    ub_cal = mc_rate_upper_bound(t1e_cal, n_sims, confidence=0.99)
    rows.append({
        "test": "Calibrated (γ_eff=0.995, γ_final=0.99): T1E ≤ α (99% CP UB)",
        "alpha": alpha,
        "observed_t1e": t1e_cal,
        "cp_upper_99ci": round(ub_cal, 4),
        "pass": ub_cal <= alpha,
    })

    # Default configuration — recorded for visibility, not for failure.
    default = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=alpha, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, gamma_final=0.975, delta_futility=0.10,
        simulate=True, n_simulations=n_sims, simulation_seed=2026,
    )
    null_row_d = next(r for r in default["simulation"]["estimates"]["oc_table"]
                      if r["true_rate"] == 0.20)
    t1e_d = null_row_d["type1_error"]
    ub_d = mc_rate_upper_bound(t1e_d, n_sims, confidence=0.99)
    # Loose envelope on the POINT ESTIMATE (not the UB) — defaults sit
    # modestly above α; this documents the operating-characteristic cost
    # of using single-α-derived γ_final without raising γ_efficacy.
    rows.append({
        "test": "Default (γ_eff=0.99, γ_final=0.975): point T1E ≤ α + 0.02",
        "alpha": alpha,
        "observed_t1e": t1e_d,
        "cp_upper_99ci": round(ub_d, 4),
        "pass": t1e_d <= alpha + 0.02,
    })
    return pd.DataFrame(rows)


def validate_power_monotonicity(client) -> pd.DataFrame:
    """Power increases monotonically across the OC table true-rate scenarios."""
    resp = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, gamma_final=0.975, delta_futility=0.10,
        simulate=True, n_simulations=3000, simulation_seed=99,
    )
    oc = sorted(resp["simulation"]["estimates"]["oc_table"], key=lambda r: r["true_rate"])
    powers = [r["power"] for r in oc]
    # Monotone non-decreasing across rates from p0 to p1+
    monotone = all(powers[i] <= powers[i + 1] + 0.02 for i in range(len(powers) - 1))
    return pd.DataFrame([{
        "test": "Power monotone non-decreasing in true rate",
        "rates": [r["true_rate"] for r in oc],
        "powers": [round(p, 3) for p in powers],
        "pass": monotone,
    }])


def validate_gamma_final_decoupling(client) -> pd.DataFrame:
    """
    Regression test for the gamma_final fix.

    Decoupling γ_efficacy (interim early-stop bar) from γ_final (final-look
    success bar) means that raising γ_efficacy should:
      (a) DECREASE the interim efficacy-stop probability, AND
      (b) NOT collapse final-look successes — they should partially
          compensate via the (more lenient) γ_final bar at the final look.

    This is sharper than checking total rejection probability alone: a bug
    that conflated the two thresholds would tank both early stops AND final
    successes when γ_efficacy went up. We assert the directional split.
    """
    n_sims = 4000
    common = dict(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        delta_futility=0.10, gamma_final=0.975,
        simulate=True, n_simulations=n_sims, simulation_seed=7,
    )

    runs = {}
    for gamma_eff in (0.95, 0.99):
        resp = client.ssr_single_arm(**common, gamma_efficacy=gamma_eff)
        oc = resp["simulation"]["estimates"]["oc_table"]
        alt_row = next(r for r in oc if r["true_rate"] == 0.40)
        # "final success" rate = total power minus interim early-efficacy stops
        # (both count as rejections; we want the portion attributable to the
        # final-look γ_final bar specifically).
        final_success_rate = alt_row["power"] - alt_row["pr_efficacy_stop"]
        runs[gamma_eff] = {
            "power": alt_row["power"],
            "early_stop": alt_row["pr_efficacy_stop"],
            "final_success": final_success_rate,
            "futility": alt_row["pr_futility_stop"],
        }

    low, high = runs[0.95], runs[0.99]
    rows = [
        {"test": "γ_eff 0.95 -> 0.99 reduces interim efficacy stops",
         "low_early_stop": round(low["early_stop"], 3),
         "high_early_stop": round(high["early_stop"], 3),
         "delta": round(high["early_stop"] - low["early_stop"], 3),
         "pass": high["early_stop"] < low["early_stop"]},
        {"test": "γ_eff 0.95 -> 0.99 does NOT collapse final-look successes",
         "low_final_success": round(low["final_success"], 3),
         "high_final_success": round(high["final_success"], 3),
         "delta": round(high["final_success"] - low["final_success"], 3),
         # final_success should INCREASE (fewer trials stopped early -> more
         # reach final). Tolerate a tiny dip from MC noise.
         "pass": high["final_success"] >= low["final_success"] - 0.02},
        {"test": "Total power preserved (>= 70% in both, spread <= 10pp)",
         "low_power": round(low["power"], 3),
         "high_power": round(high["power"], 3),
         "delta": round(high["power"] - low["power"], 3),
         "pass": (low["power"] >= 0.70 and high["power"] >= 0.70
                  and abs(low["power"] - high["power"]) <= 0.10)},
    ]
    return pd.DataFrame(rows)


def validate_input_guards(client) -> pd.DataFrame:
    """Backend rejects invalid inputs with 4xx."""
    rows = []
    cases = [
        ("Guard: p1 <= p0 rejected",
         dict(endpoint_type="binary", ssr_method="bayesian",
              p0=0.40, p1=0.20, alpha=0.025, power=0.80,
              interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
              gamma_efficacy=0.99, delta_futility=0.10, simulate=False)),
        ("Guard: alpha out of range (0)",
         dict(endpoint_type="binary", ssr_method="bayesian",
              p0=0.20, p1=0.40, alpha=0.0, power=0.80,
              interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
              gamma_efficacy=0.99, delta_futility=0.10, simulate=False)),
        ("Guard: gamma_final out of (0.5, 1)",
         dict(endpoint_type="binary", ssr_method="bayesian",
              p0=0.20, p1=0.40, alpha=0.025, power=0.80,
              interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
              gamma_efficacy=0.99, gamma_final=1.5, delta_futility=0.10,
              simulate=False)),
        ("Guard: prior_alpha <= 0",
         dict(endpoint_type="binary", ssr_method="bayesian",
              p0=0.20, p1=0.40, alpha=0.025, power=0.80,
              interim_fraction=0.5, prior_alpha=0.0, prior_beta=0.5,
              gamma_efficacy=0.99, delta_futility=0.10, simulate=False)),
    ]
    for name, payload in cases:
        resp = client.ssr_single_arm_raw(**payload)
        rows.append({
            "test": name,
            "status_code": resp.status_code,
            "pass": 400 <= resp.status_code < 500,
        })
    return pd.DataFrame(rows)


def validate_schema(client) -> pd.DataFrame:
    """Required output fields are present in both modes."""
    rows = []

    bayes = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, delta_futility=0.10, simulate=False,
    )
    ar = bayes["analytical_results"]
    required_bayes = [
        "initial_n", "interim_n", "ssr_method", "posterior_probability",
        "predictive_probability", "gamma_final_used", "prior_description",
        "decision_rule_description", "recalculation_scenarios",
        "regulatory_notes",
    ]
    missing = [k for k in required_bayes if k not in ar]
    rows.append({
        "test": "Bayesian mode: required fields present",
        "missing": missing,
        "pass": not missing,
    })

    cp = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="conditional_power",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80, interim_fraction=0.5,
        cp_futility=0.20, cp_promising_lower=0.36, cp_promising_upper=0.80,
        n_max_factor=1.5, simulate=False,
    )
    ar_cp = cp["analytical_results"]
    required_cp = [
        "initial_n", "interim_n", "ssr_method", "conditional_power",
        "conditional_power_planned", "zone", "z1", "recalculated_n",
        "inflation_factor", "n_capped", "n_max_used",
        "decision_rule_description", "recalculation_scenarios",
    ]
    missing_cp = [k for k in required_cp if k not in ar_cp]
    rows.append({
        "test": "CP mode: required fields present",
        "missing": missing_cp,
        "pass": not missing_cp,
    })

    return pd.DataFrame(rows)


def validate_sap_text_early_stop_clause(client) -> pd.DataFrame:
    """
    Regression: the Bayesian SAP description must explicitly state that an
    early-efficacy stop terminates enrollment at the interim N. Earlier
    versions said "stop for early efficacy" without naming the sample-size
    consequence, leaving readers to assume enrollment continued to the
    planned N.
    """
    resp = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, delta_futility=0.10, simulate=False,
    )
    desc = resp["analytical_results"]["decision_rule_description"]
    interim_n = resp["analytical_results"]["interim_n"]
    rows = [
        {"test": "SAP names interim N as final N on early-efficacy stop",
         "found_clause": "stop for early efficacy" in desc.lower(),
         "names_interim_n": str(interim_n) in desc and "FINAL" in desc.upper(),
         "pass": ("stop for early efficacy" in desc.lower()
                  and str(interim_n) in desc
                  and "FINAL" in desc.upper())},
        {"test": "SAP states N-floor (not reduced below planned N)",
         "found_clause": "not be reduced below" in desc.lower(),
         "names_interim_n": True,
         "pass": "not be reduced below" in desc.lower()},
    ]
    return pd.DataFrame(rows)


def validate_pp_promising_upper_pushes_n_p90(client) -> pd.DataFrame:
    """
    Regression: raising `pp_promising_upper` from 0.50 to 0.70 should keep
    more trials in the SSR zone longer, pushing N_p90 toward the N_max cap.
    With a tunable parameter, the user can now actually utilize the budget.
    """
    common = dict(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.40, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, gamma_final=0.975, delta_futility=0.10,
        n_max_absolute=200,
        simulate=True, n_simulations=3000, simulation_seed=11,
    )
    rows = []
    p_target = 0.30  # Case-B-like: between p0 and p1, where SSR matters most
    n_p90_by_threshold = {}
    for pp_upper in (0.50, 0.70):
        resp = client.ssr_single_arm(**common, pp_promising_upper=pp_upper)
        oc = resp["simulation"]["estimates"]["oc_table"]
        target_row = min(oc, key=lambda r: abs(r["true_rate"] - p_target))
        n_p90_by_threshold[pp_upper] = target_row["n_p90"]
        rows.append({
            "test": f"PP upper={pp_upper}: N_p90 at p≈{p_target}",
            "n_p90": target_row["n_p90"],
            "expected_n": round(target_row["expected_n"], 1),
            "pr_n_hits_cap": round(target_row["pr_n_hits_cap"], 3),
            "pass": True,  # informational rows
        })
    # Real assertion: raising the upper threshold must not reduce N_p90.
    rows.append({
        "test": "Raising PP upper does not shrink N_p90 (budget utilization)",
        "n_p90": n_p90_by_threshold[0.70],
        "expected_n": None,
        "pr_n_hits_cap": None,
        "pass": n_p90_by_threshold[0.70] >= n_p90_by_threshold[0.50],
    })
    return pd.DataFrame(rows)


def validate_n_floor_in_favorable_zone(client) -> pd.DataFrame:
    """
    Regression: every Bayesian sensitivity-table row whose decision is NOT
    a stop ('continue_favorable' or 'continue_ssr') must have
    recalculated_n_per_arm >= initial_n. This catches the bug pattern where
    the favorable zone silently reported N below the planned N.
    """
    resp = client.ssr_single_arm(
        endpoint_type="binary", ssr_method="bayesian",
        p0=0.20, p1=0.35, alpha=0.025, power=0.80,
        interim_fraction=0.5, prior_alpha=0.5, prior_beta=0.5,
        gamma_efficacy=0.99, delta_futility=0.10,
        n_max_absolute=200, simulate=False,
    )
    ar = resp["analytical_results"]
    initial_n = ar["initial_n"]
    rows = []
    for s in ar["recalculation_scenarios"]:
        decision = s.get("decision")
        n = s["recalculated_n_per_arm"]
        is_stop = decision in ("stop_efficacy", "stop_futility")
        # For non-stop rows, N must be >= initial_n.
        ok = is_stop or n >= initial_n
        rows.append({
            "test": f"{s['label']} (decision={decision})",
            "n": n,
            "initial_n": initial_n,
            "is_stop": is_stop,
            "pass": ok,
        })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────

def main(base_url: str = None) -> int:
    client = get_client(base_url)

    suites = [
        ("1. Initial N (Binomial Normal Approximation)", validate_initial_n),
        ("2. Posterior Probability (Beta-Binomial Conjugate)", validate_posterior_probability),
        ("3. Promising Zone Classification (CP mode)", validate_promising_zone),
        ("4. N_max Cap Enforcement", validate_n_cap),
        ("5. Seed Reproducibility", validate_seed_reproducibility),
        ("6. Type I Error Calibration", validate_type1_error),
        ("7. Power Monotonicity", validate_power_monotonicity),
        ("8. gamma_final Decoupling (regression)", validate_gamma_final_decoupling),
        ("9. Input Guards", validate_input_guards),
        ("10. Schema Contracts", validate_schema),
        ("11. SAP early-stop clause (regression)", validate_sap_text_early_stop_clause),
        ("12. pp_promising_upper raises N_p90 (regression)", validate_pp_promising_upper_pushes_n_p90),
        ("13. N-floor in favorable zone (regression)", validate_n_floor_in_favorable_zone),
    ]

    print("=" * 70)
    print("SINGLE-ARM SSR VALIDATION (Bayesian + Conditional Power)")
    print("=" * 70)

    all_pass = True
    all_frames: list[pd.DataFrame] = []
    for name, fn in suites:
        print(f"\n{name}")
        print("-" * 70)
        try:
            df = fn(client)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_pass = False
            continue
        print(df.to_string(index=False))
        # Tag each row with its suite name for the consolidated CSV output.
        df_out = df.copy()
        df_out.insert(0, "suite", name)
        all_frames.append(df_out)
        if not df["pass"].all():
            all_pass = False

    # Save consolidated results to results/ssr_single_arm_validation.csv
    if all_frames:
        os.makedirs("results", exist_ok=True)
        all_results = pd.concat(all_frames, ignore_index=True, sort=False)
        all_results.to_csv("results/ssr_single_arm_validation.csv", index=False)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        return 0
    print("SOME VALIDATIONS FAILED")
    return 1


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(base_url))
