#!/usr/bin/env python3
"""
Validate the Sample Size Calculator endpoints (Free Tier).

Three endpoints cover the public "Sample Size Calculator" landing-page
advertisement of "power analysis for continuous, binary, and time-to-event":

  * /api/v1/validation/sample-size/continuous  — two-sample normal approx
  * /api/v1/validation/sample-size/binary      — two-proportion with arcsine h
  * /api/v1/validation/sample-size/survival    — Schoenfeld log-rank formula

This script validates:
  Continuous
    1. n1 formula matches closed-form 2 * ((z_a + z_b) * sd / delta)^2
    2. Reference case vs statsmodels NormalIndPower (d=0.5, alpha=0.05, power=0.80)
    3. One-sided alpha uses a smaller z_alpha than two-sided (smaller N)
    4. Unequal allocation: r=2 -> n2 = 2*n1
    5. Power monotonicity (0.80 -> 0.90 -> 0.95 -> strictly increasing N)
    6. Alpha monotonicity (0.05 -> 0.01 -> 0.001 -> strictly increasing N)
    7. Effect size scaling: halving delta quadruples N (asymptotic)
    8. Input guard: mean1 == mean2 -> 4xx
  Binary
    9. Cohen's h = 2*arcsin(sqrt(p2)) - 2*arcsin(sqrt(p1))
   10. Reference case p1=0.3 p2=0.5 alpha=0.05 power=0.80 -> ~93/arm
   11. Rate-swap symmetry: (0.3, 0.5) == (0.5, 0.3) give same N
   12. Rare-event penalty: (0.01, 0.05) needs more N than (0.10, 0.50)
   13. Unequal allocation: n2 = r * n1 invariant
   14. Input guard: p1 == p2 -> 4xx
  Survival
   15. Schoenfeld events: d = ((z_a + z_b) / log(HR))^2 * (1+r)^2 / r
   16. Reference case HR=0.7 median=12mo -> events ~247, N ~498
   17. HR monotonicity: HR=0.9 needs more events than HR=0.5 (toward null)
   18. Power monotonicity: 0.80 -> 0.90 -> more events
   19. Unequal allocation r=2 needs more events than r=1 (loss of balance)
   20. Input guard: HR == 1.0 -> 4xx

References:
  * Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
  * Schoenfeld (1981) "The asymptotic properties of nonparametric tests for
    comparing survival distributions." Biometrika, 68, 316-319.
  * Chow, Shao, Wang, Lokhnygina (2018) "Sample Size Calculations in Clinical
    Research," 3rd ed., CRC Press.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


# ─── Reference implementations ────────────────────────────────────────

def ref_continuous_n1(mean1, mean2, sd, alpha, power, ratio, two_sided):
    """Closed-form normal-approx two-sample N per group (arm 1)."""
    z_a = sp_stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_b = sp_stats.norm.ppf(power)
    delta = abs(mean2 - mean1)
    n1 = (1 + 1 / ratio) * ((z_a + z_b) * sd / delta) ** 2
    return int(np.ceil(n1))


def ref_binary_h(p1, p2):
    """Cohen's h via arcsine transformation."""
    return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))


def ref_survival_events(hazard_ratio, alpha, power, r):
    """Schoenfeld required events for log-rank test (two-sided)."""
    log_hr = np.log(hazard_ratio)
    z_a = sp_stats.norm.ppf(1 - alpha / 2)
    z_b = sp_stats.norm.ppf(power)
    events = ((z_a + z_b) / log_hr) ** 2 * (1 + r) ** 2 / r
    return int(np.ceil(events))


# ─── Continuous tests ─────────────────────────────────────────────────

def validate_continuous_formula(client) -> pd.DataFrame:
    """n1 matches closed-form normal-approx across a range of inputs."""
    cases = [
        dict(mean1=0.0, mean2=0.5, sd=1.0, alpha=0.05, power=0.80, ratio=1.0, two_sided=True),
        dict(mean1=0.0, mean2=0.3, sd=1.0, alpha=0.05, power=0.80, ratio=1.0, two_sided=True),
        dict(mean1=10.0, mean2=12.0, sd=4.0, alpha=0.05, power=0.90, ratio=1.0, two_sided=True),
        dict(mean1=0.0, mean2=0.5, sd=1.0, alpha=0.01, power=0.80, ratio=1.0, two_sided=True),
        dict(mean1=0.0, mean2=0.5, sd=1.0, alpha=0.05, power=0.80, ratio=2.0, two_sided=True),
    ]
    rows = []
    for c in cases:
        resp = client.sample_size_continuous(**c)
        api_n1 = resp["n1"]
        ref_n1 = ref_continuous_n1(**c)
        rows.append({
            "test": f"n1 formula: μ1={c['mean1']} μ2={c['mean2']} sd={c['sd']} α={c['alpha']} 1-β={c['power']} r={c['ratio']}",
            "api_n1": api_n1,
            "ref_n1": ref_n1,
            "deviation": abs(api_n1 - ref_n1),
            "pass": abs(api_n1 - ref_n1) <= 1,
        })
    return pd.DataFrame(rows)


def validate_continuous_cohen_textbook(client) -> pd.DataFrame:
    """Reference Cohen's d values against the textbook two-sample normal-approx.

    The closed-form sample size per arm for two-sample comparison of means
    under a known σ (normal approx, two-sided α) is:
        n_per_arm = ((z_{α/2} + z_β) / d)^2
    where d = |μ2 - μ1| / σ is Cohen's d. At α=0.05, power=0.80:
        d=0.2 (small)  -> ~393/arm
        d=0.5 (medium) -> ~63/arm
        d=0.8 (large)  -> ~25/arm
    (Cohen 1988, Table 2.4.1, normal-approx formulation.)
    """
    cases = [(0.2, 393), (0.5, 63), (0.8, 25)]
    rows = []
    for d, expected_n in cases:
        resp = client.sample_size_continuous(
            mean1=0.0, mean2=d, sd=1.0, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
        )
        rows.append({
            "test": f"Cohen's d={d} -> ~{expected_n}/arm (tol ±1)",
            "api_n_per_arm": resp["n1"],
            "expected_n_per_arm": expected_n,
            "deviation": abs(resp["n1"] - expected_n),
            "pass": abs(resp["n1"] - expected_n) <= 1,
        })
    return pd.DataFrame(rows)


def validate_continuous_one_vs_two_sided(client) -> pd.DataFrame:
    """One-sided alpha requires smaller N than two-sided at the same power."""
    common = dict(mean1=0.0, mean2=0.5, sd=1.0, alpha=0.05, power=0.80, ratio=1.0)
    two = client.sample_size_continuous(**common, two_sided=True)
    one = client.sample_size_continuous(**common, two_sided=False)
    return pd.DataFrame([{
        "test": "One-sided α < two-sided α at same nominal level",
        "n_two_sided": two["n_total"],
        "n_one_sided": one["n_total"],
        "pass": one["n_total"] < two["n_total"],
    }])


def validate_continuous_unequal_allocation(client) -> pd.DataFrame:
    """Unequal allocation ratio=2: n2 = 2*n1 exactly (modulo ceiling)."""
    rows = []
    for r in (1.0, 1.5, 2.0, 3.0):
        resp = client.sample_size_continuous(
            mean1=0.0, mean2=0.5, sd=1.0, alpha=0.05, power=0.80, ratio=r, two_sided=True
        )
        expected_n2 = int(np.ceil(r * resp["n1"]))
        rows.append({
            "test": f"Allocation ratio r={r}: n2 = r * n1 (modulo ceiling)",
            "n1": resp["n1"], "n2": resp["n2"], "expected_n2": expected_n2,
            # Ceiling is applied independently to n1 and n2 server-side, so
            # allow ±1 deviation.
            "pass": abs(resp["n2"] - expected_n2) <= 1,
        })
    return pd.DataFrame(rows)


def validate_continuous_power_monotonicity(client) -> pd.DataFrame:
    """Higher power -> larger N, strictly."""
    ns = []
    for pw in (0.80, 0.90, 0.95):
        resp = client.sample_size_continuous(
            mean1=0.0, mean2=0.5, sd=1.0, alpha=0.05, power=pw, ratio=1.0, two_sided=True
        )
        ns.append((pw, resp["n_total"]))
    rows = [{"test": f"power={pw} -> N_total", "N": n, "pass": True} for pw, n in ns]
    rows.append({
        "test": "Strictly increasing: N(0.80) < N(0.90) < N(0.95)",
        "N": None,
        "pass": ns[0][1] < ns[1][1] < ns[2][1],
    })
    return pd.DataFrame(rows)


def validate_continuous_alpha_monotonicity(client) -> pd.DataFrame:
    """Smaller α -> larger N, strictly."""
    ns = []
    for a in (0.05, 0.01, 0.001):
        resp = client.sample_size_continuous(
            mean1=0.0, mean2=0.5, sd=1.0, alpha=a, power=0.80, ratio=1.0, two_sided=True
        )
        ns.append((a, resp["n_total"]))
    rows = [{"test": f"α={a} -> N_total", "N": n, "pass": True} for a, n in ns]
    rows.append({
        "test": "Strictly increasing: N(0.05) < N(0.01) < N(0.001)",
        "N": None,
        "pass": ns[0][1] < ns[1][1] < ns[2][1],
    })
    return pd.DataFrame(rows)


def validate_continuous_effect_size_scaling(client) -> pd.DataFrame:
    """Halving δ should quadruple N (asymptotic, up to ceiling tolerance)."""
    n_big = client.sample_size_continuous(
        mean1=0.0, mean2=1.0, sd=1.0, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
    )["n_total"]
    n_half = client.sample_size_continuous(
        mean1=0.0, mean2=0.5, sd=1.0, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
    )["n_total"]
    ratio_obs = n_half / n_big
    return pd.DataFrame([{
        "test": "δ halved -> N ≈ 4× larger (tol ±2% of 4.0)",
        "n_delta_1.0": n_big,
        "n_delta_0.5": n_half,
        "ratio_obs": round(ratio_obs, 4),
        "pass": abs(ratio_obs - 4.0) / 4.0 < 0.02,
    }])


def validate_continuous_guards(client) -> pd.DataFrame:
    """Backend rejects invalid inputs with 4xx."""
    cases = [
        ("Guard: mean1 == mean2 rejected",
         dict(mean1=0.5, mean2=0.5, sd=1.0, alpha=0.05, power=0.80, ratio=1.0, two_sided=True)),
    ]
    rows = []
    for name, payload in cases:
        resp = client.sample_size_continuous_raw(**payload)
        rows.append({
            "test": name,
            "status_code": resp.status_code,
            "pass": 400 <= resp.status_code < 500,
        })
    return pd.DataFrame(rows)


# ─── Binary tests ─────────────────────────────────────────────────────

def validate_binary_h_formula(client) -> pd.DataFrame:
    """Cohen's h matches 2*arcsin(√p2) - 2*arcsin(√p1)."""
    cases = [(0.30, 0.50), (0.10, 0.30), (0.50, 0.70), (0.01, 0.05), (0.80, 0.90)]
    rows = []
    for p1, p2 in cases:
        resp = client.sample_size_binary(
            p1=p1, p2=p2, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
        )
        ref_h = ref_binary_h(p1, p2)
        api_h = resp["effect_size_h"]
        rows.append({
            "test": f"h formula: p1={p1} p2={p2}",
            "api_h": round(api_h, 6),
            "ref_h": round(ref_h, 6),
            "deviation": abs(api_h - ref_h),
            "pass": abs(api_h - ref_h) < 1e-5,
        })
    return pd.DataFrame(rows)


def validate_binary_reference_case(client) -> pd.DataFrame:
    """Canonical p1=0.3 p2=0.5 α=0.05 power=0.80 reference case."""
    resp = client.sample_size_binary(
        p1=0.30, p2=0.50, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
    )
    # Expected ~93/arm via the pooled-variance two-proportion formula
    expected_lo, expected_hi = 85, 100
    return pd.DataFrame([{
        "test": "p1=0.30, p2=0.50, α=0.05, 1-β=0.80 -> ~93/arm",
        "api_n_per_arm": resp["n1"],
        "expected_range": f"[{expected_lo},{expected_hi}]",
        "pass": expected_lo <= resp["n1"] <= expected_hi,
    }])


def validate_binary_symmetry(client) -> pd.DataFrame:
    """Swapping p1 and p2 gives the same |h| and same per-arm N under r=1."""
    a = client.sample_size_binary(p1=0.30, p2=0.50, alpha=0.05, power=0.80, ratio=1.0, two_sided=True)
    b = client.sample_size_binary(p1=0.50, p2=0.30, alpha=0.05, power=0.80, ratio=1.0, two_sided=True)
    return pd.DataFrame([{
        "test": "Rate swap (0.3,0.5) <-> (0.5,0.3) -> same per-arm N",
        "n_ab": a["n1"], "n_ba": b["n1"],
        "pass": a["n1"] == b["n1"],
    }])


def validate_binary_rare_event_penalty(client) -> pd.DataFrame:
    """Rare-event contrast (0.01, 0.05) needs more N than mid-range (0.10, 0.50)."""
    rare = client.sample_size_binary(
        p1=0.01, p2=0.05, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
    )
    mid = client.sample_size_binary(
        p1=0.10, p2=0.50, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
    )
    return pd.DataFrame([{
        "test": "Rare events (0.01 vs 0.05) need more N than mid-range (0.10 vs 0.50)",
        "n_rare": rare["n_total"], "n_mid": mid["n_total"],
        "pass": rare["n_total"] > mid["n_total"],
    }])


def validate_binary_unequal_allocation(client) -> pd.DataFrame:
    """n2 = round_up(r * n1) under unequal allocation."""
    rows = []
    for r in (1.0, 1.5, 2.0, 3.0):
        resp = client.sample_size_binary(
            p1=0.30, p2=0.50, alpha=0.05, power=0.80, ratio=r, two_sided=True
        )
        expected_n2 = int(np.ceil(r * resp["n1"]))
        rows.append({
            "test": f"Binary allocation r={r}: n2 ≈ r * n1",
            "n1": resp["n1"], "n2": resp["n2"], "expected_n2": expected_n2,
            "pass": abs(resp["n2"] - expected_n2) <= 1,
        })
    return pd.DataFrame(rows)


def validate_binary_guards(client) -> pd.DataFrame:
    """Backend rejects p1 == p2 with 4xx."""
    resp = client.sample_size_binary_raw(
        p1=0.30, p2=0.30, alpha=0.05, power=0.80, ratio=1.0, two_sided=True
    )
    return pd.DataFrame([{
        "test": "Guard: p1 == p2 rejected",
        "status_code": resp.status_code,
        "pass": 400 <= resp.status_code < 500,
    }])


# ─── Survival tests ───────────────────────────────────────────────────

def validate_survival_schoenfeld(client) -> pd.DataFrame:
    """Events match Schoenfeld formula across HR and power combinations."""
    cases = [
        dict(hazard_ratio=0.7, median_control=12.0, accrual_time=12.0,
             follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, power=0.80,
             allocation_ratio=1.0),
        dict(hazard_ratio=0.5, median_control=12.0, accrual_time=12.0,
             follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, power=0.80,
             allocation_ratio=1.0),
        dict(hazard_ratio=0.7, median_control=12.0, accrual_time=12.0,
             follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, power=0.90,
             allocation_ratio=1.0),
        dict(hazard_ratio=0.7, median_control=12.0, accrual_time=12.0,
             follow_up_time=12.0, dropout_rate=0.0, alpha=0.025, power=0.80,
             allocation_ratio=1.0),
    ]
    rows = []
    for c in cases:
        resp = client.sample_size_survival(**c)
        api_ev = resp["events_required"]
        ref_ev = ref_survival_events(c["hazard_ratio"], c["alpha"], c["power"], c["allocation_ratio"])
        rows.append({
            "test": f"Schoenfeld events: HR={c['hazard_ratio']} α={c['alpha']} 1-β={c['power']}",
            "api_events": api_ev, "ref_events": ref_ev,
            "deviation": abs(api_ev - ref_ev),
            "pass": abs(api_ev - ref_ev) <= 1,
        })
    return pd.DataFrame(rows)


def validate_survival_reference_case(client) -> pd.DataFrame:
    """HR=0.7, median=12mo, 12mo accrual + 12mo follow-up reference case."""
    resp = client.sample_size_survival(
        hazard_ratio=0.7, median_control=12.0, accrual_time=12.0,
        follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, power=0.80,
        allocation_ratio=1.0,
    )
    # Schoenfeld events ~247, N depends on event probability under exponential
    return pd.DataFrame([{
        "test": "HR=0.7 reference: events in [240, 255], N_total in [300, 600]",
        "events_required": resp["events_required"],
        "n_total": resp["n_total"],
        "pass": (240 <= resp["events_required"] <= 255
                 and 300 <= resp["n_total"] <= 600),
    }])


def validate_survival_hr_monotonicity(client) -> pd.DataFrame:
    """Toward-null HR needs more events than away-from-null HR."""
    common = dict(median_control=12.0, accrual_time=12.0, follow_up_time=12.0,
                  dropout_rate=0.0, alpha=0.05, power=0.80, allocation_ratio=1.0)
    hr_09 = client.sample_size_survival(hazard_ratio=0.9, **common)
    hr_05 = client.sample_size_survival(hazard_ratio=0.5, **common)
    return pd.DataFrame([{
        "test": "HR=0.9 (near null) requires more events than HR=0.5",
        "events_hr_0.9": hr_09["events_required"],
        "events_hr_0.5": hr_05["events_required"],
        "pass": hr_09["events_required"] > hr_05["events_required"],
    }])


def validate_survival_power_monotonicity(client) -> pd.DataFrame:
    """Higher power -> more events, strictly."""
    common = dict(hazard_ratio=0.7, median_control=12.0, accrual_time=12.0,
                  follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, allocation_ratio=1.0)
    events = []
    for pw in (0.80, 0.90, 0.95):
        resp = client.sample_size_survival(power=pw, **common)
        events.append((pw, resp["events_required"]))
    rows = [{"test": f"power={pw} -> events", "events": e, "pass": True} for pw, e in events]
    rows.append({
        "test": "Strictly increasing events across power levels",
        "events": None,
        "pass": events[0][1] < events[1][1] < events[2][1],
    })
    return pd.DataFrame(rows)


def validate_survival_unequal_allocation(client) -> pd.DataFrame:
    """r=2 needs more events than r=1 (imbalance penalty)."""
    common = dict(hazard_ratio=0.7, median_control=12.0, accrual_time=12.0,
                  follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, power=0.80)
    r1 = client.sample_size_survival(allocation_ratio=1.0, **common)
    r2 = client.sample_size_survival(allocation_ratio=2.0, **common)
    return pd.DataFrame([{
        "test": "r=2 requires more events than r=1 at same HR/power",
        "events_r1": r1["events_required"],
        "events_r2": r2["events_required"],
        "pass": r2["events_required"] > r1["events_required"],
    }])


def validate_survival_guards(client) -> pd.DataFrame:
    """Backend rejects HR=1.0 with 4xx."""
    resp = client.sample_size_survival_raw(
        hazard_ratio=1.0, median_control=12.0, accrual_time=12.0,
        follow_up_time=12.0, dropout_rate=0.0, alpha=0.05, power=0.80,
        allocation_ratio=1.0,
    )
    return pd.DataFrame([{
        "test": "Guard: HR == 1.0 rejected",
        "status_code": resp.status_code,
        "pass": 400 <= resp.status_code < 500,
    }])


# ─── Main ─────────────────────────────────────────────────────────────

def main(base_url: str = None) -> int:
    client = get_client(base_url)

    suites = [
        ("1. Continuous: n1 closed-form formula", validate_continuous_formula),
        ("2. Continuous: Cohen's d textbook benchmarks (0.2/0.5/0.8)", validate_continuous_cohen_textbook),
        ("3. Continuous: one-sided vs two-sided α", validate_continuous_one_vs_two_sided),
        ("4. Continuous: unequal allocation", validate_continuous_unequal_allocation),
        ("5. Continuous: power monotonicity", validate_continuous_power_monotonicity),
        ("6. Continuous: α monotonicity", validate_continuous_alpha_monotonicity),
        ("7. Continuous: effect-size scaling (δ halved -> 4×N)", validate_continuous_effect_size_scaling),
        ("8. Continuous: input guards", validate_continuous_guards),
        ("9. Binary: Cohen's h arcsine formula", validate_binary_h_formula),
        ("10. Binary: canonical p1=0.30, p2=0.50 reference", validate_binary_reference_case),
        ("11. Binary: rate-swap symmetry", validate_binary_symmetry),
        ("12. Binary: rare-event penalty", validate_binary_rare_event_penalty),
        ("13. Binary: unequal allocation", validate_binary_unequal_allocation),
        ("14. Binary: input guards", validate_binary_guards),
        ("15. Survival: Schoenfeld events formula", validate_survival_schoenfeld),
        ("16. Survival: HR=0.7 reference case", validate_survival_reference_case),
        ("17. Survival: HR monotonicity", validate_survival_hr_monotonicity),
        ("18. Survival: power monotonicity", validate_survival_power_monotonicity),
        ("19. Survival: unequal allocation penalty", validate_survival_unequal_allocation),
        ("20. Survival: input guards", validate_survival_guards),
    ]

    print("=" * 70)
    print("SAMPLE SIZE CALCULATOR VALIDATION (continuous / binary / survival)")
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
        df_out = df.copy()
        df_out.insert(0, "suite", name)
        all_frames.append(df_out)
        if not df["pass"].all():
            all_pass = False

    if all_frames:
        os.makedirs("results", exist_ok=True)
        all_results = pd.concat(all_frames, ignore_index=True, sort=False)
        all_results.to_csv("results/sample_size_validation.csv", index=False)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        return 0
    print("SOME VALIDATIONS FAILED")
    return 1


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(base_url))
