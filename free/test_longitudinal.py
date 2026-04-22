#!/usr/bin/env python3
"""
Validate the Longitudinal / Repeated Measures math shipped in the frontend.

The Sample Size Calculator's longitudinal mode runs client-side — no backend.
This script exercises the actual TypeScript module via Node:

    frontend/src/lib/stats/sample_size.ts
imported by:
    frontend/src/app/calculators/sample-size/page.tsx

Requires Node.js 22+ (native TypeScript strip-types).

This script validates three target quantities — slope, endpoint, and change
from baseline — under two correlation structures (AR(1), CS):

   1. Slope AR(1): exact Var(β̂) via matrix form sᵀΣs / (sᵀs)²
   2. Slope AR(1): matches the simple matrix reference implementation
   3. Slope CS: closed form σ²(1-ρ) · 12 / (m(m²-1))
   4. Slope CS: matches matrix reference
   5. Endpoint (no baseline adjust): effective variance = σ²
   6. Endpoint (ANCOVA-adjusted): effective variance = σ²(1 - ρ²)
   7. Change-from-baseline CS: 2σ²(1 - ρ)
   8. Change-from-baseline AR(1): 2σ²(1 - ρ^(m-1)); m=2 recovers CS
   9. Frison-Pocock (1992) Table 2 ANCOVA vs change-from-baseline benchmark
  10. Slope AR(1) vs shipped frontend was PREVIOUSLY WRONG (asymptotic): this
      suite asserts the new exact form produces Var(β̂) much smaller than the
      old m→∞ limit at m ≥ 10
  11. Monotonicity: slope AR(1) variance decreases with m (at fixed ρ)
  12. Monotonicity: endpoint ANCOVA variance decreases with ρ (variance reduction
      via baseline adjustment)
  13. Monotonicity: change-from-baseline CS variance decreases with ρ
  14. Simulation: AR(1) slope fit from generated data — empirical power
      near target at the planned sample size
  15. Simulation: CS change-from-baseline — empirical power near target

References:
  Diggle, Heagerty, Liang & Zeger (2002) "Analysis of Longitudinal Data" §3.5
  Frison & Pocock (1992) "Repeated measures in clinical trials"
    Stat Med 11:1685-1704, Table 2
  Fitzmaurice, Laird & Ware (2011) "Applied Longitudinal Analysis"
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from common.frontend_bridge import FrontendBridge, resolve_frontend_path


# ─── Reference implementations ────────────────────────────────────────

def ref_ar1_slope_var(m: int, rho: float, sd: float = 1.0) -> float:
    """Exact Var(OLS slope) via matrix form, equally-spaced t=1..m."""
    t = np.arange(1, m + 1)
    s = t - t.mean()
    Sigma = rho ** np.abs(np.subtract.outer(t, t))
    return float(sd * sd * (s @ Sigma @ s) / (s @ s) ** 2)


def ref_cs_slope_var(m: int, rho: float, sd: float = 1.0) -> float:
    """Exact Var(OLS slope) under CS errors, equally-spaced."""
    t = np.arange(1, m + 1)
    s = t - t.mean()
    Sigma = np.eye(m) * (1 - rho) + rho
    return float(sd * sd * (s @ Sigma @ s) / (s @ s) ** 2)


# ─── 1-2. AR(1) slope ─────────────────────────────────────────────────

def validate_slope_ar1_exact(bridge: FrontendBridge) -> pd.DataFrame:
    """Frontend slope AR(1) matches the exact matrix calculation."""
    cases = [(3, 0.5), (3, 0.7), (5, 0.5), (5, 0.9), (10, 0.3), (10, 0.7),
             (20, 0.5), (20, 0.9)]
    rows = []
    for m, rho in cases:
        fe = bridge.call("slope_var_ar1", sd=1.0, rho=rho, m=m)
        ref = ref_ar1_slope_var(m, rho)
        rows.append({
            "test": f"slope AR(1) m={m} ρ={rho}",
            "frontend": round(fe, 8),
            "ref": round(ref, 8),
            "deviation": abs(fe - ref),
            "pass": abs(fe - ref) < 1e-10,
        })
    return pd.DataFrame(rows)


def validate_slope_ar1_sigma_scales(bridge: FrontendBridge) -> pd.DataFrame:
    """Doubling σ quadruples Var(β̂) (since Var ∝ σ²)."""
    base = bridge.call("slope_var_ar1", sd=1.0, rho=0.5, m=10)
    double = bridge.call("slope_var_ar1", sd=2.0, rho=0.5, m=10)
    return pd.DataFrame([{
        "test": "σ doubled -> Var quadruples",
        "base_var": round(base, 8),
        "double_var": round(double, 8),
        "ratio": round(double / base, 6),
        "pass": abs(double / base - 4.0) < 1e-10,
    }])


# ─── 3-4. CS slope ────────────────────────────────────────────────────

def validate_slope_cs_exact(bridge: FrontendBridge) -> pd.DataFrame:
    """Frontend slope CS matches the exact matrix calculation AND the closed form."""
    cases = [(3, 0.3), (3, 0.7), (5, 0.5), (10, 0.5), (10, 0.9), (20, 0.5)]
    rows = []
    for m, rho in cases:
        fe = bridge.call("slope_var_cs", sd=1.0, rho=rho, m=m)
        ref = ref_cs_slope_var(m, rho)
        closed_form = (1 - rho) * 12 / (m * (m * m - 1))
        rows.append({
            "test": f"slope CS m={m} ρ={rho}",
            "frontend": round(fe, 8),
            "matrix_ref": round(ref, 8),
            "closed_form": round(closed_form, 8),
            "pass": abs(fe - ref) < 1e-10 and abs(fe - closed_form) < 1e-10,
        })
    return pd.DataFrame(rows)


def validate_slope_cs_recovers_rho_zero(bridge: FrontendBridge) -> pd.DataFrame:
    """ρ=0 (independent observations) should recover the no-correlation case."""
    rows = []
    for m in (3, 5, 10, 20):
        fe = bridge.call("slope_var_cs", sd=1.0, rho=0.0, m=m)
        # Var = 12 / (m(m²-1)) at ρ=0
        ref = 12.0 / (m * (m * m - 1))
        rows.append({
            "test": f"slope CS ρ=0 m={m}",
            "frontend": round(fe, 8),
            "ref (12/m(m²-1))": round(ref, 8),
            "pass": abs(fe - ref) < 1e-12,
        })
    return pd.DataFrame(rows)


# ─── 5-6. Endpoint ────────────────────────────────────────────────────

def validate_endpoint_unadjusted(bridge: FrontendBridge) -> pd.DataFrame:
    """Endpoint without baseline adjustment: effVar = σ² regardless of ρ."""
    rows = []
    for rho in (0.0, 0.3, 0.7, 0.9):
        fe = bridge.call("longitudinal_effective_variance", input={
            "target": "endpoint", "correlation": "cs",
            "sd": 2.0, "rho": rho, "measurements": 2, "baselineAdjust": False,
        })
        rows.append({
            "test": f"endpoint unadjusted ρ={rho}: effVar = σ² = 4.0",
            "frontend": round(fe, 6),
            "pass": abs(fe - 4.0) < 1e-12,
        })
    return pd.DataFrame(rows)


def validate_endpoint_ancova(bridge: FrontendBridge) -> pd.DataFrame:
    """Endpoint with ANCOVA baseline adjustment: effVar = σ²(1-ρ²)."""
    rows = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        fe = bridge.call("longitudinal_effective_variance", input={
            "target": "endpoint", "correlation": "cs",
            "sd": 1.0, "rho": rho, "measurements": 2, "baselineAdjust": True,
        })
        ref = 1.0 - rho * rho
        rows.append({
            "test": f"endpoint ANCOVA ρ={rho}: effVar = 1-ρ² = {ref:.4f}",
            "frontend": round(fe, 6),
            "ref": round(ref, 6),
            "pass": abs(fe - ref) < 1e-12,
        })
    return pd.DataFrame(rows)


# ─── 7-8. Change from baseline ────────────────────────────────────────

def validate_change_cs(bridge: FrontendBridge) -> pd.DataFrame:
    """Change CS: Var(Y_m - Y_1) = 2σ²(1 - ρ)."""
    rows = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        fe = bridge.call("longitudinal_effective_variance", input={
            "target": "change", "correlation": "cs",
            "sd": 1.0, "rho": rho, "measurements": 5, "baselineAdjust": False,
        })
        ref = 2 * (1 - rho)
        rows.append({
            "test": f"change CS ρ={rho}: effVar = 2(1-ρ) = {ref:.4f}",
            "frontend": round(fe, 6),
            "ref": round(ref, 6),
            "pass": abs(fe - ref) < 1e-12,
        })
    return pd.DataFrame(rows)


def validate_change_ar1(bridge: FrontendBridge) -> pd.DataFrame:
    """Change AR(1): Var(Y_m - Y_1) = 2σ²(1 - ρ^(m-1)). m=2 matches CS."""
    rows = []
    for m, rho in [(2, 0.5), (3, 0.5), (5, 0.5), (10, 0.5), (2, 0.9), (10, 0.9)]:
        fe = bridge.call("longitudinal_effective_variance", input={
            "target": "change", "correlation": "ar1",
            "sd": 1.0, "rho": rho, "measurements": m, "baselineAdjust": False,
        })
        ref = 2 * (1 - rho ** (m - 1))
        rows.append({
            "test": f"change AR(1) m={m} ρ={rho}: effVar = 2(1-ρ^(m-1)) = {ref:.4f}",
            "frontend": round(fe, 6),
            "ref": round(ref, 6),
            "pass": abs(fe - ref) < 1e-12,
        })
    return pd.DataFrame(rows)


# ─── 9. Frison-Pocock reference ───────────────────────────────────────

def validate_frison_pocock(bridge: FrontendBridge) -> pd.DataFrame:
    """Frison & Pocock (1992) Table 2: ANCOVA vs change-from-baseline.

    For the two-group comparison of post-baseline means, FP Table 2 tabulates
    the ratio of required N under various (ρ, n_measurements) vs a simple
    post-only t-test. At ρ=0.5, post-only baseline → 1.0, change-from-baseline
    → 2(1-ρ)=1.0 (same as post-only); ANCOVA → 1-ρ²=0.75 (25% reduction).
    """
    rho = 0.5
    post = bridge.call("longitudinal_effective_variance", input={
        "target": "endpoint", "correlation": "cs",
        "sd": 1.0, "rho": rho, "measurements": 2, "baselineAdjust": False,
    })
    change = bridge.call("longitudinal_effective_variance", input={
        "target": "change", "correlation": "cs",
        "sd": 1.0, "rho": rho, "measurements": 2, "baselineAdjust": False,
    })
    ancova = bridge.call("longitudinal_effective_variance", input={
        "target": "endpoint", "correlation": "cs",
        "sd": 1.0, "rho": rho, "measurements": 2, "baselineAdjust": True,
    })
    rows = [
        {"test": "ρ=0.5 post-only vs change: equal variance (FP Table 2)",
         "post": round(post, 4), "change": round(change, 4),
         "pass": abs(post - change) < 1e-12},
        {"test": "ρ=0.5 ANCOVA 25% reduction over post-only (FP Table 2)",
         "post": round(post, 4), "ancova": round(ancova, 4),
         "ratio": round(ancova / post, 4),
         "pass": abs(ancova / post - 0.75) < 1e-10},
    ]
    return pd.DataFrame(rows)


# ─── 10. Old-vs-new slope AR(1) regression ────────────────────────────

def validate_slope_ar1_not_asymptotic(bridge: FrontendBridge) -> pd.DataFrame:
    """Confirm the shipped formula is NOT the old m→∞ asymptotic
    (1-ρ²)/[m(1-ρ)²], which was off by 2-14× in real-world (m, ρ) regimes."""
    cases = [(10, 0.5), (10, 0.7), (20, 0.5)]
    rows = []
    for m, rho in cases:
        fe = bridge.call("slope_var_ar1", sd=1.0, rho=rho, m=m)
        asymptotic = (1 - rho * rho) / (m * (1 - rho) ** 2)
        ref_exact = ref_ar1_slope_var(m, rho)
        rows.append({
            "test": f"m={m} ρ={rho}: not the old asymptotic",
            "frontend": round(fe, 6),
            "exact_matrix": round(ref_exact, 6),
            "old_asymptotic": round(asymptotic, 6),
            "exact_vs_asymp_ratio": round(ref_exact / asymptotic, 3),
            # frontend should match exact to machine precision and differ
            # meaningfully from the asymptotic
            "pass": (abs(fe - ref_exact) < 1e-10
                     and abs(fe - asymptotic) > 1e-3),
        })
    return pd.DataFrame(rows)


# ─── 11-13. Monotonicity ──────────────────────────────────────────────

def validate_slope_m_monotonicity(bridge: FrontendBridge) -> pd.DataFrame:
    """Slope variance strictly decreases with m (more measurements → tighter slope)."""
    rows = []
    for correlation in ("ar1", "cs"):
        op = "slope_var_ar1" if correlation == "ar1" else "slope_var_cs"
        prev = float("inf")
        all_dec = True
        vals = []
        for m in (3, 5, 10, 20, 50):
            v = bridge.call(op, sd=1.0, rho=0.5, m=m)
            vals.append((m, v))
            if v >= prev:
                all_dec = False
            prev = v
        rows.append({
            "test": f"slope {correlation} variance strictly decreases with m (ρ=0.5)",
            "values": ",".join(f"m={m}:{v:.5f}" for m, v in vals),
            "pass": all_dec,
        })
    return pd.DataFrame(rows)


def validate_endpoint_ancova_rho_monotonicity(bridge: FrontendBridge) -> pd.DataFrame:
    """ANCOVA endpoint variance strictly decreases as ρ grows (up to ρ=1)."""
    prev = float("inf")
    all_dec = True
    vals = []
    for rho in (0.0, 0.2, 0.5, 0.7, 0.9):
        v = bridge.call("longitudinal_effective_variance", input={
            "target": "endpoint", "correlation": "cs",
            "sd": 1.0, "rho": rho, "measurements": 2, "baselineAdjust": True,
        })
        vals.append((rho, v))
        if v >= prev:
            all_dec = False
        prev = v
    return pd.DataFrame([{
        "test": "endpoint ANCOVA variance strictly decreases with ρ",
        "values": ",".join(f"ρ={rho}:{v:.4f}" for rho, v in vals),
        "pass": all_dec,
    }])


def validate_change_cs_rho_monotonicity(bridge: FrontendBridge) -> pd.DataFrame:
    """Change-from-baseline CS variance strictly decreases with ρ."""
    prev = float("inf")
    all_dec = True
    vals = []
    for rho in (0.0, 0.2, 0.5, 0.7, 0.9):
        v = bridge.call("longitudinal_effective_variance", input={
            "target": "change", "correlation": "cs",
            "sd": 1.0, "rho": rho, "measurements": 5, "baselineAdjust": False,
        })
        vals.append((rho, v))
        if v >= prev:
            all_dec = False
        prev = v
    return pd.DataFrame([{
        "test": "change-from-baseline CS variance strictly decreases with ρ",
        "values": ",".join(f"ρ={rho}:{v:.4f}" for rho, v in vals),
        "pass": all_dec,
    }])


# ─── 14. Simulation: AR(1) slope ──────────────────────────────────────

def _simulate_ar1_slope_power(n_per_arm: int, m: int, rho: float, sd: float,
                                delta_slope: float, alpha: float,
                                rng, n_sim: int) -> float:
    """Generate two-arm AR(1) longitudinal data; fit OLS slope per subject;
    test slope difference via two-sample t-test."""
    times = np.arange(1, m + 1, dtype=float)
    tc = times - times.mean()
    denom = (tc ** 2).sum()
    reject = 0
    for _ in range(n_sim):
        slopes_c = np.empty(n_per_arm)
        slopes_t = np.empty(n_per_arm)
        for i in range(n_per_arm):
            eps_c = _draw_ar1(m, rho, sd, rng)
            y_c = eps_c  # slope 0
            slopes_c[i] = (tc * (y_c - y_c.mean())).sum() / denom
            eps_t = _draw_ar1(m, rho, sd, rng)
            y_t = delta_slope * times + eps_t
            slopes_t[i] = (tc * (y_t - y_t.mean())).sum() / denom
        _, p = sp_stats.ttest_ind(slopes_t, slopes_c, equal_var=True)
        if p < alpha:
            reject += 1
    return reject / n_sim


def _draw_ar1(m: int, rho: float, sd: float, rng) -> np.ndarray:
    """Draw an AR(1) error vector of length m with marginal variance sd²."""
    eps = np.empty(m)
    eps[0] = rng.normal(0, sd)
    sig_inn = sd * math.sqrt(1 - rho * rho)
    for j in range(1, m):
        eps[j] = rho * eps[j - 1] + rng.normal(0, sig_inn)
    return eps


def validate_ar1_slope_simulation(bridge: FrontendBridge) -> pd.DataFrame:
    """Empirical power matches target at the planned per-arm N for AR(1) slope."""
    # Target ~80% power to detect slope difference of 0.3 units/visit,
    # m=5, ρ=0.5, σ=1.0, α=0.05
    plan = bridge.call("longitudinal_sample_size", input={
        "target": "slope", "correlation": "ar1",
        "sd": 1.0, "rho": 0.5, "measurements": 5,
        "delta": 0.3, "alpha": 0.05, "power": 0.80,
        "twoSided": True, "ratio": 1.0,
    })
    n_per_arm = plan["n1"]
    rng = np.random.default_rng(0xA2B)
    n_sim = 1500
    emp = _simulate_ar1_slope_power(
        n_per_arm=n_per_arm, m=5, rho=0.5, sd=1.0, delta_slope=0.3,
        alpha=0.05, rng=rng, n_sim=n_sim,
    )
    ci_lo = float(sp_stats.beta.ppf(0.025, emp * n_sim + 0.5,
                                      n_sim - emp * n_sim + 0.5))
    return pd.DataFrame([{
        "test": f"AR(1) slope power: plan N={n_per_arm}/arm",
        "target_power": 0.80,
        "empirical_power": round(emp, 3),
        "ci_lower_95": round(ci_lo, 3),
        "n_simulations": n_sim,
        # Allow 3 pp slack below target for MC noise; this is OLS slope, not MLE,
        # so we expect empirical to sit slightly below target in small samples.
        "pass": emp >= 0.77,
    }])


# ─── 15. Simulation: CS change-from-baseline ──────────────────────────

def _simulate_cs_change_power(n_per_arm: int, m: int, rho: float, sd: float,
                                delta_change: float, alpha: float,
                                rng, n_sim: int) -> float:
    """CS covariance simulation: Y_ij = μ_i + γ·I(treat) + b_i + ε_ij
       where b_i ~ N(0, ρσ²), ε ~ N(0, (1-ρ)σ²)."""
    var_b = rho * sd * sd
    var_e = (1 - rho) * sd * sd
    reject = 0
    for _ in range(n_sim):
        ch_c = np.empty(n_per_arm)
        ch_t = np.empty(n_per_arm)
        for i in range(n_per_arm):
            b = rng.normal(0, math.sqrt(var_b))
            e1_c = rng.normal(0, math.sqrt(var_e))
            em_c = rng.normal(0, math.sqrt(var_e))
            y1_c = b + e1_c
            ym_c = b + em_c
            ch_c[i] = ym_c - y1_c  # no treatment effect

            bt = rng.normal(0, math.sqrt(var_b))
            e1_t = rng.normal(0, math.sqrt(var_e))
            em_t = rng.normal(0, math.sqrt(var_e))
            y1_t = bt + e1_t
            ym_t = bt + delta_change + em_t
            ch_t[i] = ym_t - y1_t
        _, p = sp_stats.ttest_ind(ch_t, ch_c, equal_var=True)
        if p < alpha:
            reject += 1
    return reject / n_sim


def validate_cs_change_simulation(bridge: FrontendBridge) -> pd.DataFrame:
    """CS change-from-baseline: verify empirical power meets target."""
    plan = bridge.call("longitudinal_sample_size", input={
        "target": "change", "correlation": "cs",
        "sd": 1.0, "rho": 0.5, "measurements": 2,  # baseline + 1 follow-up
        "delta": 0.5, "alpha": 0.05, "power": 0.80,
        "twoSided": True, "ratio": 1.0,
    })
    n_per_arm = plan["n1"]
    rng = np.random.default_rng(0xBEE)
    n_sim = 2000
    emp = _simulate_cs_change_power(
        n_per_arm=n_per_arm, m=2, rho=0.5, sd=1.0, delta_change=0.5,
        alpha=0.05, rng=rng, n_sim=n_sim,
    )
    ci_lo = float(sp_stats.beta.ppf(0.025, emp * n_sim + 0.5,
                                      n_sim - emp * n_sim + 0.5))
    return pd.DataFrame([{
        "test": f"CS change-from-baseline power: plan N={n_per_arm}/arm",
        "target_power": 0.80,
        "empirical_power": round(emp, 3),
        "ci_lower_95": round(ci_lo, 3),
        "n_simulations": n_sim,
        "pass": emp >= 0.77,
    }])


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default=None)
    parser.add_argument("--frontend-path", default=None)
    args = parser.parse_args()
    _ = args.base_url

    frontend_path = resolve_frontend_path(args.frontend_path)

    suites = [
        ("1. Slope AR(1) exact matrix form", validate_slope_ar1_exact),
        ("2. Slope AR(1) σ-scaling invariant", validate_slope_ar1_sigma_scales),
        ("3. Slope CS exact matrix + closed form", validate_slope_cs_exact),
        ("4. Slope CS at ρ=0 recovers no-correlation case", validate_slope_cs_recovers_rho_zero),
        ("5. Endpoint unadjusted: effVar = σ²", validate_endpoint_unadjusted),
        ("6. Endpoint ANCOVA: effVar = σ²(1-ρ²)", validate_endpoint_ancova),
        ("7. Change-from-baseline CS: 2σ²(1-ρ)", validate_change_cs),
        ("8. Change-from-baseline AR(1): 2σ²(1-ρ^(m-1))", validate_change_ar1),
        ("9. Frison-Pocock Table 2 ANCOVA benchmark", validate_frison_pocock),
        ("10. Slope AR(1) is NOT the old asymptotic (regression)", validate_slope_ar1_not_asymptotic),
        ("11. Slope variance monotone decreasing in m", validate_slope_m_monotonicity),
        ("12. Endpoint ANCOVA variance monotone in ρ", validate_endpoint_ancova_rho_monotonicity),
        ("13. Change-CS variance monotone in ρ", validate_change_cs_rho_monotonicity),
        ("14. AR(1) slope simulation power at planned N", validate_ar1_slope_simulation),
        ("15. CS change-from-baseline simulation power", validate_cs_change_simulation),
    ]

    print("=" * 70)
    print("LONGITUDINAL / REPEATED MEASURES VALIDATION (frontend TS via Node)")
    print(f"frontend: {frontend_path}")
    print("=" * 70)

    all_pass = True
    all_frames: list[pd.DataFrame] = []
    with FrontendBridge("sample_size", frontend_path) as bridge:
        for name, fn in suites:
            print(f"\n{name}")
            print("-" * 70)
            try:
                df = fn(bridge)
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
        all_results.to_csv("results/longitudinal_validation.csv", index=False)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        return 0
    print("SOME VALIDATIONS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
