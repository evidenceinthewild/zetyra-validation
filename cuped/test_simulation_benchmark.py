#!/usr/bin/env python3
"""
CUPED Simulation & Published Benchmark Validation

Extends the analytical validation with:
1. Monte Carlo simulation: generate correlated pre/post data, apply CUPED,
   verify variance reduction matches theory
2. Deng et al. (2013) reference: verify sample size reduction ratios from
   the original WSDM paper
3. Extreme correlation tests: boundary behavior at ρ → 0, ρ → 1
4. Multiple MDE / power combinations: broader parameter coverage

References:
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013) "Improving the Sensitivity
  of Online Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM)
- The key result: CUPED reduces variance by factor (1 - ρ²), equivalently
  reduces required sample size by (1 - ρ²)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
import pandas as pd
import numpy as np
from scipy import stats


# ─── Reference implementations ────────────────────────────────────────

def reference_cuped(baseline_mean, baseline_std, mde, correlation, alpha=0.05, power=0.80):
    """Analytical CUPED sample size."""
    delta = baseline_mean * mde
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n_original = int(np.ceil(2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2))
    vrf = 1 - correlation ** 2
    adjusted_std = baseline_std * np.sqrt(vrf)
    n_adjusted = int(np.ceil(2 * ((z_alpha + z_beta) * adjusted_std / delta) ** 2))
    return n_original, n_adjusted, vrf


def simulate_cuped_variance_reduction(rho, n_obs=50000, sigma=20.0, seed=42):
    """
    Monte Carlo: generate correlated (X, Y) pairs, compute CUPED-adjusted
    variance, and verify reduction factor matches theory.

    Model:
      X ~ N(100, sigma^2)  (pre-experiment metric)
      Y = mu + rho * (X - 100) + epsilon,  epsilon ~ N(0, sigma^2 * (1 - rho^2))
    so Corr(X, Y) = rho, Var(Y) = sigma^2

    CUPED adjustment:
      Y_adj = Y - b_opt * (X - E[X])
      b_opt = Cov(Y, X) / Var(X) = rho * sigma_Y / sigma_X
      Var(Y_adj) = Var(Y) * (1 - rho^2)
    """
    rng = np.random.default_rng(seed)

    mu_x = 100.0
    x = rng.normal(mu_x, sigma, size=n_obs)

    # Y with correlation rho to X
    noise_std = sigma * np.sqrt(max(1 - rho ** 2, 0))
    y = mu_x + rho * (x - mu_x) + rng.normal(0, noise_std, size=n_obs)

    # CUPED adjustment
    b_opt = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    y_adj = y - b_opt * (x - np.mean(x))

    var_original = np.var(y, ddof=1)
    var_adjusted = np.var(y_adj, ddof=1)
    empirical_vrf = var_adjusted / var_original

    return empirical_vrf, 1 - rho ** 2


# ─── Test functions ───────────────────────────────────────────────────

def validate_mc_variance_reduction(client) -> pd.DataFrame:
    """MC simulation verifies variance reduction factor = 1 - ρ²."""
    results = []

    for rho in [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]:
        empirical_vrf, theoretical_vrf = simulate_cuped_variance_reduction(rho, n_obs=100000)

        # Also check API returns correct VRF
        zetyra = client.cuped(
            baseline_mean=100, baseline_std=20, mde=0.05,
            correlation=rho, alpha=0.05, power=0.80,
        )

        # MC tolerance: with 100k samples, VRF should be within ~1% of theory
        mc_ok = abs(empirical_vrf - theoretical_vrf) < 0.02
        api_ok = abs(zetyra["variance_reduction_factor"] - theoretical_vrf) < 0.001

        results.append({
            "test": f"ρ={rho}: MC VRF ≈ 1−ρ²",
            "empirical_vrf": round(empirical_vrf, 4),
            "theoretical_vrf": round(theoretical_vrf, 4),
            "api_vrf": round(zetyra["variance_reduction_factor"], 4),
            "mc_dev": round(abs(empirical_vrf - theoretical_vrf), 4),
            "pass": mc_ok and api_ok,
        })

    return pd.DataFrame(results)


def validate_mc_sample_size_effect(client) -> pd.DataFrame:
    """
    MC simulation: verify that CUPED at n_adjusted achieves the same power
    as no-CUPED at n_original.

    Under CUPED, the effective sample size for a test of size n is
    n_eff = n / (1 - ρ²). So a test with n_adj subjects and CUPED
    should have the same power as n_original subjects without CUPED.

    Both arms are simulated independently and compared.
    """
    results = []

    rng = np.random.default_rng(123)
    n_trials = 10000
    sigma = 20.0
    mu = 100.0
    delta = 5.0  # True effect: 5% of mean
    alpha = 0.05

    for rho in [0.5, 0.7, 0.9]:
        # Get API sample sizes
        zetyra = client.cuped(
            baseline_mean=mu, baseline_std=sigma, mde=delta / mu,
            correlation=rho, alpha=alpha, power=0.80,
        )
        n_adj = zetyra["n_adjusted"]
        n_orig = zetyra["n_original"]

        # --- Arm A: no-CUPED at n_original ---
        rejections_no_cuped = 0
        rng_a = np.random.default_rng(200 + int(rho * 100))
        for _ in range(n_trials):
            y_c = rng_a.normal(mu, sigma, size=n_orig)
            y_t = rng_a.normal(mu + delta, sigma, size=n_orig)
            _, p_val = stats.ttest_ind(y_t, y_c)
            if p_val < alpha:
                rejections_no_cuped += 1
        power_no_cuped = rejections_no_cuped / n_trials

        # --- Arm B: CUPED at n_adjusted ---
        rejections_cuped = 0
        rng_b = np.random.default_rng(300 + int(rho * 100))
        noise_std = sigma * np.sqrt(max(1 - rho ** 2, 0))
        for _ in range(n_trials):
            x_c = rng_b.normal(mu, sigma, size=n_adj)
            y_c = mu + rho * (x_c - mu) + rng_b.normal(0, noise_std, size=n_adj)
            x_t = rng_b.normal(mu, sigma, size=n_adj)
            y_t = (mu + delta) + rho * (x_t - mu) + rng_b.normal(0, noise_std, size=n_adj)

            x_all = np.concatenate([x_c, x_t])
            y_all = np.concatenate([y_c, y_t])
            b_opt = np.cov(y_all, x_all)[0, 1] / np.var(x_all, ddof=1)
            y_c_adj = y_c - b_opt * (x_c - np.mean(x_all))
            y_t_adj = y_t - b_opt * (x_t - np.mean(x_all))
            _, p_val = stats.ttest_ind(y_t_adj, y_c_adj)
            if p_val < alpha:
                rejections_cuped += 1
        power_cuped = rejections_cuped / n_trials

        # Both should achieve ~80% power
        results.append({
            "test": f"ρ={rho}: no-CUPED n_orig={n_orig} achieves ~80% power",
            "empirical_power": round(power_no_cuped, 4),
            "target_power": 0.80,
            "deviation": round(abs(power_no_cuped - 0.80), 4),
            "pass": abs(power_no_cuped - 0.80) < 0.05,
        })
        results.append({
            "test": f"ρ={rho}: CUPED n_adj={n_adj} achieves ~80% power",
            "empirical_power": round(power_cuped, 4),
            "target_power": 0.80,
            "deviation": round(abs(power_cuped - 0.80), 4),
            "pass": abs(power_cuped - 0.80) < 0.05,
        })

        # Power equivalence: both designs should have similar power
        power_diff = abs(power_no_cuped - power_cuped)
        results.append({
            "test": f"ρ={rho}: power equivalence (no-CUPED @ n_orig ≈ CUPED @ n_adj)",
            "power_no_cuped": round(power_no_cuped, 4),
            "power_cuped": round(power_cuped, 4),
            "difference": round(power_diff, 4),
            "pass": power_diff < 0.06,
        })

    return pd.DataFrame(results)


def validate_deng_et_al_reduction_ratio(client) -> pd.DataFrame:
    """
    Deng et al. (2013) key result: sample size reduction = (1 - ρ²).

    From the paper:
    "The variance of the CUPED estimator is Var(Y)(1 - ρ²), where ρ is
    the correlation between the pre-experiment covariate and the metric."

    This means: n_adjusted / n_original = (1 - ρ²)

    We verify this ratio across multiple parameter combinations.
    """
    results = []

    # Various parameter combos — the ratio should always be (1 - ρ²)
    param_combos = [
        {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "alpha": 0.05, "power": 0.80},
        {"baseline_mean": 50, "baseline_std": 10, "mde": 0.10, "alpha": 0.05, "power": 0.80},
        {"baseline_mean": 200, "baseline_std": 50, "mde": 0.02, "alpha": 0.01, "power": 0.90},
        {"baseline_mean": 1000, "baseline_std": 100, "mde": 0.03, "alpha": 0.05, "power": 0.95},
    ]

    for params in param_combos:
        for rho in [0.3, 0.5, 0.7, 0.9]:
            zetyra = client.cuped(**params, correlation=rho)
            n_orig = zetyra["n_original"]
            n_adj = zetyra["n_adjusted"]

            # Theoretical ratio (using continuous formula before ceiling)
            theoretical_ratio = 1 - rho ** 2

            # Compute expected values from first principles
            z_alpha = stats.norm.ppf(1 - params["alpha"] / 2)
            z_beta = stats.norm.ppf(params["power"])
            delta = params["baseline_mean"] * params["mde"]
            sigma = params["baseline_std"]

            n_orig_float = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
            expected_orig = int(np.ceil(n_orig_float))
            n_adj_float = n_orig_float * theoretical_ratio
            expected_adj = int(np.ceil(n_adj_float))

            # Check n_original matches reference
            n_orig_ok = abs(n_orig - expected_orig) <= 1

            # Check n_adjusted matches reference
            n_adj_ok = abs(n_adj - expected_adj) <= 1

            # Check actual ratio is close to theoretical (1 - ρ²)
            # Ceiling rounding introduces noise, so allow ±0.03
            actual_ratio = n_adj / max(n_orig, 1)
            ratio_ok = abs(actual_ratio - theoretical_ratio) < 0.03

            results.append({
                "test": f"μ={params['baseline_mean']},σ={params['baseline_std']},MDE={params['mde']},ρ={rho}",
                "n_original": n_orig,
                "expected_orig": expected_orig,
                "n_adjusted": n_adj,
                "expected_adj": expected_adj,
                "ratio": round(actual_ratio, 4),
                "target_ratio": round(theoretical_ratio, 4),
                "pass": n_orig_ok and n_adj_ok and ratio_ok,
            })

    return pd.DataFrame(results)


def validate_extreme_correlations(client) -> pd.DataFrame:
    """Test boundary behavior at extreme correlations."""
    results = []

    base = {"baseline_mean": 100, "baseline_std": 20, "mde": 0.05, "alpha": 0.05, "power": 0.80}

    # Very high correlation: ρ=0.99 → VRF = 0.0199, huge sample reduction
    z_high = client.cuped(**base, correlation=0.99)
    vrf_expected = 1 - 0.99 ** 2  # 0.0199
    results.append({
        "test": "ρ=0.99: VRF ≈ 0.0199",
        "zetyra_vrf": round(z_high["variance_reduction_factor"], 6),
        "expected_vrf": round(vrf_expected, 6),
        "n_original": z_high["n_original"],
        "n_adjusted": z_high["n_adjusted"],
        "pass": abs(z_high["variance_reduction_factor"] - vrf_expected) < 0.001,
    })

    # ρ=0.99: sample size should be ~2% of original
    ratio = z_high["n_adjusted"] / z_high["n_original"]
    results.append({
        "test": "ρ=0.99: n_adjusted < 5% of n_original",
        "ratio": round(ratio, 4),
        "pass": ratio < 0.05,
    })

    # Very low correlation: ρ=0.01 → VRF ≈ 0.9999, almost no reduction
    z_low = client.cuped(**base, correlation=0.01)
    results.append({
        "test": "ρ=0.01: VRF ≈ 1 (no reduction)",
        "zetyra_vrf": round(z_low["variance_reduction_factor"], 6),
        "n_original": z_low["n_original"],
        "n_adjusted": z_low["n_adjusted"],
        "pass": abs(z_low["variance_reduction_factor"] - (1 - 0.01 ** 2)) < 0.001,
    })

    # At near-zero correlation, n_original ≈ n_adjusted
    results.append({
        "test": "ρ=0.01: n_original ≈ n_adjusted",
        "n_original": z_low["n_original"],
        "n_adjusted": z_low["n_adjusted"],
        "pass": abs(z_low["n_original"] - z_low["n_adjusted"]) <= 1,
    })

    return pd.DataFrame(results)


def validate_multiple_power_alpha(client) -> pd.DataFrame:
    """Verify CUPED works correctly across different power/alpha combinations."""
    results = []

    configs = [
        {"alpha": 0.05, "power": 0.80, "desc": "standard (α=0.05, 80%)"},
        {"alpha": 0.05, "power": 0.90, "desc": "high power (α=0.05, 90%)"},
        {"alpha": 0.01, "power": 0.80, "desc": "strict alpha (α=0.01, 80%)"},
        {"alpha": 0.01, "power": 0.90, "desc": "strict both (α=0.01, 90%)"},
    ]

    for cfg in configs:
        for rho in [0.5, 0.8]:
            zetyra = client.cuped(
                baseline_mean=100, baseline_std=20, mde=0.05,
                correlation=rho, alpha=cfg["alpha"], power=cfg["power"],
            )

            ref_orig, ref_adj, ref_vrf = reference_cuped(
                100, 20, 0.05, rho, cfg["alpha"], cfg["power"],
            )

            results.append({
                "test": f"{cfg['desc']}, ρ={rho}",
                "zetyra_n_adj": zetyra["n_adjusted"],
                "ref_n_adj": ref_adj,
                "zetyra_vrf": round(zetyra["variance_reduction_factor"], 4),
                "ref_vrf": round(ref_vrf, 4),
                "pass": zetyra["n_adjusted"] == ref_adj and abs(zetyra["variance_reduction_factor"] - ref_vrf) < 0.001,
            })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("CUPED SIMULATION & BENCHMARK VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. MC Variance Reduction (100k samples)")
    print("-" * 70)
    mc_vrf = validate_mc_variance_reduction(client)
    print(mc_vrf.to_string(index=False))
    all_frames.append(mc_vrf)

    print("\n2. MC Power Verification (10k trials)")
    print("-" * 70)
    mc_power = validate_mc_sample_size_effect(client)
    print(mc_power.to_string(index=False))
    all_frames.append(mc_power)

    print("\n3. Deng et al. (2013) Reduction Ratio: n_adj/n_orig = 1−ρ²")
    print("-" * 70)
    deng = validate_deng_et_al_reduction_ratio(client)
    print(deng.to_string(index=False))
    all_frames.append(deng)

    print("\n4. Extreme Correlations")
    print("-" * 70)
    extreme = validate_extreme_correlations(client)
    print(extreme.to_string(index=False))
    all_frames.append(extreme)

    print("\n5. Multiple Power/Alpha Combinations")
    print("-" * 70)
    multi = validate_multiple_power_alpha(client)
    print(multi.to_string(index=False))
    all_frames.append(multi)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/cuped_simulation_benchmark.csv", index=False)

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
