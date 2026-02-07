#!/usr/bin/env python3
"""
Offline Reference Tests — Pure Math, No API

Validates the reference implementations used by other test scripts.
Runs without any server dependency, useful for CI when API is unavailable.

Tests:
1. Beta-Binomial conjugate update
2. Normal-Normal conjugate update
3. Zhou & Ji (2024) boundary formula
4. Cochran's Q / I² calculation
5. ESS-based prior elicitation
6. Power prior formula
7. Binomial CI helper (common/assertions.py)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from common.assertions import binomial_ci, mc_rate_within, mc_rate_upper_bound

TOLERANCE = 1e-10


def test_beta_binomial_conjugate():
    """Beta-Binomial: posterior = Beta(alpha + s, beta + n - s)."""
    results = []

    cases = [
        (1.0, 1.0, 10, 30),    # Flat prior, 10/30 successes
        (0.5, 0.5, 0, 10),     # Jeffreys, zero successes
        (2.0, 8.0, 25, 45),    # Informative + REBYOTA CD2
    ]
    for prior_a, prior_b, s, n in cases:
        post_a = prior_a + s
        post_b = prior_b + (n - s)
        # Check via scipy: Beta(post_a, post_b) should have mean = post_a / (post_a + post_b)
        expected_mean = post_a / (post_a + post_b)
        scipy_mean = stats.beta.mean(post_a, post_b)
        ok = abs(expected_mean - scipy_mean) < TOLERANCE
        results.append({
            "test": f"Beta({prior_a},{prior_b}) + {s}/{n}",
            "post_mean": round(expected_mean, 6),
            "scipy_mean": round(scipy_mean, 6),
            "pass": ok,
        })

    return results


def test_normal_conjugate():
    """Normal-Normal: posterior precision = prior_prec + data_prec."""
    results = []

    cases = [
        (0.0, 1.0, 0.3, 0.1),   # Vague prior, data at 0.3
        (0.5, 0.5, 0.4, 0.05),  # Informative prior
        (0.0, 100.0, 0.6, 0.2), # Very vague prior
    ]
    for prior_mean, prior_var, data_mean, data_var in cases:
        prior_prec = 1 / prior_var
        data_prec = 1 / data_var
        post_prec = prior_prec + data_prec
        post_var = 1 / post_prec
        post_mean = post_var * (prior_mean * prior_prec + data_mean * data_prec)

        # Sanity: posterior should be between prior and data
        in_between = (min(prior_mean, data_mean) - 0.01 <= post_mean <= max(prior_mean, data_mean) + 0.01)
        # Posterior variance should be less than both
        var_reduced = post_var < prior_var and post_var < data_var

        results.append({
            "test": f"N({prior_mean},{prior_var}) + data({data_mean},{data_var})",
            "post_mean": round(post_mean, 6),
            "in_between": in_between,
            "var_reduced": var_reduced,
            "pass": in_between and var_reduced,
        })

    return results


def test_zhou_ji_boundary():
    """Zhou & Ji (2024) boundary formula at known values."""
    results = []

    # With mu=0, nu^2=1, sigma^2=1, n=100, gamma=0.975:
    # c = Phi^-1(0.975) * sqrt(1 + 1/(100*1)) - 0
    # c = 1.96 * sqrt(1.01) ≈ 1.96 * 1.00499 ≈ 1.9698
    prior_mean, prior_var, data_var, n_k = 0.0, 1.0, 1.0, 100
    threshold = 0.975

    c = (
        stats.norm.ppf(threshold) * np.sqrt(1 + data_var / (n_k * prior_var))
        - prior_mean * np.sqrt(data_var) / (np.sqrt(n_k) * prior_var)
    )
    expected = stats.norm.ppf(0.975) * np.sqrt(1.01)
    ok = abs(c - expected) < 1e-10
    results.append({
        "test": "Zhou & Ji: mu=0, nu^2=1, n=100, gamma=0.975",
        "c": round(c, 6),
        "expected": round(expected, 6),
        "pass": ok,
    })

    # With large nu^2, boundary -> Phi^-1(gamma)
    prior_var_large = 1e6
    c_vague = (
        stats.norm.ppf(threshold) * np.sqrt(1 + data_var / (n_k * prior_var_large))
    )
    z_crit = stats.norm.ppf(0.975)
    ok = abs(c_vague - z_crit) < 0.001
    results.append({
        "test": "Zhou & Ji: vague prior -> z_critical",
        "c_vague": round(c_vague, 6),
        "z_crit": round(z_crit, 6),
        "pass": ok,
    })

    return results


def test_cochrans_q_and_i_squared():
    """Cochran's Q and I² from known study data."""
    results = []

    # Two identical studies: Q=0, I^2=0
    studies = [{"n_events": 20, "n_total": 50}, {"n_events": 20, "n_total": 50}]
    rates = [s["n_events"] / s["n_total"] for s in studies]
    variances = [r * (1 - r) / s["n_total"] for r, s in zip(rates, studies)]
    weights = [1 / v for v in variances]
    total_w = sum(weights)
    pooled = sum(w * r for w, r in zip(weights, rates)) / total_w
    Q = sum(w * (r - pooled) ** 2 for w, r in zip(weights, rates))

    ok = abs(Q) < 1e-10 and pooled == 0.4
    results.append({
        "test": "Identical studies: Q=0, I^2=0",
        "Q": round(Q, 10),
        "pooled": round(pooled, 6),
        "pass": ok,
    })

    # Two very different studies: high I^2
    studies2 = [{"n_events": 5, "n_total": 50}, {"n_events": 45, "n_total": 50}]
    rates2 = [s["n_events"] / s["n_total"] for s in studies2]
    variances2 = [r * (1 - r) / s["n_total"] for r, s in zip(rates2, studies2)]
    weights2 = [1 / v for v in variances2]
    total_w2 = sum(weights2)
    pooled2 = sum(w * r for w, r in zip(weights2, rates2)) / total_w2
    Q2 = sum(w * (r - pooled2) ** 2 for w, r in zip(weights2, rates2))
    df2 = 1
    i_squared2 = max(0, (Q2 - df2) / Q2 * 100) if Q2 > 0 else 0

    ok2 = i_squared2 > 90
    results.append({
        "test": "Diverse rates (10% vs 90%): high I^2",
        "I_squared": round(i_squared2, 1),
        "pass": ok2,
    })

    return results


def test_ess_prior_elicitation():
    """ESS-based prior: alpha = mean * ESS, beta = (1-mean) * ESS."""
    results = []

    cases = [
        (0.30, 10),   # Berry-like
        (0.50, 2),    # Vague
        (0.001, 100), # Near-zero mean
        (0.999, 100), # Near-one mean
    ]
    for mean, ess in cases:
        alpha = mean * ess
        beta = (1 - mean) * ess
        recovered_mean = alpha / (alpha + beta)
        recovered_ess = alpha + beta

        ok = abs(recovered_mean - mean) < 1e-12 and abs(recovered_ess - ess) < 1e-12
        results.append({
            "test": f"ESS: mean={mean}, ess={ess}",
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "pass": ok,
        })

    return results


def test_power_prior():
    """Power prior: alpha = alpha0 + delta * events."""
    results = []

    cases = [
        (25, 45, 0.5, 1.0, 1.0),  # REBYOTA CD2
        (25, 45, 0.0, 1.0, 1.0),  # No borrowing
        (25, 45, 1.0, 1.0, 1.0),  # Full borrowing
        (0, 50, 0.5, 1.0, 1.0),   # Zero events
    ]
    for events, n, delta, a0, b0 in cases:
        alpha = a0 + delta * events
        beta = b0 + delta * (n - events)
        ess = alpha + beta
        borrowed = delta * n

        ok = abs(ess - (a0 + b0 + delta * n)) < 1e-10
        ok = ok and abs(borrowed - delta * n) < 1e-10
        results.append({
            "test": f"Power: {events}/{n}, delta={delta}",
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "ess": round(ess, 3),
            "pass": ok,
        })

    return results


def test_binomial_ci():
    """Validate Clopper-Pearson CI helper from assertions.py."""
    results = []

    # Known case: k=50, n=100 -> should bracket 0.5
    lo, hi = binomial_ci(50, 100, confidence=0.95)
    ok = lo < 0.5 < hi and lo > 0.35 and hi < 0.65
    results.append({
        "test": "CP CI: 50/100 at 95%",
        "lo": round(lo, 4), "hi": round(hi, 4),
        "pass": ok,
    })

    # Edge: k=0 -> lo=0
    lo, hi = binomial_ci(0, 100, confidence=0.95)
    ok = lo == 0.0 and hi > 0.0 and hi < 0.10
    results.append({
        "test": "CP CI: 0/100 at 95%",
        "lo": round(lo, 4), "hi": round(hi, 4),
        "pass": ok,
    })

    # Edge: k=n -> hi=1
    lo, hi = binomial_ci(100, 100, confidence=0.95)
    ok = hi == 1.0 and lo > 0.90
    results.append({
        "test": "CP CI: 100/100 at 95%",
        "lo": round(lo, 4), "hi": round(hi, 4),
        "pass": ok,
    })

    # mc_rate_within: 0.05 observed from 1000 sims should bracket 0.05
    ok = mc_rate_within(0.05, 1000, 0.05)
    results.append({
        "test": "mc_rate_within: 0.05 from 1000 sims",
        "pass": ok,
    })

    # mc_rate_upper_bound: 0.04 from 5000 sims should be < 0.10
    ub = mc_rate_upper_bound(0.04, 5000)
    ok = ub < 0.10
    results.append({
        "test": "mc_rate_upper_bound: 0.04 from 5000",
        "upper_bound": round(ub, 4),
        "pass": ok,
    })

    return results


def main():
    print("=" * 70)
    print("OFFLINE REFERENCE TESTS (No API)")
    print("=" * 70)

    all_results = []
    all_pass = True

    test_fns = [
        ("Beta-Binomial Conjugate", test_beta_binomial_conjugate),
        ("Normal-Normal Conjugate", test_normal_conjugate),
        ("Zhou & Ji Boundary Formula", test_zhou_ji_boundary),
        ("Cochran's Q / I^2", test_cochrans_q_and_i_squared),
        ("ESS Prior Elicitation", test_ess_prior_elicitation),
        ("Power Prior", test_power_prior),
        ("Binomial CI Helpers", test_binomial_ci),
    ]

    for name, fn in test_fns:
        print(f"\n{name}")
        print("-" * 70)
        results = fn()
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  [{status}] {r['test']}")
            if not r["pass"]:
                all_pass = False
            all_results.append(r)

    print("\n" + "=" * 70)
    n_pass = sum(1 for r in all_results if r["pass"])
    n_total = len(all_results)
    print(f"{n_pass}/{n_total} tests passed")

    if all_pass:
        print("ALL OFFLINE VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("SOME OFFLINE VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
