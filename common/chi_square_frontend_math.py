"""
Python port of the Zetyra Chi-Square Calculator's client-side math.

The chi-square calculator at frontend/src/app/calculators/chi-square/page.tsx
runs entirely in the browser — there is no backend endpoint. This module
mirrors the four numerical functions implemented there so we can validate the
shipped math against scipy/statsmodels references.

The functions here are straight Python translations of the TypeScript
implementations (Lanczos lnGamma, regularized incomplete gamma, Wilson-Hilferty
chi-square quantile with Newton refinement, classical McNemar).

IMPORTANT: Keep in sync with
  frontend/src/app/calculators/chi-square/page.tsx:24-197
If that file's math changes, update this file and re-run
  free/test_chi_square.py
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Log-gamma (Lanczos approximation — mirrors page.tsx lnGamma)
# ---------------------------------------------------------------------------
_LANCZOS_COEF = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def ln_gamma(z: float) -> float:
    if z < 0.5:
        return math.log(math.pi / math.sin(math.pi * z)) - ln_gamma(1 - z)
    z -= 1
    x = _LANCZOS_COEF[0]
    g = 7
    for i in range(1, g + 2):
        x += _LANCZOS_COEF[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def log_factorial(n: int) -> float:
    if n <= 1:
        return 0.0
    return ln_gamma(n + 1)


# ---------------------------------------------------------------------------
# Regularized incomplete gamma (mirrors regularizedGammaP / Q)
# ---------------------------------------------------------------------------
def _regularized_gamma_q(a: float, x: float) -> float:
    # Lentz: c initialized to 1/FPMIN (large), not FPMIN itself.
    c = 1e30
    d = 1.0 / (x + 1 - a)
    h = d
    for n in range(1, 200):
        an = -n * (n - a)
        bn = x + 2 * n + 1 - a
        d = bn + an * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = bn + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - ln_gamma(a)) * h


def regularized_gamma_p(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x > a + 1:
        return 1 - _regularized_gamma_q(a, x)
    s = 1.0 / a
    term = 1.0 / a
    for n in range(1, 200):
        term *= x / (a + n)
        s += term
        if abs(term) < 1e-12 * abs(s):
            break
    return s * math.exp(-x + a * math.log(x) - ln_gamma(a))


# ---------------------------------------------------------------------------
# Chi-square p-value (mirrors chiSquarePValue)
# ---------------------------------------------------------------------------
def chi_square_p_value(x: float, df: int = 1) -> float:
    if x <= 0:
        return 1.0
    if df == 1:
        # Fast path: P = erfc(sqrt(x/2)), with erf via Abramowitz 7.1.26
        u = math.sqrt(x / 2)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        t = 1.0 / (1.0 + p * u)
        erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-u * u)
        return 1 - erf
    return 1 - regularized_gamma_p(df / 2, x / 2)


# ---------------------------------------------------------------------------
# Normal quantile (Beasley-Springer-Moro — mirrors normalQuantile)
# ---------------------------------------------------------------------------
def normal_quantile(p: float) -> float:
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    if p == 0.5:
        return 0.0
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


# ---------------------------------------------------------------------------
# Chi-square critical value (mirrors chiSquareCritical)
# ---------------------------------------------------------------------------
def chi_square_critical(alpha: float, df: int) -> float:
    z = normal_quantile(1 - alpha)
    term = 1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))
    x = df * term**3
    if x < 0:
        x = 0.01
    for _ in range(20):
        p = chi_square_p_value(x, df)
        diff = p - alpha
        log_pdf = (df / 2 - 1) * math.log(x) - x / 2 - (df / 2) * math.log(2) - ln_gamma(df / 2)
        pdf_val = math.exp(log_pdf)
        if pdf_val < 1e-20:
            break
        step = diff / pdf_val
        x += step
        if x < 0:
            x = 0.001
        if abs(step) < 1e-10:
            break
    return x


# ---------------------------------------------------------------------------
# Pearson chi-square for r x c contingency (mirrors the test-mode reducer)
# 2x2 tables always use Yates correction, per the frontend.
# ---------------------------------------------------------------------------
def pearson_chi_square(table: Sequence[Sequence[int]]) -> dict:
    R = len(table)
    C = len(table[0])
    row_totals = [sum(table[r]) for r in range(R)]
    col_totals = [sum(table[r][c] for r in range(R)) for c in range(C)]
    N = sum(row_totals)
    if N <= 0 or any(t <= 0 for t in row_totals) or any(t <= 0 for t in col_totals):
        return {}
    df = (R - 1) * (C - 1)
    is_2x2 = R == 2 and C == 2
    chi_sq = 0.0
    min_expected = math.inf
    for r in range(R):
        for c in range(C):
            e = row_totals[r] * col_totals[c] / N
            if e < min_expected:
                min_expected = e
            o = table[r][c]
            if e > 0:
                if is_2x2:
                    chi_sq += (abs(o - e) - 0.5) ** 2 / e
                else:
                    chi_sq += (o - e) ** 2 / e
    p_value = chi_square_p_value(chi_sq, df)
    phi = math.sqrt(chi_sq / N)
    min_dim = min(R - 1, C - 1)
    cramers_v = math.sqrt(chi_sq / (N * min_dim)) if min_dim > 0 else phi
    return {
        "chi_sq": chi_sq,
        "df": df,
        "p_value": p_value,
        "phi": phi,
        "cramers_v": cramers_v,
        "min_expected": min_expected,
        "N": N,
        "is_2x2": is_2x2,
    }


# ---------------------------------------------------------------------------
# McNemar classical (no continuity correction — matches mcnemarResult)
# ---------------------------------------------------------------------------
def mcnemar_classical(a: int, b: int, c: int, d: int) -> dict:
    b_plus_c = b + c
    if b_plus_c <= 0:
        return {}
    chi_sq = (b - c) ** 2 / b_plus_c
    p_value = chi_square_p_value(chi_sq, 1)
    return {
        "chi_sq": chi_sq,
        "p_value": p_value,
        "discordant": b_plus_c,
        "total": a + b + c + d,
    }


# ---------------------------------------------------------------------------
# Fisher's exact 2x2 two-sided (mirrors fisherExact2x2)
# ---------------------------------------------------------------------------
def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d

    def log_hypergeom(k: int) -> float:
        if k < 0 or k > r1 or (c1 - k) < 0 or (c1 - k) > r2:
            return -math.inf
        return (
            log_factorial(r1)
            - log_factorial(k)
            - log_factorial(r1 - k)
            + log_factorial(r2)
            - log_factorial(c1 - k)
            - log_factorial(r2 - (c1 - k))
            - log_factorial(n)
            + log_factorial(c1)
            + log_factorial(c2)
        )

    p_observed = log_hypergeom(a)
    p_value = 0.0
    k_min = max(0, c1 - r2)
    k_max = min(r1, c1)
    for k in range(k_min, k_max + 1):
        lp = log_hypergeom(k)
        if lp <= p_observed + 1e-10:
            p_value += math.exp(lp)
    return min(p_value, 1.0)
