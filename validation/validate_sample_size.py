"""
Validate Zetyra Sample Size Calculators

Compares Zetyra results against:
- R pwr package (pwr.t.test, pwr.2p.test)
- scipy.stats formulas
- Schoenfeld formula for survival
- Published benchmarks
"""

import math
import pandas as pd
import numpy as np
from scipy import stats
from zetyra_client import get_client

# Tolerance for validation (1% relative difference)
TOLERANCE = 0.01
# Slightly higher tolerance for survival due to event probability estimation
SURVIVAL_TOLERANCE = 0.02


def reference_sample_size_continuous(
    mean1: float,
    mean2: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> dict:
    """
    Calculate sample size using scipy (matches R pwr::pwr.t.test).

    This is the reference implementation for validation.
    """
    delta = mean2 - mean1
    effect_size = delta / sd

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Sample size formula for unequal allocation
    n1 = (1 + 1 / ratio) * ((z_alpha + z_beta) * sd / abs(delta)) ** 2
    n2 = ratio * n1

    return {
        "n1": int(np.ceil(n1)),
        "n2": int(np.ceil(n2)),
        "n_total": int(np.ceil(n1)) + int(np.ceil(n2)),
        "effect_size": effect_size,
    }


def reference_sample_size_binary(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> dict:
    """
    Calculate sample size for binary outcomes using pooled-variance z-test.

    Uses the standard formula with pooled variance under H0 and separate
    variances under H1. Also computes Cohen's h for effect size reporting.

    Note: This differs slightly from pwr::pwr.2p.test which uses arcsine
    transformation throughout. The pooled-variance approach is standard
    in clinical trial design (Chow et al., Lachin).
    """
    # Cohen's h effect size
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    # Pooled proportion under H0
    pooled_p = (p1 + ratio * p2) / (1 + ratio)

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Variance components
    var_h0 = pooled_p * (1 - pooled_p)
    var_h1_1 = p1 * (1 - p1)
    var_h1_2 = p2 * (1 - p2)

    delta = abs(p2 - p1)
    numerator = z_alpha * np.sqrt((1 + 1 / ratio) * var_h0) + z_beta * np.sqrt(
        var_h1_1 + var_h1_2 / ratio
    )
    n1 = (numerator / delta) ** 2
    n2 = ratio * n1

    return {
        "n1": int(np.ceil(n1)),
        "n2": int(np.ceil(n2)),
        "n_total": int(np.ceil(n1)) + int(np.ceil(n2)),
        "effect_size_h": h,
    }


def reference_sample_size_survival(
    hazard_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    median_control: float = 12.0,
    accrual_time: float = 24.0,
    follow_up_time: float = 12.0,
    dropout_rate: float = 0.0,
) -> dict:
    """
    Calculate required events and sample size using Schoenfeld formula.

    This is the reference implementation for survival validation.
    Matches R survival/gsDesign methodology.

    The Schoenfeld formula (two-sided):
    d = ((z_alpha + z_beta) / log(HR))^2 * (1 + r)^2 / r

    where d = events, r = allocation ratio

    Sample size is then calculated from events / P(event).
    """
    log_hr = math.log(hazard_ratio)
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-sided
    z_beta = stats.norm.ppf(power)

    r = allocation_ratio

    # Schoenfeld formula for required events
    events = ((z_alpha + z_beta) / log_hr) ** 2 * (1 + r) ** 2 / r
    events_int = int(np.ceil(events))

    # Estimate probability of event using exponential survival
    lambda_control = np.log(2) / median_control
    lambda_treatment = lambda_control * hazard_ratio

    # Average follow-up time for uniform accrual
    avg_follow_up = follow_up_time + accrual_time / 2

    def prob_event(lambda_val, avg_fu, dropout_rate):
        """Probability of observing event accounting for dropout."""
        p_event = 1 - np.exp(-lambda_val * avg_fu)
        p_no_dropout = (1 - dropout_rate) ** (avg_fu / 12)  # Annual dropout
        return p_event * p_no_dropout

    p_event_ctrl = prob_event(lambda_control, avg_follow_up, dropout_rate)
    p_event_trt = prob_event(lambda_treatment, avg_follow_up, dropout_rate)

    # Weighted average event probability
    p_event_avg = (p_event_ctrl + r * p_event_trt) / (1 + r)

    # Total sample size = events / p_event
    n_total = events_int / p_event_avg
    n1 = n_total / (1 + r)
    n2 = r * n1

    return {
        "events_required": events_int,
        "n1": int(np.ceil(n1)),
        "n2": int(np.ceil(n2)),
        "n_total": int(np.ceil(n1)) + int(np.ceil(n2)),
        "log_hr": log_hr,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "p_event_avg": p_event_avg,
    }


def validate_continuous(client, scenarios: list) -> pd.DataFrame:
    """Validate continuous sample size against reference."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.sample_size_continuous(**scenario)

        # Get reference result
        reference = reference_sample_size_continuous(**scenario)

        # Calculate deviation
        n_deviation = abs(zetyra["n_total"] - reference["n_total"]) / reference["n_total"]
        d_deviation = abs(zetyra["effect_size"] - reference["effect_size"]) / abs(
            reference["effect_size"]
        )

        results.append({
            "scenario": str(scenario),
            "zetyra_n": zetyra["n_total"],
            "reference_n": reference["n_total"],
            "n_deviation_pct": n_deviation * 100,
            "zetyra_d": zetyra["effect_size"],
            "reference_d": reference["effect_size"],
            "d_deviation_pct": d_deviation * 100,
            "pass": n_deviation <= TOLERANCE and d_deviation <= TOLERANCE,
        })

    return pd.DataFrame(results)


def validate_binary(client, scenarios: list) -> pd.DataFrame:
    """Validate binary sample size against reference."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.sample_size_binary(**scenario)

        # Get reference result
        reference = reference_sample_size_binary(**scenario)

        # Calculate deviation
        n_deviation = abs(zetyra["n_total"] - reference["n_total"]) / reference["n_total"]
        h_deviation = abs(zetyra["effect_size_h"] - reference["effect_size_h"]) / abs(
            reference["effect_size_h"]
        )

        results.append({
            "scenario": str(scenario),
            "zetyra_n": zetyra["n_total"],
            "reference_n": reference["n_total"],
            "n_deviation_pct": n_deviation * 100,
            "zetyra_h": zetyra["effect_size_h"],
            "reference_h": reference["effect_size_h"],
            "h_deviation_pct": h_deviation * 100,
            "pass": n_deviation <= TOLERANCE and h_deviation <= TOLERANCE,
        })

    return pd.DataFrame(results)


def validate_survival(client, scenarios: list) -> pd.DataFrame:
    """Validate survival sample size against Schoenfeld formula."""
    results = []

    for scenario in scenarios:
        # Get Zetyra result
        zetyra = client.sample_size_survival(**scenario)

        # Get reference result with all params for full validation
        reference = reference_sample_size_survival(
            hazard_ratio=scenario["hazard_ratio"],
            alpha=scenario.get("alpha", 0.05),
            power=scenario.get("power", 0.80),
            allocation_ratio=scenario.get("allocation_ratio", 1.0),
            median_control=scenario["median_control"],
            accrual_time=scenario["accrual_time"],
            follow_up_time=scenario["follow_up_time"],
            dropout_rate=scenario.get("dropout_rate", 0.0),
        )

        # Calculate deviation for events (primary validation)
        events_deviation = (
            abs(zetyra["events_required"] - reference["events_required"])
            / reference["events_required"]
        )

        # Log HR should match exactly
        log_hr_deviation = (
            abs(zetyra["log_hr"] - reference["log_hr"]) / abs(reference["log_hr"])
        )

        # Sample size validation (secondary - depends on event probability estimate)
        n_total_deviation = (
            abs(zetyra["n_total"] - reference["n_total"])
            / reference["n_total"]
        )

        results.append({
            "scenario": f"HR={scenario['hazard_ratio']}, α={scenario.get('alpha', 0.05)}, power={scenario.get('power', 0.80)}",
            "zetyra_events": zetyra["events_required"],
            "reference_events": reference["events_required"],
            "events_deviation_pct": events_deviation * 100,
            "zetyra_n": zetyra["n_total"],
            "reference_n": reference["n_total"],
            "n_deviation_pct": n_total_deviation * 100,
            "pass": events_deviation <= SURVIVAL_TOLERANCE and log_hr_deviation <= TOLERANCE and n_total_deviation <= SURVIVAL_TOLERANCE,
        })

    return pd.DataFrame(results)


def validate_survival_properties(client) -> pd.DataFrame:
    """Validate mathematical properties of survival calculator."""
    results = []

    # Property 1: Smaller HR requires fewer events
    small_hr = client.sample_size_survival(
        hazard_ratio=0.6, median_control=12, accrual_time=24, follow_up_time=12
    )
    large_hr = client.sample_size_survival(
        hazard_ratio=0.9, median_control=12, accrual_time=24, follow_up_time=12
    )
    prop1_pass = small_hr["events_required"] < large_hr["events_required"]
    results.append({
        "property": "Larger effect (smaller HR) → fewer events",
        "expected": "events(HR=0.6) < events(HR=0.9)",
        "actual": f"{small_hr['events_required']} < {large_hr['events_required']}",
        "pass": prop1_pass,
    })

    # Property 2: Higher power requires more events
    power_80 = client.sample_size_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12, power=0.80
    )
    power_90 = client.sample_size_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12, power=0.90
    )
    prop2_pass = power_90["events_required"] > power_80["events_required"]
    results.append({
        "property": "Higher power → more events",
        "expected": "events(power=0.90) > events(power=0.80)",
        "actual": f"{power_90['events_required']} > {power_80['events_required']}",
        "pass": prop2_pass,
    })

    # Property 3: Lower alpha requires more events
    alpha_05 = client.sample_size_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12, alpha=0.05
    )
    alpha_01 = client.sample_size_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12, alpha=0.01
    )
    prop3_pass = alpha_01["events_required"] > alpha_05["events_required"]
    results.append({
        "property": "Lower alpha → more events",
        "expected": "events(α=0.01) > events(α=0.05)",
        "actual": f"{alpha_01['events_required']} > {alpha_05['events_required']}",
        "pass": prop3_pass,
    })

    # Property 4: HR=1 should return error (no treatment effect)
    hr_one = client.sample_size_survival(
        hazard_ratio=1.0, median_control=12, accrual_time=24, follow_up_time=12,
        allow_errors=True
    )
    # API should return 400 error for HR=1
    prop4_pass = hr_one.get("error") == 400 or "error" in str(hr_one).lower()
    results.append({
        "property": "HR=1 returns error (undefined sample size)",
        "expected": "HTTP 400 error",
        "actual": f"HTTP {hr_one.get('error')}" if hr_one.get("error") else "No error",
        "pass": prop4_pass,
    })

    # Property 5: Schoenfeld formula accuracy (known values)
    # For HR=0.7, alpha=0.05, power=0.80, r=1: events ≈ 248
    known = client.sample_size_survival(
        hazard_ratio=0.7, median_control=12, accrual_time=24, follow_up_time=12,
        alpha=0.05, power=0.80, allocation_ratio=1.0
    )
    expected_events = 248  # From manual Schoenfeld calculation
    prop5_pass = abs(known["events_required"] - expected_events) <= 2  # Allow ±2 for rounding
    results.append({
        "property": "Schoenfeld formula: HR=0.7, α=0.05, power=0.80",
        "expected": f"~{expected_events} events",
        "actual": f"{known['events_required']} events",
        "pass": prop5_pass,
    })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full sample size validation."""
    client = get_client(base_url)

    # Continuous scenarios
    continuous_scenarios = [
        {"mean1": 100, "mean2": 105, "sd": 20, "alpha": 0.05, "power": 0.80},
        {"mean1": 100, "mean2": 110, "sd": 25, "alpha": 0.05, "power": 0.90},
        {"mean1": 50, "mean2": 55, "sd": 15, "alpha": 0.01, "power": 0.80},
        {"mean1": 100, "mean2": 105, "sd": 20, "alpha": 0.05, "power": 0.80, "ratio": 2.0},
        {"mean1": 100, "mean2": 103, "sd": 20, "alpha": 0.025, "power": 0.90, "two_sided": False},
    ]

    # Binary scenarios
    binary_scenarios = [
        {"p1": 0.10, "p2": 0.15, "alpha": 0.05, "power": 0.80},
        {"p1": 0.20, "p2": 0.30, "alpha": 0.05, "power": 0.90},
        {"p1": 0.50, "p2": 0.60, "alpha": 0.01, "power": 0.80},
        {"p1": 0.10, "p2": 0.15, "alpha": 0.05, "power": 0.80, "ratio": 2.0},
        {"p1": 0.30, "p2": 0.40, "alpha": 0.025, "power": 0.90, "two_sided": False},
    ]

    # Survival scenarios
    survival_scenarios = [
        {"hazard_ratio": 0.7, "median_control": 12, "accrual_time": 24, "follow_up_time": 12},
        {"hazard_ratio": 0.75, "median_control": 10, "accrual_time": 24, "follow_up_time": 18, "alpha": 0.05, "power": 0.80},
        {"hazard_ratio": 0.8, "median_control": 24, "accrual_time": 36, "follow_up_time": 24, "alpha": 0.05, "power": 0.90},
        {"hazard_ratio": 0.65, "median_control": 8, "accrual_time": 18, "follow_up_time": 12, "alpha": 0.01, "power": 0.80},
        {"hazard_ratio": 0.7, "median_control": 12, "accrual_time": 24, "follow_up_time": 12, "allocation_ratio": 2.0},
    ]

    continuous_results = validate_continuous(client, continuous_scenarios)
    binary_results = validate_binary(client, binary_scenarios)
    survival_results = validate_survival(client, survival_scenarios)
    survival_properties = validate_survival_properties(client)

    all_pass = (
        continuous_results["pass"].all()
        and binary_results["pass"].all()
        and survival_results["pass"].all()
        and survival_properties["pass"].all()
    )

    return {
        "continuous": continuous_results,
        "binary": binary_results,
        "survival": survival_results,
        "survival_properties": survival_properties,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    import sys

    # Allow override for local testing
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA SAMPLE SIZE VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nContinuous Outcomes (Two-Sample T-Test)")
    print("-" * 70)
    print(results["continuous"].to_string(index=False))

    print("\nBinary Outcomes (Two-Proportion Z-Test)")
    print("-" * 70)
    print(results["binary"].to_string(index=False))

    print("\nSurvival Outcomes (Log-Rank Test)")
    print("-" * 70)
    print(results["survival"].to_string(index=False))

    print("\nSurvival Properties")
    print("-" * 70)
    print(results["survival_properties"].to_string(index=False))

    print("\n" + "=" * 70)
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
