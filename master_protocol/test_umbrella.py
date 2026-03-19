#!/usr/bin/env python3
"""
Validate Umbrella Trial Calculator

Tests umbrella trial design across binary, continuous, and survival endpoints
with frequentist and Bayesian analysis, including simulation-based validation.

1.  Frequentist binary: 3-substudy basic, power > 0, Bonferroni, no multiplicity, sample size sums
2.  Frequentist continuous: basic, large effect -> high power
3.  Frequentist survival: HR < 1 events_required, power reasonable
4.  Bayesian binary: posterior exceedance, strong effect -> high, null -> low
5.  Simulation: frequentist binary, type I error, Bayesian sim, survival sim
6.  Structural: control allocation, biomarker prevalences, Holm thresholds
7.  Input validation: n_substudies < 2, mismatched arrays

References:
- Park et al. (2019) Trials 20:572.
- Woodcock J, LaVange LM (2017) NEJM 377(1):62-70.
- FDA (2022) Master Protocols: Efficient Clinical Trial Design Strategies.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import mc_rate_upper_bound
import pandas as pd


# ─── Helpers ─────────────────────────────────────────────────────────

def ar(resp):
    """Extract analytical_results from response."""
    return resp["analytical_results"]


# ─── Test functions ──────────────────────────────────────────────────

def validate_frequentist_binary(client) -> pd.DataFrame:
    """Frequentist binary endpoint tests."""
    results = []

    # 1. Basic 3-substudy: per_substudy_results returned with correct count
    z = client.umbrella(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.30, 0.25],
        multiplicity_method="bonferroni",
        alpha=0.025,
    )
    a = ar(z)
    results.append({
        "test": "Freq binary: 3-substudy basic structure",
        "n_substudies": len(a["per_substudy"]),
        "pass": len(a["per_substudy"]) == 3 and a["n_substudies"] == 3,
    })

    # 2. Power > 0 for active substudies (alternative > null)
    # Sub-study 1 has alt=0.35 vs null=0.15 -- should have low p-value
    sub1 = a["per_substudy"][0]
    results.append({
        "test": "Freq binary: active substudy has p_value < 1",
        "p_value": sub1.get("p_value"),
        "pass": sub1.get("p_value") is not None and sub1["p_value"] < 1.0,
    })

    # 3. Bonferroni multiplicity: adjusted alpha = alpha/n_substudies
    expected_adj_alpha = round(0.025 / 3, 6)
    results.append({
        "test": "Freq binary: Bonferroni adjusted_alpha = alpha/3",
        "adjusted_alpha": sub1.get("adjusted_alpha"),
        "expected": expected_adj_alpha,
        "pass": sub1.get("adjusted_alpha") is not None and abs(sub1["adjusted_alpha"] - expected_adj_alpha) < 1e-5,
    })

    # 4. No multiplicity: each substudy uses raw alpha
    z_none = client.umbrella(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.30, 0.25],
        multiplicity_method="none",
        alpha=0.025,
    )
    a_none = ar(z_none)
    all_raw = all(
        s.get("adjusted_alpha") == 0.025
        for s in a_none["per_substudy"]
    )
    results.append({
        "test": "Freq binary: no multiplicity -> raw alpha",
        "pass": all_raw,
    })

    # 5. Sample sizes sum correctly (control + all treatments ~ total_n)
    n_ctrl = a["pooled_control"]["n_total"]
    n_trts = sum(s["n_treatment"] for s in a["per_substudy"])
    total_allocated = n_ctrl + n_trts
    results.append({
        "test": "Freq binary: control + treatments = total_n",
        "n_control": n_ctrl,
        "n_treatments_sum": n_trts,
        "total_allocated": total_allocated,
        "total_n": 300,
        "pass": abs(total_allocated - 300) <= 2,  # rounding tolerance
    })

    return pd.DataFrame(results)


def validate_frequentist_continuous(client) -> pd.DataFrame:
    """Frequentist continuous endpoint tests."""
    results = []

    # 6. Basic continuous: verify per_substudy results
    z = client.umbrella(
        endpoint_type="continuous",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_means=[0.0, 0.0, 0.0],
        alternative_means=[0.5, 0.3, 0.1],
        common_sd=1.0,
        multiplicity_method="bonferroni",
        alpha=0.025,
    )
    a = ar(z)
    results.append({
        "test": "Freq continuous: 3-substudy structure",
        "n_substudies": len(a["per_substudy"]),
        "pass": len(a["per_substudy"]) == 3 and a["endpoint_type"] == "continuous",
    })

    # 7. Large effect -> high power (low p-value) per substudy
    # Sub-study 1 has a large effect (0.5 - 0.0 = 0.5 with SD=1.0)
    sub1 = a["per_substudy"][0]
    results.append({
        "test": "Freq continuous: large effect -> small p-value",
        "p_value": sub1.get("p_value"),
        "pass": sub1.get("p_value") is not None and sub1["p_value"] < 0.05,
    })

    return pd.DataFrame(results)


def validate_frequentist_survival(client) -> pd.DataFrame:
    """Frequentist survival endpoint tests."""
    results = []

    # 8. Survival with HR < 1: verify events_required present
    z = client.umbrella(
        endpoint_type="survival",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=600,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        hazard_ratios=[0.6, 0.7, 0.8],
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        multiplicity_method="bonferroni",
        alpha=0.025,
    )
    a = ar(z)
    sub1 = a["per_substudy"][0]
    results.append({
        "test": "Freq survival: events_required present",
        "events_required": sub1.get("events_required"),
        "pass": sub1.get("events_required") is not None and sub1["events_required"] > 0,
    })

    # 9. Survival power estimates reasonable (p-value < 1 for HR < 1)
    results.append({
        "test": "Freq survival: HR<1 yields p_value < 1",
        "p_value": sub1.get("p_value"),
        "pass": sub1.get("p_value") is not None and sub1["p_value"] < 1.0,
    })

    return pd.DataFrame(results)


def validate_bayesian_binary(client) -> pd.DataFrame:
    """Bayesian binary endpoint tests."""
    results = []

    # 10. Bayesian analysis returns posterior_prob per substudy
    z = client.umbrella(
        endpoint_type="binary",
        analysis_type="bayesian",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.30, 0.25],
        multiplicity_method="bonferroni",
        decision_threshold=0.975,
    )
    a = ar(z)
    has_posterior = all(
        "posterior_prob" in s
        for s in a["per_substudy"]
    )
    results.append({
        "test": "Bayesian binary: posterior_prob per substudy",
        "pass": has_posterior,
    })

    # 11. Large N and strong effect -> posterior exceedance high
    z_large = client.umbrella(
        endpoint_type="binary",
        analysis_type="bayesian",
        n_substudies=2,
        total_n=2000,
        control_allocation=0.33,
        biomarker_prevalences=[0.5, 0.5],
        null_rates=[0.10, 0.10],
        alternative_rates=[0.40, 0.40],
        multiplicity_method="none",
        decision_threshold=0.975,
    )
    a_large = ar(z_large)
    sub1_large = a_large["per_substudy"][0]
    results.append({
        "test": "Bayesian binary: strong effect -> high posterior",
        "posterior_prob": sub1_large.get("posterior_prob"),
        "pass": sub1_large.get("posterior_prob", 0) > 0.90,
    })

    # 12. Under null (alt ~ null), posterior exceedance should be low
    z_null = client.umbrella(
        endpoint_type="binary",
        analysis_type="bayesian",
        n_substudies=2,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.5, 0.5],
        null_rates=[0.30, 0.30],
        alternative_rates=[0.30, 0.30],  # same as null
        multiplicity_method="none",
        decision_threshold=0.975,
    )
    a_null = ar(z_null)
    sub1_null = a_null["per_substudy"][0]
    results.append({
        "test": "Bayesian binary: null scenario -> low posterior",
        "posterior_prob": sub1_null.get("posterior_prob"),
        "pass": sub1_null.get("posterior_prob", 1) < 0.80,
    })

    return pd.DataFrame(results)


def validate_simulation_frequentist_binary(client) -> pd.DataFrame:
    """Simulation-based frequentist binary tests."""
    results = []

    # 13. Frequentist binary sim: power > 0 for active substudies
    z = client.umbrella(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=600,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.40, 0.35, 0.30],
        multiplicity_method="bonferroni",
        alpha=0.025,
        simulate=True,
        n_simulations=2000,
        simulation_seed=42,
    )
    sim = z["simulation"]
    per_sub_power = sim["estimates"]["per_substudy_power"]
    # All substudies are active (alt > null), so power should be > 0
    all_positive = all(p is not None and p > 0 for p in per_sub_power)
    results.append({
        "test": "Sim freq binary: power > 0 for active substudies",
        "per_sub_power": per_sub_power,
        "pass": all_positive,
    })

    # 14. Type I error <= 0.10 under complete null
    # FWER from simulation under null should be controlled
    fwer = sim["estimates"]["fwer"]
    upper = mc_rate_upper_bound(fwer, 2000, confidence=0.99)
    results.append({
        "test": "Sim freq binary: type I error (FWER) <= 0.10",
        "fwer": fwer,
        "upper_bound_99": round(upper, 4),
        "pass": upper <= 0.10,
    })

    return pd.DataFrame(results)


def validate_simulation_bayesian(client) -> pd.DataFrame:
    """Simulation-based Bayesian tests."""
    results = []

    # 15. Bayesian sim works
    z = client.umbrella(
        endpoint_type="binary",
        analysis_type="bayesian",
        n_substudies=3,
        total_n=600,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.40, 0.35, 0.30],
        multiplicity_method="bonferroni",
        decision_threshold=0.975,
        simulate=True,
        n_simulations=2000,
        simulation_seed=42,
    )
    sim = z["simulation"]
    results.append({
        "test": "Sim Bayesian binary: simulation runs",
        "avg_power": sim["estimates"]["average_power"],
        "pass": sim is not None and "estimates" in sim and sim["estimates"]["average_power"] >= 0,
    })

    return pd.DataFrame(results)


def validate_simulation_survival(client) -> pd.DataFrame:
    """Simulation-based survival tests."""
    results = []

    # 16. Survival sim works
    z = client.umbrella(
        endpoint_type="survival",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=600,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        hazard_ratios=[0.6, 0.7, 0.8],
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        multiplicity_method="bonferroni",
        alpha=0.025,
        simulate=True,
        n_simulations=2000,
        simulation_seed=42,
    )
    sim = z["simulation"]
    results.append({
        "test": "Sim survival: simulation runs and returns estimates",
        "avg_power": sim["estimates"]["average_power"],
        "pass": sim is not None and sim["estimates"]["average_power"] >= 0,
    })

    return pd.DataFrame(results)


def validate_structural(client) -> pd.DataFrame:
    """Structural property tests."""
    results = []

    # 17. Control allocation fraction preserved
    z = client.umbrella(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.30, 0.25],
        multiplicity_method="bonferroni",
        alpha=0.025,
    )
    a = ar(z)
    n_ctrl = a["pooled_control"]["n_total"]
    ctrl_frac = n_ctrl / 300
    results.append({
        "test": "Structural: control_allocation=0.33 -> ~33% to control",
        "n_control": n_ctrl,
        "actual_frac": round(ctrl_frac, 3),
        "pass": abs(ctrl_frac - 0.33) < 0.02,
    })

    # 18. Biomarker prevalences affect treatment arm sizes
    # With prevalences [0.6, 0.2, 0.2], first arm should be largest
    z_prev = client.umbrella(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.6, 0.2, 0.2],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.35, 0.30, 0.25],
        multiplicity_method="bonferroni",
        alpha=0.025,
    )
    a_prev = ar(z_prev)
    n_trt_0 = a_prev["per_substudy"][0]["n_treatment"]
    n_trt_1 = a_prev["per_substudy"][1]["n_treatment"]
    n_trt_2 = a_prev["per_substudy"][2]["n_treatment"]
    results.append({
        "test": "Structural: prevalence [0.6,0.2,0.2] -> arm 1 largest",
        "n_trt": [n_trt_0, n_trt_1, n_trt_2],
        "pass": n_trt_0 > n_trt_1 and n_trt_0 > n_trt_2,
    })

    # 19. Holm multiplicity produces different adjusted thresholds per substudy
    z_holm = client.umbrella(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=600,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15, 0.15],
        alternative_rates=[0.40, 0.25, 0.16],  # varied effects -> varied p-values
        multiplicity_method="holm",
        alpha=0.025,
    )
    a_holm = ar(z_holm)
    adj_alphas = [s.get("adjusted_alpha") for s in a_holm["per_substudy"]]
    # Holm step-down: thresholds differ across substudies
    unique_thresholds = len(set(adj_alphas))
    results.append({
        "test": "Structural: Holm produces different thresholds",
        "adjusted_alphas": adj_alphas,
        "n_unique": unique_thresholds,
        "pass": unique_thresholds > 1,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Input validation tests using umbrella_raw."""
    results = []

    # 20. n_substudies < 2 rejected
    resp = client.umbrella_raw(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=1,
        total_n=100,
        control_allocation=0.33,
        biomarker_prevalences=[1.0],
        null_rates=[0.15],
        alternative_rates=[0.35],
        multiplicity_method="none",
        alpha=0.025,
    )
    results.append({
        "test": "Guard: n_substudies < 2 rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # 21. Mismatched array lengths rejected
    resp2 = client.umbrella_raw(
        endpoint_type="binary",
        analysis_type="frequentist",
        n_substudies=3,
        total_n=300,
        control_allocation=0.33,
        biomarker_prevalences=[0.4, 0.35, 0.25],
        null_rates=[0.15, 0.15],  # only 2 values for 3 substudies
        alternative_rates=[0.35, 0.30, 0.25],
        multiplicity_method="bonferroni",
        alpha=0.025,
    )
    results.append({
        "test": "Guard: mismatched array lengths rejected",
        "status_code": resp2.status_code,
        "pass": resp2.status_code in (400, 422),
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("UMBRELLA TRIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    sections = [
        ("1. Frequentist Binary", validate_frequentist_binary),
        ("2. Frequentist Continuous", validate_frequentist_continuous),
        ("3. Frequentist Survival", validate_frequentist_survival),
        ("4. Bayesian Binary", validate_bayesian_binary),
        ("5. Simulation: Frequentist Binary", validate_simulation_frequentist_binary),
        ("6. Simulation: Bayesian", validate_simulation_bayesian),
        ("7. Simulation: Survival", validate_simulation_survival),
        ("8. Structural Properties", validate_structural),
        ("9. Input Guards", validate_input_guards),
    ]

    for title, fn in sections:
        print(f"\n{title}")
        print("-" * 70)
        df = fn(client)
        print(df.to_string(index=False))
        all_frames.append(df)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/umbrella_validation.csv", index=False)

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
