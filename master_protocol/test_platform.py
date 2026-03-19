#!/usr/bin/env python3
"""
Validate Platform Trial Calculator (MAMS)

Tests multi-arm multi-stage platform trial design across:
1. Frequentist binary, continuous, survival endpoints
2. Bayesian binary analysis
3. Staggered arm entry
4. Control pooling modes (concurrent_only, pooled_adjusted, pooled_naive)
5. Monte Carlo simulation (power, type I error, stopping distribution)
6. Spending functions (O'Brien-Fleming, Pocock)
7. Input validation guards

References:
- Royston, Parmar & Barthel (2011). "Designs for clinical trials with
  time-to-event outcomes based on stopping guidelines for lack of benefit."
  Trials 12:81.
- Saville & Berry (2016). "Efficiencies of platform clinical trials."
  Clinical Trials 13(6):557-565.
- FDA (2022). "Master Protocols: Efficient Clinical Trial Design Strategies."
- Woodcock & LaVange (2017). "Master protocols to study multiple therapies,
  multiple diseases, or both." NEJM 377(1):62-70.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import mc_rate_upper_bound
import pandas as pd


# ─── Helper: standard 2-arm 3-stage binary params ───────────────────

def extract_analytical(resp: dict) -> dict:
    """Extract analytical_results from the full API response, merging simulation if present."""
    ar = resp.get("analytical_results", resp)
    if "simulation" in resp and resp["simulation"]:
        ar["simulation"] = resp["simulation"]
    if "warnings" in resp:
        ar["warnings"] = resp["warnings"]
    return ar


class PlatformClient:
    """Wrapper that auto-extracts analytical_results from platform responses."""
    def __init__(self, client):
        self._client = client

    def __call__(self, **kwargs):
        resp = self._client.platform(**kwargs)
        return extract_analytical(resp)

    def raw(self, **kwargs):
        return self._client.platform_raw(**kwargs)


def base_binary_params(**overrides):
    """Return standard frequentist binary platform params with overrides."""
    params = dict(
        n_arms=2,
        n_stages=3,
        n_per_stage=100,
        endpoint_type="binary",
        analysis_type="frequentist",
        null_rate=0.15,
        alpha=0.025,
        boundary_method="spending",
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
            {"name": "Arm B", "entry_stage": 1, "response_rate": 0.30, "is_active": True},
        ],
    )
    params.update(overrides)
    return params


# ─── Test functions ──────────────────────────────────────────────────

def validate_frequentist_binary(client) -> pd.DataFrame:
    """Tests 1-5: Frequentist binary endpoint."""
    results = []

    # Test 1: Basic 2-arm 3-stage returns per_arm_results
    z = client(**base_binary_params())
    has_per_arm = "per_arm" in z and isinstance(z["per_arm"], list) and len(z["per_arm"]) == 2
    results.append({
        "test": "1. Freq binary: per_arm returned for 2-arm 3-stage",
        "n_per_arm": len(z.get("per_arm", [])),
        "pass": has_per_arm,
    })

    # Test 2: Power > 0 for active arms (response_rate > null_rate)
    powers = [arm.get("power_estimate") for arm in z.get("per_arm", [])]
    all_power_positive = all(p is not None and p > 0 for p in powers)
    results.append({
        "test": "2. Freq binary: power > 0 for active arms",
        "powers": powers,
        "pass": all_power_positive,
    })

    # Test 3: Boundaries returned per stage
    bt = z.get("boundary_table", [])
    has_boundaries = (
        len(bt) == 3
        and all("efficacy_z" in row for row in bt)
        and all("futility_z" in row for row in bt)
    )
    results.append({
        "test": "3. Freq binary: boundaries per stage",
        "n_boundary_rows": len(bt),
        "pass": has_boundaries,
    })

    # Test 4: Total N max = (n_arms + 1_control) * n_stages * n_per_stage
    # For all arms entering at stage 1: control gets n_per_stage * n_stages,
    # each arm gets n_per_stage * n_stages
    expected_max = (2 + 1) * 3 * 100  # 2 arms + 1 control, 3 stages, 100/stage
    actual_max = z.get("total_n_max", 0)
    results.append({
        "test": "4. Freq binary: total_n_max correct",
        "expected": expected_max,
        "actual": actual_max,
        "pass": actual_max == expected_max,
    })

    # Test 5: Bonferroni boundary method works
    z_bonf = client(**base_binary_params(boundary_method="bonferroni"))
    bt_bonf = z_bonf.get("boundary_table", [])
    bonf_ok = len(bt_bonf) == 3 and all("efficacy_z" in row for row in bt_bonf)
    results.append({
        "test": "5. Freq binary: Bonferroni boundary method works",
        "n_boundary_rows": len(bt_bonf),
        "pass": bonf_ok,
    })

    return pd.DataFrame(results)


def validate_bayesian_binary(client) -> pd.DataFrame:
    """Tests 6-7: Bayesian binary endpoint."""
    results = []

    # Test 6: Bayesian analysis returns posterior-based decisions
    z = client(
        n_arms=2, n_stages=3, n_per_stage=100,
        endpoint_type="binary", analysis_type="bayesian",
        null_rate=0.15, alpha=0.025,
        efficacy_threshold=0.975, futility_threshold=0.10,
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
            {"name": "Arm B", "entry_stage": 1, "response_rate": 0.30, "is_active": True},
        ],
    )

    bt = z.get("boundary_table", [])
    has_bayesian_fields = (
        len(bt) == 3
        and all("efficacy_threshold" in row for row in bt)
        and all("futility_threshold" in row for row in bt)
    )
    results.append({
        "test": "6. Bayesian binary: posterior-based boundary table",
        "n_rows": len(bt),
        "pass": has_bayesian_fields,
    })

    # Test 7: With strong effect, simulation shows high power (early stopping)
    z_sim = client(
        n_arms=1, n_stages=3, n_per_stage=200,
        endpoint_type="binary", analysis_type="bayesian",
        null_rate=0.15, alpha=0.025,
        efficacy_threshold=0.975, futility_threshold=0.10,
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.50, "is_active": True},
        ],
        simulate=True, n_simulations=1000, simulation_seed=42,
    )
    sim = z_sim.get("simulation", {})
    est = sim.get("estimates", {})
    power = est.get("any_arm_power", 0)
    results.append({
        "test": "7. Bayesian binary sim: strong effect -> high power",
        "power": power,
        "pass": power is not None and power > 0.5,
    })

    return pd.DataFrame(results)


def validate_frequentist_continuous(client) -> pd.DataFrame:
    """Tests 8-9: Frequentist continuous endpoint."""
    results = []

    # Test 8: Continuous endpoint works with mean difference
    z = client(
        n_arms=2, n_stages=3, n_per_stage=100,
        endpoint_type="continuous", analysis_type="frequentist",
        null_mean=0.0, common_sd=1.0, alpha=0.025,
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "mean_effect": 0.3, "is_active": True},
            {"name": "Arm B", "entry_stage": 1, "mean_effect": 0.5, "is_active": True},
        ],
    )

    has_per_arm = "per_arm" in z and len(z["per_arm"]) == 2
    powers = [arm.get("power_estimate") for arm in z.get("per_arm", [])]
    results.append({
        "test": "8. Freq continuous: endpoint works",
        "powers": powers,
        "pass": has_per_arm and all(p is not None for p in powers),
    })

    # Test 9: Large effect → high power
    large_effect_power = powers[1] if len(powers) > 1 else 0  # Arm B has effect=0.5
    small_effect_power = powers[0] if len(powers) > 0 else 0  # Arm A has effect=0.3
    results.append({
        "test": "9. Freq continuous: larger effect -> higher power",
        "power_small": small_effect_power,
        "power_large": large_effect_power,
        "pass": large_effect_power > small_effect_power,
    })

    return pd.DataFrame(results)


def validate_frequentist_survival(client) -> pd.DataFrame:
    """Tests 10-11: Frequentist survival endpoint."""
    results = []

    # Test 10: Survival with HR < 1 works
    z = client(
        n_arms=2, n_stages=3, n_per_stage=150,
        endpoint_type="survival", analysis_type="frequentist",
        median_control=12.0, accrual_time=24.0, follow_up_time=12.0,
        alpha=0.025, spending_function="obrien_fleming",
        control_type="concurrent_only",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "hazard_ratio": 0.7, "is_active": True},
            {"name": "Arm B", "entry_stage": 1, "hazard_ratio": 0.8, "is_active": True},
        ],
    )

    has_per_arm = "per_arm" in z and len(z["per_arm"]) == 2
    results.append({
        "test": "10. Freq survival: HR < 1 works",
        "n_per_arm": len(z.get("per_arm", [])),
        "pass": has_per_arm,
    })

    # Test 11: Events per stage present in survival results
    arm_a = z["per_arm"][0] if has_per_arm else {}
    has_events = "expected_events" in arm_a or "events_required_80pct" in arm_a
    results.append({
        "test": "11. Freq survival: events info present",
        "has_expected_events": "expected_events" in arm_a,
        "has_events_required": "events_required_80pct" in arm_a,
        "pass": has_events,
    })

    return pd.DataFrame(results)


def validate_staggered_entry(client) -> pd.DataFrame:
    """Tests 12-13: Staggered arm entry."""
    results = []

    # Test 12: Arms with entry_stage > 1 are correctly handled
    z = client(
        n_arms=2, n_stages=4, n_per_stage=100,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
            {"name": "Arm B", "entry_stage": 3, "response_rate": 0.35, "is_active": True},
        ],
    )

    arm_a = z["per_arm"][0]
    arm_b = z["per_arm"][1]
    a_entry = arm_a.get("entry_stage", 0)
    b_entry = arm_b.get("entry_stage", 0)
    results.append({
        "test": "12. Staggered: entry_stage > 1 handled",
        "arm_a_entry": a_entry,
        "arm_b_entry": b_entry,
        "pass": a_entry == 1 and b_entry == 3,
    })

    # Test 13: Late-entering arm has fewer stages of data
    a_max_stages = arm_a.get("max_stages", 0)
    b_max_stages = arm_b.get("max_stages", 0)
    results.append({
        "test": "13. Staggered: late arm fewer stages",
        "arm_a_max_stages": a_max_stages,
        "arm_b_max_stages": b_max_stages,
        "pass": a_max_stages > b_max_stages and b_max_stages == 2,
    })

    return pd.DataFrame(results)


def validate_control_modes(client) -> pd.DataFrame:
    """Tests 14-16: Control pooling modes."""
    results = []

    # Common params with staggered entry to make control type matter
    common = dict(
        n_arms=2, n_stages=4, n_per_stage=100,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        spending_function="obrien_fleming",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
            {"name": "Arm B", "entry_stage": 3, "response_rate": 0.35, "is_active": True},
        ],
    )

    # Test 14: concurrent_only
    z_conc = client(**common, control_type="concurrent_only")
    arm_b_conc = z_conc["per_arm"][1]
    # For concurrent_only, Arm B (entry_stage=3) at its final stage (stage 4)
    # should have control n = n_per_stage * (4-3+1) = 200
    b_stages_conc = arm_b_conc.get("stages", [])
    b_final_ctrl_conc = b_stages_conc[-1]["n_control_cumulative"] if b_stages_conc else 0
    results.append({
        "test": "14. concurrent_only: limited control",
        "arm_b_ctrl_final": b_final_ctrl_conc,
        "pass": b_final_ctrl_conc == 200,  # 2 stages * 100
    })

    # Test 15: pooled_adjusted
    z_adj = client(**common, control_type="pooled_adjusted")
    arm_b_adj = z_adj["per_arm"][1]
    b_stages_adj = arm_b_adj.get("stages", [])
    b_final_ctrl_adj = b_stages_adj[-1]["n_control_cumulative"] if b_stages_adj else 0
    # pooled_adjusted: concurrent full weight + non-concurrent 0.5 weight
    # concurrent stages = 2, non-concurrent = 2 (stages 1,2)
    # n_ctrl = 100*2 + 100*2*0.5 = 300
    results.append({
        "test": "15. pooled_adjusted: more control data",
        "arm_b_ctrl_final": b_final_ctrl_adj,
        "pass": b_final_ctrl_adj > b_final_ctrl_conc,
    })

    # Test 16: pooled_naive
    z_naive = client(**common, control_type="pooled_naive")
    arm_b_naive = z_naive["per_arm"][1]
    b_stages_naive = arm_b_naive.get("stages", [])
    b_final_ctrl_naive = b_stages_naive[-1]["n_control_cumulative"] if b_stages_naive else 0
    # pooled_naive: all stages pooled equally
    # n_ctrl = n_per_stage * global_stage = 100 * 4 = 400
    results.append({
        "test": "16. pooled_naive: most control data",
        "arm_b_ctrl_final": b_final_ctrl_naive,
        "pass": b_final_ctrl_naive >= b_final_ctrl_adj and b_final_ctrl_naive == 400,
    })

    return pd.DataFrame(results)


def validate_simulation(client) -> pd.DataFrame:
    """Tests 17-19: Monte Carlo simulation tests."""
    results = []

    sim_params = dict(
        simulate=True,
        n_simulations=2000,
        simulation_seed=42,
    )

    # Test 17: Binary sim: power > 0 for active arms
    z = client(
        n_arms=2, n_stages=3, n_per_stage=150,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
            {"name": "Arm B", "entry_stage": 1, "response_rate": 0.30, "is_active": True},
        ],
        **sim_params,
    )

    sim = z.get("simulation", {})
    est = sim.get("estimates", {})
    per_arm_power = est.get("per_arm_power", [])
    any_power = est.get("any_arm_power", 0)
    all_power_pos = all(p is not None and p > 0 for p in per_arm_power)
    results.append({
        "test": "17. Sim binary: power > 0 for active arms",
        "per_arm_power": per_arm_power,
        "any_arm_power": any_power,
        "pass": all_power_pos and any_power > 0,
    })

    # Test 18: Type I error ≤ 0.10 under null (generous bound for MC noise)
    # Use mc_rate_upper_bound for proper MC comparison
    z_null = client(
        n_arms=2, n_stages=3, n_per_stage=100,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.15, "is_active": False},
            {"name": "Arm B", "entry_stage": 1, "response_rate": 0.15, "is_active": False},
        ],
        **sim_params,
    )
    sim_null = z_null.get("simulation", {})
    fwer = sim_null.get("type1_error", 1.0)
    n_sims = sim_null.get("n_simulations", 2000)
    fwer_upper = mc_rate_upper_bound(fwer, n_sims, confidence=0.99)
    results.append({
        "test": "18. Sim binary: type I error <= 0.10",
        "fwer": fwer,
        "fwer_upper_99ci": round(fwer_upper, 4),
        "pass": fwer_upper <= 0.10,
    })

    # Test 19: Stopping distribution returned
    stop_dist = est.get("arm_stopping_distribution", {})
    has_stop_dist = len(stop_dist) > 0
    has_p_stop = any(
        "p_stop_per_stage" in v for v in stop_dist.values()
    ) if has_stop_dist else False
    results.append({
        "test": "19. Sim binary: stopping distribution returned",
        "n_arms_with_dist": len(stop_dist),
        "has_p_stop_per_stage": has_p_stop,
        "pass": has_stop_dist and has_p_stop,
    })

    return pd.DataFrame(results)


def validate_spending_functions(client) -> pd.DataFrame:
    """Tests 20-21: Spending function behavior."""
    results = []

    # Test 20: OBF spending: early boundaries stricter than late
    z_obf = client(**base_binary_params(spending_function="obrien_fleming"))
    bt_obf = z_obf.get("boundary_table", [])
    if len(bt_obf) >= 2:
        early_z = bt_obf[0].get("efficacy_z", 0)
        late_z = bt_obf[-1].get("efficacy_z", 0)
        obf_ok = early_z > late_z  # OBF: early boundaries are stricter (higher z)
    else:
        early_z = late_z = None
        obf_ok = False
    results.append({
        "test": "20. OBF spending: early stricter than late",
        "early_z": early_z,
        "late_z": late_z,
        "pass": obf_ok,
    })

    # Test 21: Pocock spending: boundaries more uniform
    z_poc = client(**base_binary_params(spending_function="pocock"))
    bt_poc = z_poc.get("boundary_table", [])
    if len(bt_poc) >= 2:
        poc_early = bt_poc[0].get("efficacy_z", 0)
        poc_late = bt_poc[-1].get("efficacy_z", 0)
        # Pocock boundaries should be more uniform than OBF
        poc_range = abs(poc_early - poc_late)
        obf_range = abs(early_z - late_z) if early_z is not None else 999
        poc_ok = poc_range < obf_range
    else:
        poc_early = poc_late = None
        poc_ok = False
    results.append({
        "test": "21. Pocock spending: more uniform than OBF",
        "pocock_range": round(poc_range, 4) if poc_early is not None else None,
        "obf_range": round(obf_range, 4) if early_z is not None else None,
        "pass": poc_ok,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Tests 22-24: Input validation (use platform_raw)."""
    results = []

    # Test 22: n_stages < 2 rejected
    resp = client.raw(
        n_arms=2, n_stages=1, n_per_stage=100,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
        ],
    )
    results.append({
        "test": "22. Guard: n_stages < 2 rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422, 500),
    })

    # Test 23: n_per_stage < 1 rejected
    resp = client.raw(
        n_arms=2, n_stages=3, n_per_stage=0,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        arms=[
            {"name": "Arm A", "entry_stage": 1, "response_rate": 0.35, "is_active": True},
        ],
    )
    results.append({
        "test": "23. Guard: n_per_stage < 1 rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422, 500),
    })

    # Test 24: Empty arms list uses default fallback (n_arms=2)
    z = client(
        n_arms=2, n_stages=3, n_per_stage=100,
        endpoint_type="binary", analysis_type="frequentist",
        null_rate=0.15, alpha=0.025,
        arms=[],
    )
    has_per_arm = "per_arm" in z and len(z.get("per_arm", [])) == 2
    results.append({
        "test": "24. Guard: empty arms uses default fallback",
        "n_per_arm": len(z.get("per_arm", [])),
        "pass": has_per_arm,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    raw_client = get_client(base_url)
    client = PlatformClient(raw_client)

    print("=" * 70)
    print("PLATFORM TRIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    sections = [
        ("1. Frequentist Binary", validate_frequentist_binary),
        ("2. Bayesian Binary", validate_bayesian_binary),
        ("3. Frequentist Continuous", validate_frequentist_continuous),
        ("4. Frequentist Survival", validate_frequentist_survival),
        ("5. Staggered Entry", validate_staggered_entry),
        ("6. Control Modes", validate_control_modes),
        ("7. Simulation", validate_simulation),
        ("8. Spending Functions", validate_spending_functions),
        ("9. Input Guards", validate_input_guards),
    ]

    for title, func in sections:
        print(f"\n{title}")
        print("-" * 70)
        df = func(client)
        print(df.to_string(index=False))
        all_frames.append(df)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/platform_validation.csv", index=False)

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
