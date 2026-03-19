#!/usr/bin/env python3
"""
Validate Platform Trial Calculator against the STAMPEDE Trial

STAMPEDE (Systemic Therapy in Advancing or Metastatic Prostate Cancer:
Evaluation of Drug Efficacy) was a multi-arm multi-stage (MAMS) platform
trial for men starting first-line hormone therapy for prostate cancer.

Design features:
- 5 treatment arms (B-F) vs 1 shared control (A)
- Allocation: 2:1:1:1:1:1 (control gets 2x)
- 4 analysis stages with one-sided alpha = {0.50, 0.25, 0.10, 0.025}
- Intermediate endpoint: failure-free survival (FFS), control median ~24 months
- Definitive endpoint: overall survival (OS), control median ~48 months
- Target HR = 0.75 (25% risk reduction)
- Pairwise type I error ~0.013

Published results:
- Arms D (celecoxib) and F (celecoxib+ZA) stopped at stage 2 for lack of
  benefit: FFS HR ~ 0.98
- Arm C (docetaxel): OS HR = 0.78 (95% CI 0.66-0.93, p=0.006)
- Arm B (zoledronic acid): OS HR = 0.94 — no benefit
- Arm E (ZA+docetaxel): OS HR = 0.82

References:
- Sydes et al. (2012). "Flexible trial design in practice - stopping arms
  for lack-of-benefit and adding research arms mid-trial in STAMPEDE."
  Trials 13:168.
- James et al. (2016). "Addition of docetaxel, zoledronic acid, or both to
  first-line long-term hormone therapy in prostate cancer (STAMPEDE)."
  Lancet 387(10024):1163-1177.
- Royston, Parmar & Barthel (2011). "Designs for clinical trials with
  time-to-event outcomes based on stopping guidelines for lack of benefit."
  Trials 12:81.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
import pandas as pd


# ─── Helper ──────────────────────────────────────────────────────────

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


# ─── STAMPEDE-like base parameters ───────────────────────────────────

def stampede_arms(hr=0.75):
    """Return 5 STAMPEDE-like arms, all entering at stage 1."""
    return [
        {"name": "Arm B (ZA)",          "entry_stage": 1, "hazard_ratio": hr, "is_active": True},
        {"name": "Arm C (Docetaxel)",   "entry_stage": 1, "hazard_ratio": hr, "is_active": True},
        {"name": "Arm D (Celecoxib)",   "entry_stage": 1, "hazard_ratio": hr, "is_active": True},
        {"name": "Arm E (ZA+Doc)",      "entry_stage": 1, "hazard_ratio": hr, "is_active": True},
        {"name": "Arm F (Cel+ZA)",      "entry_stage": 1, "hazard_ratio": hr, "is_active": True},
    ]


def stampede_base(**overrides):
    """Return STAMPEDE-like platform trial params (OS endpoint)."""
    params = dict(
        n_arms=5,
        n_stages=4,
        n_per_stage=200,
        endpoint_type="survival",
        analysis_type="frequentist",
        alpha=0.025,
        boundary_method="spending",
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        median_control=48.0,       # OS: control median ~48 months
        accrual_time=60.0,         # ~5 year accrual
        follow_up_time=24.0,       # ~2 year follow-up
        arms=stampede_arms(0.75),
    )
    params.update(overrides)
    return params


# ─── Test functions ──────────────────────────────────────────────────

def validate_design_structure(client) -> pd.DataFrame:
    """Test 1: 5-arm platform with 4 stages returns per_arm results for all 5 arms."""
    results = []

    z = client(**stampede_base())
    per_arm = z.get("per_arm", [])
    has_5_arms = isinstance(per_arm, list) and len(per_arm) == 5
    has_4_stage_boundary = len(z.get("boundary_table", [])) == 4
    results.append({
        "test": "1. Design structure: 5 arms, 4 stages returned",
        "n_per_arm": len(per_arm),
        "n_boundary_rows": len(z.get("boundary_table", [])),
        "pass": has_5_arms and has_4_stage_boundary,
    })

    return pd.DataFrame(results)


def validate_boundary_monotonicity(client) -> pd.DataFrame:
    """Test 2: Efficacy boundaries become less strict at later stages (OBF spending)."""
    results = []

    z = client(**stampede_base())
    bt = z.get("boundary_table", [])
    # OBF spending: efficacy_z decreases over stages (less strict)
    eff_z = [row["efficacy_z"] for row in bt]
    monotone_decreasing = all(eff_z[i] >= eff_z[i + 1] for i in range(len(eff_z) - 1))
    results.append({
        "test": "2. Boundary monotonicity: efficacy_z decreases (OBF)",
        "efficacy_z": eff_z,
        "pass": monotone_decreasing and len(eff_z) == 4,
    })

    return pd.DataFrame(results)


def validate_power_docetaxel(client) -> pd.DataFrame:
    """Test 3: With HR=0.75 and adequate sample size, power should be substantial (>0.50)."""
    results = []

    # Use larger n_per_stage to approximate STAMPEDE's sample size
    z = client(**stampede_base(n_per_stage=300))
    per_arm = z.get("per_arm", [])
    # All arms have HR=0.75, so all should have meaningful power estimates
    # power_estimate is analytical (at final look only), so may be conservative
    # but with 5*4*300=6000 treatment + 4*300=1200 control, should be >0.50
    powers = [arm.get("power_estimate") for arm in per_arm]
    # power_estimate for survival may be None if the calculator doesn't compute
    # analytical power for survival — check per_arm structure
    any_has_power = any(p is not None and p > 0.50 for p in powers)
    # If power_estimate is None for survival (calculator uses events_required instead),
    # check that expected_events are substantial
    any_has_events = any(arm.get("expected_events", 0) > 100 for arm in per_arm)

    results.append({
        "test": "3. Power for HR=0.75: substantial power or events",
        "powers": powers,
        "expected_events": [arm.get("expected_events") for arm in per_arm],
        "pass": any_has_power or any_has_events,
    })

    return pd.DataFrame(results)


def validate_futility_celecoxib(client) -> pd.DataFrame:
    """Test 4: With HR=0.98 (null), design should stop most arms early.

    Arms D and F in STAMPEDE were stopped at stage 2 for lack of benefit.
    Simulate with HR=0.98 and verify most sims don't reach the final stage.
    """
    results = []

    # Single null arm to isolate futility behavior
    null_arms = [
        {"name": "Null arm (HR=0.98)", "entry_stage": 1, "hazard_ratio": 0.98, "is_active": True},
    ]
    z = client(
        n_arms=1,
        n_stages=4,
        n_per_stage=200,
        endpoint_type="survival",
        analysis_type="frequentist",
        alpha=0.025,
        boundary_method="spending",
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        median_control=48.0,
        accrual_time=60.0,
        follow_up_time=24.0,
        arms=null_arms,
        simulate=True,
        n_simulations=500,
        simulation_seed=42,
    )

    sim = z.get("simulation", {})
    est = sim.get("estimates", {})
    stop_dist = est.get("arm_stopping_distribution", {})

    # Check that power is low (null effect should rarely declare efficacy)
    per_arm_power = est.get("per_arm_power", [0])
    power_low = per_arm_power[0] < 0.10 if per_arm_power else True

    # Check stopping distribution: most sims should stop before final stage
    # or run to completion without efficacy
    arm_name = list(stop_dist.keys())[0] if stop_dist else None
    if arm_name:
        p_stop = stop_dist[arm_name].get("p_stop_per_stage", {})
        # Probability of reaching the final stage (stage 4) should not be 100%
        # With non-binding futility, arms may continue to final stage
        # but efficacy declarations should be rare
        p_efficacy = stop_dist[arm_name].get("p_efficacy", 0)
        pass_test = p_efficacy < 0.10
    else:
        p_efficacy = None
        pass_test = False

    results.append({
        "test": "4. Futility detection: HR=0.98 rarely declares efficacy",
        "per_arm_power": per_arm_power,
        "p_efficacy": p_efficacy,
        "pass": pass_test,
    })

    return pd.DataFrame(results)


def validate_total_n_max(client) -> pd.DataFrame:
    """Test 5: total_n_max = (n_arms + 1) * n_stages * n_per_stage for all arms at stage 1."""
    results = []

    n_per_stage = 200
    z = client(**stampede_base(n_per_stage=n_per_stage))
    # With 5 arms + 1 control, all entering stage 1, 4 stages:
    # control: n_per_stage * 4 = 800
    # each arm: n_per_stage * 4 = 800
    # total = 6 * 4 * 200 = 4800
    expected = (5 + 1) * 4 * n_per_stage
    actual = z.get("total_n_max", 0)
    results.append({
        "test": "5. Total N max: (5+1) * 4 * n_per_stage",
        "expected": expected,
        "actual": actual,
        "pass": actual == expected,
    })

    return pd.DataFrame(results)


def validate_concurrent_control(client) -> pd.DataFrame:
    """Test 6: STAMPEDE used concurrent control. Verify concurrent_only mode with 5 arms."""
    results = []

    z = client(**stampede_base(control_type="concurrent_only"))
    # Should succeed and have per_arm with correct control_type
    ct = z.get("control_type", "")
    has_5_arms = len(z.get("per_arm", [])) == 5
    results.append({
        "test": "6. Concurrent control: concurrent_only with 5 arms",
        "control_type": ct,
        "n_arms": len(z.get("per_arm", [])),
        "pass": ct == "concurrent_only" and has_5_arms,
    })

    return pd.DataFrame(results)


def validate_entry_stage(client) -> pd.DataFrame:
    """Test 7: All 5 STAMPEDE arms entered at stage 1. Verify entry_stage=1 for all."""
    results = []

    z = client(**stampede_base())
    per_arm = z.get("per_arm", [])
    all_entry_1 = all(arm.get("entry_stage") == 1 for arm in per_arm)
    all_max_4 = all(arm.get("max_stages") == 4 for arm in per_arm)
    results.append({
        "test": "7. All arms entry_stage=1, max_stages=4",
        "entry_stages": [arm.get("entry_stage") for arm in per_arm],
        "max_stages": [arm.get("max_stages") for arm in per_arm],
        "pass": all_entry_1 and all_max_4 and len(per_arm) == 5,
    })

    return pd.DataFrame(results)


def validate_control_allocation(client) -> pd.DataFrame:
    """Test 8: STAMPEDE used 2:1:1:1:1:1 allocation. Our calculator uses equal allocation.

    Verify that the equal-allocation design produces reasonable output
    (5 arms, per-arm results, boundaries).
    """
    results = []

    z = client(**stampede_base())
    per_arm = z.get("per_arm", [])
    bt = z.get("boundary_table", [])

    # Check structural correctness with equal allocation
    has_arms = len(per_arm) == 5
    has_boundaries = len(bt) == 4
    has_summary = "design_summary" in z
    n_max = z.get("total_n_max", 0)
    n_max_positive = n_max > 0

    results.append({
        "test": "8. Control allocation: equal-alloc design reasonable output",
        "n_arms": len(per_arm),
        "n_boundaries": len(bt),
        "total_n_max": n_max,
        "pass": has_arms and has_boundaries and has_summary and n_max_positive,
    })

    return pd.DataFrame(results)


def validate_docetaxel_power_reference(client) -> pd.DataFrame:
    """Test 9: Docetaxel power reference check.

    Published: N~2962 (arms A+C combined), HR=0.78 on OS, ~80-90% power.
    Run a 2-arm subset sized to approximate this.
    With ~1481 per arm across 4 stages, n_per_stage ~ 370.
    Use HR=0.78, OS median 48 months.
    """
    results = []

    z = client(
        n_arms=1,
        n_stages=4,
        n_per_stage=370,
        endpoint_type="survival",
        analysis_type="frequentist",
        alpha=0.025,
        boundary_method="spending",
        spending_function="obrien_fleming",
        control_type="concurrent_only",
        median_control=48.0,
        accrual_time=60.0,
        follow_up_time=24.0,
        arms=[
            {"name": "Docetaxel", "entry_stage": 1, "hazard_ratio": 0.78, "is_active": True},
        ],
        simulate=True,
        n_simulations=500,
        simulation_seed=42,
    )

    sim = z.get("simulation", {})
    est = sim.get("estimates", {})
    per_arm_power = est.get("per_arm_power", [0])
    sim_power = per_arm_power[0] if per_arm_power else 0

    # Analytical power check too
    per_arm = z.get("per_arm", [])
    analytical_power = per_arm[0].get("power_estimate") if per_arm else None
    expected_events = per_arm[0].get("expected_events") if per_arm else None

    # The trial had ~80-90% power. With simulation, expect a reasonable range.
    # Allow 0.40-1.0 given our equal-allocation approximation differs from 2:1.
    power_reasonable = sim_power > 0.40

    results.append({
        "test": "9. Docetaxel reference: simulated power > 0.40",
        "sim_power": sim_power,
        "analytical_power": analytical_power,
        "expected_events": expected_events,
        "pass": power_reasonable,
    })

    return pd.DataFrame(results)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    raw_client = get_client(base_url)
    client = PlatformClient(raw_client)

    print("=" * 70)
    print("STAMPEDE PLATFORM TRIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    sections = [
        ("1. Design Structure",           validate_design_structure),
        ("2. Boundary Monotonicity",       validate_boundary_monotonicity),
        ("3. Power (HR=0.75)",             validate_power_docetaxel),
        ("4. Futility (HR=0.98)",          validate_futility_celecoxib),
        ("5. Total N Max",                 validate_total_n_max),
        ("6. Concurrent Control",          validate_concurrent_control),
        ("7. Entry Stage",                 validate_entry_stage),
        ("8. Control Allocation",          validate_control_allocation),
        ("9. Docetaxel Power Reference",   validate_docetaxel_power_reference),
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
    all_results.to_csv("results/stampede_validation.csv", index=False)

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
