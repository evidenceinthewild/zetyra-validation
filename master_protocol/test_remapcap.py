#!/usr/bin/env python3
"""
Validate Platform Trial Calculator Against REMAP-CAP Design Parameters

Tests the platform trial calculator using published REMAP-CAP trial results,
mapping mortality outcomes to binary success rates (higher = better).

REMAP-CAP Design:
- Bayesian adaptive platform trial for COVID-19 treatments
- Multiple domains: corticosteroids, IL-6 receptor antagonists, anticoagulation, antivirals
- Superiority threshold: posterior probability > 99%
- Futility threshold: posterior probability > 95% that OR < 1.2
- Concurrent control shared across domains

Key Published Results (binary mortality approximation):
- IL-6 RA (Tocilizumab): n=353 trt, n=402 ctrl, mortality 28.0% vs 35.8%
  -> posterior prob >99.9%, STOPPED FOR SUPERIORITY
- IL-6 RA (Sarilumab): n=48 trt, n=63 ctrl, posterior prob 99.5%
- Anticoagulation (critical): n=591 trt, n=616 ctrl, posterior 99.9% futility
  -> STOPPED FOR FUTILITY
- Lopinavir-ritonavir: n=255 trt, n=362 ctrl, mortality 35.3% vs 30.0%
  -> posterior >=99.9% futility, STOPPED FOR FUTILITY
- Hydroxychloroquine: n=50 trt, mortality 34.7%, posterior >=99.9% futility

Endpoint mapping:
- REMAP-CAP: lower mortality = better
- Our calculator: higher response_rate = better
- Mapping: response_rate = 1 - mortality_rate

References:
- Angus DC et al. (2020). "Effect of Hydrocortisone on Mortality and Organ
  Support in Patients With Severe COVID-19: The REMAP-CAP COVID-19
  Corticosteroid Domain Randomized Clinical Trial." JAMA 324(13):1317-1329.
- REMAP-CAP Investigators (2021). "Interleukin-6 Receptor Antagonists in
  Critically Ill Patients with Covid-19." NEJM 384(16):1491-1502.
- REMAP-CAP Investigators (2021). "Therapeutic Anticoagulation with Heparin
  in Critically Ill Patients with Covid-19." NEJM 385(9):777-789.
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


# ─── REMAP-CAP mortality -> success rate mapping ─────────────────────
# control mortality 35.8% -> control success (null_rate) = 0.642
# tocilizumab mortality 28.0% -> success = 0.720
# lopinavir mortality 35.3% -> success = 0.647 (barely above null)
# sarilumab: similar effect to tocilizumab
# hydroxychloroquine mortality 34.7% -> success = 0.653

CTRL_MORTALITY = 0.358
CTRL_SUCCESS = 1 - CTRL_MORTALITY  # 0.642

TOCI_MORTALITY = 0.280
TOCI_SUCCESS = 1 - TOCI_MORTALITY  # 0.720

SARI_SUCCESS = 0.710  # approximate, similar to tocilizumab

LOPI_MORTALITY = 0.353
LOPI_SUCCESS = 1 - LOPI_MORTALITY  # 0.647 (treatment is WORSE — higher mortality)

HCQ_MORTALITY = 0.347
HCQ_SUCCESS = 1 - HCQ_MORTALITY   # 0.653


# ─── Test functions ──────────────────────────────────────────────────

def validate_tocilizumab_superiority(client) -> pd.DataFrame:
    """Test 1: Tocilizumab shows superiority with REMAP-CAP-like parameters.

    With control success 0.642 and treatment success 0.720 (large effect),
    Bayesian analysis should show high posterior probability of superiority.
    """
    results = []

    z = client(
        n_arms=1,
        n_stages=3,
        n_per_stage=150,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Tocilizumab", "entry_stage": 1,
             "response_rate": TOCI_SUCCESS, "is_active": True},
        ],
    )

    per_arm = z.get("per_arm", [])
    toci = per_arm[0] if per_arm else {}
    power = toci.get("power_estimate", 0)

    results.append({
        "test": "1. Tocilizumab superiority: power > 0 with 99% threshold",
        "power_estimate": power,
        "effect_delta": round(TOCI_SUCCESS - CTRL_SUCCESS, 3),
        "pass": power is not None and power > 0,
    })

    return pd.DataFrame(results)


def validate_lopinavir_futility(client) -> pd.DataFrame:
    """Test 2: Lopinavir-ritonavir should not show superiority.

    Treatment mortality 35.3% vs control 30.0% — treatment is WORSE.
    Mapped: treatment success 0.647 vs control success 0.642.
    The effect is negligible/wrong direction. Should not show high power.
    """
    results = []

    z = client(
        n_arms=1,
        n_stages=3,
        n_per_stage=150,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Lopinavir", "entry_stage": 1,
             "response_rate": LOPI_SUCCESS, "is_active": True},
        ],
    )

    per_arm = z.get("per_arm", [])
    lopi = per_arm[0] if per_arm else {}
    power = lopi.get("power_estimate", 0)

    # With negligible effect (0.647 vs 0.642 = +0.005), power should be very low
    results.append({
        "test": "2. Lopinavir futility: power < 0.20 with negligible effect",
        "power_estimate": power,
        "effect_delta": round(LOPI_SUCCESS - CTRL_SUCCESS, 3),
        "pass": power is not None and power < 0.20,
    })

    return pd.DataFrame(results)


def validate_multi_domain(client) -> pd.DataFrame:
    """Test 3: Multi-domain platform with tocilizumab, sarilumab, lopinavir.

    Different arms should show different power estimates reflecting effect sizes.
    Tocilizumab (large effect) > Sarilumab (moderate) > Lopinavir (near null).
    """
    results = []

    z = client(
        n_arms=3,
        n_stages=3,
        n_per_stage=150,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Tocilizumab", "entry_stage": 1,
             "response_rate": TOCI_SUCCESS, "is_active": True},
            {"name": "Sarilumab", "entry_stage": 1,
             "response_rate": SARI_SUCCESS, "is_active": True},
            {"name": "Lopinavir", "entry_stage": 1,
             "response_rate": LOPI_SUCCESS, "is_active": True},
        ],
    )

    per_arm = z.get("per_arm", [])
    has_3_arms = len(per_arm) == 3
    powers = [arm.get("power_estimate", 0) for arm in per_arm]

    # Tocilizumab power > Lopinavir power (largest vs smallest effect)
    if len(powers) == 3:
        toci_power, sari_power, lopi_power = powers
        ordering_ok = toci_power > lopi_power
    else:
        toci_power = sari_power = lopi_power = None
        ordering_ok = False

    results.append({
        "test": "3. Multi-domain: 3 arms with different powers",
        "toci_power": toci_power,
        "sari_power": sari_power,
        "lopi_power": lopi_power,
        "pass": has_3_arms and ordering_ok,
    })

    return pd.DataFrame(results)


def validate_sample_size(client) -> pd.DataFrame:
    """Test 4: REMAP-CAP-like sample sizes (~350/arm) produce meaningful results.

    With n_per_stage=120 and 3 stages, each arm gets ~360 total,
    matching REMAP-CAP tocilizumab arm (n=353).
    """
    results = []

    z = client(
        n_arms=1,
        n_stages=3,
        n_per_stage=120,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Tocilizumab", "entry_stage": 1,
             "response_rate": TOCI_SUCCESS, "is_active": True},
        ],
    )

    total_n_max = z.get("total_n_max", 0)
    per_arm = z.get("per_arm", [])
    power = per_arm[0].get("power_estimate", 0) if per_arm else 0

    # total_n_max = (1 arm + 1 control) * 3 stages * 120 = 720
    expected_max = 2 * 3 * 120
    results.append({
        "test": "4. Sample size: ~360/arm produces meaningful power",
        "total_n_max": total_n_max,
        "expected_max": expected_max,
        "power": power,
        "pass": total_n_max == expected_max and power is not None and power > 0,
    })

    return pd.DataFrame(results)


def validate_threshold_effect(client) -> pd.DataFrame:
    """Test 5: Higher efficacy threshold (0.99 vs 0.95) reduces efficacy declarations.

    With a marginal effect, 99% threshold should yield lower power than 95%.
    Use sarilumab-like moderate effect to see the difference.
    """
    results = []

    common = dict(
        n_arms=1,
        n_stages=3,
        n_per_stage=100,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Sarilumab", "entry_stage": 1,
             "response_rate": SARI_SUCCESS, "is_active": True},
        ],
    )

    z_strict = client(**common, efficacy_threshold=0.99)
    z_lenient = client(**common, efficacy_threshold=0.95)

    per_arm_strict = z_strict.get("per_arm", [])
    per_arm_lenient = z_lenient.get("per_arm", [])
    power_strict = per_arm_strict[0].get("power_estimate", 0) if per_arm_strict else 0
    power_lenient = per_arm_lenient[0].get("power_estimate", 0) if per_arm_lenient else 0

    results.append({
        "test": "5. Threshold effect: 99% threshold <= 95% threshold power",
        "power_99pct": power_strict,
        "power_95pct": power_lenient,
        "pass": power_strict <= power_lenient,
    })

    return pd.DataFrame(results)


def validate_concurrent_control(client) -> pd.DataFrame:
    """Test 6: REMAP-CAP uses concurrent control. Verify concurrent_only mode
    with Bayesian analysis returns valid results.
    """
    results = []

    z = client(
        n_arms=2,
        n_stages=3,
        n_per_stage=120,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Tocilizumab", "entry_stage": 1,
             "response_rate": TOCI_SUCCESS, "is_active": True},
            {"name": "Lopinavir", "entry_stage": 1,
             "response_rate": LOPI_SUCCESS, "is_active": True},
        ],
    )

    bt = z.get("boundary_table", [])
    has_bayesian_fields = (
        len(bt) == 3
        and all("efficacy_threshold" in row for row in bt)
        and all("futility_threshold" in row for row in bt)
    )
    per_arm = z.get("per_arm", [])
    has_2_arms = len(per_arm) == 2

    results.append({
        "test": "6. Concurrent control: Bayesian boundaries + 2 arms returned",
        "n_boundary_rows": len(bt),
        "n_per_arm": len(per_arm),
        "pass": has_bayesian_fields and has_2_arms,
    })

    return pd.DataFrame(results)


def validate_staggered_entry(client) -> pd.DataFrame:
    """Test 7: REMAP-CAP domains opened at different times.

    Model IL-6 domain entering at stage 1 and anticoagulation at stage 2.
    Late-entering arm should have fewer stages of data.
    """
    results = []

    z = client(
        n_arms=2,
        n_stages=4,
        n_per_stage=120,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "IL-6 RA", "entry_stage": 1,
             "response_rate": TOCI_SUCCESS, "is_active": True},
            {"name": "Anticoagulation", "entry_stage": 2,
             "response_rate": 0.645, "is_active": True},
        ],
    )

    per_arm = z.get("per_arm", [])
    if len(per_arm) == 2:
        il6 = per_arm[0]
        anticoag = per_arm[1]
        il6_entry = il6.get("entry_stage", 0)
        anticoag_entry = anticoag.get("entry_stage", 0)
        il6_max = il6.get("max_stages", 0)
        anticoag_max = anticoag.get("max_stages", 0)
        stagger_ok = (
            il6_entry == 1
            and anticoag_entry == 2
            and il6_max > anticoag_max
        )
    else:
        il6_entry = anticoag_entry = il6_max = anticoag_max = None
        stagger_ok = False

    results.append({
        "test": "7. Staggered entry: late domain has fewer stages",
        "il6_entry": il6_entry,
        "anticoag_entry": anticoag_entry,
        "il6_max_stages": il6_max,
        "anticoag_max_stages": anticoag_max,
        "pass": stagger_ok,
    })

    return pd.DataFrame(results)


def validate_strong_effect_simulation(client) -> pd.DataFrame:
    """Test 8: Tocilizumab effect (mortality 28% vs 36%) is large.

    With 500 simulations and REMAP-CAP-like parameters, power should be high.
    """
    results = []

    z = client(
        n_arms=1,
        n_stages=3,
        n_per_stage=150,
        endpoint_type="binary",
        analysis_type="bayesian",
        null_rate=CTRL_SUCCESS,
        alpha=0.025,
        efficacy_threshold=0.99,
        futility_threshold=0.01,
        control_type="concurrent_only",
        arms=[
            {"name": "Tocilizumab", "entry_stage": 1,
             "response_rate": TOCI_SUCCESS, "is_active": True},
        ],
        simulate=True,
        n_simulations=500,
        simulation_seed=42,
    )

    sim = z.get("simulation", {})
    est = sim.get("estimates", {})
    power = est.get("any_arm_power", 0)

    results.append({
        "test": "8. Strong effect simulation: tocilizumab power > 0.50",
        "sim_power": power,
        "n_simulations": sim.get("n_simulations", 0),
        "pass": power is not None and power > 0.50,
    })

    return pd.DataFrame(results)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    raw_client = get_client(base_url)
    client = PlatformClient(raw_client)

    print("=" * 70)
    print("REMAP-CAP PLATFORM TRIAL VALIDATION")
    print("=" * 70)

    all_frames = []

    sections = [
        ("1. Tocilizumab Superiority", validate_tocilizumab_superiority),
        ("2. Lopinavir Futility", validate_lopinavir_futility),
        ("3. Multi-Domain Structure", validate_multi_domain),
        ("4. Sample Size Check", validate_sample_size),
        ("5. High Threshold Reduces False Positives", validate_threshold_effect),
        ("6. Concurrent Control Mode", validate_concurrent_control),
        ("7. Staggered Entry", validate_staggered_entry),
        ("8. Strong Effect Simulation", validate_strong_effect_simulation),
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
    all_results.to_csv("results/remapcap_validation.csv", index=False)

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
