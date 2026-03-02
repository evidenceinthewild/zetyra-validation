#!/usr/bin/env python3
"""
Validate Unblinded Sample Size Re-estimation (SSR)

Tests unblinded SSR across continuous, binary, and survival endpoints:
1. Zone classification (unfavorable, promising, favorable)
2. Conditional power consistent with zone
3. No change in favorable zone (CP already high)
4. N increases in promising zone
5. Structural properties
6. Input guards
7. Schema contract

References:
- Mehta & Pocock (2011) "Adaptive Increase in Sample Size When Interim
  Results are Promising: A Practical Guide with Examples"
- Chen, DeMets & Lan (2004) "Increasing the Sample Size When the
  Unblinded Interim Result is Promising"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import assert_schema
import pandas as pd


# ─── Test functions ──────────────────────────────────────────────────

def validate_continuous_zones(client) -> pd.DataFrame:
    """Validate zone classification for continuous outcomes."""
    results = []

    # Favorable: effect at design value -> high CP -> no increase
    z_fav = client.ssr_unblinded(
        endpoint_type="continuous",
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
        cp_futility=0.20,
        cp_promising_lower=0.20,
        cp_promising_upper=0.80,
    )
    schema_errors = assert_schema(z_fav, "ssr_unblinded")

    results.append({
        "test": "Continuous: no observed -> favorable zone",
        "zone": z_fav["zone"],
        "cp": z_fav["conditional_power"],
        "increase": z_fav["sample_size_increase"],
        "pass": z_fav["zone"] == "favorable" and z_fav["sample_size_increase"] == 0 and len(schema_errors) == 0,
    })

    # Promising: smaller observed effect -> moderate CP
    z_prom = client.ssr_unblinded(
        endpoint_type="continuous",
        effect_size=0.3,
        observed_effect=0.15,  # Half the planned effect
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
        cp_futility=0.10,
        cp_promising_lower=0.10,
        cp_promising_upper=0.80,
    )
    results.append({
        "test": "Continuous: smaller effect -> promising zone",
        "zone": z_prom["zone"],
        "cp": z_prom["conditional_power"],
        "pass": z_prom["zone"] == "promising" and z_prom["sample_size_increase"] > 0,
    })

    # Unfavorable: small effect, CP between futility and promising thresholds
    z_unf = client.ssr_unblinded(
        endpoint_type="continuous",
        effect_size=0.3,
        observed_effect=0.05,  # Very small effect
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
        cp_futility=0.05,
        cp_promising_lower=0.30,
        cp_promising_upper=0.80,
    )
    results.append({
        "test": "Continuous: small effect -> unfavorable/futility",
        "zone": z_unf["zone"],
        "cp": z_unf["conditional_power"],
        "pass": z_unf["zone"] in ("unfavorable", "futility"),
    })

    # Regression: different effect_size values produce different initial N
    z_small = client.ssr_unblinded(
        endpoint_type="continuous",
        effect_size=0.1,  # Smaller effect -> larger N
        alpha=0.025, power=0.90, interim_fraction=0.5, n_max_factor=2.0,
    )
    z_large = client.ssr_unblinded(
        endpoint_type="continuous",
        effect_size=0.5,  # Larger effect -> smaller N
        alpha=0.025, power=0.90, interim_fraction=0.5, n_max_factor=2.0,
    )
    results.append({
        "test": "Continuous: smaller effect -> larger N",
        "n_small_effect": z_small["initial_n_total"],
        "n_large_effect": z_large["initial_n_total"],
        "pass": z_small["initial_n_total"] > z_large["initial_n_total"],
    })

    return pd.DataFrame(results)


def validate_binary_zones(client) -> pd.DataFrame:
    """Validate zone classification for binary outcomes."""
    results = []

    # Favorable: observed rates match planned
    z = client.ssr_unblinded(
        endpoint_type="binary",
        p_control=0.20,
        p_treatment=0.35,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
    )
    schema_errors = assert_schema(z, "ssr_unblinded")

    results.append({
        "test": "Binary: no observed -> favorable zone",
        "zone": z["zone"],
        "cp": z["conditional_power"],
        "pass": z["zone"] == "favorable" and len(schema_errors) == 0,
    })

    # Promising: observed rates closer together (CP ~0.42, solidly in promising)
    z_prom = client.ssr_unblinded(
        endpoint_type="binary",
        p_control=0.20,
        p_treatment=0.35,
        observed_p_control=0.22,
        observed_p_treatment=0.30,  # Smaller observed difference -> moderate CP
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
        cp_futility=0.05,
        cp_promising_lower=0.10,
        cp_promising_upper=0.80,
    )
    results.append({
        "test": "Binary: smaller difference -> promising",
        "zone": z_prom["zone"],
        "cp": z_prom["conditional_power"],
        "pass": z_prom["zone"] == "promising" and z_prom["sample_size_increase"] > 0,
    })

    return pd.DataFrame(results)


def validate_survival_zones(client) -> pd.DataFrame:
    """Validate zone classification for survival outcomes."""
    results = []

    # Favorable: HR at design value
    z = client.ssr_unblinded(
        endpoint_type="survival",
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
    )
    schema_errors = assert_schema(z, "ssr_unblinded")

    results.append({
        "test": "Survival: baseline -> favorable zone",
        "zone": z["zone"],
        "cp": z["conditional_power"],
        "initial_n": z["initial_n_total"],
        "pass": z["zone"] == "favorable" and z["initial_n_total"] > 0 and len(schema_errors) == 0,
    })

    # Promising: HR weaker than planned (CP ~0.32, solidly in promising)
    z_prom = client.ssr_unblinded(
        endpoint_type="survival",
        hazard_ratio=0.7,
        observed_hr=0.85,  # Weaker than planned 0.7
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=3.0,
        cp_futility=0.10,
        cp_promising_lower=0.10,
        cp_promising_upper=0.80,
    )
    results.append({
        "test": "Survival: weaker HR -> promising",
        "zone": z_prom["zone"],
        "cp": z_prom["conditional_power"],
        "increase": z_prom["sample_size_increase"],
        "pass": z_prom["zone"] == "promising" and z_prom["sample_size_increase"] > 0,
    })

    # N increases in promising zone
    results.append({
        "test": "Survival: promising -> N increases",
        "initial_n": z_prom["initial_n_total"],
        "recalculated_n": z_prom["recalculated_n_total"],
        "pass": z_prom["recalculated_n_total"] > z_prom["initial_n_total"],
    })

    # CP in valid range
    results.append({
        "test": "Survival: CP in (0,1)",
        "cp": z["conditional_power"],
        "pass": 0 < z["conditional_power"] <= 1,
    })

    return pd.DataFrame(results)


def validate_properties(client) -> pd.DataFrame:
    """Cross-endpoint structural properties."""
    results = []

    for etype, kwargs in [
        ("continuous", {"effect_size": 0.3}),
        ("binary", {"p_control": 0.20, "p_treatment": 0.35}),
        ("survival", {"hazard_ratio": 0.7, "median_control": 12, "accrual_time": 24, "follow_up_time": 12}),
    ]:
        z = client.ssr_unblinded(
            endpoint_type=etype, alpha=0.025, power=0.90,
            interim_fraction=0.5, n_max_factor=2.0, **kwargs,
        )

        # n_per_arm * 2 ≈ n_total
        diff = abs(z["initial_n_per_arm"] * 2 - z["initial_n_total"])
        results.append({
            "test": f"{etype}: n_per_arm * 2 = n_total",
            "n_per_arm": z["initial_n_per_arm"],
            "n_total": z["initial_n_total"],
            "pass": diff <= 1,
        })

        # Zone is a valid string
        results.append({
            "test": f"{etype}: zone is valid",
            "zone": z["zone"],
            "pass": z["zone"] in ("futility", "unfavorable", "promising", "favorable"),
        })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return 400/422."""
    results = []

    resp = client.ssr_unblinded_raw(endpoint_type="continuous")
    results.append({
        "test": "Guard: continuous without effect_size",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    resp = client.ssr_unblinded_raw(endpoint_type="binary")
    results.append({
        "test": "Guard: binary without rates",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    resp = client.ssr_unblinded_raw(endpoint_type="survival")
    results.append({
        "test": "Guard: survival without HR",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Threshold ordering: cp_futility > cp_promising_lower should be rejected
    resp = client.ssr_unblinded_raw(
        endpoint_type="continuous",
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
        cp_futility=0.50,
        cp_promising_lower=0.20,
        cp_promising_upper=0.80,
    )
    results.append({
        "test": "Guard: cp_futility > cp_promising_lower",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    # Threshold ordering: cp_promising_lower > cp_promising_upper should be rejected
    resp = client.ssr_unblinded_raw(
        endpoint_type="continuous",
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=2.0,
        cp_futility=0.10,
        cp_promising_lower=0.90,
        cp_promising_upper=0.80,
    )
    results.append({
        "test": "Guard: cp_promising_lower > cp_promising_upper",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422),
    })

    return pd.DataFrame(results)


def validate_cap_enforcement(client) -> pd.DataFrame:
    """Verify n_max_factor cap is enforced."""
    results = []

    # Continuous: small observed effect + high variance -> promising & capped
    # CP ~0.15 (solidly in promising with wide thresholds), n_max_factor=1.3 triggers cap
    z = client.ssr_unblinded(
        endpoint_type="continuous",
        effect_size=0.3,
        observed_effect=0.12,
        observed_variance=2.0,
        alpha=0.025,
        power=0.90,
        interim_fraction=0.5,
        n_max_factor=1.3,
        cp_futility=0.05,
        cp_promising_lower=0.05,
        cp_promising_upper=0.95,
    )

    # Must land in promising
    results.append({
        "test": "Cap scenario: zone is promising",
        "zone": z["zone"],
        "cp": z["conditional_power"],
        "pass": z["zone"] == "promising",
    })

    # Cap must be enforced
    max_allowed = int(z["initial_n_total"] * 1.3) + 2  # +2 for rounding
    results.append({
        "test": "Cap: n_capped is true",
        "n_capped": z["n_capped"],
        "pass": z["n_capped"] is True,
    })
    results.append({
        "test": "Cap: recalculated <= initial * factor",
        "recalculated_n": z["recalculated_n_total"],
        "max_allowed": max_allowed,
        "pass": z["recalculated_n_total"] <= max_allowed,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("UNBLINDED SSR VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Continuous Zones")
    print("-" * 70)
    c_results = validate_continuous_zones(client)
    print(c_results.to_string(index=False))
    all_frames.append(c_results)

    print("\n2. Binary Zones")
    print("-" * 70)
    b_results = validate_binary_zones(client)
    print(b_results.to_string(index=False))
    all_frames.append(b_results)

    print("\n3. Survival Zones")
    print("-" * 70)
    s_results = validate_survival_zones(client)
    print(s_results.to_string(index=False))
    all_frames.append(s_results)

    print("\n4. Cross-Endpoint Properties")
    print("-" * 70)
    p_results = validate_properties(client)
    print(p_results.to_string(index=False))
    all_frames.append(p_results)

    print("\n5. Input Guards")
    print("-" * 70)
    g_results = validate_input_guards(client)
    print(g_results.to_string(index=False))
    all_frames.append(g_results)

    print("\n6. Cap Enforcement")
    print("-" * 70)
    cap_results = validate_cap_enforcement(client)
    print(cap_results.to_string(index=False))
    all_frames.append(cap_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/ssr_unblinded_validation.csv", index=False)

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
