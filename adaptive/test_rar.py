#!/usr/bin/env python3
"""
Validate Response-Adaptive Randomization (RAR) Calculator

Tests analytical allocations (Rosenberger optimal, Neyman, Thompson),
simulation operating characteristics (power, type I error, allocation
trajectories), input validation guards, and reference value checks.

Covers binary, continuous, and survival endpoints with DBCD, Thompson,
and Neyman methods across two-arm and multi-arm designs.

References:
- Rosenberger et al. (2001) "Optimal allocation for binary responses"
- Hu & Zhang (2004) "Asymptotic properties of DBCD"
- Tymofyeyev, Rosenberger & Hu (2007) "Multi-arm DBCD"
- Wathen & Thall (2017) "Thompson sampling for multi-arm AR"
- Robertson et al. (2023) "Comprehensive review of RAR procedures"
- FDA (2019) "Adaptive Designs for Clinical Trials of Drugs and Biologics"
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
from common.assertions import mc_rate_upper_bound
import pandas as pd


# ─── Test functions ──────────────────────────────────────────────────


def validate_analytical_binary(client) -> pd.DataFrame:
    """Validate analytical allocations for binary endpoints."""
    results = []

    # Test 1: Binary DBCD — rosenberger_optimal_allocation sums to 1.0
    z = client.rar(
        method="dbcd",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
    )
    ar = z["analytical_results"]
    alloc_sum = sum(ar["rosenberger_optimal_allocation"])
    results.append({
        "test": "Binary DBCD: rosenberger allocation sums to 1.0",
        "alloc_sum": round(alloc_sum, 6),
        "pass": abs(alloc_sum - 1.0) < 1e-6,
    })

    # Test 2: Binary DBCD — rosenberger favors higher-rate arm
    # arm_rates=[0.20, 0.40] → arm 1 (higher rate) gets more allocation
    alloc = ar["rosenberger_optimal_allocation"]
    results.append({
        "test": "Binary DBCD: rosenberger favors higher-rate arm",
        "arm_0_alloc": alloc[0],
        "arm_1_alloc": alloc[1],
        "pass": alloc[1] > alloc[0],
    })

    # Test 3: Binary Neyman — allocation sums to 1.0
    z_ney = client.rar(
        method="neyman",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
    )
    ar_ney = z_ney["analytical_results"]
    ney_sum = sum(ar_ney["neyman_allocation"])
    results.append({
        "test": "Binary Neyman: allocation sums to 1.0",
        "alloc_sum": round(ney_sum, 6),
        "pass": abs(ney_sum - 1.0) < 1e-6,
    })

    # Test 4: Binary Thompson — allocations sum to 1.0 and in (0,1)
    z_th = client.rar(
        method="thompson",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
    )
    ar_th = z_th["analytical_results"]
    # Thompson analytical still returns rosenberger/neyman/equal allocations
    eq_sum = sum(ar_th["equal_allocation"])
    eq_in_range = all(0 < x < 1 for x in ar_th["equal_allocation"])
    results.append({
        "test": "Binary Thompson: equal allocation sums to 1.0 and in (0,1)",
        "alloc_sum": round(eq_sum, 6),
        "in_range": eq_in_range,
        "pass": abs(eq_sum - 1.0) < 1e-6 and eq_in_range,
    })

    return pd.DataFrame(results)


def validate_analytical_continuous(client) -> pd.DataFrame:
    """Validate analytical allocations for continuous endpoints."""
    results = []

    # Test 5: Continuous DBCD — allocations with different means
    z = client.rar(
        method="dbcd",
        endpoint_type="continuous",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_means=[0.0, 0.5],
        common_sd=1.0,
    )
    ar = z["analytical_results"]
    alloc_sum = sum(ar["rosenberger_optimal_allocation"])
    results.append({
        "test": "Continuous DBCD: rosenberger allocation sums to 1.0",
        "allocation": ar["rosenberger_optimal_allocation"],
        "pass": abs(alloc_sum - 1.0) < 1e-6,
    })

    return pd.DataFrame(results)


def validate_analytical_survival(client) -> pd.DataFrame:
    """Validate analytical allocations for survival endpoints."""
    results = []

    # Test 6: Survival DBCD — analytical computes with HR=0.7
    z = client.rar(
        method="dbcd",
        endpoint_type="survival",
        n_arms=2,
        n_total=300,
        alpha=0.025,
        hazard_ratio=0.7,
        median_control=12.0,
        accrual_time=24.0,
        follow_up_time=12.0,
    )
    ar = z["analytical_results"]
    alloc_sum = sum(ar["rosenberger_optimal_allocation"])
    results.append({
        "test": "Survival DBCD: rosenberger allocation sums to 1.0",
        "allocation": ar["rosenberger_optimal_allocation"],
        "pass": abs(alloc_sum - 1.0) < 1e-6 and ar["endpoint_type"] == "survival",
    })

    # Test 9: events_required_80pct present for survival
    results.append({
        "test": "Survival: events_required_80pct present",
        "has_key": "events_required_80pct" in ar,
        "pass": "events_required_80pct" in ar and ar["events_required_80pct"]["per_comparison"] > 0,
    })

    return pd.DataFrame(results)


def validate_analytical_multiarm(client) -> pd.DataFrame:
    """Validate multi-arm analytical allocations."""
    results = []

    # Test 7: Multi-arm (3 arms) — returns 3-element allocation vectors
    z = client.rar(
        method="dbcd",
        endpoint_type="binary",
        n_arms=3,
        n_total=300,
        alpha=0.025,
        arm_rates=[0.20, 0.35, 0.50],
    )
    ar = z["analytical_results"]
    n_rosenberger = len(ar["rosenberger_optimal_allocation"])
    n_neyman = len(ar["neyman_allocation"])
    n_equal = len(ar["equal_allocation"])
    results.append({
        "test": "Multi-arm (3): returns 3-element allocation vectors",
        "n_rosenberger": n_rosenberger,
        "n_neyman": n_neyman,
        "n_equal": n_equal,
        "pass": n_rosenberger == 3 and n_neyman == 3 and n_equal == 3,
    })

    # Test 8: Equal rates — rosenberger optimal should be ~equal
    z_eq = client.rar(
        method="dbcd",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.30, 0.30],
    )
    ar_eq = z_eq["analytical_results"]
    alloc = ar_eq["rosenberger_optimal_allocation"]
    results.append({
        "test": "Equal rates: rosenberger ~equal allocation",
        "arm_0": alloc[0],
        "arm_1": alloc[1],
        "pass": abs(alloc[0] - 0.5) < 0.01 and abs(alloc[1] - 0.5) < 0.01,
    })

    return pd.DataFrame(results)


def validate_simulation_binary(client) -> pd.DataFrame:
    """Validate simulation operating characteristics for binary endpoints."""
    results = []

    # Test 10: Binary DBCD simulation — power > 0 under H1
    z = client.rar(
        method="dbcd",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
        simulate=True,
        n_simulations=500,
        simulation_seed=42,
    )
    sim = z["simulation"]
    results.append({
        "test": "Sim Binary DBCD: power > 0 under H1",
        "power": sim["power"],
        "pass": sim["power"] > 0,
    })

    # Test 11: Binary DBCD type I error ≤ 0.10 (with MC uncertainty)
    # Use mc_rate_upper_bound to account for MC noise
    t1e_upper = mc_rate_upper_bound(sim["type1_error"], 500, confidence=0.99)
    results.append({
        "test": "Sim Binary DBCD: type I error upper bound ≤ 0.10",
        "type1_error": sim["type1_error"],
        "upper_bound_99": round(t1e_upper, 4),
        "pass": t1e_upper <= 0.10,
    })

    # Test 12: Simulation returns arm_sample_size_distribution
    has_dist = "arm_sample_size_distribution" in sim["estimates"]
    dist = sim["estimates"].get("arm_sample_size_distribution", {})
    has_arms = "arm_0" in dist and "arm_1" in dist
    results.append({
        "test": "Sim Binary DBCD: arm_sample_size_distribution present",
        "has_dist": has_dist,
        "has_arms": has_arms,
        "pass": has_dist and has_arms,
    })

    # Test 13: Simulation returns allocation_trajectories
    has_traj = "allocation_trajectories" in sim["estimates"]
    traj = sim["estimates"].get("allocation_trajectories", {})
    has_traj_arms = "arm_0" in traj and "arm_1" in traj
    results.append({
        "test": "Sim Binary DBCD: allocation_trajectories present",
        "has_traj": has_traj,
        "has_traj_arms": has_traj_arms,
        "pass": has_traj and has_traj_arms,
    })

    # Test 14: Thompson simulation — power > 0
    z_th = client.rar(
        method="thompson",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
        simulate=True,
        n_simulations=500,
        simulation_seed=42,
    )
    sim_th = z_th["simulation"]
    results.append({
        "test": "Sim Thompson: power > 0",
        "power": sim_th["power"],
        "pass": sim_th["power"] > 0,
    })

    return pd.DataFrame(results)


def validate_input_guards(client) -> pd.DataFrame:
    """Validate invalid inputs return errors."""
    results = []

    # Test 15: Invalid method rejected
    resp = client.rar_raw(
        method="invalid",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
    )
    results.append({
        "test": "Guard: invalid method rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422, 500),
    })

    # Test 16: Invalid endpoint_type rejected
    resp = client.rar_raw(
        method="dbcd",
        endpoint_type="invalid",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
    )
    results.append({
        "test": "Guard: invalid endpoint_type rejected",
        "status_code": resp.status_code,
        "pass": resp.status_code in (400, 422, 500),
    })

    # Test 17: delta too large for n_arms
    # allocation_bounds_delta=0.3 with n_arms=4 means 4*0.3=1.2 > 1.0
    resp = client.rar_raw(
        method="thompson",
        endpoint_type="binary",
        n_arms=4,
        n_total=400,
        alpha=0.025,
        arm_rates=[0.20, 0.30, 0.40, 0.50],
        allocation_bounds_delta=0.30,
        simulate=True,
        n_simulations=100,
        simulation_seed=42,
    )
    # This may succeed (projection clamps) or error — accept either
    # as long as the server does not crash with a 500 unhandled error.
    # If the server returns 200, it handled it gracefully; if 400, it validated.
    results.append({
        "test": "Guard: delta too large for n_arms",
        "status_code": resp.status_code,
        "pass": resp.status_code in (200, 400, 422),
    })

    return pd.DataFrame(results)


def validate_reference_checks(client) -> pd.DataFrame:
    """Validate known reference values for allocation formulas."""
    results = []

    # Test 18: Rosenberger optimal for arm_rates=[0.20, 0.40]
    # Optimal: rho_k = sqrt(p_k) / sum(sqrt(p_j))
    # arm 1 allocation = sqrt(0.40) / (sqrt(0.20) + sqrt(0.40))
    #                   = 0.6325 / (0.4472 + 0.6325) = 0.6325 / 1.0797 ≈ 0.5858
    expected_arm1 = math.sqrt(0.40) / (math.sqrt(0.20) + math.sqrt(0.40))
    z = client.rar(
        method="dbcd",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.20, 0.40],
    )
    ar = z["analytical_results"]
    actual_arm1 = ar["rosenberger_optimal_allocation"][1]
    results.append({
        "test": "Ref: rosenberger arm_rates=[0.20,0.40] arm1 ≈ 0.586",
        "expected": round(expected_arm1, 4),
        "actual": actual_arm1,
        "pass": abs(actual_arm1 - expected_arm1) < 0.005,
    })

    # Test 19: Equal arm_rates=[0.30, 0.30] → rosenberger ~0.50 each
    z_eq = client.rar(
        method="dbcd",
        endpoint_type="binary",
        n_arms=2,
        n_total=200,
        alpha=0.025,
        arm_rates=[0.30, 0.30],
    )
    ar_eq = z_eq["analytical_results"]
    alloc = ar_eq["rosenberger_optimal_allocation"]
    results.append({
        "test": "Ref: equal rates [0.30,0.30] → ~0.50 each",
        "arm_0": alloc[0],
        "arm_1": alloc[1],
        "pass": abs(alloc[0] - 0.50) < 0.005 and abs(alloc[1] - 0.50) < 0.005,
    })

    # Test 20: Neyman allocation for binary proportional to sqrt(p*(1-p))
    # arm_rates=[0.20, 0.40]
    # sqrt(0.20*0.80) = sqrt(0.16) = 0.4000
    # sqrt(0.40*0.60) = sqrt(0.24) = 0.4899
    # arm 0: 0.4000 / (0.4000 + 0.4899) = 0.4000 / 0.8899 ≈ 0.4495
    # arm 1: 0.4899 / 0.8899 ≈ 0.5505
    sd0 = math.sqrt(0.20 * 0.80)
    sd1 = math.sqrt(0.40 * 0.60)
    expected_ney0 = sd0 / (sd0 + sd1)
    expected_ney1 = sd1 / (sd0 + sd1)
    ney_alloc = ar["neyman_allocation"]
    results.append({
        "test": "Ref: Neyman binary proportional to sqrt(p(1-p))",
        "expected_arm0": round(expected_ney0, 4),
        "actual_arm0": ney_alloc[0],
        "expected_arm1": round(expected_ney1, 4),
        "actual_arm1": ney_alloc[1],
        "pass": abs(ney_alloc[0] - expected_ney0) < 0.005 and abs(ney_alloc[1] - expected_ney1) < 0.005,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(base_url)

    print("=" * 70)
    print("RAR (RESPONSE-ADAPTIVE RANDOMIZATION) VALIDATION")
    print("=" * 70)

    all_frames = []

    print("\n1. Analytical — Binary Endpoint")
    print("-" * 70)
    r1 = validate_analytical_binary(client)
    print(r1.to_string(index=False))
    all_frames.append(r1)

    print("\n2. Analytical — Continuous Endpoint")
    print("-" * 70)
    r2 = validate_analytical_continuous(client)
    print(r2.to_string(index=False))
    all_frames.append(r2)

    print("\n3. Analytical — Survival Endpoint")
    print("-" * 70)
    r3 = validate_analytical_survival(client)
    print(r3.to_string(index=False))
    all_frames.append(r3)

    print("\n4. Analytical — Multi-Arm & Equal Rates")
    print("-" * 70)
    r4 = validate_analytical_multiarm(client)
    print(r4.to_string(index=False))
    all_frames.append(r4)

    print("\n5. Simulation — Binary Endpoint")
    print("-" * 70)
    r5 = validate_simulation_binary(client)
    print(r5.to_string(index=False))
    all_frames.append(r5)

    print("\n6. Input Guards")
    print("-" * 70)
    r6 = validate_input_guards(client)
    print(r6.to_string(index=False))
    all_frames.append(r6)

    print("\n7. Reference Value Checks")
    print("-" * 70)
    r7 = validate_reference_checks(client)
    print(r7.to_string(index=False))
    all_frames.append(r7)

    # Save results
    os.makedirs("results", exist_ok=True)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv("results/rar_validation.csv", index=False)

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
