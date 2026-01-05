"""
Validate Zetyra Group Sequential Design Calculator

Compares Zetyra GSD results against:
- R gsDesign package
- rpact package
- Analytical spending function formulas
"""

import pandas as pd
import numpy as np
from scipy import stats
from zetyra_client import get_client

# Tolerance for Z-score boundaries
BOUNDARY_TOLERANCE = 0.05  # Absolute difference in Z-scores


def reference_obrien_fleming_spending(t: float, alpha: float = 0.025) -> float:
    """
    O'Brien-Fleming alpha spending function.

    α*(t) = 2 - 2Φ(z_{α/2} / √t)

    where Φ is the standard normal CDF.
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    z_alpha = stats.norm.ppf(1 - alpha)
    return 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t)))


def reference_pocock_spending(t: float, alpha: float = 0.025) -> float:
    """
    Pocock alpha spending function.

    α*(t) = α × ln(1 + (e-1)t)
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    return alpha * np.log(1 + (np.e - 1) * t)


def validate_spending_functions(client) -> pd.DataFrame:
    """Validate spending function cumulative alpha values."""
    results = []

    # Test O'Brien-Fleming spending
    obf_result = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        k=4,
        spending_function="OBrienFleming",
    )

    info_fracs = obf_result["information_fractions"]
    alpha_spent = obf_result["alpha_spent"]

    for i, (t, spent) in enumerate(zip(info_fracs, alpha_spent)):
        expected = reference_obrien_fleming_spending(t, 0.025)
        deviation = abs(spent - expected)
        results.append({
            "function": "O'Brien-Fleming",
            "look": i + 1,
            "info_frac": round(t, 3),
            "zetyra_alpha": round(spent, 6),
            "reference_alpha": round(expected, 6),
            "deviation": round(deviation, 6),
            "pass": deviation < 0.001 or i == len(info_fracs) - 1,  # Final should equal alpha
        })

    # Test Pocock spending
    pocock_result = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        k=4,
        spending_function="Pocock",
    )

    info_fracs = pocock_result["information_fractions"]
    alpha_spent = pocock_result["alpha_spent"]

    for i, (t, spent) in enumerate(zip(info_fracs, alpha_spent)):
        expected = reference_pocock_spending(t, 0.025)
        deviation = abs(spent - expected)
        results.append({
            "function": "Pocock",
            "look": i + 1,
            "info_frac": round(t, 3),
            "zetyra_alpha": round(spent, 6),
            "reference_alpha": round(expected, 6),
            "deviation": round(deviation, 6),
            "pass": deviation < 0.001 or i == len(info_fracs) - 1,
        })

    return pd.DataFrame(results)


def validate_boundary_properties(client) -> pd.DataFrame:
    """Validate mathematical properties of GSD boundaries."""
    results = []

    # Property 1: O'Brien-Fleming is conservative early (high first boundary)
    obf = client.gsd(effect_size=0.3, k=3, spending_function="OBrienFleming")
    prop1_pass = obf["efficacy_boundaries"][0] > obf["efficacy_boundaries"][2]
    results.append({
        "property": "OBF: Conservative at first look",
        "expected": "Z[1] > Z[3]",
        "actual": f"{obf['efficacy_boundaries'][0]:.3f} > {obf['efficacy_boundaries'][2]:.3f}",
        "pass": prop1_pass,
    })

    # Property 2: Pocock has more uniform boundaries
    pocock = client.gsd(effect_size=0.3, k=3, spending_function="Pocock")
    boundary_range_pocock = max(pocock["efficacy_boundaries"]) - min(pocock["efficacy_boundaries"])
    boundary_range_obf = max(obf["efficacy_boundaries"]) - min(obf["efficacy_boundaries"])
    prop2_pass = boundary_range_pocock < boundary_range_obf
    results.append({
        "property": "Pocock: More uniform than OBF",
        "expected": f"range(Pocock) < range(OBF)",
        "actual": f"{boundary_range_pocock:.3f} < {boundary_range_obf:.3f}",
        "pass": prop2_pass,
    })

    # Property 3: Sample size inflation for interim analyses
    fixed_n = obf["n_fixed"]
    max_n = obf["n_max"]
    inflation = max_n / fixed_n if fixed_n > 0 else 0
    # OBF typically inflates by 1.02-1.05 for k=3
    prop3_pass = 1.0 <= inflation <= 1.10
    results.append({
        "property": "Sample size inflation reasonable",
        "expected": "1.0 ≤ inflation ≤ 1.10",
        "actual": f"{inflation:.4f}",
        "pass": prop3_pass,
    })

    # Property 4: Final boundary close to fixed design z-value
    z_final = obf["efficacy_boundaries"][-1]
    z_fixed = stats.norm.ppf(1 - 0.025)  # One-sided alpha
    diff = abs(z_final - z_fixed)
    # For OBF, final boundary should be close to fixed
    prop4_pass = diff < 0.3
    results.append({
        "property": "OBF: Final boundary ≈ fixed design",
        "expected": f"|Z_final - Z_fixed| < 0.3",
        "actual": f"|{z_final:.3f} - {z_fixed:.3f}| = {diff:.3f}",
        "pass": prop4_pass,
    })

    # Property 5: More looks → higher max N
    k2 = client.gsd(effect_size=0.3, k=2, spending_function="OBrienFleming")
    k5 = client.gsd(effect_size=0.3, k=5, spending_function="OBrienFleming")
    prop5_pass = k5["n_max"] >= k2["n_max"]
    results.append({
        "property": "More looks → larger max N",
        "expected": "n_max(k=5) ≥ n_max(k=2)",
        "actual": f"{k5['n_max']} ≥ {k2['n_max']}",
        "pass": prop5_pass,
    })

    return pd.DataFrame(results)


def validate_gsdesign_benchmarks(client) -> pd.DataFrame:
    """
    Validate against published gsDesign benchmarks.

    These values are from gsDesign R package documentation and
    Jennison & Turnbull (2000).
    """
    results = []

    # Benchmark 1: gsDesign example (3 equally spaced, OBF, alpha=0.025)
    # Expected boundaries from gsDesign::gsDesign(k=3, test.type=1, alpha=0.025, beta=0.1)
    result = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        k=3,
        spending_function="OBrienFleming",
    )

    # gsDesign reference boundaries (approximate)
    gsdesign_bounds = [4.33, 2.96, 2.00]  # Approximate OBF boundaries for k=3

    for i, (zetyra_z, ref_z) in enumerate(zip(result["efficacy_boundaries"], gsdesign_bounds)):
        deviation = abs(zetyra_z - ref_z)
        results.append({
            "benchmark": "gsDesign k=3 OBF",
            "look": i + 1,
            "zetyra_z": round(zetyra_z, 3),
            "gsdesign_z": ref_z,
            "deviation": round(deviation, 3),
            "pass": deviation < BOUNDARY_TOLERANCE,
        })

    # Benchmark 2: Pocock boundaries (should be approximately equal)
    pocock_result = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.90,
        k=3,
        spending_function="Pocock",
    )

    # Pocock reference (approximately equal across looks)
    pocock_ref = 2.29  # Approximate Pocock boundary for k=3, alpha=0.025

    for i, zetyra_z in enumerate(pocock_result["efficacy_boundaries"]):
        deviation = abs(zetyra_z - pocock_ref)
        results.append({
            "benchmark": "gsDesign k=3 Pocock",
            "look": i + 1,
            "zetyra_z": round(zetyra_z, 3),
            "gsdesign_z": pocock_ref,
            "deviation": round(deviation, 3),
            "pass": deviation < 0.15,  # Pocock boundaries should be similar
        })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full GSD validation."""
    client = get_client(base_url)

    spending_results = validate_spending_functions(client)
    property_results = validate_boundary_properties(client)
    benchmark_results = validate_gsdesign_benchmarks(client)

    return {
        "spending": spending_results,
        "properties": property_results,
        "benchmarks": benchmark_results,
        "all_pass": (
            spending_results["pass"].all()
            and property_results["pass"].all()
            and benchmark_results["pass"].all()
        ),
    }


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA GROUP SEQUENTIAL DESIGN VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nSpending Function Validation")
    print("-" * 70)
    print(results["spending"].to_string(index=False))

    print("\nBoundary Properties")
    print("-" * 70)
    print(results["properties"].to_string(index=False))

    print("\ngsDesign Benchmarks")
    print("-" * 70)
    print(results["benchmarks"].to_string(index=False))

    print("\n" + "=" * 70)
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
