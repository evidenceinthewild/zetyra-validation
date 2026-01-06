"""
Validate Zetyra Group Sequential Design Calculator

Compares Zetyra GSD results against:
- R gsDesign package (8 benchmark designs: OF_2 through OF_5, Pocock_2 through Pocock_4)
- Published clinical trials (HPTN 083, HeartMate II)
- Analytical spending function formulas
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from zetyra_client import get_client

# Tolerance for Z-score boundaries
BOUNDARY_TOLERANCE = 0.05  # Absolute difference in Z-scores

# Path to reference data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def reference_obrien_fleming_spending(t: float, alpha: float = 0.025) -> float:
    """
    O'Brien-Fleming alpha spending function (one-sided).

    α*(t) = 2 - 2Φ(z_α / √t)

    where Φ is the standard normal CDF and z_α is the one-sided critical value.
    Note: For two-sided tests, use alpha/2 as the input alpha.
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
        # Final look should exactly equal alpha, intermediate looks should match spending function
        is_final = i == len(info_fracs) - 1
        if is_final:
            # At final look, cumulative alpha must equal total alpha
            passes = abs(spent - 0.025) < 0.0001
        else:
            passes = deviation < 0.001
        results.append({
            "function": "O'Brien-Fleming",
            "look": i + 1,
            "info_frac": round(t, 3),
            "zetyra_alpha": round(spent, 6),
            "reference_alpha": round(expected, 6),
            "deviation": round(deviation, 6),
            "pass": passes,
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
        # Final look should exactly equal alpha, intermediate looks should match spending function
        is_final = i == len(info_fracs) - 1
        if is_final:
            # At final look, cumulative alpha must equal total alpha
            passes = abs(spent - 0.025) < 0.0001
        else:
            passes = deviation < 0.001
        results.append({
            "function": "Pocock",
            "look": i + 1,
            "info_frac": round(t, 3),
            "zetyra_alpha": round(spent, 6),
            "reference_alpha": round(expected, 6),
            "deviation": round(deviation, 6),
            "pass": passes,
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
    # GSD with non-binding futility typically inflates by 1.02-1.20 for k=3
    # Higher inflation is expected when futility bounds are included
    prop3_pass = 1.0 <= inflation <= 1.20
    results.append({
        "property": "Sample size inflation reasonable",
        "expected": "1.0 ≤ inflation ≤ 1.20",
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
    Validate against all gsDesign benchmarks from CSV.

    Tests 8 design configurations (OF_2 through OF_5, Pocock_2 through Pocock_4)
    with 24 total boundary comparisons against gsDesign R package.
    """
    results = []

    # Load reference boundaries from CSV
    csv_path = os.path.join(DATA_DIR, "gsd_reference_boundaries.csv")
    ref_df = pd.read_csv(csv_path)

    # Group by design to test each configuration
    designs = ref_df.groupby("design")

    for design_name, design_df in designs:
        # Extract parameters from first row
        first_row = design_df.iloc[0]
        k = int(first_row["looks"])
        alpha = float(first_row["alpha"])
        spending_fn = first_row["spending_function"]

        # Call Zetyra API
        zetyra_result = client.gsd(
            effect_size=0.3,
            alpha=alpha,
            power=0.80,  # beta=0.20 means power=0.80
            k=k,
            spending_function=spending_fn,
        )

        # Compare each boundary
        for _, row in design_df.iterrows():
            look = int(row["look"])
            ref_z = float(row["z_boundary"])

            # Get Zetyra boundary for this look (0-indexed)
            zetyra_z = zetyra_result["efficacy_boundaries"][look - 1]
            deviation = abs(zetyra_z - ref_z)

            results.append({
                "design": design_name,
                "look": look,
                "zetyra_z": round(zetyra_z, 3),
                "gsdesign_z": ref_z,
                "deviation": round(deviation, 3),
                "pass": deviation < BOUNDARY_TOLERANCE,
            })

    return pd.DataFrame(results)


def validate_published_trials(client) -> pd.DataFrame:
    """
    Validate against published clinical trials with explicit GSD parameters.

    Tests:
    - HPTN 083-style (2021): 4-look O'Brien-Fleming design (gsDesign reference)
    - HeartMate II (2009): 3-look with unequal info fractions (27%, 67%, 100%)

    Note: We use gsDesign R package reference values rather than exact published
    boundaries since published trials may use different software implementations.
    """
    results = []

    # HPTN 083-style: HIV Prevention Trial design parameters
    # 4-look O'Brien-Fleming, alpha=0.025, one-sided
    # gsDesign reference: gsDesign(k=4, alpha=0.025, test.type=1, sfu='OF')
    # gsDesign Z-boundaries: [4.049, 2.863, 2.337, 2.024]
    hptn_boundaries = [4.049, 2.863, 2.337, 2.024]  # gsDesign reference values
    hptn_info_fracs = [0.25, 0.50, 0.75, 1.00]

    hptn_result = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.80,
        k=4,
        timing=hptn_info_fracs,
        spending_function="OBrienFleming",
    )

    for i, (zetyra_z, ref_z) in enumerate(zip(hptn_result["efficacy_boundaries"], hptn_boundaries)):
        deviation = abs(zetyra_z - ref_z)
        results.append({
            "trial": "HPTN 083-style",
            "look": i + 1,
            "info_frac": hptn_info_fracs[i],
            "zetyra_z": round(zetyra_z, 3),
            "gsdesign_z": ref_z,
            "deviation": round(deviation, 3),
            "pass": deviation < BOUNDARY_TOLERANCE,
        })

    # HeartMate II: LVAD Trial (NEJM 2009)
    # 3-look O'Brien-Fleming with unequal info fractions: 27%, 67%, 100%
    # This tests the alpha-spending function with non-standard timing
    heartmate_info_fracs = [0.27, 0.67, 1.00]

    heartmate_result = client.gsd(
        effect_size=0.3,
        alpha=0.025,  # One-sided (original was two-sided 0.05)
        power=0.80,
        k=3,
        timing=heartmate_info_fracs,
        spending_function="OBrienFleming",
    )

    # Verify boundaries follow expected O'Brien-Fleming pattern with unequal spacing
    # Property: First boundary should be highest, decreasing toward final
    boundaries = heartmate_result["efficacy_boundaries"]
    monotonic_decreasing = all(boundaries[i] >= boundaries[i+1] for i in range(len(boundaries)-1))

    results.append({
        "trial": "HeartMate II",
        "look": "all",
        "info_frac": str(heartmate_info_fracs),
        "zetyra_z": str([round(z, 3) for z in boundaries]),
        "published_z": "Monotonic decreasing",
        "deviation": 0.0 if monotonic_decreasing else 1.0,
        "pass": monotonic_decreasing,
    })

    # Check that info fractions match requested values
    returned_fracs = heartmate_result["information_fractions"]
    fracs_match = all(
        abs(returned_fracs[i] - heartmate_info_fracs[i]) < 0.01
        for i in range(len(heartmate_info_fracs))
    )
    results.append({
        "trial": "HeartMate II",
        "look": "info_fracs",
        "info_frac": str(heartmate_info_fracs),
        "zetyra_z": str([round(f, 2) for f in returned_fracs]),
        "published_z": "Match requested",
        "deviation": 0.0 if fracs_match else 1.0,
        "pass": fracs_match,
    })

    return pd.DataFrame(results)


def run_validation(base_url: str = None) -> dict:
    """Run full GSD validation."""
    client = get_client(base_url)

    spending_results = validate_spending_functions(client)
    property_results = validate_boundary_properties(client)
    benchmark_results = validate_gsdesign_benchmarks(client)
    published_results = validate_published_trials(client)

    return {
        "spending": spending_results,
        "properties": property_results,
        "benchmarks": benchmark_results,
        "published_trials": published_results,
        "all_pass": (
            spending_results["pass"].all()
            and property_results["pass"].all()
            and benchmark_results["pass"].all()
            and published_results["pass"].all()
        ),
    }


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("ZETYRA GROUP SEQUENTIAL DESIGN VALIDATION")
    print("=" * 70)

    results = run_validation(base_url)

    print("\nSpending Function Validation (8 tests)")
    print("-" * 70)
    print(results["spending"].to_string(index=False))

    print("\nBoundary Properties (5 tests)")
    print("-" * 70)
    print(results["properties"].to_string(index=False))

    print("\ngsDesign Benchmarks (24 boundary comparisons across 8 designs)")
    print("-" * 70)
    print(results["benchmarks"].to_string(index=False))

    print("\nPublished Trial Validation (HPTN 083-style, HeartMate II)")
    print("-" * 70)
    print(results["published_trials"].to_string(index=False))

    # Summary statistics
    total_tests = (
        len(results["spending"])
        + len(results["properties"])
        + len(results["benchmarks"])
        + len(results["published_trials"])
    )
    passed_tests = (
        results["spending"]["pass"].sum()
        + results["properties"]["pass"].sum()
        + results["benchmarks"]["pass"].sum()
        + results["published_trials"]["pass"].sum()
    )

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
    if results["all_pass"]:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    sys.exit(0 if results["all_pass"] else 1)
