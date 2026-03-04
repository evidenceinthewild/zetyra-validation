#!/usr/bin/env python3
"""
Property checks for GSD with unequal information fractions

Uses HeartMate II-inspired timing (0.27, 0.67, 1.00) to verify that
Zetyra handles non-equal spacing correctly: monotonicity, info fraction
echo, and the expected behavior that earlier first looks raise the first
boundary.

Note: This is a structural/shape check, not a trial replication. The
HeartMate II paper does not publish the exact z-score boundary values
needed for a numerical comparison.

Reference:
- Slaughter et al. (2009) NEJM "Advanced Heart Failure Treated with
  Continuous-Flow Left Ventricular Assist Device"
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.zetyra_client import get_client
import pandas as pd


def validate_heartmate(base_url: str = None) -> pd.DataFrame:
    """Property checks for GSD with HeartMate II-inspired unequal info fractions."""
    client = get_client(base_url)
    results = []

    # HeartMate II: 3-look O'Brien-Fleming with unequal spacing
    info_fracs = [0.27, 0.67, 1.00]

    zetyra = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.80,
        k=3,
        timing=info_fracs,
        spending_function="OBrienFleming",
    )

    boundaries = zetyra["efficacy_boundaries"]

    # Property 1: Boundaries should be monotonically decreasing for OBF
    monotonic = all(boundaries[i] >= boundaries[i+1] for i in range(len(boundaries)-1))
    results.append({
        "property": "Monotonically decreasing",
        "expected": "b1 ≥ b2 ≥ b3",
        "actual": f"{boundaries[0]:.3f} ≥ {boundaries[1]:.3f} ≥ {boundaries[2]:.3f}",
        "pass": monotonic,
    })

    # Property 2: Information fractions should match requested
    returned_fracs = zetyra["information_fractions"]
    fracs_match = all(
        abs(returned_fracs[i] - info_fracs[i]) < 0.01
        for i in range(len(info_fracs))
    )
    results.append({
        "property": "Info fractions match",
        "expected": str(info_fracs),
        "actual": str([round(f, 2) for f in returned_fracs]),
        "pass": fracs_match,
    })

    # Property 3: First boundary should be higher than k=3 equal spacing
    equal_spacing = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.80,
        k=3,
        spending_function="OBrienFleming",
    )
    # With earlier first look (0.27 vs 0.33), first boundary should be higher
    first_higher = boundaries[0] >= equal_spacing["efficacy_boundaries"][0]
    results.append({
        "property": "Earlier look → higher boundary",
        "expected": "b1(0.27) ≥ b1(0.33)",
        "actual": f"{boundaries[0]:.3f} ≥ {equal_spacing['efficacy_boundaries'][0]:.3f}",
        "pass": first_higher,
    })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("GSD UNEQUAL SPACING (HEARTMATE II-INSPIRED)")
    print("=" * 70)

    results = validate_heartmate(base_url)
    print(results.to_string(index=False))

    # Save results
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/heartmate_validation.csv", index=False)

    print("\n" + "=" * 70)
    if results["pass"].all():
        print("✅ ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
