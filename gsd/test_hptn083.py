#!/usr/bin/env python3
"""
Validate Zetyra GSD against HPTN 083 trial design

HPTN 083 was a Phase 3 HIV prevention trial using a 4-look
O'Brien-Fleming group sequential design.

Reference:
- Landovitz et al. (2021) NEJM "Cabotegravir for HIV Prevention"
- gsDesign reference: gsDesign(k=4, alpha=0.025, test.type=1, sfu='OF')
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.zetyra_client import get_client
import pandas as pd

BOUNDARY_TOLERANCE = 0.05


def validate_hptn083(base_url: str = None) -> pd.DataFrame:
    """Validate against HPTN 083 design parameters."""
    client = get_client(base_url)
    results = []

    # HPTN 083 design: 4-look O'Brien-Fleming, alpha=0.025, one-sided
    # gsDesign reference boundaries: [4.049, 2.863, 2.337, 2.024]
    reference_boundaries = [4.049, 2.863, 2.337, 2.024]
    info_fracs = [0.25, 0.50, 0.75, 1.00]

    zetyra = client.gsd(
        effect_size=0.3,
        alpha=0.025,
        power=0.80,
        k=4,
        timing=info_fracs,
        spending_function="OBrienFleming",
    )

    for i, (zetyra_z, ref_z) in enumerate(zip(zetyra["efficacy_boundaries"], reference_boundaries)):
        deviation = abs(zetyra_z - ref_z)
        results.append({
            "trial": "HPTN 083",
            "look": i + 1,
            "info_frac": info_fracs[i],
            "zetyra_z": round(zetyra_z, 4),
            "reference_z": ref_z,
            "deviation": round(deviation, 4),
            "pass": deviation < BOUNDARY_TOLERANCE,
        })

    return pd.DataFrame(results)


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("HPTN 083 TRIAL REPLICATION")
    print("=" * 70)

    results = validate_hptn083(base_url)
    print(results.to_string(index=False))

    # Save results
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/hptn083_validation.csv", index=False)

    print("\n" + "=" * 70)
    if results["pass"].all():
        print("✅ ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
