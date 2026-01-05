#!/usr/bin/env python3
"""
Zetyra Full Validation Suite

Runs all validation scripts and generates a summary report.
"""

import sys
import pandas as pd
from datetime import datetime

# Import validation modules
from validate_sample_size import run_validation as validate_sample_size
from validate_cuped import run_validation as validate_cuped
from validate_gsd import run_validation as validate_gsd
from validate_bayesian import run_validation as validate_bayesian


def run_all_validations(base_url: str = None) -> dict:
    """Run all validation suites."""
    results = {}

    print("=" * 70)
    print("ZETYRA FULL VALIDATION SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Sample Size
    print("\n[1/4] Validating Sample Size Calculators...")
    try:
        results["sample_size"] = validate_sample_size(base_url)
        status = "✅ PASS" if results["sample_size"]["all_pass"] else "❌ FAIL"
        print(f"      Sample Size: {status}")
    except Exception as e:
        print(f"      Sample Size: ❌ ERROR - {e}")
        results["sample_size"] = {"all_pass": False, "error": str(e)}

    # CUPED
    print("\n[2/4] Validating CUPED Calculator...")
    try:
        results["cuped"] = validate_cuped(base_url)
        status = "✅ PASS" if results["cuped"]["all_pass"] else "❌ FAIL"
        print(f"      CUPED: {status}")
    except Exception as e:
        print(f"      CUPED: ❌ ERROR - {e}")
        results["cuped"] = {"all_pass": False, "error": str(e)}

    # GSD
    print("\n[3/4] Validating Group Sequential Design...")
    try:
        results["gsd"] = validate_gsd(base_url)
        status = "✅ PASS" if results["gsd"]["all_pass"] else "❌ FAIL"
        print(f"      GSD: {status}")
    except Exception as e:
        print(f"      GSD: ❌ ERROR - {e}")
        results["gsd"] = {"all_pass": False, "error": str(e)}

    # Bayesian
    print("\n[4/4] Validating Bayesian Predictive Power...")
    try:
        results["bayesian"] = validate_bayesian(base_url)
        status = "✅ PASS" if results["bayesian"]["all_pass"] else "❌ FAIL"
        print(f"      Bayesian: {status}")
    except Exception as e:
        print(f"      Bayesian: ❌ ERROR - {e}")
        results["bayesian"] = {"all_pass": False, "error": str(e)}

    return results


def generate_summary(results: dict) -> pd.DataFrame:
    """Generate validation summary table."""
    summary = []

    for calculator, result in results.items():
        summary.append({
            "Calculator": calculator.replace("_", " ").title(),
            "Status": "PASS" if result.get("all_pass", False) else "FAIL",
            "Error": result.get("error", ""),
        })

    return pd.DataFrame(summary)


def main():
    # Allow override for local testing
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    if base_url:
        print(f"Using custom API URL: {base_url}")

    results = run_all_validations(base_url)

    # Generate summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    summary = generate_summary(results)
    print(summary.to_string(index=False))

    # Overall result
    all_pass = all(r.get("all_pass", False) for r in results.values())

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
