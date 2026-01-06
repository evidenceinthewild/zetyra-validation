#!/usr/bin/env python
"""Generate validation report."""

import sys
from datetime import datetime

# Add validation directory to path
sys.path.insert(0, 'validation')

from validate_sample_size import run_validation as validate_ss
from validate_cuped import run_validation as validate_cuped
from validate_gsd import run_validation as validate_gsd
from validate_bayesian import run_validation as validate_bayesian


def generate_report(base_url: str = None):
    """Generate markdown validation report."""
    report_lines = []
    report_lines.append('# Zetyra Validation Results')
    report_lines.append('')
    report_lines.append(f'**Validation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_lines.append(f'**API URL:** {base_url or "https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation"}')
    report_lines.append('')
    report_lines.append('## Summary')
    report_lines.append('')
    report_lines.append('| Calculator | Status | Tests Passed |')
    report_lines.append('|------------|--------|--------------|')

    # Sample Size (includes continuous, binary, survival)
    ss_results = validate_ss(base_url)
    ss_pass = ss_results['all_pass']
    ss_tests = (
        len(ss_results['continuous'])
        + len(ss_results['binary'])
        + len(ss_results['survival'])
        + len(ss_results['survival_properties'])
    )
    report_lines.append(f"| Sample Size | {'✅ PASS' if ss_pass else '❌ FAIL'} | {ss_tests} |")

    # CUPED
    cuped_results = validate_cuped(base_url)
    cuped_pass = cuped_results['all_pass']
    cuped_tests = len(cuped_results['numerical']) + len(cuped_results['properties'])
    report_lines.append(f"| CUPED | {'✅ PASS' if cuped_pass else '❌ FAIL'} | {cuped_tests} |")

    # GSD
    gsd_results = validate_gsd(base_url)
    gsd_pass = gsd_results['all_pass']
    gsd_tests = len(gsd_results['spending']) + len(gsd_results['properties']) + len(gsd_results['benchmarks'])
    report_lines.append(f"| Group Sequential Design | {'✅ PASS' if gsd_pass else '❌ FAIL'} | {gsd_tests} |")

    # Bayesian
    bayesian_results = validate_bayesian(base_url)
    bayesian_pass = bayesian_results['all_pass']
    bayesian_tests = len(bayesian_results['continuous']) + len(bayesian_results['binary']) + len(bayesian_results['properties'])
    report_lines.append(f"| Bayesian | {'✅ PASS' if bayesian_pass else '❌ FAIL'} | {bayesian_tests} |")

    all_pass = ss_pass and cuped_pass and gsd_pass and bayesian_pass
    total_tests = ss_tests + cuped_tests + gsd_tests + bayesian_tests

    report_lines.append('')
    report_lines.append(f'**Total Tests: {total_tests}**')
    report_lines.append(f'**Overall Status: {"✅ ALL PASSED" if all_pass else "❌ SOME FAILED"}**')

    report_lines.append('')
    report_lines.append('## Detailed Results')
    report_lines.append('')

    # Sample Size Details
    report_lines.append('### Sample Size Calculator')
    report_lines.append('')
    report_lines.append('#### Continuous Outcomes')
    report_lines.append('```')
    report_lines.append(ss_results['continuous'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Binary Outcomes')
    report_lines.append('```')
    report_lines.append(ss_results['binary'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Survival Outcomes')
    report_lines.append('```')
    report_lines.append(ss_results['survival'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Survival Properties')
    report_lines.append('```')
    report_lines.append(ss_results['survival_properties'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')

    # CUPED Details
    report_lines.append('### CUPED Calculator')
    report_lines.append('')
    report_lines.append('#### Numerical Validation')
    report_lines.append('```')
    report_lines.append(cuped_results['numerical'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Properties')
    report_lines.append('```')
    report_lines.append(cuped_results['properties'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')

    # GSD Details
    report_lines.append('### Group Sequential Design')
    report_lines.append('')
    report_lines.append('#### Spending Functions')
    report_lines.append('```')
    report_lines.append(gsd_results['spending'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Properties')
    report_lines.append('```')
    report_lines.append(gsd_results['properties'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### gsDesign Benchmarks')
    report_lines.append('```')
    report_lines.append(gsd_results['benchmarks'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')

    # Bayesian Details
    report_lines.append('### Bayesian Calculator')
    report_lines.append('')
    report_lines.append('#### Continuous Posterior')
    report_lines.append('```')
    report_lines.append(bayesian_results['continuous'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Binary Posterior')
    report_lines.append('```')
    report_lines.append(bayesian_results['binary'].to_string(index=False))
    report_lines.append('```')
    report_lines.append('')
    report_lines.append('#### Properties')
    report_lines.append('```')
    report_lines.append(bayesian_results['properties'].to_string(index=False))
    report_lines.append('```')

    return '\n'.join(report_lines), all_pass, total_tests


if __name__ == '__main__':
    import os
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    print('Generating validation report...')
    report, all_pass, total_tests = generate_report(base_url)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    with open('results/validation_summary.md', 'w') as f:
        f.write(report)

    print(f'Report saved to results/validation_summary.md')
    print(f'Total tests: {total_tests}')
    print(f'Overall status: {"ALL PASSED" if all_pass else "SOME FAILED"}')

    sys.exit(0 if all_pass else 1)
