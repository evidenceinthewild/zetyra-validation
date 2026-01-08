# Contributing to Zetyra Validation

We welcome contributions! Here's how you can help:

## Adding New Test Cases

1. Fork the repository
2. Add test to appropriate module (`gsd/`, `cuped/`, `bayesian/`)
3. Ensure test uses established benchmark (gsDesign, analytical formulas, etc.)
4. Document expected vs. actual results
5. Run tests locally to verify they pass
6. Submit pull request

## Test Structure

Each test module should:
- Import from `common.zetyra_client`
- Define clear tolerance thresholds
- Save results to `results/` subdirectory
- Print pass/fail summary
- Exit with appropriate status code

Example:
```python
from common.zetyra_client import get_client

TOLERANCE = 0.01

def main():
    client = get_client()
    # Run validations...

    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED")
        sys.exit(1)
```

## Reporting Issues

- Use GitHub Issues
- Include: test name, expected result, actual result
- Attach CSV output if applicable
- Note the API endpoint being tested

## Code Style

- Python: Follow PEP 8
- R: Use tidyverse style guide
- Include docstrings for all functions
- Keep tests self-contained and reproducible
