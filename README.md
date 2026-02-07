# Zetyra Validation Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18253308.svg)](https://doi.org/10.5281/zenodo.18253308)
![Tests](https://img.shields.io/badge/tests-169%20passed-success)
![Coverage](https://img.shields.io/badge/coverage-GSD%20%7C%20CUPED%20%7C%20Bayesian%20Toolkit-blue)
![Accuracy](https://img.shields.io/badge/max%20deviation-0.0046%20z--score-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Independent validation of Zetyra statistical calculators against reference implementations and published benchmarks.

## Summary of Results

| Calculator | Tests | Status | Reference |
|------------|-------|--------|-----------|
| GSD | 30 | ✅ Pass | gsDesign R package |
| CUPED | 12 | ✅ Pass | Analytical formulas |
| Bayesian Predictive Power | 17 | ✅ Pass | Conjugate priors |
| Prior Elicitation | 22 | ✅ Pass | ESS formula, scipy.optimize |
| Bayesian Borrowing | 18 | ✅ Pass | Power prior, Cochran's Q |
| Bayesian Sample Size | 14 | ✅ Pass | Binomial CI, MC search |
| Bayesian Two-Arm | 13 | ✅ Pass | Binomial CI, MC search |
| Bayesian Sequential | 20 | ✅ Pass | Zhou & Ji (2024) |
| Offline References | 23 | ✅ Pass | Pure math (no API) |

**Total: 169 tests across 12 scripts, all passing.**

[See detailed results →](docs/validation_results_summary.md)

## Repository Structure

```
zetyra-validation/
├── README.md
├── LICENSE
├── requirements.txt
├── common/                              # Shared utilities
│   ├── __init__.py
│   ├── zetyra_client.py                 # API client (12 endpoints)
│   └── assertions.py                    # Binomial CI, schema contracts
├── gsd/
│   ├── test_gsdesign_benchmark.R        # 23 gsDesign comparisons
│   ├── test_hptn083.py                  # HPTN 083 replication
│   ├── test_heartmate.py                # HeartMate II replication
│   └── results/
├── cuped/
│   ├── test_analytical.py               # Variance reduction formula
│   └── results/
└── bayesian/
    ├── test_beta_binomial.py            # Beta-Binomial conjugate PP
    ├── test_normal_conjugate.py         # Normal-Normal conjugate PP
    ├── test_prior_elicitation.py        # ESS, historical, quantile matching
    ├── test_bayesian_borrowing.py       # Power prior, MAP, heterogeneity
    ├── test_bayesian_sample_size.py     # Single-arm MC sample size search
    ├── test_bayesian_two_arm.py         # Two-arm MC sample size search
    ├── test_bayesian_sequential.py      # Posterior probability boundaries
    ├── test_offline_references.py       # Pure-math tests (no API)
    └── results/
```

## What's Validated

### Bayesian Toolkit (v1.1)

Each of the 6 Bayesian calculators has a dedicated test suite covering:

- **Analytical correctness** — conjugate posteriors, boundary formulas, ESS derivations compared against closed-form references
- **Monte Carlo calibration** — type I error and power checked with Clopper-Pearson binomial CIs (scales with simulation count)
- **Schema contracts** — response keys, types, and value bounds validated for every API call
- **Input guards** — invalid inputs return 400/422 with the offending field named
- **Boundary conditions** — extreme priors, zero/all events, single-look designs
- **Invariants** — higher power → larger n, larger effect → smaller n, higher discount → higher ESS
- **Seed reproducibility** — identical seeds produce identical MC results
- **Symmetry** — null hypothesis gives same type I regardless of label swap

### Offline References

23 pure-math tests run without any API dependency:
- Beta-Binomial and Normal-Normal conjugate updates
- Zhou & Ji (2024) boundary formula (including vague-prior → frequentist convergence)
- Cochran's Q / I² heterogeneity
- ESS-based prior elicitation
- Power prior discounting
- Clopper-Pearson CI helpers

## Running Validations

### Prerequisites

```bash
# Python
pip install -r requirements.txt

# R (for GSD validation only)
install.packages(c("gsDesign", "httr", "jsonlite"))
```

### Run Tests

```bash
# All Bayesian tests (against local server)
cd bayesian
for f in test_*.py; do python "$f" http://localhost:8000; done

# Offline tests (no server needed)
python test_offline_references.py

# GSD
cd gsd
python test_hptn083.py
python test_heartmate.py
Rscript test_gsdesign_benchmark.R

# CUPED
cd cuped
python test_analytical.py
```

### Example Output

```
$ python bayesian/test_bayesian_two_arm.py http://localhost:8000

======================================================================
BAYESIAN TWO-ARM VALIDATION
======================================================================

1. Two-Arm MC Validation (Binomial CI)
----------------------------------------------------------------------
                                                 test  rec_n_per_arm  type1  type1_ub  power  power_lb  pass
      Superiority: ctrl=0.30, treat=0.50, flat priors             80 0.0390    0.0515 0.8105    0.7869  True
PUNCH CD3 rates: ctrl=0.624, treat=0.712, flat priors            350 0.0445    0.0578 0.8060    0.7823  True
     Large effect: ctrl=0.20, treat=0.50, flat priors             60 0.0500    0.0639 0.9755    0.9651  True

2. Directional Properties
----------------------------------------------------------------------
                                 test  small_effect_n  large_effect_n  pass
  Property: larger effect → smaller n           280.0            40.0  True
Property: higher threshold → larger n           125.0           175.0  True

...

======================================================================
ALL VALIDATIONS PASSED
```

## API Endpoints

All validations use Zetyra's public validation API:

```
https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation
```

Endpoints:
- `POST /sample-size/continuous`
- `POST /sample-size/binary`
- `POST /sample-size/survival`
- `POST /cuped`
- `POST /gsd`
- `POST /bayesian/continuous`
- `POST /bayesian/binary`
- `POST /bayesian/prior-elicitation`
- `POST /bayesian/borrowing`
- `POST /bayesian/sample-size-single-arm`
- `POST /bayesian/two-arm`
- `POST /bayesian/sequential`

## Assertion Helpers

`common/assertions.py` provides shared validation infrastructure:

- **`binomial_ci(k, n)`** — Clopper-Pearson exact CI for MC rate estimates
- **`mc_rate_within(rate, n_sims, target)`** — check if target is consistent with observed MC rate
- **`mc_rate_upper_bound / mc_rate_lower_bound`** — one-sided CI bounds for type I / power checks
- **`assert_schema(response, contract)`** — validate response keys, types, and bounds against contracts (supports strict and non-strict lower bounds)

## Troubleshooting

### API Connection Issues

```bash
# Check if API is accessible
curl https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation/health
```

### Running Against Local Server

```bash
# Start backend
cd /path/to/Zetyra/backend
.venv/bin/python3 -m uvicorn app.main:app --port 8000

# Run tests with local URL
python bayesian/test_beta_binomial.py http://localhost:8000
```

### R Package Installation Fails

```r
# Install from CRAN mirror
install.packages("gsDesign", repos="https://cloud.r-project.org")
```

### Python Import Errors

```bash
# Ensure you're in project root
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## References

1. **GSD**: Jennison & Turnbull (2000) *Group Sequential Methods*
2. **CUPED**: Deng et al. (2013) *Improving Online Controlled Experiments* (WSDM)
3. **Bayesian**: Gelman et al. (2013) *Bayesian Data Analysis*
4. **gsDesign**: Anderson (2022) *gsDesign R package*
5. **Bayesian Sequential**: Zhou & Ji (2024) *Bayesian sequential monitoring*
6. **Prior Elicitation**: Morita, Thall & Müller (2008) *Determining ESS of a parametric prior*

## License

MIT License - see [LICENSE](LICENSE)
