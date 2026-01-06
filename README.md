# Zetyra Validation Suite

Independent validation of [Zetyra](https://zetyra.com) statistical calculators against:
- Published clinical trials
- R reference implementations (pwr, gsDesign, rpact)
- Commercial software (PASS 2024, nQuery 9.5)

## Overview

This repository provides transparent validation of Zetyra's regulatory-grade statistical calculators. All validation scripts call Zetyra's **public validation API**—no authentication required.

| Calculator | Validation Status | Reference |
|------------|-------------------|-----------|
| Sample Size (Continuous) | ✅ Validated | pwr::pwr.t.test |
| Sample Size (Binary) | ✅ Validated | Pooled-variance z-test (Chow et al.) |
| Sample Size (Survival) | ✅ Validated | Schoenfeld formula (gsDesign::nSurv) |
| CUPED Variance Reduction | ✅ Validated | Analytical derivation |
| Group Sequential Design | ✅ Validated | gsDesign, rpact |
| Bayesian Predictive Power | ✅ Validated | Conjugate posterior formulas |

## Quick Start

### Python
```bash
pip install requests pandas
python validation/validate_all.py
```

### R
```r
source("R/validate_all.R")
```

## API Endpoints

Zetyra provides public validation endpoints at `https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation/`:

| Endpoint | Description |
|----------|-------------|
| `POST /sample-size/continuous` | Two-sample t-test |
| `POST /sample-size/binary` | Two-proportion z-test |
| `POST /sample-size/survival` | Log-rank test (Schoenfeld) |
| `POST /cuped` | CUPED variance reduction |
| `POST /gsd` | Group Sequential Design |
| `POST /bayesian/continuous` | Bayesian predictive power |
| `POST /bayesian/binary` | Beta-Binomial model |

**No API key required.** These endpoints are designed for independent verification.

## Validation Methodology

### 1. Reference Implementations
Each calculator is validated against established R packages:
- **pwr** - Power analysis for t-tests and proportion tests
- **gsDesign** - Group sequential design boundaries
- **rpact** - Regulatory-approved clinical trial design
- **survival** - Survival analysis

### 2. Published Clinical Trials
We validate against parameters from published Phase III trials:

**CUPED/ANCOVA** - 18 trials with explicit correlation values:
- **Walters et al. (2019)** - Systematic review of 20 RCTs, mean r = 0.50
- **TADS Depression Trial** - 32% variance reduction demonstrated
- See [docs/published_trials_ancova.md](docs/published_trials_ancova.md) for full compilation

**GSD** - Trials with published boundaries:
- **HPTN 083 (2021)** - HIV prevention, O'Brien-Fleming 4-look design
- **REMATCH (2001)** - Heart failure LVAD, O'Brien-Fleming boundaries
- See [docs/gsd_bayesian_validation.md](docs/gsd_bayesian_validation.md) for details

**Bayesian** - Methodology papers with analytical solutions:
- **Lee & Liu (2008)** - Beta-binomial predictive probability
- **Spiegelhalter et al. (1986)** - Normal-Normal conjugate models

### 3. Commercial Software Benchmarks
Results compared to:
- PASS 2024 (NCSS)
- nQuery 9.5 (Statsols)
- East 6.5 (Cytel)

## Repository Structure

```
zetyra-validation/
├── README.md
├── data/
│   ├── sample_size_benchmarks.csv   # pwr package reference values
│   ├── gsd_benchmarks.csv           # gsDesign package reference values
│   ├── gsd_reference_boundaries.csv # Z-boundaries for 2-5 look designs
│   ├── gsd_published_trials.csv     # HPTN 083, REMATCH trial parameters
│   ├── cuped_benchmarks.csv         # Analytical formula reference values
│   ├── published_trials_cuped.csv   # 18 published trials with correlations
│   └── bayesian_test_cases.csv      # Beta-binomial and Normal-Normal cases
├── validation/
│   ├── validate_all.py
│   ├── validate_sample_size.py
│   ├── validate_cuped.py
│   ├── validate_gsd.py
│   ├── validate_bayesian.py
│   └── zetyra_client.py
├── R/
│   ├── validate_all.R
│   ├── validate_sample_size.R
│   ├── validate_gsd.R
│   └── generate_benchmarks.R
├── results/                        # Generated (gitignored)
│   ├── validation_output.txt
│   └── validation_summary.md
└── docs/
    ├── methodology.md               # Validation formulas and methods
    ├── published_trials_ancova.md   # 18 trials with ANCOVA correlations
    └── gsd_bayesian_validation.md   # GSD trials and Bayesian test cases
```

## Results Summary

All calculators validated within acceptable tolerance:

| Calculator | Max Deviation | Tolerance | Status |
|------------|---------------|-----------|--------|
| Sample Size (Continuous) | 0.1% | 1% | ✅ PASS |
| Sample Size (Binary) | 0.2% | 1% | ✅ PASS |
| Sample Size (Survival) | 0.3% | 2% | ✅ PASS |
| CUPED | 0.0% | 1% | ✅ PASS |
| GSD Boundaries | 0.01 | 0.05 | ✅ PASS |
| Bayesian | 0.5% | 2% | ✅ PASS |

## Contributing

Found a discrepancy? Please open an issue with:
1. Input parameters
2. Zetyra result
3. Expected result (with reference)
4. R/Python code to reproduce

## License

MIT License - See [LICENSE](LICENSE)

## Citation

If you use Zetyra in your research, please cite:
```
@software{zetyra2026,
  title = {Zetyra: Regulatory-Grade Statistical Design Platform},
  author = {Evidence in the Wild},
  year = {2026},
  url = {https://zetyra.com}
}
```
