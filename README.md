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
| Sample Size (Binary) | ✅ Validated | pwr::pwr.2p.test |
| Sample Size (Survival) | ✅ Validated | gsDesign::nSurv |
| CUPED Variance Reduction | ✅ Validated | Manual derivation |
| Group Sequential Design | ✅ Validated | gsDesign, rpact |
| Bayesian Predictive Power | ✅ Validated | Manual derivation |

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

Zetyra provides public validation endpoints at `https://api.zetyra.com/api/v1/validation/`:

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

### 2. Published Trials
We validate against parameters from published Phase III trials:
- BEACON CRC (2019) - Survival endpoint
- KEYNOTE-189 (2018) - GSD with interim analysis
- Adaptive platform trials from COVID-19

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
│   ├── sample_size_benchmarks.csv
│   ├── gsd_benchmarks.csv
│   └── published_trials.csv
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
├── results/
│   ├── validation_report.csv
│   └── validation_summary.md
└── docs/
    └── methodology.md
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
@software{zetyra2025,
  title = {Zetyra: Regulatory-Grade Statistical Design Platform},
  author = {Evidence in the Wild},
  year = {2025},
  url = {https://zetyra.com}
}
```
