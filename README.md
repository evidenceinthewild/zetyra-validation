# Zetyra Validation Suite

![Tests](https://img.shields.io/badge/tests-51%20passed-success)
![Coverage](https://img.shields.io/badge/coverage-GSD%20%7C%20CUPED%20%7C%20Bayesian-blue)
![Accuracy](https://img.shields.io/badge/max%20deviation-0.0046%20z--score-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Independent validation of Zetyra statistical calculators against reference implementations and published benchmarks.

## Summary of Results

| Calculator | Tests | Status | Reference |
|------------|-------|--------|-----------|
| GSD | 30 | ✅ Pass | gsDesign R package |
| CUPED | 12 | ✅ Pass | Analytical formulas |
| Bayesian | 9 | ✅ Pass | Conjugate priors |

[See detailed results →](docs/validation_results_summary.md)

## Repository Structure

```
zetyra-validation/
├── README.md
├── LICENSE
├── requirements.txt
├── common/                          # Shared utilities
│   ├── __init__.py
│   └── zetyra_client.py             # API client
├── gsd/
│   ├── test_gsdesign_benchmark.R    # 23 gsDesign comparisons
│   ├── test_hptn083.py              # HPTN 083 replication
│   ├── test_heartmate.py            # HeartMate II replication
│   └── results/
├── cuped/
│   ├── test_analytical.py           # Variance reduction formula
│   └── results/
└── bayesian/
    ├── test_beta_binomial.py        # Beta-Binomial conjugate
    ├── test_normal_conjugate.py     # Normal-Normal conjugate
    └── results/
```

## Running Validations

### Prerequisites

```bash
# Python
pip install -r requirements.txt

# R (for GSD validation)
install.packages(c("gsDesign", "httr", "jsonlite"))
```

### Run Tests

```bash
# GSD (from gsd/ directory)
cd gsd
python test_hptn083.py
python test_heartmate.py
Rscript test_gsdesign_benchmark.R

# CUPED (from cuped/ directory)
cd cuped
python test_analytical.py

# Bayesian (from bayesian/ directory)
cd bayesian
python test_beta_binomial.py
python test_normal_conjugate.py
```

### Example Output

```bash
$ cd gsd && python test_hptn083.py

======================================================================
HPTN 083 TRIAL REPLICATION
======================================================================
   trial  look  info_frac  zetyra_z  reference_z  deviation  pass
HPTN 083     1       0.25    4.0444        4.049     0.0046  True
HPTN 083     2       0.50    2.8598        2.863     0.0032  True
HPTN 083     3       0.75    2.3351        2.337     0.0019  True
HPTN 083     4       1.00    2.0222        2.024     0.0018  True

======================================================================
✅ ALL VALIDATIONS PASSED
```

## Validation Details

### Group Sequential Design (GSD)

Validates against the gold-standard **gsDesign** R package:

- **8 design configurations**: OF_2 through OF_5, Pocock_2 through Pocock_4
- **24 boundary comparisons**: All within 0.05 z-score tolerance
- **Published trials**: HPTN 083 (HIV prevention), HeartMate II (LVAD)

Key formulas validated:
- O'Brien-Fleming: `b_k = c / √t_k`
- Pocock: constant boundary across looks
- Alpha spending: `α*(t) = 2[1 - Φ(z_α/√t)]` (OBF)

### CUPED

Validates variance reduction against analytical formulas:

- **Variance reduction factor**: `VRF = 1 - ρ²`
- **Sample size reduction**: proportional to VRF
- **Symmetry**: `|ρ|` determines reduction

### Bayesian Predictive Power

Validates conjugate posterior calculations:

- **Beta-Binomial**: `π|x ~ Beta(α + x, β + n - x)`
- **Normal-Normal**: Precision-weighted posterior
- **Predictive probability**: Beta-binomial analytical formula

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

## Troubleshooting

### API Connection Issues

If tests fail with connection errors:

```bash
# Check if API is accessible
curl https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation/health
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

## License

MIT License - see [LICENSE](LICENSE)
