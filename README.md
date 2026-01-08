# Zetyra Validation Suite

Independent validation of Zetyra statistical calculators against reference implementations and published benchmarks.

## Summary of Results

| Calculator | Tests | Status | Reference |
|------------|-------|--------|-----------|
| GSD | 42 | ✅ Pass | gsDesign R package |
| CUPED | 15 | ✅ Pass | Analytical formulas |
| Bayesian | 26 | ✅ Pass | Conjugate priors |
| Sample Size | 10 | ✅ Pass | pwr R package |

## Repository Structure

```
zetyra-validation/
├── README.md
├── LICENSE
├── gsd/
│   ├── test_gsdesign_benchmark.R    # 24 gsDesign comparisons
│   ├── test_hptn083.py              # HPTN 083 replication
│   ├── test_heartmate.py            # HeartMate II replication
│   └── results/
│       └── gsd_validation_results.csv
├── cuped/
│   ├── test_analytical.py           # Variance reduction formula
│   └── results/
│       └── cuped_validation_results.csv
├── bayesian/
│   ├── test_beta_binomial.py        # Beta-Binomial conjugate
│   ├── test_normal_conjugate.py     # Normal-Normal conjugate
│   └── results/
│       └── bayesian_validation_results.csv
└── validation/                      # Shared utilities
    ├── zetyra_client.py             # API client
    ├── validate_all.py              # Run all validations
    └── ...
```

## Running Validations

### Prerequisites

```bash
# Python
pip install -r requirements.txt

# R (for GSD validation)
install.packages(c("gsDesign", "httr", "jsonlite"))
```

### Run All Tests

```bash
# Python validation suite
cd validation
python validate_all.py

# R GSD benchmarks
cd gsd
Rscript test_gsdesign_benchmark.R
```

### Run Individual Tests

```bash
# GSD
python gsd/test_hptn083.py
python gsd/test_heartmate.py
Rscript gsd/test_gsdesign_benchmark.R

# CUPED
python cuped/test_analytical.py

# Bayesian
python bayesian/test_beta_binomial.py
python bayesian/test_normal_conjugate.py
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

## References

1. **GSD**: Jennison & Turnbull (2000) *Group Sequential Methods*
2. **CUPED**: Deng et al. (2013) *Improving Online Controlled Experiments* (WSDM)
3. **Bayesian**: Gelman et al. (2013) *Bayesian Data Analysis*
4. **gsDesign**: Anderson (2022) *gsDesign R package*

## License

MIT License - see [LICENSE](LICENSE)
