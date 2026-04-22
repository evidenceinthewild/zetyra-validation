# Zetyra Validation Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18879839.svg)](https://doi.org/10.5281/zenodo.18879839)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18880066.svg)](https://doi.org/10.5281/zenodo.18880066)
![Tests](https://img.shields.io/badge/tests-758%20passed-success)
![Coverage](https://img.shields.io/badge/coverage-GSD%20%7C%20CUPED%20%7C%20Bayesian%20%7C%20SSR%20%7C%20RAR%20%7C%20Master%20Protocol-blue)
![Accuracy](https://img.shields.io/badge/max%20deviation-0.034%20z--score-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Independent validation of Zetyra statistical calculators against reference implementations and published benchmarks.

## Summary of Results

| Calculator | Tests | Status | Reference |
|------------|-------|--------|-----------|
| Sample Size (continuous / binary / survival) | 48 | ✅&nbsp;Pass | Cohen (1988) textbook, Schoenfeld (1981), scipy |
| Chi-Square (client-side math) | 55 | ✅&nbsp;Pass | scipy.stats.chi2, chi2_contingency, fisher_exact |
| GSD | 30 | ✅&nbsp;Pass | gsDesign R package |
| GSD PACIFIC OS | 17 | ✅&nbsp;Pass | Antonia et al. (2018) NEJM, Lan-DeMets OBF |
| GSD MONALEESA-7 OS | 20 | ✅&nbsp;Pass | Im et al. (2019) NEJM, Lan-DeMets OBF |
| GSD Survival/TTE | 15 | ✅&nbsp;Pass | Schoenfeld (1983), gsDesign |
| GSD Survival gsDesign Benchmark | 36 | ✅&nbsp;Pass | gsDesign R package (boundaries, alpha spending) |
| CUPED | 12 | ✅&nbsp;Pass | Analytical formulas |
| CUPED Simulation Benchmark | 43 | ✅&nbsp;Pass | MC simulation, Deng et al. (2013) |
| Bayesian Predictive Power | 17 | ✅&nbsp;Pass | Conjugate priors |
| Bayesian Survival | 21 | ✅&nbsp;Pass | Normal-Normal conjugate on log(HR) |
| Bayesian Survival Benchmark | 25 | ✅&nbsp;Pass | Conjugate oracle, MC PP cross-validation |
| Prior Elicitation | 22 | ✅&nbsp;Pass | ESS formula, scipy.optimize |
| Bayesian Borrowing | 18 | ✅&nbsp;Pass | Power prior, Cochran's Q |
| Bayesian Sample Size | 26 | ✅&nbsp;Pass | Binomial CI, MC search (binary + continuous) |
| Bayesian Two-Arm | 24 | ✅&nbsp;Pass | Binomial CI, MC search (binary + continuous) |
| Bayesian Sequential | 20 | ✅&nbsp;Pass | Zhou & Ji (2024) |
| Bayesian Sequential Table 3 | 27 | ✅&nbsp;Pass | Zhou & Ji (2024) Table 3 + companion R code |
| Bayesian Sequential Survival | 24 | ✅&nbsp;Pass | Zhou & Ji (2024) + Schoenfeld |
| Bayesian Sequential Survival Benchmark | 24 | ✅&nbsp;Pass | Zhou & Ji formula + Type I error + convergence |
| SSR Blinded | 20 | ✅&nbsp;Pass | Conditional power formulas |
| SSR Unblinded | 21 | ✅&nbsp;Pass | Zone classification, CP thresholds |
| SSR Single-Arm (Phase II ORR) | 13 | ✅&nbsp;Pass | Beta-Binomial conjugate, Lee & Liu (2008), Saville et al. (2014) |
| NCT03377023 Replication (Nivo+Ipi+Nintedanib NSCLC) | 13 | ✅&nbsp;Pass | Real Bayesian Phase II with published interim+final outcomes |
| SSR gsDesign Benchmark | 14 | ✅&nbsp;Pass | gsDesign R package, reference formulas |
| RAR (Adaptive Randomization) | 20 | ✅&nbsp;Pass | Rosenberger optimal, DBCD, Thompson, Neyman |
| Minimization (Pocock-Simon) | 17 | ✅&nbsp;Pass | Imbalance reduction, pure random benchmark |
| Basket Trial | 21 | ✅&nbsp;Pass | Independent, BHM, EXNEX; Beta-Binomial conjugate |
| Umbrella Trial | 21 | ✅&nbsp;Pass | Frequentist/Bayesian; binary, continuous, survival |
| Platform Trial (MAMS) | 24 | ✅&nbsp;Pass | Boundaries, staggered entry, control modes |
| I-SPY 2 Replication | 10 | ✅&nbsp;Pass | Barker et al. (2009), published pCR rates |
| STAMPEDE Replication | 9 | ✅&nbsp;Pass | Sydes et al. (2012), MAMS boundaries, OS/FFS |
| REMAP-CAP Replication | 8 | ✅&nbsp;Pass | Angus et al. (2020), Bayesian platform |
| Offline References | 23 | ✅&nbsp;Pass | Pure math (no API) |

**Total: 758 tests across 37 scripts, all passing.**

## Repository Structure

```
zetyra-validation/
├── README.md
├── LICENSE
├── requirements.txt
├── common/                              # Shared utilities
│   ├── __init__.py
│   ├── zetyra_client.py                 # API client (25 endpoints)
│   ├── assertions.py                    # Binomial CI, schema contracts
│   └── chi_square_frontend_math.py      # Python port of client-side χ² math
├── free/                                # Free-tier calculators
│   ├── test_sample_size.py              # Two-sample continuous/binary/survival
│   └── test_chi_square.py               # χ² p-value, critical, Pearson, McNemar, Fisher
├── gsd/
│   ├── test_gsdesign_benchmark.R        # 23 gsDesign comparisons
│   ├── test_hptn083.py                  # HPTN 083 replication
│   ├── test_heartmate.py                # HeartMate II replication
│   ├── test_pacific.py                  # PACIFIC OS replication (NSCLC)
│   ├── test_monaleesa7.py               # MONALEESA-7 OS replication (breast cancer)
│   ├── test_gsd_survival.py             # GSD survival/TTE boundaries
│   ├── test_gsd_survival_benchmark.R    # GSD survival vs gsDesign R package
│   └── results/
├── cuped/
│   ├── test_analytical.py               # Variance reduction formula
│   ├── test_simulation_benchmark.py     # MC simulation + Deng et al. (2013)
│   └── results/
├── bayesian/
│   ├── test_beta_binomial.py            # Beta-Binomial conjugate PP
│   ├── test_normal_conjugate.py         # Normal-Normal conjugate PP
│   ├── test_prior_elicitation.py        # ESS, historical, quantile matching
│   ├── test_bayesian_borrowing.py       # Power prior, MAP, heterogeneity
│   ├── test_bayesian_sample_size.py     # Single-arm MC sample size search
│   ├── test_bayesian_two_arm.py         # Two-arm MC sample size search
│   ├── test_bayesian_sequential.py      # Posterior probability boundaries
│   ├── test_zhou_ji_table3.py          # Zhou & Ji (2024) Table 3 cross-validation
│   ├── test_bayesian_sequential_survival.py  # Sequential survival boundaries
│   ├── test_bayesian_sequential_survival_benchmark.py  # Survival Zhou & Ji cross-validation
│   ├── test_bayesian_survival.py        # Bayesian predictive power (survival)
│   ├── test_bayesian_survival_benchmark.py  # Survival PP conjugate oracle + MC cross-validation
│   ├── test_offline_references.py       # Pure-math tests (no API)
│   └── results/
├── ssr/
│   ├── test_ssr_blinded.py                       # Blinded sample size re-estimation
│   ├── test_ssr_unblinded.py                     # Unblinded SSR with zone classification
│   ├── test_ssr_single_arm.py                    # Single-Arm Phase II ORR (Bayesian + CP)
│   ├── test_nct03377023.py                       # NCT03377023 end-to-end (Nivo+Ipi+Nintedanib NSCLC)
│   └── test_ssr_rpact_benchmark.R                # SSR cross-validation against gsDesign
├── adaptive/                            # Adaptive randomization
│   ├── test_rar.py                      # RAR: DBCD, Thompson, Neyman (binary, continuous, survival)
│   └── test_minimization.py             # Pocock-Simon covariate-adaptive minimization
└── master_protocol/                     # Master protocol designs
    ├── test_basket.py                   # Basket trial: independent, BHM, EXNEX
    ├── test_umbrella.py                 # Umbrella trial: frequentist/Bayesian × 3 endpoints
    ├── test_platform.py                 # Platform trial: MAMS, staggered entry, control modes
    ├── test_ispy2.py                    # I-SPY 2 replication (basket, pCR, Bayesian)
    ├── test_stampede.py                 # STAMPEDE replication (platform, MAMS, survival)
    └── test_remapcap.py                 # REMAP-CAP replication (platform, Bayesian, binary)
```

## What's Validated

### Free-Tier Calculators (v2.3)

- **Sample Size (continuous / binary / survival)** — 48 tests covering the public `/validation/sample-size/{continuous,binary,survival}` endpoints. Continuous: closed-form two-sample normal-approx, Cohen's d={0.2, 0.5, 0.8} textbook benchmarks, one-sided vs two-sided, unequal allocation, monotonicity in α/power/effect, input guards. Binary: Cohen's h arcsine formula, canonical p₁=0.30 p₂=0.50 → ~93/arm reference, rate-swap symmetry, rare-event penalty, allocation, guards. Survival: Schoenfeld required events vs closed-form, HR=0.7 median=12mo reference, HR/power monotonicity, unequal-allocation penalty, guards.
- **Chi-Square Calculator (client-side math)** — 55 tests. The calculator runs entirely in the browser, so the Python port in `common/chi_square_frontend_math.py` mirrors the frontend's numerical functions and validates them against `scipy.stats.chi2`, `chi2_contingency`, and `fisher_exact`. Covers the p-value survival function (df ∈ {1, 2, 5, 10, 20}, Abramowitz erf fast path + Numerical Recipes Lentz continued fraction), Wilson-Hilferty critical values, Pearson 2×2 with Yates correction, r×c tables without correction, φ and Cramér's V effect sizes, classical McNemar, two-sided Fisher's exact, and underflow edge cases.

### Bayesian Toolkit (v1.2)

Each of the 6 Bayesian calculators has a dedicated test suite covering:

- **Analytical correctness** — conjugate posteriors, boundary formulas, ESS derivations compared against closed-form references
- **Monte Carlo calibration** — type I error and power checked with Clopper-Pearson binomial CIs (scales with simulation count)
- **Schema contracts** — response keys, types, and value bounds validated for every API call
- **Input guards** — invalid inputs return 400/422 with the offending field named
- **Boundary conditions** — extreme priors, zero/all events, single-look designs
- **Invariants** — higher power → larger n, larger effect → smaller n, higher discount → higher ESS
- **Seed reproducibility** — identical seeds produce identical MC results
- **Symmetry** — null hypothesis gives same type I regardless of label swap

**Continuous endpoints (v1.2):** Bayesian Sample Size (single-arm) and Two-Arm now support Normal-Normal conjugate models alongside the original Beta-Binomial. Continuous-specific tests cover:
- Analytical posterior correctness (closed-form Normal-Normal conjugate update)
- MC calibration of type I error and power with Clopper-Pearson CIs
- Vague-prior convergence to frequentist z-test sample size
- Monotonicity invariants (larger effect → smaller n, larger variance → larger n)
- Input guards for missing continuous fields

### Real-World Trial Replications

Six published clinical trials are replicated against Zetyra's calculators:

- **HPTN 083** (HIV prevention) — 4-look O'Brien-Fleming GSD, z-score boundaries matched to gsDesign within 0.005
- **HeartMate II** (LVAD) — 3-look OBF with unequal info fractions, structural properties verified
- **PACIFIC** (durvalumab, Stage III NSCLC OS) — 3-look Lan-DeMets OBF survival GSD, reference z-scores matched within 0.022 (looks 1–2: 0.000, look 3: 0.022); trial crossing at 299 events verified
- **MONALEESA-7** (ribociclib, HR+ breast cancer OS) — 3-look Lan-DeMets OBF survival GSD, reference z-scores matched within 0.006 (looks 1–2: 0.000, look 3: 0.006); crossing at look 2 (p=0.00973) verified
- **REBYOTA / PUNCH CD2+CD3** (*C. difficile*) — Bayesian borrowing, prior elicitation, two-arm sample size with real Phase 2b/3 data
- **NCT03377023** (nivolumab + ipilimumab + nintedanib in NSCLC, Moffitt Cancer Center; Chen et al. 2019, JTO 2021, JCO 2023) — Bayesian two-stage Phase II design with predictive-probability futility monitoring. End-to-end replication includes both arms' published OCs (power, Type I, P(early stop)) and **direct assertions of the SAP's actual decision rules**: (1) Arm B (ICI-treated) interim rule at r₁=2/20: `PPoS(r=2) = 0.307 > d_futility = 0.20` → continue (matches trial's published decision); rule at the SAP's stopping boundary r₁=1: `PPoS(r=1) = 0.081 ≤ 0.20` → stop (matches SAP's "≤1 responder stop" boundary). (2) Arm B final 6/28 evaluable → posterior 0.997 → crosses the SAP's 0.95 success threshold (matches trial's positive result). (3) Arm A (ICI-naïve) final 9/22 evaluable → posterior 0.880 → below the 0.95 threshold (Arm A enrolled only 22/40 planned, so even at 40.9% ORR the truncated sample doesn't reach the formal Bayesian success criterion — a real finding the design rule surfaces). Calculator-correctness checks also verify Zetyra's posterior formula matches scipy reference to 4 decimals at both the planning assumption and the actual interim/final counts.

### CUPED Simulation Benchmark

Beyond the analytical formula checks, the CUPED calculator is validated with:
- **Monte Carlo variance reduction** — 100k correlated (X, Y) samples verify VRF = 1 − ρ² empirically
- **MC power verification** — 10k simulated experiments confirm n_adjusted achieves target 80% power
- **Deng et al. (2013) reduction ratio** — n_adjusted / n_original = 1 − ρ² verified across 16 parameter combinations
- **Extreme correlations** — ρ = 0.01 (no reduction) and ρ = 0.99 (98% reduction)

### Survival/TTE Endpoints

Three calculators now support time-to-event outcomes via the Schoenfeld variance mapping `Var(log HR) = 4/d`:

- **GSD Survival** — event-driven group sequential boundaries with O'Brien-Fleming / Pocock spending, sample size from event probability, allocation ratio support
- **GSD Survival gsDesign Benchmark** — z-score boundaries, cumulative alpha spending, and Schoenfeld event counts cross-validated against gsDesign R package across 5 spending configurations (OBF k=3,4,5; Pocock k=3,4)
- **Bayesian Sequential Survival** — posterior probability boundaries mapped from the Normal-Normal conjugate framework (`data_variance=4`, `n_k = events/2`)
- **Bayesian Sequential Survival Benchmark** — Zhou & Ji boundary formula verified across 4 event schedules, Type I error controlled via MC multivariate normal, vague-prior convergence to Φ⁻¹(γ), futility boundaries verified
- **Bayesian Predictive Power (Survival)** — interim HR → posterior on log(HR) scale → predictive probability of final success, with HR-scale credible intervals
- **Bayesian Survival PP Benchmark** — 5 conjugate posterior oracle checks, independent MC predictive probability cross-validation, frequentist convergence (vague prior PP ≈ conditional power), known-outcome edge cases

### Sample Size Re-estimation (SSR)

- **Blinded SSR** — variance/rate re-estimation at interim with conditional power, supports continuous, binary, and survival endpoints
- **Unblinded SSR** — four-zone classification (futility, unfavorable, promising, favorable) based on conditional power thresholds, with sample size inflation caps
- **Single-Arm SSR (Phase II oncology ORR)** — Bayesian (posterior + predictive-probability) and frequentist (conditional-power promising-zone) interim decision rules against a fixed historical control rate. 13 suites cover analytical posterior matching `1 − F_Beta(p₀; α+r, β+n−r)`, initial N from the one-sample binomial normal approximation, four-zone CP classification, N_max cap enforcement, seed reproducibility (bit-identical OC tables across 9 numeric columns), Type I error calibration with Clopper-Pearson upper bounds, power monotonicity, the **γ_efficacy / γ_final decoupling regression** (raising γ_efficacy 0.95→0.99 with γ_final pinned at 0.975 reduces interim early stops without collapsing final-look success), input guards (4xx for invalid p₁≤p₀ / α=0 / γ_final out of range / prior_α≤0), required schema fields in both modes, the Bayesian SAP early-stop clause regression, the **`pp_promising_upper` regression** (raising the promising-zone upper bound does not shrink N_p90), and the **N-floor regression** (every Bayesian non-stop sensitivity row has `recalculated_n ≥ initial_n`).
- **gsDesign cross-validation** — sample size formulas, conditional power, zone classification, and binary rate re-estimation verified against reference formulas and gsDesign R package

### Adaptive Randomization (v2.0)

- **RAR** — Response-adaptive randomization with DBCD, Thompson sampling, and Neyman allocation. Analytical tests verify Rosenberger optimal allocation formula (√p / Σ√p), Neyman allocation (proportional to √(p(1-p))), and allocation sum invariants. Simulation tests verify power > 0 under H1, type I error controlled, arm sample size distributions, and allocation trajectories. Reference checks match closed-form Rosenberger values within machine precision. Binary, continuous, and survival endpoints covered.
- **Minimization** — Pocock-Simon covariate-adaptive minimization. Analytical tests verify pure random imbalance benchmarks. Simulation tests verify minimization reduces imbalance vs pure random, higher p_randomization → lower imbalance, and deterministic assignment (p=1.0) achieves very low imbalance. Input guards for factor levels, prevalences, and imbalance function validated.

### Master Protocol (v2.1)

- **Basket Trial** — Independent, BHM (Berry et al. 2013), and EXNEX (Neuenschwander et al. 2016) analyses. Analytical tests verify Beta-Binomial conjugate posterior mean ((1+s)/(2+n)), BHM shrinkage property (posteriors between independent estimate and grand mean), and EXNEX convergence to BHM/independent at extreme weights. Simulation tests verify per-basket power, type I error, and FWER reporting.
- **Umbrella Trial** — Frequentist and Bayesian analyses across binary, continuous, and survival endpoints. Tests verify Bonferroni/Holm multiplicity adjustment, shared control allocation, biomarker prevalence effects, and per-substudy power. Simulation tests cover frequentist binary, Bayesian, and survival paths.
- **Platform Trial** — Multi-arm multi-stage with staggered arm entry and three control pooling modes. Tests verify O'Brien-Fleming and Pocock spending boundaries, per-arm power estimates, total N max formula, concurrent/pooled/naive control allocation, and input guards. Simulation tests verify power and FWER control.

### Real-World Adaptive Trial Replications (v2.2)

Three landmark adaptive trials are replicated against Zetyra's master protocol calculators:

- **I-SPY 2** (breast cancer, Barker et al. 2009) — Bayesian basket trial with pCR endpoint across biomarker-defined signatures. Validates graduation decisions for veliparib (TNBC), pembrolizumab (TNBC, HR+/HER2-), and neratinib (HER2+/HR-) using published pCR rates. Beta-Binomial conjugate posterior verified against scipy reference values.
- **STAMPEDE** (prostate cancer, Sydes et al. 2012) — MAMS platform trial with 5 arms, 4 stages, and survival endpoints. Validates boundary structure (OBF spending), power estimates for docetaxel-like effects (HR=0.78), futility detection for celecoxib-like null effects (HR=0.98), and total sample size calculations. Simulation confirms early stopping behavior.
- **REMAP-CAP** (COVID-19, Angus et al. 2020) — Bayesian adaptive platform with 99% posterior probability thresholds. Validates tocilizumab superiority detection (mortality 28.0% vs 35.8%), lopinavir futility (null effect), multi-domain structure, staggered arm entry, and concurrent control handling. Simulation confirms strong effect detection.

### Bayesian Sequential Cross-Validation

Zhou & Ji (2024) Table 3 provides exact numerical boundary values for two prior configurations (conservative and vague). The cross-validation:
- Reproduces all 10 Table 3 boundary values within ±0.02
- Verifies Type I error = 0.05 via multivariate normal Monte Carlo integration
- Runs 15 additional scenarios with varied priors and data variances

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

# R (for GSD and SSR benchmark validations)
install.packages(c("gsDesign", "httr", "jsonlite"))
```

### Run Tests

```bash
# All Python tests (against local server)
for f in free/test_*.py bayesian/test_*.py gsd/test_*.py cuped/test_*.py ssr/test_*.py adaptive/test_*.py master_protocol/test_*.py; do
  python "$f" http://localhost:8000
done

# Offline tests (no server needed)
python bayesian/test_offline_references.py
python free/test_chi_square.py  # validates the client-side frontend math

# R-based GSD benchmarks
cd gsd && Rscript test_gsdesign_benchmark.R
cd gsd && Rscript test_gsd_survival_benchmark.R

# R-based SSR benchmark
cd ssr && Rscript test_ssr_rpact_benchmark.R
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
      Superiority: ctrl=0.30, treat=0.50, flat priors             80 0.0500    0.0639 0.8355    0.8131  True
PUNCH CD3 rates: ctrl=0.624, treat=0.712, flat priors            400 0.0490    0.0628 0.8345    0.8121  True
     Large effect: ctrl=0.20, treat=0.50, flat priors             40 0.0415    0.0544 0.8850    0.8654  True

...

======================================================================
CONTINUOUS ENDPOINT TESTS
======================================================================

6. Continuous Two-Arm MC Validation
----------------------------------------------------------------------
                                              test  rec_n_per_arm  type1  type1_ub  power  power_lb  pass
     Continuous: Moderate: δ=0.5, σ²=1, flat prior             50 0.0433    0.0538 0.8007    0.7812  True
Continuous: Small effect: δ=0.3, σ²=2, vague prior            360 0.0473    0.0582 0.8943    0.8791  True

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
- `POST /gsd/survival`
- `POST /bayesian/continuous`
- `POST /bayesian/binary`
- `POST /bayesian/survival`
- `POST /bayesian/prior-elicitation`
- `POST /bayesian/borrowing`
- `POST /bayesian/sample-size-single-arm`
- `POST /bayesian/two-arm`
- `POST /bayesian/sequential`
- `POST /bayesian/sequential/survival`
- `POST /ssr/blinded`
- `POST /ssr/unblinded`
- `POST /ssr-single-arm`
- `POST /rar`
- `POST /minimization`
- `POST /basket`
- `POST /umbrella`
- `POST /platform`

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
7. **Survival**: Schoenfeld (1983) *Sample-size formula for the proportional-hazards regression model*
8. **SSR**: Cui, Hung & Wang (1999) *Modification of sample size in group sequential clinical trials*
9. **PACIFIC**: Antonia et al. (2018) NEJM 379:2342-2350 *Overall Survival with Durvalumab*
10. **MONALEESA-7**: Im et al. (2019) NEJM 381:307-316 *Overall Survival with Ribociclib*
11. **Mehta & Pocock**: Mehta & Pocock (2011) *Adaptive increase in sample size when interim results are promising*
12. **Bayesian PP**: Spiegelhalter, Abrams & Myles (2004) *Bayesian Approaches to Clinical Trials*
13. **RAR**: Rosenberger et al. (2001) *Optimal adaptive designs for binary response trials*
14. **RAR**: Hu & Zhang (2004) *Asymptotic properties of doubly adaptive biased coin designs*
15. **Basket**: Berry et al. (2013) *Bayesian Hierarchical Models for Basket Trials*
16. **EXNEX**: Neuenschwander et al. (2016) *Robust exchangeability designs*
17. **Platform**: Saville & Berry (2016) *Efficiencies of platform clinical trials*
18. **Master Protocol**: FDA (2022) *Master Protocols: Efficient Clinical Trial Design Strategies*
19. **I-SPY 2**: Barker et al. (2009) *I-SPY 2: An Adaptive Breast Cancer Trial Design*
20. **STAMPEDE**: Sydes et al. (2012) *Flexible trial design in practice — stopping arms for lack-of-benefit*
21. **REMAP-CAP**: Angus et al. (2020) *Effect of Hydrocortisone on Mortality and Organ Support* JAMA

## Citation

If you use this validation suite in your work, please cite the accompanying white papers:

```bibtex
@software{qian2026zetyra,
  author    = {Qian, Lu},
  title     = {Zetyra: A Validated Suite of Statistical Calculators for Efficient Clinical Trial Design},
  version   = {2.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18879839},
  url       = {https://doi.org/10.5281/zenodo.18879839}
}

@software{qian2026zetyra_bayesian,
  author    = {Qian, Lu},
  title     = {Zetyra Bayesian Toolkit: A Comprehensive Suite of Validated Bayesian Calculators for Clinical Trial Design},
  version   = {1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18880066},
  url       = {https://doi.org/10.5281/zenodo.18880066}
}
```

## License

MIT License - see [LICENSE](LICENSE)
