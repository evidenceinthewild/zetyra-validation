# Validation Results Summary

Detailed breakdown of validation results comparing Zetyra calculators against reference implementations.

**Total: 169 tests across 12 scripts, all passing.**

---

## GSD Validation Against gsDesign

**Maximum deviation: 0.0046 z-score** (target: 0.05)
**Mean deviation: 0.0012 z-score**

### Boundary Comparisons by Design

| Design | Looks | Boundaries Tested | Max Dev | Status |
|--------|-------|-------------------|---------|--------|
| OF_2   | 2     | 2                 | 0.0000  | ✅ Pass |
| OF_3   | 3     | 3                 | 0.0001  | ✅ Pass |
| OF_4   | 4     | 4                 | 0.0017  | ✅ Pass |
| OF_5   | 5     | 5                 | 0.0046  | ✅ Pass |
| Pocock_2 | 2   | 2                 | 0.0000  | ✅ Pass |
| Pocock_3 | 3   | 3                 | 0.0002  | ✅ Pass |
| Pocock_4 | 4   | 4                 | 0.0008  | ✅ Pass |

[See detailed results →](../gsd/results/gsd_validation_results.csv)

### Published Trial Replications

#### HPTN 083 (HIV Prevention Trial)
- **Design**: 4-look O'Brien-Fleming
- **Reference**: gsDesign R package
- **Max deviation**: 0.0046 z-score

| Look | Info Frac | Zetyra | Reference | Deviation |
|------|-----------|--------|-----------|-----------|
| 1    | 0.25      | 4.0444 | 4.049     | 0.0046    |
| 2    | 0.50      | 2.8598 | 2.863     | 0.0032    |
| 3    | 0.75      | 2.3351 | 2.337     | 0.0019    |
| 4    | 1.00      | 2.0222 | 2.024     | 0.0018    |

[See detailed results →](../gsd/results/hptn083_validation.csv)

#### HeartMate II (LVAD Trial)
- **Design**: 3-look O'Brien-Fleming with unequal spacing
- **Info fractions**: [0.27, 0.67, 1.00]
- **All property tests passed**

[See detailed results →](../gsd/results/heartmate_validation.csv)

---

## CUPED Validation

**Formula validated**: VRF = 1 - ρ²

### Numerical Validation

| Correlation | Zetyra n_adj | Reference n_adj | Zetyra VRF | Reference VRF | Status |
|-------------|--------------|-----------------|------------|---------------|--------|
| 0.0         | 252          | 252             | 1.00       | 1.00          | ✅ Pass |
| 0.3         | 230          | 229             | 0.91       | 0.91          | ✅ Pass |
| 0.5         | 189          | 189             | 0.75       | 0.75          | ✅ Pass |
| 0.7         | 129          | 129             | 0.51       | 0.51          | ✅ Pass |
| 0.9         | 48           | 48              | 0.19       | 0.19          | ✅ Pass |

### Property Tests

| Property | Expected | Actual | Status |
|----------|----------|--------|--------|
| Zero correlation → no reduction | n_original == n_adjusted | 252 == 252 | ✅ Pass |
| VRF = 1 - ρ² (ρ=0.3) | 0.9100 | 0.9100 | ✅ Pass |
| VRF = 1 - ρ² (ρ=0.5) | 0.7500 | 0.7500 | ✅ Pass |
| VRF = 1 - ρ² (ρ=0.7) | 0.5100 | 0.5100 | ✅ Pass |
| VRF = 1 - ρ² (ρ=0.9) | 0.1900 | 0.1900 | ✅ Pass |
| Symmetry: \|ρ\| determines reduction | n(ρ=0.7) == n(ρ=-0.7) | 129 == 129 | ✅ Pass |
| Higher ρ → smaller n | n(ρ=0.8) < n(ρ=0.5) | 91 < 189 | ✅ Pass |

[See detailed results →](../cuped/results/cuped_validation_results.csv)

---

## Bayesian Predictive Power

**17 tests: conjugate posteriors, predictive probability properties**

### Beta-Binomial Conjugate

**Formula validated**: π|x ~ Beta(α + x, β + n - x)

| Scenario | Control Posterior | Treatment Posterior | Status |
|----------|-------------------|---------------------|--------|
| C=30/100, T=45/100 | Beta(31, 71) | Beta(46, 56) | ✅ Pass |
| C=25/100, T=35/100 | Beta(25.5, 75.5) | Beta(35.5, 65.5) | ✅ Pass |
| C=15/75, T=25/75 | Beta(17, 68) | Beta(27, 58) | ✅ Pass |

[See detailed results →](../bayesian/results/bayesian_validation_results.csv)

### Normal-Normal Conjugate

**Formula validated**: Precision-weighted posterior

| Scenario | Zetyra Mean | Reference Mean | Zetyra Var | Reference Var | Status |
|----------|-------------|----------------|------------|---------------|--------|
| prior=(0.0, 1.0) | 0.2727 | 0.2727 | 0.0909 | 0.0909 | ✅ Pass |
| prior=(0.5, 0.5) | 0.4091 | 0.4091 | 0.0455 | 0.0455 | ✅ Pass |
| prior=(0.0, 2.0) | 0.5455 | 0.5455 | 0.1818 | 0.1818 | ✅ Pass |

### Predictive Probability Properties

| Property | Expected | Actual | Status |
|----------|----------|--------|--------|
| Strong effect → high PP | PP > 0.7 | PP = 1.000 | ✅ Pass |
| Null effect → low PP | PP < 0.3 | PP = 0.042 | ✅ Pass |
| Optimistic prior → higher PP | PP(prior=0.3) ≥ PP(prior=0.0) | 0.382 ≥ 0.296 | ✅ Pass |

[See detailed results →](../bayesian/results/normal_conjugate_validation.csv)

---

## Prior Elicitation

**22 tests across 3 elicitation methods**

### ESS-Based Prior

**Formula validated**: α = mean × ESS, β = (1−mean) × ESS

| Scenario | Zetyra α | Reference α | Zetyra β | Reference β | Status |
|----------|----------|-------------|----------|-------------|--------|
| Weakly informative (ESS=2) | 0.60 | 0.60 | 1.40 | 1.40 | ✅ Pass |
| Berry et al. (2010) | 2.50 | 2.50 | 7.50 | 7.50 | ✅ Pass |
| Vague prior (ESS=2) | 1.00 | 1.00 | 1.00 | 1.00 | ✅ Pass |
| Moderate informative (ESS=20) | 3.00 | 3.00 | 17.00 | 17.00 | ✅ Pass |

### Historical Prior (REBYOTA PUNCH CD2)

Uses power prior formula: α = 1 + δ×events, β = 1 + δ×(n−events)

**Data source**: REBYOTA PUNCH CD2 Phase 2b — 25/45 responders, two-dose arm (FDA BLA 125739)

| Discount (δ) | Zetyra α | Reference α | Zetyra β | Reference β | Status |
|--------------|----------|-------------|----------|-------------|--------|
| 0.0 (no borrow) | 1.00 | 1.00 | 1.00 | 1.00 | ✅ Pass |
| 0.1 | 3.50 | 3.50 | 3.00 | 3.00 | ✅ Pass |
| 0.5 | 13.50 | 13.50 | 11.00 | 11.00 | ✅ Pass |
| 1.0 (full borrow) | 26.00 | 26.00 | 21.00 | 21.00 | ✅ Pass |

### Quantile Matching

Validated against scipy.optimize (Nelder-Mead) reference implementation.

| Scenario | Target Quantiles | Deviation | Status |
|----------|------------------|-----------|--------|
| Berry: median≈0.25, 90%CI=[0.10,0.40] | q05=0.10, q50=0.25, q95=0.40 | < 0.02 | ✅ Pass |
| Tight: median≈0.50, 90%CI=[0.40,0.60] | q05=0.40, q50=0.50, q95=0.60 | < 0.02 | ✅ Pass |

---

## Bayesian Borrowing

**18 tests covering power prior, MAP prior, and heterogeneity**

### Power Prior (REBYOTA)

**Data source**: REBYOTA PUNCH CD2 Phase 2b — 25/45 responders (FDA BLA 125739)

| Scenario | Zetyra α | Reference α | Zetyra ESS | Reference ESS | Status |
|----------|----------|-------------|------------|---------------|--------|
| REBYOTA δ=0.5 | 13.50 | 13.50 | 24.50 | 24.50 | ✅ Pass |
| REBYOTA δ=0.0 (no borrow) | 1.00 | 1.00 | 2.00 | 2.00 | ✅ Pass |
| REBYOTA δ=1.0 (full borrow) | 26.00 | 26.00 | 47.00 | 47.00 | ✅ Pass |
| PUNCH CD3 δ=0.5 | 64.00 | 64.00 | 90.50 | 90.50 | ✅ Pass |

### MAP Prior — Heterogeneity (Cochran's Q / I²)

| Scenario | Expected I² | Status |
|----------|-------------|--------|
| Low heterogeneity (similar rates ~0.21) | 0–30% | ✅ Pass |
| High heterogeneity (diverse rates) | 70–100% | ✅ Pass |
| Two similar studies | 0–30% | ✅ Pass |
| REBYOTA CD2+CD3 (real trial data) | 40–90% | ✅ Pass |

The REBYOTA CD2+CD3 scenario combines Phase 2b (55.6% response, n=45) and Phase 3 (71.2% response, n=177) data to validate heterogeneity detection across trial phases.

### Invariants & Symmetry

| Property | Result | Status |
|----------|--------|--------|
| Higher discount → higher ESS | ESS(δ=0.8) > ESS(δ=0.2) | ✅ Pass |
| Identical studies → I²=0 | I² < 1.0% | ✅ Pass |

---

## Bayesian Sample Size (Single-Arm)

**14 tests: analytical posteriors, MC search, properties**

### Analytical Posterior

| Scenario | Formula | Status |
|----------|---------|--------|
| Beta(1,1) uninformative, null=0.10, alt=0.25 | Beta(α+r, β+n−r) | ✅ Pass |
| Beta(2,8) informative, null=0.10, alt=0.25 | Beta(α+r, β+n−r) | ✅ Pass |
| REBYOTA prior, null=0.45, alt=0.65 | Beta(13.5+r, 11+n−r) | ✅ Pass |

The REBYOTA scenario uses prior parameters α=13.5, β=11.0 derived from PUNCH CD2 Phase 2b data (25/45 responders at 50% discount).

### MC Sample Size Search & Properties

Type I error and power validated with Clopper-Pearson binomial CIs.

| Property | Result | Status |
|----------|--------|--------|
| Higher threshold → larger n | n(0.99) > n(0.95) | ✅ Pass |
| Larger effect → smaller n | n(Δ=0.10) > n(Δ=0.20) | ✅ Pass |
| Seed reproducibility | Identical across runs | ✅ Pass |

---

## Bayesian Two-Arm Design

**13 tests: MC search, properties, symmetry**

### MC Validation (REBYOTA PUNCH CD3)

**Data source**: REBYOTA PUNCH CD3 Phase 3 — 126/177 treatment (71.2%), 53/85 placebo (62.4%) (FDA BLA 125739)

Type I error bounds are Clopper-Pearson upper confidence limits; power bounds are lower confidence limits (2000 simulations).

| Scenario | n per arm | Type I UB | Power LB | Status |
|----------|-----------|-----------|----------|--------|
| Superiority: ctrl=0.30, treat=0.50 | ~80 | ≤ 0.08 | ≥ 0.70 | ✅ Pass |
| PUNCH CD3: ctrl=0.624, treat=0.712 | ~350 | ≤ 0.08 | ≥ 0.70 | ✅ Pass |
| Large effect: ctrl=0.20, treat=0.50 | ~60 | ≤ 0.08 | ≥ 0.70 | ✅ Pass |

### Properties & Symmetry

| Property | Result | Status |
|----------|--------|--------|
| Larger effect → smaller n | n(Δ=0.30) < n(Δ=0.10) | ✅ Pass |
| Higher threshold → larger n | n(γ=0.975) > n(γ=0.95) | ✅ Pass |
| Null same-seed → identical results | Exact match | ✅ Pass |
| Mirror symmetry around 0.5 | (0.30,0.50) ≈ (0.50,0.70) | ✅ Pass |
| Same Δ, different base rates | Within tolerance | ✅ Pass |

---

## Bayesian Sequential Monitoring

**20 tests: analytical boundaries, structural properties, invariants**

### Analytical Boundaries (Zhou & Ji 2024)

**Formula**: c_k = Φ⁻¹(γ) × √(1 + σ²/(n_k × ν²)) − μ × √(σ²) / (√(n_k) × ν²)

Reference: Zhou, T., & Ji, Y. (2024) "On Bayesian Sequential Clinical Trial Designs," *NEJSDS*, 2(1), 136–151.

| Scenario | Looks | Max Deviation | Status |
|----------|-------|---------------|--------|
| Zhou & Ji (2024) example | 3 | < 0.0001 | ✅ Pass |
| Informative prior (μ=0.5, ν²=0.5) | 3 | < 0.0001 | ✅ Pass |
| Large sample (n=50,100,150,200) | 4 | < 0.0001 | ✅ Pass |

### Structural Properties

| Property | Result | Status |
|----------|--------|--------|
| Efficacy boundaries decrease with n | Monotonically decreasing | ✅ Pass |
| Futility < Efficacy at each look | Strict inequality | ✅ Pass |
| Vague prior → converges to Φ⁻¹(γ) | Boundary → 1.96 as n → ∞ | ✅ Pass |
| Higher threshold → higher boundary | c(0.99) > c(0.975) | ✅ Pass |

---

## Offline Reference Tests

**23 pure-math tests (no API dependency)**

| Category | Tests | Status |
|----------|-------|--------|
| Beta-Binomial conjugate update | 3 | ✅ Pass |
| Normal-Normal conjugate update | 3 | ✅ Pass |
| Zhou & Ji boundary formula | 4 | ✅ Pass |
| Vague-prior → frequentist convergence | 2 | ✅ Pass |
| Cochran's Q / I² heterogeneity | 3 | ✅ Pass |
| ESS-based prior elicitation | 2 | ✅ Pass |
| Power prior discounting | 3 | ✅ Pass |
| Clopper-Pearson CI helpers | 3 | ✅ Pass |
