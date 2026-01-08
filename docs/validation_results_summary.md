# Validation Results Summary

Detailed breakdown of validation results comparing Zetyra calculators against reference implementations.

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

## Bayesian Validation

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
