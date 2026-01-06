# Zetyra Validation Results

**Validation Date:** 2026-01-05 16:31:39
**API URL:** http://localhost:8080/api/v1/validation

## Summary

| Calculator | Status | Tests Passed |
|------------|--------|--------------|
| Sample Size | ✅ PASS | 10 |
| CUPED | ✅ PASS | 15 |
| Group Sequential Design | ✅ PASS | 19 |
| Bayesian | ✅ PASS | 10 |

**Total Tests: 54**
**Overall Status: ✅ ALL PASSED**

## Detailed Results

### Sample Size Calculator

#### Continuous Outcomes
```
                                                                                scenario  zetyra_n  reference_n  n_deviation_pct  zetyra_d  reference_d  d_deviation_pct  pass
                     {'mean1': 100, 'mean2': 105, 'sd': 20, 'alpha': 0.05, 'power': 0.8}       504          504              0.0  0.250000     0.250000           0.0000  True
                     {'mean1': 100, 'mean2': 110, 'sd': 25, 'alpha': 0.05, 'power': 0.9}       264          264              0.0  0.400000     0.400000           0.0000  True
                       {'mean1': 50, 'mean2': 55, 'sd': 15, 'alpha': 0.01, 'power': 0.8}       422          422              0.0  0.333333     0.333333           0.0001  True
       {'mean1': 100, 'mean2': 105, 'sd': 20, 'alpha': 0.05, 'power': 0.8, 'ratio': 2.0}       566          566              0.0  0.250000     0.250000           0.0000  True
{'mean1': 100, 'mean2': 103, 'sd': 20, 'alpha': 0.025, 'power': 0.9, 'two_sided': False}      1868         1868              0.0  0.150000     0.150000           0.0000  True
```

#### Binary Outcomes
```
                                                                scenario  zetyra_n  reference_n  n_deviation_pct  zetyra_h  reference_h  h_deviation_pct  pass
                    {'p1': 0.1, 'p2': 0.15, 'alpha': 0.05, 'power': 0.8}      1372         1372              0.0  0.151898     0.151898         0.000183  True
                     {'p1': 0.2, 'p2': 0.3, 'alpha': 0.05, 'power': 0.9}       784          784              0.0  0.231984     0.231984         0.000113  True
                     {'p1': 0.5, 'p2': 0.6, 'alpha': 0.01, 'power': 0.8}      1154         1154              0.0  0.201358     0.201358         0.000039  True
      {'p1': 0.1, 'p2': 0.15, 'alpha': 0.05, 'power': 0.8, 'ratio': 2.0}      1577         1577              0.0  0.151898     0.151898         0.000183  True
{'p1': 0.3, 'p2': 0.4, 'alpha': 0.025, 'power': 0.9, 'two_sided': False}       954          954              0.0  0.210159     0.210159         0.000036  True
```

### CUPED Calculator

#### Numerical Validation
```
 correlation  zetyra_n_orig  ref_n_orig  zetyra_n_adj  ref_n_adj  zetyra_vrf  ref_vrf  n_deviation_pct  pass
         0.0            252         252           252        252        1.00     1.00             0.00  True
         0.3            252         252           230        229        0.91     0.91             0.44  True
         0.5            252         252           189        189        0.75     0.75             0.00  True
         0.7            252         252           129        129        0.51     0.51             0.00  True
         0.9            252         252            48         48        0.19     0.19             0.00  True
        -0.6            252         252           162        161        0.64     0.64             0.62  True
         0.6            142         142            91         91        0.64     0.64             0.00  True
         0.8           1323        1323           477        477        0.36     0.36             0.00  True
```

#### Properties
```
                          property                 expected     actual  pass
   Zero correlation → no reduction n_original == n_adjusted 252 == 252  True
              VRF = 1 - ρ² (ρ=0.3)                   0.9100     0.9100  True
              VRF = 1 - ρ² (ρ=0.5)                   0.7500     0.7500  True
              VRF = 1 - ρ² (ρ=0.7)                   0.5100     0.5100  True
              VRF = 1 - ρ² (ρ=0.9)                   0.1900     0.1900  True
Symmetry: |ρ| determines reduction    n(ρ=0.7) == n(ρ=-0.7) 129 == 129  True
              Higher ρ → smaller n      n(ρ=0.8) < n(ρ=0.5)   91 < 189  True
```

### Group Sequential Design

#### Spending Functions
```
       function  look  info_frac  zetyra_alpha  reference_alpha  deviation  pass
O'Brien-Fleming     1       0.25      0.000089         0.000089        0.0  True
O'Brien-Fleming     2       0.50      0.005575         0.005575        0.0  True
O'Brien-Fleming     3       0.75      0.023625         0.023625        0.0  True
O'Brien-Fleming     4       1.00      0.025000         0.025000        0.0  True
         Pocock     1       0.25      0.008934         0.008934        0.0  True
         Pocock     2       0.50      0.015503         0.015503        0.0  True
         Pocock     3       0.75      0.020700         0.020700        0.0  True
         Pocock     4       1.00      0.025000         0.025000        0.0  True
```

#### Properties
```
                          property                   expected                  actual  pass
   OBF: Conservative at first look                Z[1] > Z[3]           3.472 > 2.004  True
     Pocock: More uniform than OBF range(Pocock) < range(OBF)           0.000 < 1.467  True
  Sample size inflation reasonable     1.0 ≤ inflation ≤ 1.20                  1.1627  True
OBF: Final boundary ≈ fixed design  |Z_final - Z_fixed| < 0.3 |2.004 - 1.960| = 0.044  True
         More looks → larger max N    n_max(k=5) ≥ n_max(k=2)               610 ≥ 509  True
```

#### gsDesign Benchmarks
```
          benchmark  look  zetyra_z  gsdesign_z  deviation  pass
   gsDesign k=3 OBF     1     3.472       3.471      0.001  True
   gsDesign k=3 OBF     2     2.455       2.454      0.001  True
   gsDesign k=3 OBF     3     2.004       2.004      0.000  True
gsDesign k=3 Pocock     1     2.297       2.289      0.008  True
gsDesign k=3 Pocock     2     2.297       2.289      0.008  True
gsDesign k=3 Pocock     3     2.297       2.289      0.008  True
```

### Bayesian Calculator

#### Continuous Posterior
```
                        scenario  zetyra_mean  ref_mean  mean_dev  zetyra_var  ref_var  var_dev  pass
 prior=(0.0,1.0), data=(0.3,0.1)       0.2727    0.2727  0.000027      0.0909   0.0909      0.0  True
prior=(0.5,0.5), data=(0.4,0.05)       0.4091    0.4091  0.000009      0.0455   0.0455      0.0  True
 prior=(0.0,2.0), data=(0.6,0.2)       0.5455    0.5455  0.000045      0.1818   0.1818      0.0  True
```

#### Binary Posterior
```
 control treatment ctrl_alpha    ctrl_beta  trt_alpha  pass
(30/100)  (45/100) 31.0 == 31   71.0 == 71 46.0 == 46  True
(50/200)  (70/200) 52.0 == 52 152.0 == 152 72.0 == 72  True
 (10/50)   (20/50) 11.0 == 11   41.0 == 41 21.0 == 21  True
```

#### Properties
```
                                          property                      expected        actual  pass
                           Strong effect → high PP                      PP > 0.7    PP = 1.000  True
                              Null effect → low PP                      PP < 0.3    PP = 0.042  True
More precise interim → higher PP (positive effect)    PP(var=0.05) ≥ PP(var=0.2) 0.892 ≥ 0.490  True
                      Optimistic prior → higher PP PP(prior=0.3) ≥ PP(prior=0.0) 0.382 ≥ 0.296  True
```