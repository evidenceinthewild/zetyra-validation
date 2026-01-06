# Clinical Trials Reporting ANCOVA Baseline-Outcome Correlations and Variance Reduction

Published clinical trials **rarely report explicit baseline-outcome correlation coefficients** in their primary publications—this statistical parameter is typically used for sample size calculations but not disclosed. However, this systematic search identified **18 trials and key methodology papers** that explicitly report quantitative correlation values (r, ρ) or ANCOVA variance reduction percentages, providing validated benchmarks for statistical calculator accuracy.

The most comprehensive empirical source is the Walters et al. (2019) systematic review of 20 RCTs, which established a **mean baseline-outcome correlation of r = 0.50** (median 0.51, range -0.13 to 0.91) across 464 correlations from 7,173 participants. This finding validates the design factor formula (1 - ρ²), indicating ANCOVA typically achieves **25% variance reduction** at the median correlation level.

---

## Key methodology papers with empirical trial data

Three foundational papers provide the statistical framework and empirical validation for ANCOVA efficiency calculations:

**Frison L, Pocock SJ (1992)** established the variance reduction formula in *Statistics in Medicine* (11:1685-1704). Their analysis of repeated-measures designs identified typical correlation patterns: **ρ = 0.7** for within-period correlations (pre-pre, post-post) and **ρ = 0.5** for between-period correlations (baseline to follow-up). The design factor (1 - ρ²) determines sample size reduction—with ρ = 0.5, ANCOVA requires only 75% of the sample needed for unadjusted analysis.

**Vickers AJ, Altman DG (2001)** demonstrated practical ANCOVA application in their BMJ Statistics Note (323:1123-1124) using the **Kleinhenz acupuncture trial for rotator cuff tendonitis** (n=52). They showed ANCOVA detected a **12.7-point difference** (95% CI: 4.1-21.3, p=0.005) compared to change-score analysis (10.8 points, p=0.014), with the regression slope b = 0.71 and within-group baseline-change correlation r = -0.25. Their worked example demonstrated that with **r = 0.6**, ANCOVA requires only 54 patients versus 85 for follow-up score analysis—a **36% reduction**.

**Walters SJ et al. (2019)** published the definitive empirical reference in *Trials* (20:566), analyzing **464 correlations from 20 UK RCTs** with 7,173 baseline participants. Overall results: **mean r = 0.50, median r = 0.51, SD = 0.15, range -0.13 to 0.91**. Correlations declined approximately 0.003 per month of follow-up. The interquartile range of 0.41-0.60 suggests most RCTs can assume 20-35% variance reduction from ANCOVA.

---

## Trials with explicit variance reduction reporting

The Benkeser et al. (2019) paper in *Biometrics* (75:1391-1400) re-analyzed three neuropsychiatric trials demonstrating **variance reductions of 4% to 32%**:

| Trial | Disease | N | Endpoint | Variance Reduction | Treatment Effect (ANCOVA) |
|-------|---------|---|----------|-------------------|--------------------------|
| **TADS (FLX arm)** | Adolescent depression | 109 | CDRS-R | **32%** | -4.36 points (p=0.01) |
| **METS** | Schizophrenia | 146 | Body weight | ~15% | -3.0 kg vs -1.0 kg |
| **MCI** | Mild cognitive impairment | 512 | CDR-SB | **4%** | -3.6 points (SE 1.6) |

The **Treatment for Adolescents with Depression Study (TADS)** provides the clearest demonstration of ANCOVA's clinical impact. The unadjusted 95% CI (-6.0 to 3.2) included zero, but ANCOVA adjustment—primarily through baseline CDRS-R correlation—yielded CI (-8.1 to -0.6), excluding zero and achieving statistical significance. Original citation: TADS Team, JAMA 2004;292:807-820.

The **MCI Trial** (Petersen et al., NEJM 2005;352:2379-2388; NCT00006286) showed only 4% variance reduction because baseline CDR-SB scores had weak correlation with 18-month outcomes—other covariates (ADAS-Cog, MMSE, ADL scores) provided minimal additional efficiency.

---

## Individual RCTs with reported correlation coefficients

### Oncology trials

**S1609 DART Trial** (Othus M et al., JNCI 2024;116:673-680; NCT02834013): This Phase II rare cancers immunotherapy basket trial (n=638 evaluable) reported exceptionally high correlations between tumor size change and survival outcomes. **Pearson correlations: r = -0.94 (median PFS), r = -0.90 (1-year OS), r = -0.84 (median OS)**—all p<0.001. C-statistics for regression models: 0.768 (PFS), 0.685 (OS). While these correlations reflect tumor change-survival relationships rather than traditional baseline-endpoint correlations, they demonstrate the strong predictive value of continuous tumor measurements.

**AIM-High Melanoma Trial** (Dixon S et al., Br J Cancer 2006;94:492-498): Interferon-alpha RCT (n=444 baseline) using EORTC QLQ-C30 quality of life outcomes. **Mean baseline-follow-up correlation r = 0.46 (median 0.49, range 0.16-0.74)** across 60 correlations at 6, 12, 18, and 24 months. Physical Functioning showed highest correlation (r = 0.52-0.68); Nausea/Vomiting showed lowest (r = 0.16-0.30). At the mean correlation of 0.46, expected variance reduction = 21%.

### Musculoskeletal and pain trials

**SELF Trial** (Littlewood C et al., Clinical Rehabilitation 2016;30:686-696): Self-managed exercise versus physiotherapy for rotator cuff tendinopathy (n=85). **SPADI Total correlation r = 0.50** (baseline to 3 months); SPADI Disability r = 0.49; SPADI Pain r = 0.36. The trial explicitly used r = 0.50 in ANCOVA-based sample size calculation, reducing requirements from 91 per group (POST method) to 34 per group—a **63% reduction**.

**Acupuncture Low Back Pain Trial** (Thomas KJ et al., HTA 2005;9:1-109): Acupuncture care for chronic low back pain (n=239 baseline, 217 with correlation data). **Mean SF-36 correlation r = 0.44 (median 0.45)**; Physical Functioning showed r = 0.62 (highest); Role-Physical showed r = 0.20 (lowest).

**Knee Replacement Physiotherapy Trial** (Mitchell C et al., J Eval Clin Pract 2005;11:283-292): Pre/post-operative home physiotherapy for total knee replacement (n=151 baseline, 114 for correlations). **Mean correlation r = 0.45 (median 0.48)**; WOMAC Physical Function r = 0.46; WOMAC Pain r = 0.26; WOMAC Stiffness r = 0.09; SF-36 Physical Functioning r = 0.65.

### Mental health and depression trials

**PoNDER Trial** (Morrell CJ et al., BMJ 2009;338:a3045): Cluster RCT of health visitor training for postnatal depression (n=2,659—largest in review). **Mean correlation r = 0.44 (median 0.47)**; EPDS correlations r = 0.47-0.52; CORE-OM Total r = 0.53-0.58; CORE-OM Functioning r = 0.54-0.58.

**Lifestyle Matters Trial** (Mountain G et al., Age Ageing 2017;46:627-634): Preventive lifestyle intervention for older adults (n=288). **Mean correlation r = 0.66 (range 0.45-0.88)**—highest among reviewed trials. PHQ-9 depression correlations r = 0.53-0.76, demonstrating that depression scales reliably show higher baseline-outcome correlations.

**STEPWISE Trial** (Holt RIG et al., HTA 2018;22:1-160): Structured lifestyle education for weight loss in schizophrenia (n=412). **Mean correlation r = 0.53 (range 0.24-0.72)**; BPRS, EQ-5D, PHQ-9, and RAND SF-36 endpoints measured at 3 and 12 months.

### Respiratory and cardiovascular trials

**COPD Pulmonary Rehabilitation Trial** (Waterhouse JC et al., HTA 2010;14:1-v): Community versus hospital pulmonary rehabilitation (n=238 baseline, 172 for correlations). **Mean correlation r = 0.53 (median 0.54)**; EQ-5D r = 0.55; SF-36 Physical Functioning r = 0.68 (highest); SF-36 Role-Physical r = 0.37 (lowest). Time points: 2, 6, 12, 18 months.

**GMC Diabetes Trial** (analyzed in Lu K et al., BMJ Open 2016;6:e013096): Type 2 diabetes fasting LDL cholesterol trial (n=195 completers). **Baseline-to-12-month LDL-C correlation approximately r = 0.50**. ANCOVA treatment effect: -11.2 mg/dL (95% CI: -19.2 to -3.3, p<0.05).

**GLP-1 Receptor Agonist Diabetes Trial** (Jones AG et al., PLoS ONE 2016;11:e0152428): Sitagliptin for type 2 diabetes (n=257). **Pre-baseline to baseline HbA1c correlation r = 0.81**; R² = 0.19 for baseline HbA1c explaining HbA1c change variance. Regression coefficient: β = -0.44 mmol/mol (95% CI: -0.58 to -0.29).

### Ophthalmology trials

**PEDIG Amblyopia Trial** (Pediatric Eye Disease Investigator Group, Ophthalmology 2010; PMC2864338): Bangerter filters versus patching for moderate amblyopia (n=170, ages 3-10). **Baseline to 24-week visual acuity correlation r = 0.20**. Statistical parameters: α = 0.05 one-sided, power = 90%, non-inferiority limit = 0.075 logMAR, SD = 0.16 logMAR. At r = 0.20, expected variance reduction = only 4%.

**SCORE Study** (SCORE Research Group, Ophthalmology 2009; PMC2851408): Multicenter Phase III trials for retinal vein occlusion. **CRVO: baseline OCT thickness-visual acuity correlation r = -0.27** (95% CI: -0.38 to -0.16); **BRVO: r = -0.28** (95% CI: -0.37 to -0.19). R² < 10% of visual acuity variance explained by OCT center point thickness.

---

## Summary table of correlation and variance reduction values

| Study | Year | Therapeutic Area | Endpoint | N | Correlation (r) | Variance Reduction |
|-------|------|-----------------|----------|---|-----------------|-------------------|
| Walters et al. (20 RCTs) | 2019 | Multiple | PROMs | 7,173 | **0.50 (mean)** | 25% |
| TADS (FLX arm) | 2004 | Depression | CDRS-R | 109 | High | **32%** |
| MCI Trial | 2005 | Cognitive impairment | CDR-SB | 512 | Moderate | **4%** |
| SELF Trial | 2016 | Shoulder pain | SPADI | 85 | **0.50** | 25% |
| PoNDER Trial | 2009 | Postnatal depression | EPDS | 2,659 | **0.44** | 19% |
| COPD Rehabilitation | 2010 | COPD | SF-36, EQ-5D | 238 | **0.53** | 28% |
| AIM-High Melanoma | 2006 | Oncology | EORTC QLQ-C30 | 444 | **0.46** | 21% |
| Lifestyle Matters | 2017 | Geriatrics | PHQ-9, EQ-5D | 288 | **0.66** | 44% |
| STEPWISE | 2018 | Schizophrenia | Multiple | 412 | **0.53** | 28% |
| Back Pain Acupuncture | 2005 | Low back pain | SF-36 | 239 | **0.44** | 19% |
| Knee Replacement | 2005 | Osteoarthritis | WOMAC, SF-36 | 151 | **0.45** | 20% |
| S1609 DART | 2024 | Rare cancers | Tumor-survival | 638 | **-0.84 to -0.94** | N/A |
| GMC Diabetes | 2016 | Type 2 diabetes | LDL-C | 195 | **~0.50** | 25% |
| GLP-1RA Diabetes | 2016 | Type 2 diabetes | HbA1c | 257 | **0.81** | R²=0.19 |
| PEDIG Amblyopia | 2010 | Amblyopia | Visual acuity | 170 | **0.20** | 4% |
| SCORE CRVO | 2009 | Retinal vein occlusion | VA-OCT | Variable | **-0.27** | <10% |
| Kleinhenz Acupuncture | 1999 | Shoulder pain | Pain scale | 52 | b=0.71 | ~36% |

---

## Correlations by outcome measure type

The Walters systematic review provides stratified correlation estimates useful for prospective sample size calculations:

| Outcome Measure | Mean r | Median r | Range | N Correlations |
|-----------------|--------|----------|-------|----------------|
| PHQ-9 (Depression) | **0.66** | 0.66 | 0.53-0.76 | 6 |
| SF-36 Physical Functioning | **0.64** | 0.63 | 0.01-0.91 | 29 |
| SF-36 General Health | 0.60 | 0.58 | 0.49-0.79 | 29 |
| SF-36 Mental Health | 0.57 | 0.57 | 0.37-0.83 | 27 |
| EQ-5D Utility | 0.55 | 0.54 | 0.32-0.87 | 29 |
| SF-6D | 0.50 | 0.48 | 0.37-0.64 | 14 |
| SPADI (Shoulder Pain) | 0.47 | 0.47 | 0.44-0.50 | 3 |
| VAS Pain | 0.41 | 0.41 | 0.33-0.48 | 4 |
| WOMAC Pain (Knee) | **0.26** | 0.26 | — | 1 |

**Key pattern**: Stable trait-like measures (physical functioning, depression severity) show correlations of **0.60-0.70**, while state-like or symptomatic measures (pain, role functioning) show correlations of **0.30-0.50**.

---

## Variance reduction formula validation

The empirical data consistently validates the Frison-Pocock formula:

**Design Factor = (1 - ρ²)**

| Correlation (ρ) | Variance Factor | Sample Size Reduction |
|-----------------|-----------------|----------------------|
| 0.20 | 0.96 | 4% |
| 0.30 | 0.91 | 9% |
| 0.40 | 0.84 | 16% |
| **0.50** | **0.75** | **25%** |
| 0.60 | 0.64 | 36% |
| 0.70 | 0.51 | 49% |
| 0.80 | 0.36 | 64% |
| 0.90 | 0.19 | 81% |

The Egbewale et al. (2014) simulation study in *BMC Medical Research Methodology* (14:49) confirmed that ANCOVA maintains optimal precision across all correlation levels ≥0.3 and remains **unbiased regardless of baseline imbalance magnitude or direction**.

---

## Conclusion

This compilation provides **18 published trials and methodology papers with explicit quantitative data** for validating ANCOVA statistical calculators. The empirical consensus supports a default baseline-outcome correlation assumption of **r = 0.50** for patient-reported outcomes, yielding approximately **25% variance reduction**. Depression and physical functioning scales reliably exceed r = 0.60 (>35% reduction), while pain and role-functioning measures typically fall in the r = 0.40-0.50 range (15-25% reduction). Correlations decline approximately 0.003 per month of follow-up. The TADS trial demonstrates the most dramatic clinical impact—32% variance reduction converting a non-significant finding to p = 0.01—providing a compelling benchmark for calculator validation.
