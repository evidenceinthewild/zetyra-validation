# GSD Validation Provenance

Group sequential design is the FDA-facing calculator, so its validation state
is recorded here in full rather than summarised in the README. This file
changes on every backend deploy; the README does not.

**Last run: 2026-08-20 against production revision `zetyra-backend-00052-fvn`
(`https://api.zetyra.com`). 9 of 9 suites passed.**

These suites exercise the deployed API, not a local checkout, so the status
above describes that revision and nothing else. **Re-run after any backend
deploy.**

## Suites

| Script | Result | What it pins |
|---|---|---|
| `gsd/test_gsdesign_benchmark.R` | 32/32 | Efficacy boundaries k ≤ 4 (27 numerical, 5 rejection) vs gsDesign 3.10.1 |
| `gsd/test_gsd_futility.py` | 46/46 | Futility bounds, max-N inflation, E[N\|H₀], `test.type=4` |
| `gsd/test_gsd_hsd_gamma.py` | 226/226 | User-selectable Hwang-Shih-DeCani γ |
| `gsd/test_gsd_survival_benchmark.R` | 47/47 | Survival boundaries and alpha spending |
| `gsd/test_gsd_survival.py` | pass | Survival/TTE designs (Schoenfeld) |
| `gsd/test_pacific.py` | pass | Antonia et al. (2018) NEJM, Lan-DeMets OBF |
| `gsd/test_monaleesa7.py` | pass | Im et al. (2019) NEJM, Lan-DeMets OBF |
| `gsd/test_hptn083.py` | pass | 4-look OBF replication |
| `gsd/test_heartmate.py` | pass | Published trial replication |

The R benchmark takes a base URL that **already includes**
`/api/v1/validation`; passing a bare host yields 404. Every other script takes
a bare host.

## Certified design domain: k ≤ 4

The API refuses `k > 4` at the schema layer (HTTP 422), so no five-look design
can be produced or measured. The cap is operational, not statistical: the
worst measured k=5 solve took **69.75s** against a **30s** budget, and MVN
solve time above four looks is unbounded rather than merely slow.

The benchmark does not delete the k=5 configuration; it asserts the rejection.
`OF_5` appears in the results as five rows carrying
`rejected 422 less_than_equal on k, le=4`, so the ceiling is *tested* rather
than merely documented, and the gsDesign reference values stay recorded so
raising the ceiling restores coverage without regenerating anything.

## Boundary agreement

Current, from `gsd/test_gsdesign_benchmark.R` against `00052-fvn`, written to
`results/gsd_validation_results.csv`:

| | |
|---|---|
| Assertions | 32 (27 numerical, 5 rejection) |
| Numerical cases | 9 executed, 9 passed |
| Max deviation | **0.0001** z-score |
| Tolerance | 0.0005 |

26 of the 27 numerical boundaries reproduce gsDesign exactly at the 4 dp
gsDesign reports. One does not: **OF_4 look 3**, gsDesign 2.3590 against the
engine's 2.3591. The underlying value sits near 2.35905, so a difference far
below the tolerance flips the fourth decimal on rounding. It is recorded
rather than described as 0.0000, because a headline that says "exactly" while
one cell disagrees is the kind of small inaccuracy this document exists to
prevent. References are computed by gsDesign in R at run time, not
read from a stored copy.

### The 0.034 figure, and where it came from

Until 2026-07-30 the README badge read **0.034 z-score**. That number was
taken from `gsd/results/gsd_validation_results.csv` — a **stale duplicate**,
last written 2026-03-02, with the same filename as the live artifact but a
different path. Nothing regenerated it. The live file,
`results/gsd_validation_results.csv`, already showed 0.0.

The duplicate has been deleted. Two facts made it misleading rather than
merely old:

  * its headline came from the fifth look of a five-look design, and k=5 has
    since become a rejection case rather than a design; and

  * its k ≤ 4 figures (0.0117 at OF_4 look 4, 0.0033 at Pocock_4) predate the
    canonical drift-space solve, so they understated the engine.

### Construction identity

A tolerance only means something between two implementations of the *same*
design, so each family names the gsDesign call that defines it:

| Family | gsDesign construction |
|---|---|
| `OF_k` | Lan-DeMets O'Brien-Fleming, `sfu=sfLDOF` |
| `Pocock_k` | Classical Wang-Tsiatis Pocock, `sfu="Pocock"` |
| `OFparam_k` | Classical parametric O'Brien-Fleming, `sfu="OF"` |

All with `test.type=1`. The engine's non-binding futility does not move the
efficacy bounds, so efficacy-only references are the correct comparison.

## Discrimination controls

Passing is not evidence unless failing was possible.

`gsd/test_gsd_hsd_gamma.py` validates the user-selectable HSD shape parameter
γ, which shipped in revision `zetyra-backend-00043-8mt` (2026-07-29). With the
endpoint deliberately patched to accept γ and discard it, the suite falls to
**74/226**. That control is the point: boundary agreement alone cannot detect a
dropped γ, because the design still matches gsDesign — just at the wrong γ.

`gsd/test_gate_sabotage.py` is the equivalent control for the validation gate
itself.

## Revision history

| Revision | Date | Change |
|---|---|---|
| `00052-fvn` | 2026-08-20 | Production tip. Finalized-manuscript composed-pipeline artifact (45-cell fixture, constant-rate vs trend-active estimand split). GSD engine unchanged |
| `00051-47g` | 2026-08-10 | Stripe configuration moved to Secret Manager after a CI-built promotion dropped the image-baked `.env` |
| `00049-sgn` | 2026-08-10 | `min-instances=1`; cold starts were running 10-24s against Stripe's ~10s budget |
| `00048-jrr` | 2026-08-01 | Rule B blinded nuisance-parameter SSR |
| `00047-m92` | 2026-07-30 | First promotion of a CI-gated runtime artifact rather than a source deploy |
| `00046-szb` | 2026-07-29 | Previous provenance baseline |
| `00044-l8d` | 2026-07-29 | Protocol/doc hotfix |
| `00043-8mt` | 2026-07-29 | Selectable HSD γ. Defaults unchanged, so every design produced before it reproduces exactly — omitting γ is identical to the historical fixed −4, and `methodology_version` stays 2 |
| `00042-7dx` | 2026-07-28 | First with the corrected futility derivation (β spent under the alternative, solved jointly with sample size) and deterministic MVN integration |

CSVs in `results/` and `gsd/results/` are the artifacts of the latest run.

## Toolchain that produced this run

| | |
|---|---|
| R | 4.6.1 |
| gsDesign | 3.10.1 |
| httr / jsonlite | 1.4.8 / 2.0.0 |
| Validation repo | `2ef9d59` |

gsDesign 3.10.1 is the pinned reference and is what makes the comparison
meaningful; the R *runtime* differs from the CI gate host, which resolves R
4.3.3 via `r-lib/actions/setup-r`. Both produce the same boundaries, which is
the point of pinning gsDesign rather than R.
