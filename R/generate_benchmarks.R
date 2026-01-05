#!/usr/bin/env Rscript
#' Generate Reference Benchmarks from R Packages
#'
#' Creates CSV files with reference values from:
#' - pwr (sample size)
#' - gsDesign (group sequential)
#'
#' These benchmarks are used to validate Zetyra results.
#'
#' Usage:
#'   Rscript generate_benchmarks.R

library(pwr)

# Check for optional packages
has_gsDesign <- require(gsDesign, quietly = TRUE)

cat("=" , rep("=", 68), "\n", sep = "")
cat("GENERATING REFERENCE BENCHMARKS\n")
cat("=" , rep("=", 68), "\n\n", sep = "")

# =============================================================================
# Sample Size Benchmarks (pwr package)
# =============================================================================

cat("Generating sample size benchmarks...\n")

sample_size_benchmarks <- data.frame(
  test_type = character(),
  effect_size = numeric(),
  alpha = numeric(),
  power = numeric(),
  n_per_group = numeric(),
  package = character(),
  stringsAsFactors = FALSE
)

# T-test benchmarks
for (d in c(0.2, 0.3, 0.5, 0.8)) {
  for (power in c(0.80, 0.90)) {
    result <- pwr.t.test(d = d, sig.level = 0.05, power = power, type = "two.sample")
    sample_size_benchmarks <- rbind(sample_size_benchmarks, data.frame(
      test_type = "two_sample_t",
      effect_size = d,
      alpha = 0.05,
      power = power,
      n_per_group = ceiling(result$n),
      package = "pwr"
    ))
  }
}

# Two-proportion benchmarks
for (h in c(0.2, 0.3, 0.5)) {
  for (power in c(0.80, 0.90)) {
    result <- pwr.2p.test(h = h, sig.level = 0.05, power = power)
    sample_size_benchmarks <- rbind(sample_size_benchmarks, data.frame(
      test_type = "two_proportion",
      effect_size = h,
      alpha = 0.05,
      power = power,
      n_per_group = ceiling(result$n),
      package = "pwr"
    ))
  }
}

write.csv(sample_size_benchmarks, "../data/sample_size_benchmarks.csv", row.names = FALSE)
cat(sprintf("  Saved %d sample size benchmarks\n", nrow(sample_size_benchmarks)))

# =============================================================================
# GSD Benchmarks (gsDesign package)
# =============================================================================

if (has_gsDesign) {
  cat("Generating GSD benchmarks...\n")

  gsd_benchmarks <- data.frame(
    k = integer(),
    spending_function = character(),
    alpha = numeric(),
    power = numeric(),
    look = integer(),
    info_frac = numeric(),
    efficacy_z = numeric(),
    futility_z = numeric(),
    cumulative_alpha = numeric(),
    package = character(),
    stringsAsFactors = FALSE
  )

  # O'Brien-Fleming
  for (k in c(2, 3, 4, 5)) {
    design <- gsDesign(k = k, test.type = 2, alpha = 0.025, beta = 0.10,
                       sfu = sfLDOF, sfl = sfLDPocock)

    for (i in 1:k) {
      gsd_benchmarks <- rbind(gsd_benchmarks, data.frame(
        k = k,
        spending_function = "OBrienFleming",
        alpha = 0.025,
        power = 0.90,
        look = i,
        info_frac = design$timing[i],
        efficacy_z = design$upper$bound[i],
        futility_z = design$lower$bound[i],
        cumulative_alpha = cumsum(design$upper$prob[, 1])[i],
        package = "gsDesign"
      ))
    }
  }

  # Pocock
  for (k in c(2, 3, 4, 5)) {
    design <- gsDesign(k = k, test.type = 2, alpha = 0.025, beta = 0.10,
                       sfu = sfLDPocock, sfl = sfLDPocock)

    for (i in 1:k) {
      gsd_benchmarks <- rbind(gsd_benchmarks, data.frame(
        k = k,
        spending_function = "Pocock",
        alpha = 0.025,
        power = 0.90,
        look = i,
        info_frac = design$timing[i],
        efficacy_z = design$upper$bound[i],
        futility_z = design$lower$bound[i],
        cumulative_alpha = cumsum(design$upper$prob[, 1])[i],
        package = "gsDesign"
      ))
    }
  }

  write.csv(gsd_benchmarks, "../data/gsd_benchmarks.csv", row.names = FALSE)
  cat(sprintf("  Saved %d GSD benchmarks\n", nrow(gsd_benchmarks)))
} else {
  cat("  Skipping GSD benchmarks (gsDesign not installed)\n")
}

# =============================================================================
# CUPED Reference Values
# =============================================================================

cat("Generating CUPED reference values...\n")

cuped_benchmarks <- data.frame(
  baseline_mean = numeric(),
  baseline_std = numeric(),
  mde = numeric(),
  correlation = numeric(),
  alpha = numeric(),
  power = numeric(),
  variance_reduction_factor = numeric(),
  n_original = numeric(),
  n_adjusted = numeric(),
  stringsAsFactors = FALSE
)

for (rho in c(0.0, 0.3, 0.5, 0.7, 0.9)) {
  # Calculate using formulas
  baseline_mean <- 100
  baseline_std <- 20
  mde <- 0.05
  alpha <- 0.05
  power <- 0.80

  delta <- baseline_mean * mde
  z_alpha <- qnorm(1 - alpha/2)
  z_beta <- qnorm(power)

  n_original <- ceiling(2 * ((z_alpha + z_beta) * baseline_std / delta)^2)

  vrf <- 1 - rho^2
  adjusted_std <- baseline_std * sqrt(vrf)
  n_adjusted <- ceiling(2 * ((z_alpha + z_beta) * adjusted_std / delta)^2)

  cuped_benchmarks <- rbind(cuped_benchmarks, data.frame(
    baseline_mean = baseline_mean,
    baseline_std = baseline_std,
    mde = mde,
    correlation = rho,
    alpha = alpha,
    power = power,
    variance_reduction_factor = vrf,
    n_original = n_original,
    n_adjusted = n_adjusted
  ))
}

write.csv(cuped_benchmarks, "../data/cuped_benchmarks.csv", row.names = FALSE)
cat(sprintf("  Saved %d CUPED benchmarks\n", nrow(cuped_benchmarks)))

# =============================================================================
# Summary
# =============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("BENCHMARKS GENERATED\n")
cat(rep("=", 70), "\n", sep = "")
cat("Files saved to ../data/\n")
cat("  - sample_size_benchmarks.csv\n")
if (has_gsDesign) cat("  - gsd_benchmarks.csv\n")
cat("  - cuped_benchmarks.csv\n")
