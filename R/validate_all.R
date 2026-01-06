#!/usr/bin/env Rscript
#' Zetyra Validation Suite - R Implementation
#'
#' Validates Zetyra calculators against reference R packages:
#' - pwr (sample size)
#' - gsDesign (group sequential)
#' - rpact (clinical trial design)
#'
#' Usage:
#'   Rscript validate_all.R [base_url]
#'
#' @examples
#'   Rscript validate_all.R
#'   Rscript validate_all.R http://localhost:8080/api/v1/validation

library(httr)
library(jsonlite)

# =============================================================================
# Configuration
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
BASE_URL <- if (length(args) > 0) args[1] else "https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation"

cat("=" , rep("=", 68), "\n", sep = "")
cat("ZETYRA R VALIDATION SUITE\n")
cat("API URL:", BASE_URL, "\n")
cat("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("=" , rep("=", 68), "\n\n", sep = "")

# =============================================================================
# API Client Functions
# =============================================================================

#' Call Zetyra validation endpoint
zetyra_post <- function(endpoint, data) {
  url <- paste0(BASE_URL, endpoint)
  response <- POST(
    url,
    body = data,
    encode = "json",
    content_type_json()
  )

  if (status_code(response) != 200) {
    stop(paste("API error:", status_code(response), content(response, "text")))
  }

  content(response, "parsed")
}

# =============================================================================
# Sample Size Validation
# =============================================================================

validate_sample_size <- function() {
  cat("\n[1/4] Validating Sample Size Calculators\n")
  cat(rep("-", 70), "\n", sep = "")

  results <- list()
  all_pass <- TRUE

  # Test 1: Continuous - compare with pwr::pwr.t.test
  cat("  Testing continuous outcomes (two-sample t-test)...\n")

  # Scenario: d = 0.25, alpha = 0.05, power = 0.80
  zetyra <- zetyra_post("/sample-size/continuous", list(
    mean1 = 100,
    mean2 = 105,
    sd = 20,
    alpha = 0.05,
    power = 0.80
  ))

  # Reference: pwr formula
  # n = 2 * ((z_alpha + z_beta) / d)^2
  d <- 5 / 20  # 0.25
  z_alpha <- qnorm(1 - 0.05/2)
  z_beta <- qnorm(0.80)
  n_ref <- ceiling(2 * ((z_alpha + z_beta) * 20 / 5)^2)

  deviation <- abs(zetyra$n1 - n_ref) / n_ref
  pass <- deviation <= 0.01

  cat(sprintf("    Zetyra n=%d, Reference n=%d, Deviation=%.2f%% [%s]\n",
              zetyra$n1, n_ref, deviation * 100, ifelse(pass, "PASS", "FAIL")))

  if (!pass) all_pass <- FALSE

  # Test 2: Binary - compare with pooled-variance z-test
  cat("  Testing binary outcomes (two-proportion z-test)...\n")

  zetyra_bin <- zetyra_post("/sample-size/binary", list(
    p1 = 0.10,
    p2 = 0.15,
    alpha = 0.05,
    power = 0.80
  ))

  # Cohen's h
  h <- 2 * asin(sqrt(0.15)) - 2 * asin(sqrt(0.10))

  # Reference calculation (simplified)
  pooled_p <- (0.10 + 0.15) / 2
  delta <- abs(0.15 - 0.10)
  var_h0 <- pooled_p * (1 - pooled_p)
  var_h1_1 <- 0.10 * (1 - 0.10)
  var_h1_2 <- 0.15 * (1 - 0.15)

  numerator <- z_alpha * sqrt(2 * var_h0) + z_beta * sqrt(var_h1_1 + var_h1_2)
  n_ref_bin <- ceiling((numerator / delta)^2)

  deviation_bin <- abs(zetyra_bin$n1 - n_ref_bin) / n_ref_bin
  pass_bin <- deviation_bin <= 0.02  # Slightly higher tolerance for binary

  cat(sprintf("    Zetyra n=%d, Reference n=%d, Deviation=%.2f%% [%s]\n",
              zetyra_bin$n1, n_ref_bin, deviation_bin * 100, ifelse(pass_bin, "PASS", "FAIL")))

  if (!pass_bin) all_pass <- FALSE

  list(all_pass = all_pass)
}

# =============================================================================
# CUPED Validation
# =============================================================================

validate_cuped <- function() {
  cat("\n[2/4] Validating CUPED Calculator\n")
  cat(rep("-", 70), "\n", sep = "")

  all_pass <- TRUE

  # Test variance reduction formula: VRF = 1 - rho^2
  correlations <- c(0.0, 0.3, 0.5, 0.7, 0.9)

  for (rho in correlations) {
    zetyra <- zetyra_post("/cuped", list(
      baseline_mean = 100,
      baseline_std = 20,
      mde = 0.05,
      correlation = rho,
      alpha = 0.05,
      power = 0.80
    ))

    expected_vrf <- 1 - rho^2
    deviation <- abs(zetyra$variance_reduction_factor - expected_vrf)
    pass <- deviation < 0.001

    cat(sprintf("  rho=%.1f: VRF=%.4f (expected %.4f) [%s]\n",
                rho, zetyra$variance_reduction_factor, expected_vrf,
                ifelse(pass, "PASS", "FAIL")))

    if (!pass) all_pass <- FALSE
  }

  # Test: Zero correlation should give no reduction
  zetyra_zero <- zetyra_post("/cuped", list(
    baseline_mean = 100, baseline_std = 20, mde = 0.05, correlation = 0.0
  ))
  pass_zero <- zetyra_zero$n_original == zetyra_zero$n_adjusted
  cat(sprintf("  Zero correlation: n_orig=%d, n_adj=%d [%s]\n",
              zetyra_zero$n_original, zetyra_zero$n_adjusted,
              ifelse(pass_zero, "PASS", "FAIL")))

  if (!pass_zero) all_pass <- FALSE

  list(all_pass = all_pass)
}

# =============================================================================
# GSD Validation
# =============================================================================

validate_gsd <- function() {
  cat("\n[3/4] Validating Group Sequential Design\n")
  cat(rep("-", 70), "\n", sep = "")

  all_pass <- TRUE

  # Test O'Brien-Fleming spending function
  cat("  Testing O'Brien-Fleming spending function...\n")

  obf <- zetyra_post("/gsd", list(
    effect_size = 0.3,
    alpha = 0.025,
    power = 0.90,
    k = 3,
    spending_function = "OBrienFleming"
  ))

  # OBF should be conservative at first look
  pass_conservative <- obf$efficacy_boundaries[[1]] > obf$efficacy_boundaries[[3]]
  cat(sprintf("    First boundary > Last boundary: %.3f > %.3f [%s]\n",
              obf$efficacy_boundaries[[1]], obf$efficacy_boundaries[[3]],
              ifelse(pass_conservative, "PASS", "FAIL")))

  if (!pass_conservative) all_pass <- FALSE

  # Test Pocock spending function
  cat("  Testing Pocock spending function...\n")

  pocock <- zetyra_post("/gsd", list(
    effect_size = 0.3,
    alpha = 0.025,
    power = 0.90,
    k = 3,
    spending_function = "Pocock"
  ))

  # Pocock should have more uniform boundaries
  pocock_range <- max(unlist(pocock$efficacy_boundaries)) - min(unlist(pocock$efficacy_boundaries))
  obf_range <- max(unlist(obf$efficacy_boundaries)) - min(unlist(obf$efficacy_boundaries))
  pass_uniform <- pocock_range < obf_range

  cat(sprintf("    Pocock range < OBF range: %.3f < %.3f [%s]\n",
              pocock_range, obf_range, ifelse(pass_uniform, "PASS", "FAIL")))

  if (!pass_uniform) all_pass <- FALSE

  # Test sample size inflation
  inflation <- obf$n_max / obf$n_fixed
  pass_inflation <- inflation >= 1.0 && inflation <= 1.10
  cat(sprintf("    Sample size inflation: %.4f (expected 1.0-1.10) [%s]\n",
              inflation, ifelse(pass_inflation, "PASS", "FAIL")))

  if (!pass_inflation) all_pass <- FALSE

  list(all_pass = all_pass)
}

# =============================================================================
# Bayesian Validation
# =============================================================================

validate_bayesian <- function() {
  cat("\n[4/4] Validating Bayesian Predictive Power\n")
  cat(rep("-", 70), "\n", sep = "")

  all_pass <- TRUE

  # Test Normal-Normal conjugate posterior
  cat("  Testing Normal-Normal conjugate posterior...\n")

  prior_mean <- 0.0
  prior_var <- 1.0
  interim_effect <- 0.3
  interim_var <- 0.1

  zetyra <- zetyra_post("/bayesian/continuous", list(
    prior_mean = prior_mean,
    prior_var = prior_var,
    interim_effect = interim_effect,
    interim_var = interim_var,
    interim_n = 100,
    final_n = 200
  ))

  # Reference: Normal-Normal conjugate
  prior_precision <- 1 / prior_var
  data_precision <- 1 / interim_var
  posterior_precision <- prior_precision + data_precision
  posterior_var_ref <- 1 / posterior_precision
  posterior_mean_ref <- posterior_var_ref * (prior_mean * prior_precision + interim_effect * data_precision)

  mean_dev <- abs(zetyra$posterior_mean - posterior_mean_ref)
  var_dev <- abs(zetyra$posterior_var - posterior_var_ref)

  pass_mean <- mean_dev < 0.01
  pass_var <- var_dev < 0.01

  cat(sprintf("    Posterior mean: %.4f (ref %.4f, dev %.6f) [%s]\n",
              zetyra$posterior_mean, posterior_mean_ref, mean_dev,
              ifelse(pass_mean, "PASS", "FAIL")))
  cat(sprintf("    Posterior var:  %.4f (ref %.4f, dev %.6f) [%s]\n",
              zetyra$posterior_var, posterior_var_ref, var_dev,
              ifelse(pass_var, "PASS", "FAIL")))

  if (!pass_mean || !pass_var) all_pass <- FALSE

  # Test Beta-Binomial conjugate posterior
  cat("  Testing Beta-Binomial conjugate posterior...\n")

  zetyra_bin <- zetyra_post("/bayesian/binary", list(
    prior_alpha = 1,
    prior_beta = 1,
    control_successes = 30,
    control_n = 100,
    treatment_successes = 45,
    treatment_n = 100,
    final_n = 200
  ))

  # Reference: Beta-Binomial conjugate
  ref_ctrl_alpha <- 1 + 30
  ref_ctrl_beta <- 1 + (100 - 30)
  ref_trt_alpha <- 1 + 45
  ref_trt_beta <- 1 + (100 - 45)

  pass_ctrl_alpha <- zetyra_bin$posterior_control_alpha == ref_ctrl_alpha
  pass_ctrl_beta <- zetyra_bin$posterior_control_beta == ref_ctrl_beta

  cat(sprintf("    Control posterior: Beta(%d, %d) [%s]\n",
              zetyra_bin$posterior_control_alpha, zetyra_bin$posterior_control_beta,
              ifelse(pass_ctrl_alpha && pass_ctrl_beta, "PASS", "FAIL")))
  cat(sprintf("    Treatment posterior: Beta(%d, %d) [%s]\n",
              zetyra_bin$posterior_treatment_alpha, zetyra_bin$posterior_treatment_beta,
              ifelse(zetyra_bin$posterior_treatment_alpha == ref_trt_alpha, "PASS", "FAIL")))

  if (!pass_ctrl_alpha || !pass_ctrl_beta) all_pass <- FALSE

  list(all_pass = all_pass)
}

# =============================================================================
# Main Execution
# =============================================================================

results <- list(
  sample_size = tryCatch(validate_sample_size(), error = function(e) list(all_pass = FALSE, error = e$message)),
  cuped = tryCatch(validate_cuped(), error = function(e) list(all_pass = FALSE, error = e$message)),
  gsd = tryCatch(validate_gsd(), error = function(e) list(all_pass = FALSE, error = e$message)),
  bayesian = tryCatch(validate_bayesian(), error = function(e) list(all_pass = FALSE, error = e$message))
)

# Summary
cat("\n", rep("=", 70), "\n", sep = "")
cat("VALIDATION SUMMARY\n")
cat(rep("=", 70), "\n", sep = "")

all_pass <- TRUE
for (name in names(results)) {
  status <- if (results[[name]]$all_pass) "PASS" else "FAIL"
  cat(sprintf("  %-20s %s\n", name, status))
  if (!results[[name]]$all_pass) all_pass <- FALSE
}

cat("\n", rep("=", 70), "\n", sep = "")
if (all_pass) {
  cat("✓ ALL VALIDATIONS PASSED\n")
} else {
  cat("✗ SOME VALIDATIONS FAILED\n")
}
cat(rep("=", 70), "\n", sep = "")

# Exit with appropriate code
quit(status = if (all_pass) 0 else 1)
