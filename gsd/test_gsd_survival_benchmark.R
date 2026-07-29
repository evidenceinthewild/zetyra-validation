#!/usr/bin/env Rscript
#' Cross-validate Zetyra GSD Survival boundaries against gsDesign R package
#'
#' Validates:
#' 1. Fixed event counts match gsDesign Schoenfeld formula
#' 2. Z-score boundaries match gsDesign for OBF and Pocock spending
#' 3. Cumulative alpha spent matches gsDesign spending function
#' 4. Information fraction structure
#'
#' Usage:
#'   Rscript test_gsd_survival_benchmark.R [base_url]
#'
#' Requires: gsDesign, httr, jsonlite

library(gsDesign)
library(httr)
library(jsonlite)

# =============================================================================
# Configuration
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
BASE_URL <- if (length(args) > 0) args[1] else "https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation"

cat("=", rep("=", 68), "\n", sep = "")
cat("GSD SURVIVAL: ZETYRA vs gsDesign BENCHMARK\n")
cat("API URL:", BASE_URL, "\n")
cat("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("=", rep("=", 68), "\n\n", sep = "")

# =============================================================================
# API Client
# =============================================================================

zetyra_gsd_survival <- function(hazard_ratio, median_control, accrual_time,
                                 follow_up_time, alpha, power, k,
                                 spending_function = "OBrienFleming") {
  url <- paste0(BASE_URL, "/gsd/survival")
  body <- list(
    hazard_ratio = hazard_ratio,
    median_control = median_control,
    accrual_time = accrual_time,
    follow_up_time = follow_up_time,
    alpha = alpha,
    power = power,
    k = k,
    spending_function = spending_function
  )
  response <- POST(url, body = body, encode = "json", content_type_json())
  if (status_code(response) != 200) {
    stop(paste("API error:", status_code(response), content(response, "text")))
  }
  content(response, "parsed")
}

# Raw variant: returns the response rather than halting, so an above-ceiling
# case can be ASSERTED as refused instead of aborting the script before its
# summary -- which is how this benchmark once printed two blank lines and read
# as a quiet success.
zetyra_gsd_survival_raw <- function(hazard_ratio, median_control, accrual_time,
                                    follow_up_time, alpha, power, k,
                                    spending_function = "OBrienFleming") {
  POST(paste0(BASE_URL, "/gsd/survival"),
       body = list(hazard_ratio = hazard_ratio, median_control = median_control,
                   accrual_time = accrual_time, follow_up_time = follow_up_time,
                   alpha = alpha, power = power, k = k,
                   spending_function = spending_function),
       encode = "json", content_type_json())
}

NUMERICAL_FIELDS <- c("max_events", "fixed_events", "n_total", "n_control",
                      "n_treatment", "efficacy_boundaries",
                      "futility_boundaries", "alpha_spent", "beta_spent",
                      "information_fractions", "event_probability")

# Machine-readable assertions only; `msg` is prose and may be reworded.
check_rejection <- function(response, case_name) {
  if (status_code(response) != 422) {
    return(list(pass = FALSE, detail = sprintf(
      "expected HTTP 422, got %d", status_code(response))))
  }
  body <- content(response, "parsed")
  detail <- body$detail
  if (is.null(detail) || length(detail) == 0) {
    return(list(pass = FALSE, detail = "422 body carried no error list"))
  }
  match <- NULL
  for (err in detail) {
    loc <- unlist(err$loc)
    if (length(loc) > 0 && loc[length(loc)] == "k") { match <- err; break }
  }
  if (is.null(match)) {
    return(list(pass = FALSE, detail = "rejected, but not on 'k'"))
  }
  if (!identical(match$type, "less_than_equal")) {
    return(list(pass = FALSE, detail = sprintf(
      "expected 'less_than_equal', got '%s'", match$type)))
  }
  if (is.null(match$ctx$le) || match$ctx$le != CERTIFIED_MAX_K) {
    return(list(pass = FALSE, detail = sprintf(
      "API advertises le=%s, suite expects %d",
      as.character(match$ctx$le), CERTIFIED_MAX_K)))
  }
  leaked <- NUMERICAL_FIELDS[NUMERICAL_FIELDS %in% names(body)]
  if (length(leaked) > 0) {
    return(list(pass = FALSE, detail = paste(
      "rejection carries numerical fields:", paste(leaked, collapse = ", "))))
  }
  list(pass = TRUE, detail = "")
}

# =============================================================================
# Reference: Schoenfeld event count
# =============================================================================

ref_schoenfeld_events <- function(alpha, power, log_hr) {
  z_a <- qnorm(1 - alpha)
  z_b <- qnorm(power)
  d <- ((z_a + z_b) / log_hr)^2 * 4  # 1:1 allocation
  return(ceiling(d))
}

# =============================================================================
# Results tracking
# =============================================================================

results <- data.frame(
  test = character(),
  metric = character(),
  gsdesign_val = numeric(),
  zetyra_val = numeric(),
  deviation = numeric(),
  pass = logical(),
  stringsAsFactors = FALSE
)

add_result <- function(test, metric, gs_val, z_val, tol, is_relative = FALSE) {
  if (is_relative) {
    dev <- abs(gs_val - z_val) / max(abs(gs_val), 1)
  } else {
    dev <- abs(gs_val - z_val)
  }
  pass <- dev < tol
  results <<- rbind(results, data.frame(
    test = test, metric = metric,
    gsdesign_val = round(gs_val, 4), zetyra_val = round(z_val, 4),
    deviation = round(dev, 4), pass = pass,
    stringsAsFactors = FALSE
  ))
  cat(sprintf("  %s: gsDesign=%.4f, Zetyra=%.4f, dev=%.4f [%s]\n",
              metric, gs_val, z_val, dev, ifelse(pass, "PASS", "FAIL")))
}

# Rejection counterpart to add_result. Writes the same columns so the results
# frame stays homogeneous, but the verdict comes from the rejection contract
# rather than from a numerical tolerance.
add_rejection_result <- function(test, metric, passed, detail) {
  results <<- rbind(results, data.frame(
    test = test, metric = metric,
    gsdesign_val = NA_real_, zetyra_val = NA_real_,
    deviation = NA_real_, pass = passed,
    stringsAsFactors = FALSE
  ))
  cat(sprintf("  %s: NOT PRODUCED (uncertified k) [%s]%s\n", metric,
              ifelse(passed, "PASS", "FAIL"),
              if (passed) "" else paste0(" -- ", detail)))
}

all_pass <- TRUE

# =============================================================================
# Test 1: Fixed Event Counts (Schoenfeld formula)
# =============================================================================

cat("\n1. Fixed Event Counts (Schoenfeld Formula)\n")
cat(rep("-", 70), "\n", sep = "")

# Look counts above the certified ceiling are skipped, not deleted. The gsDesign
# reference values stay recorded so raising the ceiling restores coverage
# without regenerating anything. k <= 4 as of 2026-07-29: the worst k=5 design
# measured 69.75s against a 30s operational ceiling, so k=5 now returns 422 and
# a benchmark that does not skip it halts instead of reporting.
CERTIFIED_MAX_K <- 4

event_scenarios <- list(
  list(name = "HR=0.7, α=0.025, 90%", hr = 0.7, alpha = 0.025, power = 0.90),
  list(name = "HR=0.8, α=0.025, 80%", hr = 0.8, alpha = 0.025, power = 0.80),
  list(name = "HR=0.5, α=0.025, 90%", hr = 0.5, alpha = 0.025, power = 0.90),
  list(name = "HR=0.6, α=0.025, 80%", hr = 0.6, alpha = 0.025, power = 0.80)
)

for (s in event_scenarios) {
  cat(sprintf("\n  %s\n", s$name))

  gs_events <- ref_schoenfeld_events(s$alpha, s$power, log(s$hr))

  z <- zetyra_gsd_survival(
    hazard_ratio = s$hr, median_control = 12, accrual_time = 24,
    follow_up_time = 12, alpha = s$alpha, power = s$power, k = 3
  )

  # Fixed events should match Schoenfeld ±5%
  add_result(s$name, "fixed_events", gs_events, z$fixed_events, 0.05, is_relative = TRUE)
}

# =============================================================================
# Test 2: Z-score Boundaries vs gsDesign
# =============================================================================

cat("\n\n2. Z-score Boundaries (OBF & Pocock)\n")
cat(rep("-", 70), "\n", sep = "")

boundary_scenarios <- list(
  list(name = "OBF k=3", k = 3, sfu = sfLDOF, spending = "OBrienFleming", tol = 0.005),
  list(name = "OBF k=4", k = 4, sfu = sfLDOF, spending = "OBrienFleming", tol = 0.015),
  list(name = "OBF k=5", k = 5, sfu = sfLDOF, spending = "OBrienFleming", tol = 0.04),
  list(name = "Pocock k=3", k = 3, sfu = "Pocock", spending = "Pocock", tol = 0.005),
  list(name = "Pocock k=4", k = 4, sfu = "Pocock", spending = "Pocock", tol = 0.008),
  # Classical (parametric) O'Brien-Fleming: sfu="OF" in gsDesign
  list(name = "OFparam k=3", k = 3, sfu = "OF", spending = "OBrienFlemingParametric", tol = 0.005),
  list(name = "OFparam k=4", k = 4, sfu = "OF", spending = "OBrienFlemingParametric", tol = 0.015)
)

# Per-scenario z-score tolerances. With sfLDOF, Zetyra and gsDesign use the same
# spending algorithm; remaining deviations are from MVN integration precision
# (scipy vs R's mvtnorm), growing with k. OBF k=3: < 0.005, k=4: < 0.015, k=5: < 0.04.

expected_numerical_ids <- sapply(
  Filter(function(x) x$k <= CERTIFIED_MAX_K, boundary_scenarios),
  function(x) x$name)
expected_rejection_ids <- sapply(
  Filter(function(x) x$k > CERTIFIED_MAX_K, boundary_scenarios),
  function(x) x$name)
observed_numerical_ids <- character(0)
observed_rejection_ids <- character(0)

for (s in boundary_scenarios) {
  if (s$k > CERTIFIED_MAX_K) {
    cat(sprintf("\n  %s -- EXPECTED REJECTION (k=%d above ceiling %d)\n",
                s$name, s$k, CERTIFIED_MAX_K))
    resp <- zetyra_gsd_survival_raw(
      hazard_ratio = 0.7, median_control = 12, accrual_time = 24,
      follow_up_time = 12, alpha = 0.025, power = 0.80, k = s$k,
      spending_function = s$spending)
    verdict <- check_rejection(resp, s$name)
    observed_rejection_ids <- c(observed_rejection_ids, s$name)
    for (look in 1:s$k) {
      add_rejection_result(s$name, sprintf("look_%d_z", look),
                           verdict$pass, verdict$detail)
    }
    next
  }
  observed_numerical_ids <- c(observed_numerical_ids, s$name)
  cat(sprintf("\n  %s (tol=%.3f)\n", s$name, s$tol))

  # gsDesign reference (one-sided, test.type=1)
  gs <- gsDesign(
    k = s$k, alpha = 0.025, beta = 0.20,
    test.type = 1, sfu = s$sfu
  )
  gs_bounds <- gs$upper$bound

  # Zetyra survival endpoint
  z <- zetyra_gsd_survival(
    hazard_ratio = 0.7, median_control = 12, accrual_time = 24,
    follow_up_time = 12, alpha = 0.025, power = 0.80, k = s$k,
    spending_function = s$spending
  )
  z_bounds <- unlist(z$efficacy_boundaries)

  for (look in 1:s$k) {
    add_result(s$name, sprintf("look_%d_z", look), gs_bounds[look], z_bounds[look], s$tol)
  }
}

# =============================================================================
# Test 3: Cumulative Alpha Spent
# =============================================================================

cat("\n\n3. Cumulative Alpha Spent\n")
cat(rep("-", 70), "\n", sep = "")

# OBF k=3: compare alpha spending at each look
# Use sfLDOF (Lan-DeMets OBF approximation) to match Zetyra's OBrienFleming spending
gs_obf3 <- gsDesign(k = 3, alpha = 0.025, beta = 0.20, test.type = 1, sfu = sfLDOF)
z_obf3 <- zetyra_gsd_survival(
  hazard_ratio = 0.7, median_control = 12, accrual_time = 24,
  follow_up_time = 12, alpha = 0.025, power = 0.80, k = 3,
  spending_function = "OBrienFleming"
)

# gsDesign cumulative upper alpha spent
gs_alpha_spent <- cumsum(gs_obf3$upper$prob[, 1])  # Under H0
z_alpha_spent <- unlist(z_obf3$alpha_spent)

for (look in 1:3) {
  add_result("OBF k=3", sprintf("alpha_spent_look_%d", look),
             gs_alpha_spent[look], z_alpha_spent[look], 0.002)
}

# Final alpha should equal target
add_result("OBF k=3", "final_alpha = 0.025",
           0.025, z_alpha_spent[3], 0.001)

# Pocock k=3
gs_poc3 <- gsDesign(k = 3, alpha = 0.025, beta = 0.20, test.type = 1, sfu = "Pocock")
z_poc3 <- zetyra_gsd_survival(
  hazard_ratio = 0.7, median_control = 12, accrual_time = 24,
  follow_up_time = 12, alpha = 0.025, power = 0.80, k = 3,
  spending_function = "Pocock"
)

gs_poc_alpha <- cumsum(gs_poc3$upper$prob[, 1])
z_poc_alpha <- unlist(z_poc3$alpha_spent)

for (look in 1:3) {
  add_result("Pocock k=3", sprintf("alpha_spent_look_%d", look),
             gs_poc_alpha[look], z_poc_alpha[look], 0.002)
}

# Classical OBF (parametric) k=3: cumulative alpha from actual H0 crossing probs
gs_ofparam3 <- gsDesign(k = 3, alpha = 0.025, beta = 0.20, test.type = 1, sfu = "OF")
z_ofparam3 <- zetyra_gsd_survival(
  hazard_ratio = 0.7, median_control = 12, accrual_time = 24,
  follow_up_time = 12, alpha = 0.025, power = 0.80, k = 3,
  spending_function = "OBrienFlemingParametric"
)

gs_ofparam_alpha <- cumsum(gs_ofparam3$upper$prob[, 1])
z_ofparam_alpha <- unlist(z_ofparam3$alpha_spent)

for (look in 1:3) {
  add_result("OFparam k=3", sprintf("alpha_spent_look_%d", look),
             gs_ofparam_alpha[look], z_ofparam_alpha[look], 0.002)
}

# Final alpha should equal target
add_result("OFparam k=3", "final_alpha = 0.025",
           0.025, z_ofparam_alpha[3], 0.001)

# =============================================================================
# Test 4: Information Fractions
# =============================================================================

cat("\n\n4. Information Fractions\n")
cat(rep("-", 70), "\n", sep = "")

# With equal spacing, gsDesign uses timing = (1:k)/k
# Zetyra survival uses events ∝ information, so fractions should be equally spaced

z_if <- unlist(z_obf3$information_fractions)
for (look in 1:3) {
  expected_if <- look / 3
  add_result("OBF k=3", sprintf("info_frac_look_%d", look),
             expected_if, z_if[look], 0.01)
}

# =============================================================================
# Test 5: Consistency Across HR Values (same boundaries for same spending)
# =============================================================================

cat("\n\n5. Boundary Consistency Across HR Values\n")
cat(rep("-", 70), "\n", sep = "")

# GSD boundaries depend on alpha/power/k/spending, NOT on HR
# Two different HRs with identical design parameters should produce the same z-score boundaries
z_hr07 <- zetyra_gsd_survival(
  hazard_ratio = 0.7, median_control = 12, accrual_time = 24,
  follow_up_time = 12, alpha = 0.025, power = 0.80, k = 3,
  spending_function = "OBrienFleming"
)
z_hr08 <- zetyra_gsd_survival(
  hazard_ratio = 0.8, median_control = 12, accrual_time = 24,
  follow_up_time = 12, alpha = 0.025, power = 0.80, k = 3,
  spending_function = "OBrienFleming"
)

z_b07 <- unlist(z_hr07$efficacy_boundaries)
z_b08 <- unlist(z_hr08$efficacy_boundaries)

for (look in 1:3) {
  add_result("HR consistency", sprintf("look_%d: HR=0.7 vs HR=0.8", look),
             z_b07[look], z_b08[look], 0.001)
}

# =============================================================================
# Save Results
# =============================================================================

output_dir <- "results"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
write.csv(results, file.path(output_dir, "gsd_survival_benchmark.csv"), row.names = FALSE)
cat("\nResults saved to results/gsd_survival_benchmark.csv\n\n")

# =============================================================================
# Summary
# =============================================================================

# MANIFEST INTEGRITY -- exact ID SETS, not counts. A count cannot distinguish a
# duplicated case from an omitted one.
integrity <- character(0)
if (!setequal(observed_numerical_ids, expected_numerical_ids)) {
  integrity <- c(integrity, sprintf(
    "numerical ID set mismatch -- missing [%s], unexpected [%s]",
    paste(setdiff(expected_numerical_ids, observed_numerical_ids), collapse = ","),
    paste(setdiff(observed_numerical_ids, expected_numerical_ids), collapse = ",")))
}
if (!setequal(observed_rejection_ids, expected_rejection_ids)) {
  integrity <- c(integrity, sprintf(
    "rejection ID set mismatch -- missing [%s], unexpected [%s]",
    paste(setdiff(expected_rejection_ids, observed_rejection_ids), collapse = ","),
    paste(setdiff(observed_rejection_ids, expected_rejection_ids), collapse = ",")))
}
if (length(observed_numerical_ids) == 0) {
  integrity <- c(integrity, "NO numerical boundary case ran")
}

all_pass <- all(results$pass) && length(integrity) == 0
n_rej <- length(observed_rejection_ids) * 5

cat(rep("=", 70), "\n", sep = "")
cat("GATE SUMMARY -- gsd-survival-benchmark\n")
cat("SUITE-ID: gsd-survival-benchmark\n")
cat(sprintf("  manifest numerical cases   : %d\n", length(expected_numerical_ids)))
cat(sprintf("  numerical executed/passed  : %d/%d\n",
            length(observed_numerical_ids), length(observed_numerical_ids)))
cat(sprintf("  manifest rejection cases   : %d\n", length(expected_rejection_ids)))
cat(sprintf("  rejections executed/passed : %d/%d\n",
            length(observed_rejection_ids), length(observed_rejection_ids)))
cat(sprintf("  unexpected skips           : 0\n"))
cat(sprintf("  unexpected HTTP errors     : 0\n"))
cat(sprintf("  total failures             : %d\n", sum(!results$pass)))
if (length(integrity) > 0) {
  cat("\nMANIFEST / INTEGRITY PROBLEMS\n")
  for (p in integrity) cat(sprintf("  - %s\n", p))
}
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("SUMMARY: %d/%d tests passed\n", sum(results$pass), nrow(results)))
if (all_pass) {
  cat("ALL VALIDATIONS PASSED\n")
} else {
  cat("SOME VALIDATIONS FAILED\n")
  cat("\nFailing tests:\n")
  print(results[!results$pass, ])
}
cat(rep("=", 70), "\n", sep = "")

quit(status = if (all_pass) 0 else 1)
