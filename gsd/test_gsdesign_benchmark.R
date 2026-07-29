#!/usr/bin/env Rscript
#' Validate Zetyra GSD boundaries against gsDesign R package
#'
#' This script compares Zetyra's Group Sequential Design boundaries
#' against the gold-standard gsDesign R package.
#'
#' Usage:
#'   Rscript test_gsdesign_benchmark.R [base_url]
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

cat("=" , rep("=", 68), "\n", sep = "")
cat("ZETYRA GSD vs gsDesign BENCHMARK\n")
cat("API URL:", BASE_URL, "\n")
cat("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("=" , rep("=", 68), "\n\n", sep = "")

# =============================================================================
# API Client
# =============================================================================

zetyra_gsd <- function(effect_size, alpha, power, k, spending_function, timing = NULL) {
  url <- paste0(BASE_URL, "/gsd")
  body <- list(
    effect_size = effect_size,
    alpha = alpha,
    power = power,
    k = k,
    spending_function = spending_function
  )
  if (!is.null(timing)) {
    body$timing <- timing
  }

  response <- POST(url, body = body, encode = "json", content_type_json())
  if (status_code(response) != 200) {
    stop(paste("API error:", status_code(response)))
  }
  content(response, "parsed")
}

# Raw variant: returns the response instead of halting. An above-ceiling case
# is EXPECTED to be refused, so a helper that stops on non-200 cannot assert
# it -- and stopping is exactly how this script printed two blank lines and
# read as a quiet success when the ceiling first dropped.
zetyra_gsd_raw <- function(effect_size, alpha, power, k, spending_function) {
  POST(paste0(BASE_URL, "/gsd"),
       body = list(effect_size = effect_size, alpha = alpha, power = power,
                   k = k, spending_function = spending_function),
       encode = "json", content_type_json())
}

# Numerical result fields. Any of these in a rejection body would mean a design
# was produced for an uncertified input.
NUMERICAL_FIELDS <- c("n_max", "n_fixed", "efficacy_boundaries",
                      "futility_boundaries", "alpha_spent", "beta_spent",
                      "information_fractions", "expected_n_h0", "expected_n_h1")

# Assert one above-ceiling case is refused, and refused for the right reason.
# Machine-readable fields only: `msg` is prose and may be reworded at any time.
check_rejection <- function(response, case_name) {
  if (status_code(response) != 422) {
    return(list(pass = FALSE, detail = sprintf(
      "expected HTTP 422, got %d -- uncertified k was not refused",
      status_code(response))))
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
    return(list(pass = FALSE, detail = paste(
      "rejected, but not on 'k' -- refusing for the wrong reason passes a",
      "naive status check while the real bound is unenforced")))
  }
  if (!identical(match$type, "less_than_equal")) {
    return(list(pass = FALSE, detail = sprintf(
      "expected error type 'less_than_equal', got '%s'", match$type)))
  }
  ceiling_val <- match$ctx$le
  if (is.null(ceiling_val) || ceiling_val != CERTIFIED_MAX_K) {
    return(list(pass = FALSE, detail = sprintf(
      "API advertises ceiling le=%s, suite expects %d",
      as.character(ceiling_val), CERTIFIED_MAX_K)))
  }
  leaked <- NUMERICAL_FIELDS[NUMERICAL_FIELDS %in% names(body)]
  if (length(leaked) > 0) {
    return(list(pass = FALSE, detail = paste(
      "rejection body carries numerical fields:", paste(leaked, collapse = ", "))))
  }
  list(pass = TRUE, detail = "")
}

# =============================================================================
# Benchmark Tests
# =============================================================================

BOUNDARY_TOLERANCE <- 0.05  # 0.05 z-score units

results <- data.frame(
  design = character(),
  look = integer(),
  gsdesign_z = numeric(),
  zetyra_z = numeric(),
  deviation = numeric(),
  pass = logical(),
  stringsAsFactors = FALSE
)

all_pass <- TRUE

# Test configurations matching gsd_reference_boundaries.csv
# Look counts above the certified ceiling are skipped, not deleted. The gsDesign
# reference values stay recorded so raising the ceiling restores coverage
# without regenerating anything. k <= 4 as of 2026-07-29: the worst k=5 design
# measured 69.75s against a 30s operational ceiling, so k=5 now returns 422 and
# a benchmark that does not skip it halts instead of reporting.
CERTIFIED_MAX_K <- 4

test_configs <- list(
  list(name = "OF_2", k = 2, sfu = sfLDOF, spending = "OBrienFleming"),
  list(name = "OF_3", k = 3, sfu = sfLDOF, spending = "OBrienFleming"),
  list(name = "OF_4", k = 4, sfu = sfLDOF, spending = "OBrienFleming"),
  list(name = "OF_5", k = 5, sfu = sfLDOF, spending = "OBrienFleming"),
  list(name = "Pocock_2", k = 2, sfu = "Pocock", spending = "Pocock"),
  list(name = "Pocock_3", k = 3, sfu = "Pocock", spending = "Pocock"),
  list(name = "Pocock_4", k = 4, sfu = "Pocock", spending = "Pocock"),
  # Classical (parametric) O'Brien-Fleming: sfu="OF" in gsDesign
  list(name = "OFparam_2", k = 2, sfu = "OF", spending = "OBrienFlemingParametric"),
  list(name = "OFparam_3", k = 3, sfu = "OF", spending = "OBrienFlemingParametric"),
  list(name = "OFparam_4", k = 4, sfu = "OF", spending = "OBrienFlemingParametric")
)

cat("Testing", length(test_configs), "design configurations...\n\n")

# Manifest, declared up front so the run is measured against an expectation
# rather than against whatever it happened to do.
expected_numerical_ids <- sapply(
  Filter(function(c) c$k <= CERTIFIED_MAX_K, test_configs), function(c) c$name)
expected_rejection_ids <- sapply(
  Filter(function(c) c$k > CERTIFIED_MAX_K, test_configs), function(c) c$name)
observed_numerical_ids <- character(0)
observed_rejection_ids <- character(0)

for (config in test_configs) {
  if (config$k > CERTIFIED_MAX_K) {
    # EXPECTED REJECTION. Emits the same per-look labels the case would have
    # emitted numerically, so the assertion count is unchanged and only the
    # meaning differs: each row now asserts the value is NOT produced.
    cat(sprintf("Testing %s (k=%d) -- EXPECTED REJECTION...\n",
                config$name, config$k))
    resp <- zetyra_gsd_raw(effect_size = 0.3, alpha = 0.025, power = 0.80,
                           k = config$k, spending_function = config$spending)
    verdict <- check_rejection(resp, config$name)
    observed_rejection_ids <- c(observed_rejection_ids, config$name)

    for (look in 1:config$k) {
      results <- rbind(results, data.frame(
        design = config$name, look = look,
        gsdesign_z = NA_real_, zetyra_z = NA_real_, deviation = NA_real_,
        pass = verdict$pass,
        note = if (verdict$pass)
          sprintf("rejected 422 less_than_equal on k, le=%d", CERTIFIED_MAX_K)
          else verdict$detail,
        stringsAsFactors = FALSE))
      cat(sprintf("  Look %d: NOT PRODUCED (uncertified k) [%s]\n",
                  look, if (verdict$pass) "PASS" else "FAIL"))
    }
    if (!verdict$pass) { all_pass <- FALSE; cat(sprintf("  -> %s\n", verdict$detail)) }
    cat("\n")
    next
  }
  observed_numerical_ids <- c(observed_numerical_ids, config$name)
  cat(sprintf("Testing %s (k=%d)...\n", config$name, config$k))

  # gsDesign reference
  gs <- gsDesign(
    k = config$k,
    alpha = 0.025,
    beta = 0.20,
    test.type = 1,  # One-sided
    sfu = config$sfu
  )
  gs_boundaries <- gs$upper$bound

  # Zetyra result
  zetyra <- zetyra_gsd(
    effect_size = 0.3,
    alpha = 0.025,
    power = 0.80,
    k = config$k,
    spending_function = config$spending
  )
  zetyra_boundaries <- unlist(zetyra$efficacy_boundaries)

  # Compare each boundary
  for (look in 1:config$k) {
    gs_z <- gs_boundaries[look]
    z_z <- zetyra_boundaries[look]
    deviation <- abs(gs_z - z_z)
    pass <- deviation < BOUNDARY_TOLERANCE

    results <- rbind(results, data.frame(
      design = config$name,
      look = look,
      gsdesign_z = round(gs_z, 4),
      zetyra_z = round(z_z, 4),
      deviation = round(deviation, 4),
      pass = pass,
      note = "",
      stringsAsFactors = FALSE
    ))

    if (!pass) all_pass <- FALSE

    cat(sprintf("  Look %d: gsDesign=%.4f, Zetyra=%.4f, dev=%.4f [%s]\n",
                look, gs_z, z_z, deviation, ifelse(pass, "PASS", "FAIL")))
  }
  cat("\n")
}

# =============================================================================
# Save Results
# =============================================================================

output_dir <- "results"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write.csv(results, file.path(output_dir, "gsd_validation_results.csv"), row.names = FALSE)
cat("Results saved to results/gsd_validation_results.csv\n\n")

# =============================================================================
# Summary
# =============================================================================

# MANIFEST INTEGRITY. Compared as exact ID SETS, not counts: a count cannot
# distinguish a duplicated case from an omitted one, since dropping one and
# duplicating another leaves the total unchanged.
integrity <- character(0)
if (length(unique(expected_numerical_ids)) != length(expected_numerical_ids) ||
    length(unique(expected_rejection_ids)) != length(expected_rejection_ids)) {
  integrity <- c(integrity, "manifest contains duplicate case IDs")
}
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
  integrity <- c(integrity, paste(
    "NO numerical case ran. A suite that asserts only rejections proves the",
    "API refuses work, not that it computes correctly."))
}
if (length(integrity) > 0) all_pass <- FALSE

n_rejection_rows <- sum(!is.na(results$note) & nzchar(results$note))
cat(rep("=", 70), "\n", sep = "")
cat("GATE SUMMARY -- gsdesign-benchmark\n")
cat("SUITE-ID: gsdesign-benchmark\n")
cat(sprintf("  manifest numerical cases   : %d\n", length(expected_numerical_ids)))
cat(sprintf("  numerical executed/passed  : %d/%d\n",
            length(observed_numerical_ids),
            length(observed_numerical_ids) - length(unique(
              results$design[!results$pass & !nzchar(results$note)]))))
cat(sprintf("  manifest rejection cases   : %d\n", length(expected_rejection_ids)))
cat(sprintf("  rejections executed/passed : %d/%d\n",
            length(observed_rejection_ids),
            length(observed_rejection_ids) - length(unique(
              results$design[!results$pass & nzchar(results$note)]))))
cat(sprintf("  unexpected skips           : 0\n"))
cat(sprintf("  unexpected HTTP errors     : 0\n"))
cat(sprintf("  total failures             : %d\n", sum(!results$pass)))
if (length(integrity) > 0) {
  cat("\nMANIFEST / INTEGRITY PROBLEMS\n")
  for (p in integrity) cat(sprintf("  - %s\n", p))
}
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("SUMMARY: %d/%d assertions passed (%d numerical, %d rejection)\n",
            sum(results$pass), nrow(results),
            nrow(results) - n_rejection_rows, n_rejection_rows))
if (all_pass) {
  cat("✓ ALL VALIDATIONS PASSED\n")
} else {
  cat("✗ SOME VALIDATIONS FAILED\n")
}
cat(rep("=", 70), "\n", sep = "")

quit(status = if (all_pass) 0 else 1)
