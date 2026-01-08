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
test_configs <- list(
  list(name = "OF_2", k = 2, sfu = "OF", spending = "OBrienFleming"),
  list(name = "OF_3", k = 3, sfu = "OF", spending = "OBrienFleming"),
  list(name = "OF_4", k = 4, sfu = "OF", spending = "OBrienFleming"),
  list(name = "OF_5", k = 5, sfu = "OF", spending = "OBrienFleming"),
  list(name = "Pocock_2", k = 2, sfu = "Pocock", spending = "Pocock"),
  list(name = "Pocock_3", k = 3, sfu = "Pocock", spending = "Pocock"),
  list(name = "Pocock_4", k = 4, sfu = "Pocock", spending = "Pocock")
)

cat("Testing", length(test_configs), "design configurations...\n\n")

for (config in test_configs) {
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

output_dir <- file.path(dirname(sys.frame(1)$ofile), "results")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write.csv(results, file.path(output_dir, "gsd_validation_results.csv"), row.names = FALSE)
cat("Results saved to results/gsd_validation_results.csv\n\n")

# =============================================================================
# Summary
# =============================================================================

cat(rep("=", 70), "\n", sep = "")
cat(sprintf("SUMMARY: %d/%d boundary comparisons passed\n",
            sum(results$pass), nrow(results)))
if (all_pass) {
  cat("✓ ALL VALIDATIONS PASSED\n")
} else {
  cat("✗ SOME VALIDATIONS FAILED\n")
}
cat(rep("=", 70), "\n", sep = "")

quit(status = if (all_pass) 0 else 1)
