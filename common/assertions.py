"""
Shared assertion helpers for Zetyra validation suite.

- binomial_ci: Clopper-Pearson exact interval for MC rate estimates
- mc_rate_within: Check if target rate is consistent with observed MC rate
- assert_schema: Validate response keys and types against contracts
"""

from scipy import stats


def binomial_ci(k: int, n: int, confidence: float = 0.99) -> tuple:
    """
    Clopper-Pearson exact confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    confidence : float
        Confidence level (default 0.99 for stringent MC checks).

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper bounds of the interval.
    """
    alpha = 1 - confidence
    if k == 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def mc_rate_within(observed_rate: float, n_sims: int, target: float,
                   tolerance: float = 0.0, confidence: float = 0.99) -> bool:
    """
    Check if a target rate is consistent with an MC-estimated rate.

    Uses Clopper-Pearson CI around the observed rate, then checks whether
    `target` falls within [lo - tolerance, hi + tolerance].

    Parameters
    ----------
    observed_rate : float
        MC-estimated rate (e.g., type I error or power).
    n_sims : int
        Number of MC simulations used.
    target : float
        Target rate to check against.
    tolerance : float
        Extra slack beyond the CI bounds.
    confidence : float
        CI confidence level.
    """
    k = round(observed_rate * n_sims)
    k = max(0, min(k, n_sims))
    lo, hi = binomial_ci(k, n_sims, confidence)
    return (lo - tolerance) <= target <= (hi + tolerance)


def mc_rate_upper_bound(observed_rate: float, n_sims: int,
                        confidence: float = 0.99) -> float:
    """
    Upper bound of Clopper-Pearson CI for an MC-estimated rate.

    Useful for type I error checks: assert upper_bound <= max_allowed.
    """
    k = round(observed_rate * n_sims)
    k = max(0, min(k, n_sims))
    _, hi = binomial_ci(k, n_sims, confidence)
    return hi


def mc_rate_lower_bound(observed_rate: float, n_sims: int,
                        confidence: float = 0.99) -> float:
    """
    Lower bound of Clopper-Pearson CI for an MC-estimated rate.

    Useful for power checks: assert lower_bound >= min_required.
    """
    k = round(observed_rate * n_sims)
    k = max(0, min(k, n_sims))
    lo, _ = binomial_ci(k, n_sims, confidence)
    return lo


# ─── Schema contracts ─────────────────────────────────────────────────

SCHEMA_CONTRACTS = {
    "bayesian_binary": {
        "required": [
            "predictive_probability", "posterior_control_alpha",
            "posterior_control_beta", "posterior_treatment_alpha",
            "posterior_treatment_beta", "posterior_control_mean",
            "posterior_treatment_mean", "recommendation", "inputs",
        ],
        "types": {
            "predictive_probability": (int, float),
            "posterior_control_alpha": (int, float),
            "posterior_control_beta": (int, float),
            "posterior_treatment_alpha": (int, float),
            "posterior_treatment_beta": (int, float),
            "recommendation": str,
        },
        "bounds": {
            "predictive_probability": (0.0, 1.0, False),
            "posterior_control_alpha": (1e-10, None),
            "posterior_control_beta": (1e-10, None),
            "posterior_treatment_alpha": (1e-10, None),
            "posterior_treatment_beta": (1e-10, None),
        },
    },
    "bayesian_continuous": {
        "required": [
            "predictive_probability", "posterior_mean", "posterior_var",
            "credible_interval_lower", "credible_interval_upper",
            "recommendation", "inputs",
        ],
        "types": {
            "predictive_probability": (int, float),
            "posterior_mean": (int, float),
            "posterior_var": (int, float),
            "recommendation": str,
        },
        "bounds": {
            "predictive_probability": (0.0, 1.0, False),
            "posterior_var": (1e-10, None),
        },
        "enums": {
            "recommendation": ["stop_for_efficacy", "continue", "stop_for_futility"],
        },
        "ordering": [
            ("credible_interval_lower", "credible_interval_upper"),
        ],
    },
    "prior_elicitation": {
        "required": ["alpha", "beta", "mean", "variance", "ess", "quantiles", "inputs"],
        "types": {
            "alpha": (int, float),
            "beta": (int, float),
            "mean": (int, float),
            "variance": (int, float),
            "ess": (int, float),
            "quantiles": dict,
        },
        "bounds": {
            "alpha": (1e-10, None),
            "beta": (1e-10, None),
            "mean": (0.0, 1.0),
            "variance": (0.0, None),
            "ess": (1e-10, None),
        },
    },
    "bayesian_borrowing": {
        "required": [
            "effective_alpha", "effective_beta", "ess_total",
            "ess_from_historical", "prior_mean", "inputs",
        ],
        "types": {
            "effective_alpha": (int, float),
            "effective_beta": (int, float),
            "ess_total": (int, float),
            "prior_mean": (int, float),
        },
        "bounds": {
            "effective_alpha": (1e-10, None),
            "effective_beta": (1e-10, None),
            "ess_total": (1e-10, None),
            "prior_mean": (0.0, 1.0),
        },
    },
    "bayesian_sample_size_single_arm": {
        "required": [
            "recommended_n", "type1_error", "power", "constraints_met",
            "posterior_at_alt_alpha", "posterior_at_alt_beta", "inputs",
        ],
        "types": {
            "recommended_n": int,
            "type1_error": (int, float),
            "power": (int, float),
            "constraints_met": bool,
        },
        "bounds": {
            "type1_error": (0.0, 1.0, False),
            "power": (0.0, 1.0, False),
        },
    },
    "bayesian_sample_size_single_arm_continuous": {
        "required": [
            "recommended_n", "type1_error", "power", "constraints_met",
            "endpoint_type", "posterior_at_alt_mean", "posterior_at_alt_variance",
            "inputs",
        ],
        "types": {
            "recommended_n": int,
            "type1_error": (int, float),
            "power": (int, float),
            "constraints_met": bool,
            "endpoint_type": str,
            "posterior_at_alt_mean": (int, float),
            "posterior_at_alt_variance": (int, float),
        },
        "bounds": {
            "type1_error": (0.0, 1.0, False),
            "power": (0.0, 1.0, False),
            "posterior_at_alt_variance": (0.0, None, False),
        },
        "enums": {
            "endpoint_type": ["continuous"],
        },
    },
    "bayesian_two_arm": {
        "required": [
            "recommended_n_per_arm", "n_total", "type1_error", "power",
            "constraints_met", "inputs",
        ],
        "types": {
            "recommended_n_per_arm": int,
            "n_total": int,
            "type1_error": (int, float),
            "power": (int, float),
            "constraints_met": bool,
        },
        "bounds": {
            "type1_error": (0.0, 1.0, False),
            "power": (0.0, 1.0, False),
        },
    },
    "bayesian_two_arm_continuous": {
        "required": [
            "recommended_n_per_arm", "n_total", "type1_error", "power",
            "constraints_met", "endpoint_type", "posterior_at_alt_mean",
            "posterior_at_alt_variance", "inputs",
        ],
        "types": {
            "recommended_n_per_arm": int,
            "n_total": int,
            "type1_error": (int, float),
            "power": (int, float),
            "constraints_met": bool,
            "endpoint_type": str,
            "posterior_at_alt_mean": (int, float),
            "posterior_at_alt_variance": (int, float),
        },
        "bounds": {
            "type1_error": (0.0, 1.0, False),
            "power": (0.0, 1.0, False),
            "posterior_at_alt_variance": (0.0, None, False),
        },
        "enums": {
            "endpoint_type": ["continuous"],
        },
    },
    "bayesian_sequential": {
        "required": [
            "endpoint_type", "efficacy_boundaries", "futility_boundaries",
            "n_looks", "inputs",
        ],
        "types": {
            "endpoint_type": str,
            "efficacy_boundaries": list,
            "futility_boundaries": list,
            "n_looks": int,
        },
        "enums": {
            "endpoint_type": ["continuous", "survival"],
        },
    },
    "gsd_survival": {
        "required": [
            "max_events", "fixed_events", "n_total", "n_control", "n_treatment",
            "inflation_factor", "efficacy_boundaries", "futility_boundaries",
            "information_fractions", "alpha_spent", "event_probability", "inputs",
        ],
        "types": {
            "max_events": int,
            "fixed_events": int,
            "n_total": int,
            "n_control": int,
            "n_treatment": int,
            "inflation_factor": (int, float),
            "efficacy_boundaries": list,
            "futility_boundaries": list,
            "event_probability": (int, float),
        },
        "bounds": {
            "max_events": (0, None),
            "fixed_events": (0, None),
            "n_total": (0, None),
            "event_probability": (0.0, 1.0, False),
        },
    },
    "ssr_blinded": {
        "required": [
            "initial_n_total", "initial_n_per_arm", "interim_n",
            "blinded_variance_estimate", "recalculated_n_total",
            "recalculated_n_per_arm", "inflation_factor",
            "sample_size_increase", "n_capped", "conditional_power", "inputs",
        ],
        "types": {
            "initial_n_total": int,
            "initial_n_per_arm": int,
            "interim_n": int,
            "blinded_variance_estimate": (int, float),
            "recalculated_n_total": int,
            "recalculated_n_per_arm": int,
            "inflation_factor": (int, float),
            "sample_size_increase": int,
            "n_capped": bool,
            "conditional_power": (int, float),
        },
        "bounds": {
            "initial_n_total": (0, None),
            "conditional_power": (0.0, 1.0, False),
            "inflation_factor": (0.0, None, False),
        },
    },
    "ssr_unblinded": {
        "required": [
            "initial_n_total", "initial_n_per_arm", "interim_n",
            "conditional_power", "zone", "recalculated_n_total",
            "recalculated_n_per_arm", "inflation_factor",
            "sample_size_increase", "n_capped", "inputs",
        ],
        "types": {
            "initial_n_total": int,
            "initial_n_per_arm": int,
            "interim_n": int,
            "conditional_power": (int, float),
            "zone": str,
            "recalculated_n_total": int,
            "recalculated_n_per_arm": int,
            "inflation_factor": (int, float),
            "sample_size_increase": int,
            "n_capped": bool,
        },
        "bounds": {
            "initial_n_total": (0, None),
            "conditional_power": (0.0, 1.0, False),
            "inflation_factor": (0.0, None, False),
        },
        "enums": {
            "zone": ["futility", "unfavorable", "promising", "favorable"],
        },
    },
    "bayesian_survival": {
        "required": [
            "predictive_probability", "posterior_log_hr_mean",
            "posterior_log_hr_variance", "hr_posterior_mean",
            "credible_interval_lower", "credible_interval_upper",
            "recommendation", "inputs",
        ],
        "types": {
            "predictive_probability": (int, float),
            "posterior_log_hr_mean": (int, float),
            "posterior_log_hr_variance": (int, float),
            "hr_posterior_mean": (int, float),
            "recommendation": str,
        },
        "bounds": {
            "predictive_probability": (0.0, 1.0, False),
            "posterior_log_hr_variance": (1e-10, None),
            "hr_posterior_mean": (1e-10, None),
            "credible_interval_lower": (0.0, None, False),  # HR scale: must be positive
            "credible_interval_upper": (0.0, None, False),
        },
        "enums": {
            "recommendation": ["stop_for_efficacy", "continue", "stop_for_futility"],
        },
        "ordering": [
            ("credible_interval_lower", "credible_interval_upper"),
        ],
    },
}


def assert_schema(response: dict, contract_name: str) -> list:
    """
    Validate response against a schema contract.

    Returns a list of error strings (empty if valid).
    """
    contract = SCHEMA_CONTRACTS.get(contract_name)
    if contract is None:
        return [f"Unknown contract: {contract_name}"]

    errors = []

    # Check required keys
    for key in contract.get("required", []):
        if key not in response:
            errors.append(f"Missing key: {key}")

    # Check types
    for key, expected_type in contract.get("types", {}).items():
        if key in response and response[key] is not None:
            if not isinstance(response[key], expected_type):
                errors.append(
                    f"{key}: expected {expected_type}, got {type(response[key]).__name__}"
                )

    # Check bounds — tuples are (lo, hi) or (lo, hi, strict_lower)
    # strict_lower=True  (default): val <= lo  is an error  (excludes boundary)
    # strict_lower=False:           val < lo   is an error  (includes boundary)
    for key, bound in contract.get("bounds", {}).items():
        if key in response and response[key] is not None:
            val = response[key]
            if isinstance(val, (int, float)):
                lo = bound[0]
                hi = bound[1]
                strict = bound[2] if len(bound) > 2 else True
                if lo is not None:
                    if strict and val <= lo:
                        errors.append(f"{key}={val} <= {lo}")
                    elif not strict and val < lo:
                        errors.append(f"{key}={val} < {lo}")
                if hi is not None and val > hi:
                    errors.append(f"{key}={val} > {hi}")

    # Check enum constraints — {key: [allowed_values]}
    for key, allowed in contract.get("enums", {}).items():
        if key in response and response[key] is not None:
            if response[key] not in allowed:
                errors.append(f"{key}={response[key]!r} not in {allowed}")

    # Check ordering constraints — list of (key_lo, key_hi) tuples
    for key_lo, key_hi in contract.get("ordering", []):
        if key_lo in response and key_hi in response:
            lo_val = response[key_lo]
            hi_val = response[key_hi]
            if lo_val is not None and hi_val is not None:
                if isinstance(lo_val, (int, float)) and isinstance(hi_val, (int, float)):
                    if lo_val >= hi_val:
                        errors.append(f"{key_lo}={lo_val} >= {key_hi}={hi_val}")

    return errors
