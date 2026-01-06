"""
Zetyra Validation API Client

A simple client for calling Zetyra's public validation endpoints.
No authentication required - these endpoints are designed for
independent verification of calculator results.
"""

import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass


# Default to production API; override for local testing
DEFAULT_BASE_URL = "https://zetyra-backend-394439308230.us-central1.run.app/api/v1/validation"


@dataclass
class ZetyraClient:
    """Client for Zetyra validation API."""

    base_url: str = DEFAULT_BASE_URL
    timeout: int = 30

    def _post(self, endpoint: str, data: Dict[str, Any], allow_errors: bool = False) -> Dict[str, Any]:
        """Make POST request to validation endpoint."""
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data, timeout=self.timeout)
        if allow_errors and response.status_code >= 400:
            return {"error": response.status_code, "detail": response.text}
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Sample Size Calculators
    # =========================================================================

    def sample_size_continuous(
        self,
        mean1: float,
        mean2: float,
        sd: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0,
        two_sided: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate sample size for continuous outcomes (two-sample t-test).

        Args:
            mean1: Mean in group 1 (control)
            mean2: Mean in group 2 (treatment)
            sd: Common standard deviation
            alpha: Type I error rate (default: 0.05)
            power: Statistical power (default: 0.80)
            ratio: Allocation ratio n2/n1 (default: 1.0)
            two_sided: Two-sided test (default: True)

        Returns:
            dict with n1, n2, n_total, effect_size, z_alpha, z_beta, inputs
        """
        return self._post("/sample-size/continuous", {
            "mean1": mean1,
            "mean2": mean2,
            "sd": sd,
            "alpha": alpha,
            "power": power,
            "ratio": ratio,
            "two_sided": two_sided,
        })

    def sample_size_binary(
        self,
        p1: float,
        p2: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0,
        two_sided: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate sample size for binary outcomes (two-proportion z-test).

        Args:
            p1: Proportion in group 1 (control)
            p2: Proportion in group 2 (treatment)
            alpha: Type I error rate (default: 0.05)
            power: Statistical power (default: 0.80)
            ratio: Allocation ratio n2/n1 (default: 1.0)
            two_sided: Two-sided test (default: True)

        Returns:
            dict with n1, n2, n_total, effect_size_h, pooled_p, z_alpha, z_beta, inputs
        """
        return self._post("/sample-size/binary", {
            "p1": p1,
            "p2": p2,
            "alpha": alpha,
            "power": power,
            "ratio": ratio,
            "two_sided": two_sided,
        })

    def sample_size_survival(
        self,
        hazard_ratio: float,
        median_control: float,
        accrual_time: float,
        follow_up_time: float,
        alpha: float = 0.05,
        power: float = 0.80,
        dropout_rate: float = 0.0,
        allocation_ratio: float = 1.0,
        allow_errors: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate sample size for survival outcomes (log-rank test).

        Uses the Schoenfeld formula.

        Args:
            hazard_ratio: Hazard ratio (treatment/control)
            median_control: Median survival in control arm
            accrual_time: Accrual period duration
            follow_up_time: Follow-up period after accrual
            alpha: Type I error rate (default: 0.05)
            power: Statistical power (default: 0.80)
            dropout_rate: Annual dropout rate (default: 0.0)
            allocation_ratio: Allocation ratio (default: 1.0)
            allow_errors: Return error dict instead of raising (default: False)

        Returns:
            dict with events_required, n1, n2, n_total, log_hr, z_alpha, z_beta, inputs
        """
        return self._post("/sample-size/survival", {
            "hazard_ratio": hazard_ratio,
            "median_control": median_control,
            "accrual_time": accrual_time,
            "follow_up_time": follow_up_time,
            "alpha": alpha,
            "power": power,
            "dropout_rate": dropout_rate,
            "allocation_ratio": allocation_ratio,
        }, allow_errors=allow_errors)

    # =========================================================================
    # CUPED Calculator
    # =========================================================================

    def cuped(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde: float,
        correlation: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Calculate CUPED variance reduction.

        Args:
            baseline_mean: Baseline mean of the metric
            baseline_std: Baseline standard deviation
            mde: Minimum detectable effect (relative, e.g., 0.05 for 5%)
            correlation: Correlation between covariate and metric
            alpha: Type I error rate (default: 0.05)
            power: Statistical power (default: 0.80)

        Returns:
            dict with n_original, n_adjusted, variance_reduction_pct,
            variance_reduction_factor, r_squared, inputs
        """
        return self._post("/cuped", {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "mde": mde,
            "correlation": correlation,
            "alpha": alpha,
            "power": power,
        })

    # =========================================================================
    # Group Sequential Design
    # =========================================================================

    def gsd(
        self,
        effect_size: float,
        alpha: float = 0.025,
        power: float = 0.90,
        k: int = 3,
        timing: Optional[list] = None,
        spending_function: str = "OBrienFleming",
        beta_spending_function: Optional[str] = None,
        binding_futility: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate Group Sequential Design boundaries.

        Args:
            effect_size: Standardized effect size (Cohen's d)
            alpha: One-sided significance level (default: 0.025)
            power: Statistical power (default: 0.90)
            k: Number of interim looks (default: 3)
            timing: Information fractions (default: equal spacing)
            spending_function: Alpha spending function
                ("OBrienFleming", "Pocock", "HwangShihDecani")
            beta_spending_function: Beta spending function for futility
            binding_futility: Binding futility boundaries (default: False)

        Returns:
            dict with n_max, n_fixed, inflation_factor, efficacy_boundaries,
            futility_boundaries, information_fractions, alpha_spent, inputs
        """
        data = {
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "k": k,
            "spending_function": spending_function,
            "binding_futility": binding_futility,
        }
        if timing:
            data["timing"] = timing
        if beta_spending_function:
            data["beta_spending_function"] = beta_spending_function
        return self._post("/gsd", data)

    # =========================================================================
    # Bayesian Predictive Power
    # =========================================================================

    def bayesian_continuous(
        self,
        interim_effect: float,
        interim_var: float,
        interim_n: int,
        final_n: int,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        success_threshold: float = 0.95,
        n_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """
        Calculate Bayesian predictive power for continuous outcomes.

        Args:
            interim_effect: Observed effect at interim
            interim_var: Variance of interim estimate
            interim_n: Sample size at interim
            final_n: Planned final sample size
            prior_mean: Prior mean for treatment effect (default: 0.0)
            prior_var: Prior variance (default: 1.0)
            success_threshold: Posterior probability threshold (default: 0.95)
            n_simulations: Monte Carlo simulations (default: 10000)

        Returns:
            dict with predictive_probability, posterior_mean, posterior_var,
            credible_interval_lower, credible_interval_upper, recommendation, inputs
        """
        return self._post("/bayesian/continuous", {
            "interim_effect": interim_effect,
            "interim_var": interim_var,
            "interim_n": interim_n,
            "final_n": final_n,
            "prior_mean": prior_mean,
            "prior_var": prior_var,
            "success_threshold": success_threshold,
            "n_simulations": n_simulations,
        })

    def bayesian_binary(
        self,
        control_successes: int,
        control_n: int,
        treatment_successes: int,
        treatment_n: int,
        final_n: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        success_threshold: float = 0.95,
        n_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """
        Calculate Bayesian predictive power for binary outcomes.

        Args:
            control_successes: Successes in control arm
            control_n: Total in control arm
            treatment_successes: Successes in treatment arm
            treatment_n: Total in treatment arm
            final_n: Planned final N per arm
            prior_alpha: Beta prior alpha (default: 1.0)
            prior_beta: Beta prior beta (default: 1.0)
            success_threshold: Posterior probability threshold (default: 0.95)
            n_simulations: Monte Carlo simulations (default: 10000)

        Returns:
            dict with predictive_probability, posterior parameters, recommendation, inputs
        """
        return self._post("/bayesian/binary", {
            "control_successes": control_successes,
            "control_n": control_n,
            "treatment_successes": treatment_successes,
            "treatment_n": treatment_n,
            "final_n": final_n,
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "success_threshold": success_threshold,
            "n_simulations": n_simulations,
        })


# Convenience function for quick testing
def get_client(base_url: Optional[str] = None) -> ZetyraClient:
    """Get a Zetyra client instance."""
    if base_url:
        return ZetyraClient(base_url=base_url)
    return ZetyraClient()


if __name__ == "__main__":
    # Quick test
    client = get_client()

    print("Testing Zetyra Validation API...")
    print("=" * 50)

    # Test sample size
    result = client.sample_size_continuous(
        mean1=100, mean2=105, sd=20, alpha=0.05, power=0.80
    )
    print(f"Sample Size (Continuous): n={result['n_total']}, d={result['effect_size']}")

    # Test CUPED
    result = client.cuped(
        baseline_mean=100, baseline_std=20, mde=0.05, correlation=0.7
    )
    print(f"CUPED: n_original={result['n_original']}, n_adjusted={result['n_adjusted']}")

    print("=" * 50)
    print("All endpoints responding!")
