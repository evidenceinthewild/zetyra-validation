"""
Zetyra API Client for Validation

Provides a simple interface to call Zetyra's public validation endpoints.
"""

import requests
from typing import Optional

DEFAULT_BASE_URL = "https://zetyra-backend-394439308230.us-central1.run.app"


class ZetyraClient:
    """Client for Zetyra validation API endpoints."""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.session = requests.Session()

    def _post(self, endpoint: str, data: dict, allow_errors: bool = False) -> dict:
        """Make POST request to API endpoint."""
        url = f"{self.base_url}/api/v1/validation{endpoint}"
        response = self.session.post(url, json=data, timeout=120)
        if not allow_errors:
            response.raise_for_status()
        return response.json()

    def _post_raw(self, endpoint: str, data: dict):
        """Make POST request and return raw requests.Response (no raise_for_status)."""
        url = f"{self.base_url}/api/v1/validation{endpoint}"
        return self.session.post(url, json=data, timeout=120)

    def sample_size_continuous(self, **kwargs) -> dict:
        """Calculate sample size for continuous outcomes."""
        return self._post("/sample-size/continuous", kwargs)

    def sample_size_binary(self, **kwargs) -> dict:
        """Calculate sample size for binary outcomes."""
        return self._post("/sample-size/binary", kwargs)

    def sample_size_survival(self, **kwargs) -> dict:
        """Calculate sample size for survival outcomes."""
        return self._post("/sample-size/survival", kwargs, allow_errors=kwargs.pop("allow_errors", False))

    def cuped(self, **kwargs) -> dict:
        """Calculate CUPED variance reduction."""
        return self._post("/cuped", kwargs)

    def gsd(self, **kwargs) -> dict:
        """Calculate Group Sequential Design boundaries."""
        return self._post("/gsd", kwargs)

    def bayesian_continuous(self, **kwargs) -> dict:
        """Calculate Bayesian predictive power for continuous outcomes."""
        return self._post("/bayesian/continuous", kwargs)

    def bayesian_binary(self, **kwargs) -> dict:
        """Calculate Bayesian predictive power for binary outcomes."""
        return self._post("/bayesian/binary", kwargs)

    def prior_elicitation(self, **kwargs) -> dict:
        """Calculate Bayesian prior elicitation."""
        return self._post("/bayesian/prior-elicitation", kwargs)

    def bayesian_borrowing(self, **kwargs) -> dict:
        """Calculate Bayesian historical borrowing prior."""
        return self._post("/bayesian/borrowing", kwargs)

    def bayesian_sample_size_single_arm(self, **kwargs) -> dict:
        """Calculate Bayesian sample size (single-arm binary)."""
        return self._post("/bayesian/sample-size-single-arm", kwargs)

    def bayesian_two_arm(self, **kwargs) -> dict:
        """Calculate Bayesian two-arm sample size."""
        return self._post("/bayesian/two-arm", kwargs)

    def bayesian_sequential(self, **kwargs) -> dict:
        """Calculate Bayesian sequential design boundaries."""
        return self._post("/bayesian/sequential", kwargs)


def get_client(base_url: str = None) -> ZetyraClient:
    """Get a configured Zetyra client."""
    return ZetyraClient(base_url)
