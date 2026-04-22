#!/usr/bin/env python3
"""
Real-world replication: DAPA-HF trial (dapagliflozin in HFrEF).

DAPA-HF (McMurray et al., Eur J Heart Fail 2019 — design paper;
NEJM 2019 — results) was an international Phase III event-driven survival
trial comparing dapagliflozin to placebo on a composite of cardiovascular
death or worsening heart failure in chronic heart failure with reduced
ejection fraction.

Published design parameters (McMurray et al., Eur J Heart Fail 2019):
    Hazard ratio (assumed):  HR = 0.80
    Alpha (one-sided):        0.025
    Target power:             0.90
    Allocation ratio:         1:1
    Required events:          844 primary-endpoint events
    Placebo event rate:       ~11% per year
    Planned sample size:      ~4500 (actual: 4744 randomized)

Source:
    McMurray JJV et al. (2019). A trial to evaluate the effect of the
    sodium-glucose co-transporter 2 inhibitor dapagliflozin on morbidity
    and mortality in patients with heart failure and reduced left
    ventricular ejection fraction (DAPA-HF).
    Eur J Heart Fail, 21(5), 665–675. doi:10.1002/ejhf.1432
    (Design paper, PMID 30895697, PMC6607736.)

This script validates:
   1. Schoenfeld formula matches the published 844-event target under
      (HR=0.80, one-sided α=0.025, power=0.90, 1:1 allocation) — the
      backend's two-sided α=0.05 is numerically equivalent to one-sided
      α=0.025 (same z_α), so the same target emerges.
   2. HR sensitivity: a smaller effect (HR=0.85) requires materially more
      events; a larger effect (HR=0.75) materially fewer.
   3. Power sensitivity: 80% power needs fewer events; 95% power needs
      more. All scale as (z_α + z_β)².
   4. Allocation imbalance penalty: 2:1 randomization needs more events
      than 1:1 to preserve the same power against HR=0.80.
   5. The total-N of ~4500 is consistent with the 11% annual placebo rate
      and the event-driven target.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy import stats as sp_stats

from common.zetyra_client import get_client


DAPA_HR = 0.80
DAPA_ALPHA_ONE_SIDED = 0.025  # equivalent two-sided 0.05
DAPA_POWER = 0.90
DAPA_RATIO = 1.0
DAPA_TARGET_EVENTS = 844
DAPA_TOTAL_N = 4500  # design target; 4744 actually randomized


def ref_schoenfeld(hr: float, alpha_two_sided: float, power: float, r: float) -> int:
    """Reference Schoenfeld events for log-rank test."""
    z_a = sp_stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = sp_stats.norm.ppf(power)
    log_hr = math.log(hr)
    events = ((z_a + z_b) / log_hr) ** 2 * (1 + r) ** 2 / r
    return int(math.ceil(events))


def validate_dapa_primary_events(client) -> pd.DataFrame:
    """Schoenfeld events at the DAPA-HF design matches the published 844."""
    resp = client.sample_size_survival(
        hazard_ratio=DAPA_HR,
        median_control=12.0,  # DAPA-HF's 11%/yr is a hazard, not a median —
                              # we plug in a placeholder median here; only
                              # events_required is what we're validating.
        accrual_time=18.0,
        follow_up_time=24.0,
        dropout_rate=0.0,
        alpha=0.05,           # two-sided 0.05 ≡ one-sided 0.025
        power=DAPA_POWER,
        allocation_ratio=DAPA_RATIO,
    )
    api_events = resp["events_required"]
    ref = ref_schoenfeld(DAPA_HR, 0.05, DAPA_POWER, DAPA_RATIO)
    return pd.DataFrame([{
        "test": "DAPA-HF: Schoenfeld events match published 844 target (±2)",
        "api_events": api_events,
        "ref_events": ref,
        "published_target": DAPA_TARGET_EVENTS,
        "pass": abs(api_events - DAPA_TARGET_EVENTS) <= 2 and abs(api_events - ref) <= 1,
    }])


def validate_dapa_hr_sensitivity(client) -> pd.DataFrame:
    """HR sensitivity: weaker effect needs more events, stronger fewer."""
    rows = []
    base = dict(
        median_control=12.0, accrual_time=18.0, follow_up_time=24.0,
        dropout_rate=0.0, alpha=0.05, power=DAPA_POWER,
        allocation_ratio=DAPA_RATIO,
    )
    events_at_hr = {}
    for hr in (0.75, 0.80, 0.85):
        resp = client.sample_size_survival(hazard_ratio=hr, **base)
        events_at_hr[hr] = resp["events_required"]
        rows.append({
            "test": f"Events required at HR={hr}",
            "events": resp["events_required"], "pass": True,
        })
    rows.append({
        "test": "Monotone: events(0.85) > events(0.80) > events(0.75)",
        "events": None,
        "pass": events_at_hr[0.85] > events_at_hr[0.80] > events_at_hr[0.75],
    })
    return pd.DataFrame(rows)


def validate_dapa_power_sensitivity(client) -> pd.DataFrame:
    """Power sensitivity: more events needed as power requirement rises."""
    rows = []
    base = dict(
        hazard_ratio=DAPA_HR, median_control=12.0, accrual_time=18.0,
        follow_up_time=24.0, dropout_rate=0.0, alpha=0.05,
        allocation_ratio=DAPA_RATIO,
    )
    events_at_power = {}
    for pw in (0.80, 0.90, 0.95):
        resp = client.sample_size_survival(power=pw, **base)
        events_at_power[pw] = resp["events_required"]
        rows.append({
            "test": f"Events at power={pw}",
            "events": resp["events_required"], "pass": True,
        })
    rows.append({
        "test": "Monotone: events(0.80) < events(0.90) < events(0.95)",
        "events": None,
        "pass": events_at_power[0.80] < events_at_power[0.90] < events_at_power[0.95],
    })
    return pd.DataFrame(rows)


def validate_dapa_allocation_ratio(client) -> pd.DataFrame:
    """2:1 randomization imposes an events-count penalty vs 1:1."""
    base = dict(
        hazard_ratio=DAPA_HR, median_control=12.0, accrual_time=18.0,
        follow_up_time=24.0, dropout_rate=0.0, alpha=0.05, power=DAPA_POWER,
    )
    r1 = client.sample_size_survival(allocation_ratio=1.0, **base)
    r2 = client.sample_size_survival(allocation_ratio=2.0, **base)
    r3 = client.sample_size_survival(allocation_ratio=3.0, **base)
    return pd.DataFrame([{
        "test": "Allocation penalty: events(r=3) > events(r=2) > events(r=1)",
        "events_r1": r1["events_required"],
        "events_r2": r2["events_required"],
        "events_r3": r3["events_required"],
        "pass": (r3["events_required"] > r2["events_required"]
                 > r1["events_required"]),
    }])


def validate_dapa_total_n_consistent(client) -> pd.DataFrame:
    """DAPA-HF expected total N is ~4500; our engine's total (which depends on
    the exponential-survival event-probability conversion) should be in the
    same order of magnitude when we plug in a median that produces the
    published 11%/yr hazard over a combined ~3-year horizon.

    11%/yr ⇒ hazard λ = 0.11, so exponential median = ln2/λ ≈ 6.30 years ≈
    75.6 months. We use 76 months and check that total N lands between
    3,500 and 6,500 (a generous band that encompasses the 4500 design).
    """
    resp = client.sample_size_survival(
        hazard_ratio=DAPA_HR,
        median_control=76.0,
        accrual_time=18.0,
        follow_up_time=24.0,
        dropout_rate=0.0,
        alpha=0.05, power=DAPA_POWER, allocation_ratio=DAPA_RATIO,
    )
    return pd.DataFrame([{
        "test": "DAPA-HF total N consistent with 4500-design band",
        "events_required": resp["events_required"],
        "n_total": resp["n_total"],
        "design_target": DAPA_TOTAL_N,
        "pass": 3500 <= resp["n_total"] <= 6500,
    }])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default=None,
                        help="backend base URL")
    args = parser.parse_args()
    client = get_client(args.base_url)

    suites = [
        ("1. Schoenfeld events matches 844 target", validate_dapa_primary_events),
        ("2. HR sensitivity monotonicity", validate_dapa_hr_sensitivity),
        ("3. Power sensitivity monotonicity", validate_dapa_power_sensitivity),
        ("4. Allocation ratio penalty", validate_dapa_allocation_ratio),
        ("5. Total N in 4500-design band", validate_dapa_total_n_consistent),
    ]

    print("=" * 70)
    print("DAPA-HF (McMurray 2019) SURVIVAL SAMPLE SIZE REPLICATION")
    print(f"HR={DAPA_HR}, one-sided α={DAPA_ALPHA_ONE_SIDED}, "
          f"power={DAPA_POWER}, target events={DAPA_TARGET_EVENTS}")
    print("=" * 70)

    all_pass = True
    all_frames: list[pd.DataFrame] = []
    for name, fn in suites:
        print(f"\n{name}")
        print("-" * 70)
        try:
            df = fn(client)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_pass = False
            continue
        print(df.to_string(index=False))
        df_out = df.copy()
        df_out.insert(0, "suite", name)
        all_frames.append(df_out)
        if not df["pass"].all():
            all_pass = False

    if all_frames:
        os.makedirs("results", exist_ok=True)
        all_results = pd.concat(all_frames, ignore_index=True, sort=False)
        all_results.to_csv("results/dapa_hf_validation.csv", index=False)

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
        return 0
    print("SOME VALIDATIONS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
