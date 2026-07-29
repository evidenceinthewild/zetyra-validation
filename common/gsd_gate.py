"""
Shared gate machinery for the GSD validation suites.

WHY THIS EXISTS
---------------
When the certified look ceiling dropped to k<=4, the suites' above-ceiling
cases started returning 422 and the scripts were changed to SKIP them. The
totals silently fell from 226/226 to 200/200 and from 46/46 to 32/32, and both
still printed "ALL VALIDATIONS PASSED". Coverage was removed and the gate
reported success.

That is the failure mode this whole programme keeps hitting: a plausible number
that nobody compared against an expected one. The same day produced three more
instances -- an R benchmark that halted before its summary and printed two
blank lines (indistinguishable from success in a terminal), and a handover that
asserted a branch pointer without running `git rev-parse`.

So a case above the ceiling is not skipped. It is asserted to be REJECTED, with
the same rigour a numerical case is asserted to be correct. Coverage is
preserved; only the meaning of the assertion changes.

WHY ID SETS AND NOT COUNTS
--------------------------
A count cannot distinguish a duplicated case from an omitted one: drop one and
duplicate another and the total is unchanged. Every suite therefore compares
the exact SET of executed case IDs against the manifest, and asserts IDs are
unique before doing so.

WHAT IS ASSERTED FOR A REJECTION
--------------------------------
Machine-readable fields only. `msg` is human prose that may be reworded at any
time, and a gate that asserts on prose either breaks on a copy edit or gets
loosened until it asserts nothing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

# The certified ceiling this gate expects the API to advertise. Asserted from
# the machine-readable `ctx.le`, so a silent change to the deployed bound fails
# here rather than being absorbed.
CERTIFIED_MAX_K = 4

# Pydantic v2 error type for an `le=` bound. Stable across message rewordings.
EXPECTED_REJECTION_TYPE = "less_than_equal"

# Any of these appearing in a rejection response would mean the API produced a
# design for an uncertified input.
NUMERICAL_RESULT_FIELDS = (
    "n_max", "n_fixed", "efficacy_boundaries", "futility_boundaries",
    "alpha_spent", "beta_spent", "information_fractions", "expected_n_h0",
    "expected_n_h1", "max_events", "fixed_events", "resolved_spending",
)


@dataclass
class GateResult:
    """One assertion outcome, numerical or rejection."""
    case_id: str
    kind: str          # "numerical" | "rejection"
    passed: bool
    detail: str = ""


@dataclass
class Gate:
    """Tracks manifest vs observed and decides the exit code.

    Construct with the manifest partition, record outcomes as they happen, then
    call `finish()`. `finish()` never returns -- it prints the structured
    summary and exits, so a suite cannot fall off the end silently.
    """

    suite: str
    suite_id: str
    expected_numerical: Set[str]
    expected_rejection: Set[str]
    results: List[GateResult] = field(default_factory=list)
    unexpected_skips: List[str] = field(default_factory=list)
    unexpected_http: List[str] = field(default_factory=list)

    # -- recording ---------------------------------------------------------

    def record_numerical(self, case_id: str, passed: bool, detail: str = "") -> None:
        self.results.append(GateResult(case_id, "numerical", passed, detail))

    def record_rejection(self, case_id: str, passed: bool, detail: str = "") -> None:
        self.results.append(GateResult(case_id, "rejection", passed, detail))

    def record_skip(self, case_id: str, why: str) -> None:
        """A skip is a TERMINAL FAILURE, never a pass.

        Skipping is how coverage disappeared in the first place. If a case
        genuinely cannot run it belongs out of the manifest, with the removal
        visible in a diff -- not silently absent from a run.
        """
        self.unexpected_skips.append(f"{case_id}: {why}")

    def record_http_error(self, case_id: str, what: str) -> None:
        """500s, timeouts and malformed JSON are failures, not absences."""
        self.unexpected_http.append(f"{case_id}: {what}")

    # -- derived -----------------------------------------------------------

    def _observed(self, kind: str) -> List[str]:
        return [r.case_id for r in self.results if r.kind == kind]

    def _duplicates(self, ids: Iterable[str]) -> Set[str]:
        seen, dupes = set(), set()
        for i in ids:
            if i in seen:
                dupes.add(i)
            seen.add(i)
        return dupes

    # -- reporting ---------------------------------------------------------

    def finish(self) -> None:
        num_obs = self._observed("numerical")
        rej_obs = self._observed("rejection")
        num_set, rej_set = set(num_obs), set(rej_obs)

        num_dupes = self._duplicates(num_obs)
        rej_dupes = self._duplicates(rej_obs)

        num_missing = self.expected_numerical - num_set
        num_extra = num_set - self.expected_numerical
        rej_missing = self.expected_rejection - rej_set
        rej_extra = rej_set - self.expected_rejection

        num_passed = sum(1 for r in self.results
                         if r.kind == "numerical" and r.passed)
        rej_passed = sum(1 for r in self.results
                         if r.kind == "rejection" and r.passed)
        failures = [r for r in self.results if not r.passed]

        print("\n" + "=" * 78)
        print(f"GATE SUMMARY -- {self.suite}")
        # Machine-readable identity and counts. The parent runner parses these
        # rather than the human table, so a suite cannot claim to be a
        # different suite or omit a field without failing schema validation.
        print(f"SUITE-ID: {self.suite_id}")
        print("=" * 78)
        print(f"  manifest numerical cases   : {len(self.expected_numerical)}")
        print(f"  numerical executed/passed  : {len(num_obs)}/{num_passed}")
        print(f"  manifest rejection cases   : {len(self.expected_rejection)}")
        print(f"  rejections executed/passed : {len(rej_obs)}/{rej_passed}")
        print(f"  unexpected skips           : {len(self.unexpected_skips)}")
        print(f"  unexpected HTTP errors     : {len(self.unexpected_http)}")
        print(f"  total failures             : {len(failures)}")

        problems: List[str] = []
        if not num_obs:
            problems.append(
                "NO numerical case ran. A suite that asserts only rejections "
                "proves the API refuses work, not that it computes correctly."
            )
        for label, s in (("duplicate numerical IDs", num_dupes),
                         ("duplicate rejection IDs", rej_dupes),
                         ("numerical IDs missing from run", num_missing),
                         ("numerical IDs not in manifest", num_extra),
                         ("rejection IDs missing from run", rej_missing),
                         ("rejection IDs not in manifest", rej_extra)):
            if s:
                problems.append(f"{label}: {sorted(s)}")
        if self.unexpected_skips:
            problems.append("skips (terminal): " + "; ".join(self.unexpected_skips))
        if self.unexpected_http:
            problems.append("HTTP errors: " + "; ".join(self.unexpected_http))

        if failures:
            print("\nFAILED ASSERTIONS")
            for r in failures:
                print(f"  [{r.kind}] {r.case_id}: {r.detail}")
        if problems:
            print("\nMANIFEST / INTEGRITY PROBLEMS")
            for p in problems:
                print(f"  - {p}")

        ok = not failures and not problems
        print("\n" + ("ALL VALIDATIONS PASSED" if ok else "GATE FAILED"))
        print("=" * 78)
        sys.stdout.flush()
        sys.exit(0 if ok else 1)


def check_rejection(response, case_id: str, gate: Gate,
                    expected_field: str = "k") -> None:
    """Assert one above-ceiling case is refused, and refused for the right reason.

    Asserts machine-readable fields only: HTTP status, Pydantic error `type`,
    the `loc` naming the offending field, and `ctx.le` carrying the ceiling.
    `msg` is prose and is deliberately not asserted -- a gate that depends on
    wording either breaks on a copy edit or gets loosened until it means
    nothing.
    """
    if response.status_code != 422:
        gate.record_rejection(
            case_id, False,
            f"expected HTTP 422, got {response.status_code}. An uncertified "
            f"look count was not refused."
        )
        return

    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        gate.record_http_error(case_id, f"malformed JSON in 422 body: {exc}")
        gate.record_rejection(case_id, False, "response body was not JSON")
        return

    detail = body.get("detail")
    if not isinstance(detail, list) or not detail:
        gate.record_rejection(
            case_id, False, f"422 body has no error list: {body!r}")
        return

    match = None
    for err in detail:
        loc = [str(p) for p in err.get("loc", [])]
        if loc and loc[-1] == expected_field:
            match = err
            break
    if match is None:
        locs = [e.get("loc") for e in detail]
        gate.record_rejection(
            case_id, False,
            f"rejected, but not on '{expected_field}' -- error locations were "
            f"{locs}. Refusing for the wrong reason passes a naive status "
            f"check while the real bound is unenforced."
        )
        return

    if match.get("type") != EXPECTED_REJECTION_TYPE:
        gate.record_rejection(
            case_id, False,
            f"expected error type {EXPECTED_REJECTION_TYPE!r}, got "
            f"{match.get('type')!r}"
        )
        return

    ceiling = (match.get("ctx") or {}).get("le")
    if ceiling != CERTIFIED_MAX_K:
        gate.record_rejection(
            case_id, False,
            f"API advertises ceiling le={ceiling}, gate expects "
            f"{CERTIFIED_MAX_K}. The deployed bound and this suite disagree."
        )
        return

    leaked = [f for f in NUMERICAL_RESULT_FIELDS if f in body]
    if leaked:
        gate.record_rejection(
            case_id, False,
            f"rejection body carries numerical result fields {leaked} -- a "
            f"design was produced for an uncertified input."
        )
        return

    gate.record_rejection(case_id, True)


def post_case(session_or_requests, url: str, payload: Dict[str, Any],
              case_id: str, gate: Gate, timeout: int = 300):
    """POST with failure classification. Returns the response, or None.

    Timeouts and transport errors are recorded as failures rather than allowed
    to raise, so one unreachable case cannot abort the run before its summary
    -- which is precisely how the R benchmark printed two blank lines and
    looked like success.
    """
    try:
        return session_or_requests.post(url, json=payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - any transport failure is a failure
        gate.record_http_error(case_id, f"{type(exc).__name__}: {exc}")
        return None
