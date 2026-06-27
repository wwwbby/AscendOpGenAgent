"""Pure helpers for baseline anchor ownership (new workflow).

The new workflow runs a single subprocess per round that invokes the
user's test script + perf script. The perf script measures BOTH gen
(triton kernel) and base (CANN reference) timings, so there is no
separate "ref pass" and no sticky baseline — every round re-measures
base. These helpers simply record/refresh the anchor from each round's
fresh base_latency_us.

Field-name compatibility: eval_assemble still writes `ref_latency_us`
as an alias for `base_latency_us` so the legacy field reads below keep
working; `baseline_source` is now `"base"` (was `"ref"`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def valid_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0 < value < float("inf")


def valid_per_shape(values: Any) -> Optional[list[float]]:
    if (isinstance(values, list) and values
            and all(valid_metric(v) for v in values)):
        return [float(v) for v in values]
    return None


def current_fingerprint(num_cases: Any) -> dict[str, int]:
    """Fingerprint kept for shape parity; sticky reuse is gone but the
    field is still written so old progress files don't break."""
    return {"num_cases": int(num_cases or 1)}


def fingerprint_mismatch(stored: Any,
                         current: dict[str, int]) -> Optional[dict[str, Any]]:
    """Return changed keys, or None. Kept for callers that still pass a
    fingerprint; in the new workflow this never triggers a re-anchor
    because base is re-measured every round."""
    if not isinstance(stored, dict) or not stored:
        return None
    mismatch = {
        k: (stored.get(k), current[k])
        for k in current
        if stored.get(k) != current[k]
    }
    return mismatch or None


def exact_fingerprint_match(stored: Any,
                            current: dict[str, int]) -> bool:
    if not isinstance(stored, dict):
        return False
    return stored.get("num_cases") == current["num_cases"]


@dataclass(frozen=True)
class StickyOverride:
    metric: float
    per_shape_us: Optional[list[float]]


@dataclass(frozen=True)
class StickyDecision:
    override: Optional[StickyOverride]
    mismatch: Optional[dict[str, Any]] = None


def sticky_override_from_progress(progress: Any,
                                  fingerprint: dict[str, int]
                                  ) -> StickyDecision:
    """New workflow has no sticky baseline — always returns None.

    Kept for call-site compatibility (eval_request still imports this
    symbol); callers that consult `decision.override` will simply always
    fall through to a fresh eval, which is what we want now.
    """
    return StickyDecision(None)


@dataclass(frozen=True)
class AnchorDecision:
    metric: Optional[float]
    source: Optional[str]
    per_shape_us: Optional[list[float]]
    fingerprint: Optional[dict[str, int]]
    reused_existing: bool
    changed: bool
    message: Optional[str] = None


def _anchor_tuple(progress: Any) -> tuple[Any, Any, Any, Any]:
    return (
        progress.get("baseline_metric"),
        progress.get("baseline_source"),
        progress.get("baseline_per_shape_us"),
        progress.get("baseline_fingerprint"),
    )


def _changed(progress: Any, decision: AnchorDecision) -> bool:
    return _anchor_tuple(progress) != (
        decision.metric,
        decision.source,
        decision.per_shape_us,
        decision.fingerprint,
    )


def _base_from_metrics(metrics: dict[str, Any]) -> Optional[float]:
    """Read the perf-script-measured base latency. Falls back to the
    legacy `ref_latency_us` alias written by eval_assemble."""
    value = metrics.get("base_latency_us")
    if not valid_metric(value):
        value = metrics.get("ref_latency_us")
    return float(value) if valid_metric(value) else None


def _per_shape_from_metrics(metrics: dict[str, Any]) -> Optional[list[float]]:
    return valid_per_shape(metrics.get("per_shape_base_us"))


def resolve_baseline_init_anchor(progress: Any, metrics: dict[str, Any],
                                 ) -> AnchorDecision:
    """Choose the anchor written by round-0 baseline initialization.

    New workflow: every round's perf script produces a fresh base
    latency, so we just adopt it when present. No sticky reuse path.
    """
    base_metric = _base_from_metrics(metrics)
    fp = current_fingerprint(metrics.get("num_cases") or 1)

    if base_metric is not None:
        return AnchorDecision(
            metric=base_metric,
            source="base",
            per_shape_us=_per_shape_from_metrics(metrics),
            fingerprint=fp,
            reused_existing=False,
            changed=True,
            message=(f"baseline = base_latency_us = {base_metric} "
                     f"(perf script CANN measurement)"),
        )

    # No valid perf base latency → no baseline. Seed timing is NOT a
    # substitute; run_baseline_init's gate refuses to commit and parks
    # the task at BASELINE for retry.
    return AnchorDecision(
        metric=None,
        source="none",
        per_shape_us=None,
        fingerprint=None,
        reused_existing=False,
        changed=True,
        message="no valid base_latency_us; baseline unmeasured (not committed)",
    )


def refresh_round_anchor(progress: Any,
                         metrics: dict[str, Any]) -> AnchorDecision:
    """Refresh Progress.baseline_* after a normal optimization round.

    New workflow: each round's perf script re-measures base, so we
    simply adopt the fresh value when present and keep the existing
    anchor otherwise (e.g. when the kernel crashed mid-perf and base
    wasn't measured this round).
    """
    base_metric = _base_from_metrics(metrics)
    fp = current_fingerprint(metrics.get("num_cases") or 1)

    existing_metric = progress.get("baseline_metric")
    existing_source = progress.get("baseline_source")
    existing_per_shape = valid_per_shape(
        progress.get("baseline_per_shape_us"))
    existing_fp = progress.get("baseline_fingerprint")

    if base_metric is not None:
        decision = AnchorDecision(
            metric=base_metric,
            source="base",
            per_shape_us=_per_shape_from_metrics(metrics),
            fingerprint=fp,
            reused_existing=False,
            changed=True,
            message=(f"captured baseline_metric={base_metric:.2f}us "
                     f"(source=base, perf re-measured this round)"),
        )
        return AnchorDecision(
            metric=decision.metric,
            source=decision.source,
            per_shape_us=decision.per_shape_us,
            fingerprint=decision.fingerprint,
            reused_existing=decision.reused_existing,
            changed=_changed(progress, decision),
            message=decision.message,
        )

    return AnchorDecision(
        metric=(float(existing_metric) if valid_metric(existing_metric)
                else None),
        source=existing_source,
        per_shape_us=existing_per_shape,
        fingerprint=existing_fp,
        reused_existing=True,
        changed=False,
        message=None,
    )
