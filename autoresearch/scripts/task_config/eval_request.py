"""Build canonical eval requests before invoking the subprocess.

New workflow: the user supplies test (pytest) + perf scripts. The perf
script measures both gen (triton kernel) and base (CANN reference) timing
on every eval, so sticky baseline reuse is gone — there's no separate
ref pass to skip. Case-count probing is gone too: the test/perf scripts
manage their own case matrices internally.

This module now owns only timeout scaling. It returns data only; the
runner executes and the assembler interprets the response.
"""
from __future__ import annotations

from dataclasses import dataclass

from .loader import TaskConfig


@dataclass(frozen=True)
class EvalRequest:
    task_dir: str
    config: TaskConfig
    num_cases: int
    timeout: int
    # Retained as None for API back-compat with callers that still read
    # these fields (worker HTTP form, _log_request). Always None in the
    # new workflow — every eval re-measures the base via the perf script.
    override_base_us: float | None = None
    override_base_per_shape_us: list[float] | None = None

    @property
    def sticky(self) -> bool:
        return False  # new workflow: no sticky baseline

    def sticky_note(self) -> str:
        return ""


def effective_timeout(config: TaskConfig, num_cases: int) -> int:
    """config.eval_timeout is the budget for the eval subprocess. In the
    new workflow a single subprocess runs both the test script (pytest)
    and the perf script, so we no longer scale by num_cases — the test
    script manages its own case matrix and the wall-clock cost is bounded
    by the script itself, not by autoresearch's case count."""
    return int(config.eval_timeout)


def build_eval_request(task_dir: str, config: TaskConfig) -> EvalRequest:
    # num_cases is informational in the new workflow (test/perf scripts
    # manage their own case matrices). We pass config.num_cases through
    # unchanged so downstream code that reads it (e.g. timeout scaling
    # logs, history records) keeps working.
    num_cases = max(int(getattr(config, "num_cases", 1) or 1), 1)
    timeout = effective_timeout(config, num_cases)
    return EvalRequest(
        task_dir=task_dir,
        config=config,
        num_cases=num_cases,
        timeout=timeout,
        # Always None: perf script re-measures base on every eval.
        override_base_us=None,
        override_base_per_shape_us=None,
    )
