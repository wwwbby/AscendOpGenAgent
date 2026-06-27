"""autoresearch eval orchestrator (new workflow: subprocess-based).

The user supplies three files (written into the task_dir by scaffold):
  - kernel.py     — the editable kernel under optimization
  - <test_file>   — a python script whose `__main__` block runs all
                    correctness cases and prints `... passed!` per case
  - <perf_file>   — a python script that prints timing lines to stdout

This entrypoint runs the test and perf scripts as subprocesses, parses
their stdout, and writes a single JSON sidecar (`.eval_result.json`)
in the same schema the old ref/kernel flow produced, so
`eval_runner.local_eval` / `eval_assemble.assemble_eval_result` keep
working with minimal changes.

Test-script contract:
  - Invoked as `python <test_file>` (the script's `__main__` block is
    responsible for running all cases and printing a summary line per
    case; pass/fail is decided by exit code + stdout).
  - exit 0  ⇒ correctness = True
  - exit !=0 ⇒ correctness = False, error_source = "kernel"
  - The test script may print `... passed!` lines to indicate per-case
    progress; we count those for `num_cases`. If the script raises an
    exception during import/collection (e.g. SyntaxError, ModuleNotFound
    before reaching `__main__`), the traceback in stderr is tagged
    error_source = "infra" (test script itself broken).

Perf-script contract:
  - Invoked as `python <perf_file>`. Must print (to stdout) lines:
      triton:  median=X.XXms   ← gen (kernel under optimization)
      cann:    median=X.XXms   ← base (reference / CANN)
    The tokens `triton` and `cann` are matched case-insensitively at
    the start of a whitespace-split token. `median` is the default
    metric key; `--perf-metric p20` switches the parser to p20 etc.
  - If the perf script exits non-zero OR no triton line is found,
    profile_gen is None ⇒ downstream outcome = KERNEL_FAIL.
  - If no cann line is found, profile_base is None (baseline unavailable
    this round; baseline_metric will stay unset and the task remains
    in BASELINE phase, mirroring the old "ref measured = required"
    hard gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure `scripts/` is on sys.path so `from utils.json_io import ...`
# resolves when this file is launched as a subprocess by eval_runner
# with cwd=task_dir (the user's directory in in-place mode, or
# ar_tasks/<op>/ in copy mode). Without this, Python puts only this
# file's own directory (scripts/engine/) on sys.path[0], not scripts/,
# and `import utils` fails.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import re
import traceback


# ---------------------------------------------------------------------------
# stdout parsing for the perf script
# ---------------------------------------------------------------------------

# Matches lines like:
#   triton:  median=1.23ms
#   cann:    median=2.34ms
#   triton:  p20=1.10ms
# We pick the `key=value` pair where key == --perf-metric (default median).
# Units: `ms` (milliseconds) is the only unit we parse; `us`/`µs` also
# accepted and converted to microseconds directly.
_PERF_LINE_RE = re.compile(
    r"^\s*(?P<source>triton|cann)\b[^\n=]*?"
    r"(?P<key>median|p20|p80|mean|avg|min|max)\s*=\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|us|µs|s)?\b",
    re.IGNORECASE,
)


def parse_perf_stdout(stdout: str, metric_key: str = "median") -> dict:
    """Extract gen (triton) and base (cann) timings from perf stdout.

    Returns {"gen_us": float|None, "base_us": float|None, "raw_lines": [...]}.
    The last matching line per source wins (perf scripts often print
    multiple shapes; the final one is the summary).
    """
    gen_us: float | None = None
    base_us: float | None = None
    raw_lines: list[str] = []
    for line in stdout.splitlines():
        m = _PERF_LINE_RE.match(line)
        if not m:
            continue
        if m.group("key").lower() != metric_key.lower():
            continue
        val = float(m.group("value"))
        unit = (m.group("unit") or "ms").lower()
        if unit == "ms":
            us = val * 1000.0
        elif unit in ("us", "µs"):
            us = val
        elif unit == "s":
            us = val * 1_000_000.0
        else:
            continue
        raw_lines.append(line.strip())
        src = m.group("source").lower()
        if src == "triton":
            gen_us = us  # last wins
        elif src == "cann":
            base_us = us
    return {"gen_us": gen_us, "base_us": base_us, "raw_lines": raw_lines}


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: list[str], cwd: str, env: dict,
                    timeout: int) -> tuple[int, str, str]:
    """Run cmd, return (rc, stdout, stderr). rc=124 on timeout."""
    import subprocess
    popen_kwargs = {
        "cwd": cwd, "env": env,
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        return 1, "", f"failed to launch: {e}"
    try:
        out, err = proc.communicate(timeout=timeout)
        return (proc.returncode or 0,
                (out or b"").decode(errors="replace"),
                (err or b"").decode(errors="replace"))
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            proc.kill()
            out, err = b"", b""
        return (124,
                (out or b"").decode(errors="replace"),
                (err or b"").decode(errors="replace")
                + f"\n[eval_kernel] timed out after {timeout}s")


def _build_env(device_id: int, task_dir: str | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    # Ensure the task_dir is on sys.path so the test/perf scripts' own
    # `from <kernel_module> import ...` statements resolve. On POSIX,
    # `python <script>` with cwd=task_dir usually does this implicitly
    # (cwd lands in sys.path[0]), but Windows doesn't always, and an
    # explicit PYTHONPATH is the reliable cross-platform fix. This is
    # especially important in in-place mode where the kernel file is
    # imported by its original module name (e.g.
    # `sparse_flash_attention_triton`).
    if task_dir:
        abs_task = os.path.abspath(task_dir)
        existing = env.get("PYTHONPATH", "")
        if existing:
            env["PYTHONPATH"] = abs_task + os.pathsep + existing
        else:
            env["PYTHONPATH"] = abs_task
    return env


# ---------------------------------------------------------------------------
# phase runners
# ---------------------------------------------------------------------------

def run_test_phase(task_dir: str, test_file: str, device_id: int,
                   timeout: int, test_filter: str | None,
                   env: dict) -> dict:
    """Run `python <test_file>` and decide correctness from exit code.

    Contract for the test script's `__main__` block:
      - Run all cases; print one line per case on success (commonly
        `... passed!`); on failure, raise an exception (let it propagate
        so the exit code is non-zero) or call `sys.exit(nonzero)`.

    Exit-code semantics:
      0   — script ran to completion without raising; correctness = True.
            `num_cases` is best-effort counted from `... passed!` lines.
      !=0 — script raised or sys.exit(nonzero). We distinguish:
            * Infra failure (test script itself broken): the process
              died BEFORE reaching `__main__` execution — i.e. stderr
              contains a Python traceback whose topmost frame is at
              module import / collection time (SyntaxError, ImportError,
              NameError at module level). Tagged error_source="infra".
            * Kernel failure (kernel produced wrong output): the
              traceback's topmost frame is inside `__main__` or any
              function the test calls (assertion, ValueError from
              shape mismatch, etc.). Tagged error_source="kernel".
            The heuristic is conservative: if we can't tell, we fall
            back to "kernel" (the more actionable of the two for the
            optimisation loop, since INFRA_FAIL parks the task at
            BASELINE until the user fixes the script).
    """
    test_path = os.path.join(task_dir, test_file)
    if not os.path.isfile(test_path):
        return {
            "correctness": False,
            "error_source": "infra",
            "error": f"test file not found: {test_file}",
            "num_cases": 0,
            "per_case": [],
            "diagnostics": [],
            "failed_indices": [],
        }

    # NOTE: `test_filter` is no longer used — pytest's `-k` filter has no
    # equivalent in `python <file>` mode. Kept in the signature for
    # call-site compatibility (eval_runner passes it through).
    del test_filter

    cmd = [sys.executable, test_file]
    rc, stdout, stderr = _run_subprocess(
        cmd, cwd=task_dir, env=env, timeout=timeout)
    combined = (stdout + "\n" + stderr).strip()

    if rc == 0:
        # Count per-case "passed!" lines the script printed in __main__.
        n = _count_passed_lines(stdout)
        return {
            "correctness": True,
            "error_source": None,
            "error": None,
            "num_cases": n,
            "per_case": [],
            "diagnostics": [],
            "failed_indices": [],
            "log": combined[-2048:],
        }

    # Non-zero exit — decide infra vs kernel by inspecting the traceback.
    error_source = _classify_failure(stderr, stdout)
    detail = _tail(combined, 400)
    return {
        "correctness": False,
        "error_source": error_source,
        "error": f"test script failed (rc={rc}): {detail}",
        "num_cases": 0,
        "per_case": [],
        "diagnostics": [],
        "failed_indices": [],
        "log": combined[-2048:],
    }


def _tail(s: str, n: int) -> str:
    s = s.strip()
    return s[-n:] if len(s) > n else s


# Lines like `golden test (D=512, ...) passed!` — the user's test scripts
# print one per successful case in __main__.
_PASSED_LINE_RE = re.compile(r"\bpassed!\s*$", re.IGNORECASE)


def _count_passed_lines(stdout: str) -> int:
    """Count `... passed!` lines the test script printed in __main__."""
    n = 0
    for line in stdout.splitlines():
        if _PASSED_LINE_RE.search(line.strip()):
            n += 1
    return n


# Module-level / import-time failure signatures. If the traceback's
# topmost (last) frame is at module scope (no function context, line
# points at a top-level statement), we treat it as infra: the test
# script itself failed to load, not a kernel correctness issue.
_IMPORT_ERROR_PATTERNS = (
    "SyntaxError",
    "IndentationError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError:",         # usually a top-level typo
    "AttributeError: module",
)


def _classify_failure(stderr: str, stdout: str) -> str:
    """Return "infra" or "kernel" based on where the script failed.

    Heuristics (checked in order):
      1. stderr contains a Python traceback → look at the exception type
         on the last `Traceback (most recent call last):` block.
         - SyntaxError / ImportError / ModuleNotFoundError / top-level
           NameError / AttributeError on a module → "infra"
         - everything else (AssertionError, ValueError, RuntimeError
           from kernel launch, etc.) → "kernel"
      2. No traceback found (script called sys.exit(nonzero) without
         raising) → "kernel" (the script deliberately signalled
         failure; that's the test script's verdict on the kernel).
    """
    text = stderr if stderr else stdout
    if "Traceback (most recent call last):" not in text:
        return "kernel"
    # Take the last line that looks like an exception type line
    # (matches `ExcType: message` or `ExcType` at start of line).
    exc_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Heuristic: exception lines usually start with a capitalised
        # CamelCase identifier optionally followed by `:` and a message.
        if re.match(r"^[A-Z][A-Za-z0-9_]*(Error|Exception|Warning)", stripped
                    ) or ":" in stripped and re.match(
                        r"^[A-Z][A-Za-z0-9_]*", stripped):
            exc_line = stripped
    for pat in _IMPORT_ERROR_PATTERNS:
        if pat in exc_line:
            return "infra"
    return "kernel"


def run_perf_phase(task_dir: str, perf_file: str, device_id: int,
                   timeout: int, perf_metric: str,
                   env: dict) -> dict:
    """Run the perf script, parse stdout for gen/base timing.

    Returns {"profile_gen": {...}|None, "profile_base": {...}|None,
             "log": str, "rc": int}.
    """
    perf_path = os.path.join(task_dir, perf_file)
    if not os.path.isfile(perf_path):
        return {
            "profile_gen": None,
            "profile_base": None,
            "log": f"perf file not found: {perf_file}",
            "rc": 1,
        }

    cmd = [sys.executable, perf_file]
    rc, stdout, stderr = _run_subprocess(
        cmd, cwd=task_dir, env=env, timeout=timeout)
    combined = (stdout + "\n" + stderr).strip()

    parsed = parse_perf_stdout(stdout, metric_key=perf_metric)
    gen_us = parsed["gen_us"]
    base_us = parsed["base_us"]

    def _block(us: float | None) -> dict | None:
        if us is None or not (0 < us < float("inf")):
            return None
        return {
            "avg_time_us": us,
            "execution_time_us": us,
            "execution_time_ms": us / 1000.0,
            "per_shape": [{"avg_time_us": us, "idx": 0}],
            "num_cases": 1,
            "method": "perf_script",
        }

    return {
        "profile_gen": _block(gen_us),
        "profile_base": _block(base_us),
        "log": combined[-4096:],
        "rc": rc,
        "raw_perf_lines": parsed["raw_lines"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="autoresearch eval orchestrator (subprocess workflow)")
    ap.add_argument("--task-dir", required=True,
                    help="task directory containing kernel + test + perf scripts")
    ap.add_argument("--op-name", required=True,
                    help="operator name (recorded in result; not used for file naming)")
    ap.add_argument("--kernel-file", required=True,
                    help="kernel module name without .py (informational; the "
                         "test/perf scripts import the kernel themselves)")
    ap.add_argument("--test-file", required=True,
                    help="test script filename (with .py); invoked as "
                         "`python <test_file>`. The script's __main__ block "
                         "runs all cases and prints `... passed!` per case")
    ap.add_argument("--perf-file", required=True,
                    help="perf script filename (with .py); must print "
                         "`triton:  median=X.XXms` and `cann:    median=X.XXms`")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--phases", default="test,perf",
                    help="comma-separated subset of {test, perf}")
    ap.add_argument("--perf-metric", default="median",
                    help="metric key to parse from perf stdout "
                         "(median|p20|p80|mean|avg|min|max)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-phase wall-clock timeout in seconds")
    ap.add_argument("--output", default=None,
                    help="JSON sidecar path (default: <task_dir>/.eval_result.json)")
    args = ap.parse_args()

    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    valid = {"test", "perf"}
    bad = requested - valid
    if bad:
        print(f"unknown phase(s): {sorted(bad)}; valid: {sorted(valid)}",
              file=sys.stderr)
        sys.exit(2)

    task_dir = os.path.abspath(args.task_dir)
    env = _build_env(args.device_id, task_dir)

    result: dict = {
        "verify": None,
        "profile_base": None,
        "profile_gen": None,
        "ok": True,
        "errors": [],
    }
    out_path = args.output or os.path.join(task_dir, ".eval_result.json")

    from utils.json_io import sanitize_floats

    def _write_and_exit(rc: int) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sanitize_floats(result), f, default=str)
        print(f"[eval_kernel] result -> {out_path}", file=sys.stderr)
        sys.exit(rc)

    # ---- test phase ----
    if "test" in requested:
        try:
            verify_block = run_test_phase(
                task_dir, args.test_file, args.device_id,
                args.timeout, None, env)
            result["verify"] = verify_block
            if not verify_block.get("correctness"):
                # Don't run perf if correctness already failed — saves
                # device time and avoids confusing a crash during perf
                # with a correctness regression.
                requested.discard("perf")
                if verify_block.get("error_source") == "infra":
                    result["ok"] = False
                    result["errors"].append({
                        "phase": "test",
                        "error": verify_block.get("error", ""),
                    })
        except Exception as e:
            result["ok"] = False
            result["errors"].append({
                "phase": "test", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            result["verify"] = {
                "correctness": False,
                "error_source": "infra",
                "error": f"test phase crashed: {type(e).__name__}: {e}",
                "num_cases": 0, "per_case": [], "diagnostics": [],
                "failed_indices": [],
            }
            requested.discard("perf")

    # ---- perf phase ----
    if "perf" in requested:
        try:
            perf = run_perf_phase(
                task_dir, args.perf_file, args.device_id,
                args.timeout, args.perf_metric, env)
            result["profile_gen"] = perf.get("profile_gen")
            result["profile_base"] = perf.get("profile_base")
            if perf.get("rc", 0) != 0 and perf.get("profile_gen") is None:
                # perf script crashed AND no timing parsed → kernel-side
                # failure (the kernel under test may have crashed during
                # perf). Tag as kernel error_source for DIAGNOSE.
                if result["verify"] is None:
                    result["verify"] = {
                        "correctness": False,
                        "error_source": "kernel",
                        "error": f"perf script failed (rc={perf.get('rc')}): "
                                 f"{_tail(perf.get('log', ''), 400)}",
                        "num_cases": 0, "per_case": [], "diagnostics": [],
                        "failed_indices": [],
                    }
                else:
                    # verify passed but perf crashed — still a kernel
                    # issue (kernel works for tiny test inputs but
                    # crashes on perf-sized inputs).
                    result["verify"]["error_source"] = "kernel"
                    result["verify"]["error"] = (
                        f"perf script failed (rc={perf.get('rc')}): "
                        f"{_tail(perf.get('log', ''), 400)}")
                    result["verify"]["correctness"] = False
        except Exception as e:
            result["ok"] = False
            result["errors"].append({
                "phase": "perf", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })

    _write_and_exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
