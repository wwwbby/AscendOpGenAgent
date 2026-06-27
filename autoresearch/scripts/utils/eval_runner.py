"""Local subprocess driver for autoresearch eval (new workflow).

The new workflow runs a single subprocess: `eval_kernel.py` with
`--phases test,perf`. Inside that subprocess, eval_kernel runs the
user's pytest-style test script and the user's perf script, then writes
a single `.eval_result.json` sidecar.

The old workflow split eval into two subprocesses (ref pass + kernel
pass) so a kernel UB overflow couldn't kill ref timing. The new workflow
doesn't need that split: the perf script measures both gen (triton
kernel) and base (CANN reference) in one run, and if the kernel crashes
the perf script, the test script has already run in the same subprocess
— we just won't have a base_time that round, and the task stays in
BASELINE until a round succeeds.

Public surface:
  - detect_local_backend() -> (ok, why)
  - local_eval(task_dir, op_name, kernel_file, test_file, perf_file,
               timeout, device_id) -> (verify_resp, profile_resp)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ascend runtime probe (unchanged from old workflow)
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = r"""
import sys
try:
    import torch
except Exception as e:
    print(f"NO: torch missing or broken: {e}")
    sys.exit(1)
try:
    import torch_npu  # noqa: F401
except Exception as e:
    print(f"NO: torch_npu missing: {e}")
    sys.exit(1)
try:
    n = torch.npu.device_count()
except Exception as e:
    print(f"NO: torch.npu unavailable: {e}")
    sys.exit(1)
print(f"OK: npu devices={n}")
sys.exit(0)
"""

_DETECT_CACHE: list = []


def detect_local_backend() -> tuple[bool, str]:
    """Probe whether this machine can run Ascend NPU eval locally."""
    if _DETECT_CACHE:
        return _DETECT_CACHE[0]
    probe_env = os.environ.copy()
    probe_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=30, env=probe_env,
        )
    except subprocess.TimeoutExpired:
        result = (False, "ascend probe timed out (>30s)")
    except Exception as e:
        result = (False, f"ascend probe failed to launch: {e}")
    else:
        line = (r.stdout or r.stderr or "").strip().splitlines()
        msg = line[-1] if line else "(no output)"
        result = (r.returncode == 0, msg)
    _DETECT_CACHE.append(result)
    return result


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _build_env(device_id: int) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_subprocess(cmd: list[str], cwd: str, env: dict,
                    timeout: int) -> tuple[int, str, str]:
    """subprocess.run wrapper. Returns (rc, stdout, stderr). rc=124 on
    timeout."""
    popen_kwargs = {
        "cwd": cwd, "env": env,
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        return 1, "", f"failed to launch eval: {e}"
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return (proc.returncode or 0,
                (stdout or b"").decode(errors="replace"),
                (stderr or b"").decode(errors="replace"))
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
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = b"", b""
        return (124,
                (stdout or b"").decode(errors="replace"),
                (stderr or b"").decode(errors="replace")
                + f"\n[eval_runner] eval timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------

def _avg_us(d: Optional[dict]) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    v = d.get("avg_time_us")
    if isinstance(v, (int, float)) and 0 < v < float("inf"):
        return float(v)
    return None


def _read_sidecar(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning("eval_runner: cannot parse %s: %s", path, e)
        return {}


# ---------------------------------------------------------------------------
# local_eval (single subprocess)
# ---------------------------------------------------------------------------

def local_eval(task_dir: str, op_name: str,
               kernel_file: str, test_file: str, perf_file: str,
               timeout: int, device_id: int = 0,
               **_legacy) -> tuple[dict, dict]:
    """Run eval_kernel.py once with --phases test,perf.

    The perf script measures both gen (triton kernel) and base (CANN
    reference) timing in the same subprocess. No sticky baseline, no
    ref pass — every eval re-measures everything.

    Returns (verify_resp, profile_resp) in the shape
    `eval_assemble.assemble_eval_result` consumes.

    `_legacy` swallows warmup/repeats/override_base_* kwargs that old
    callers still pass — they're ignored in the new workflow.
    """
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_script = os.path.join(scripts_dir, "engine", "eval_kernel.py")
    abs_task = os.path.abspath(task_dir)
    env = _build_env(device_id)

    sidecar_path = os.path.join(abs_task, ".eval_result.json")
    try:
        os.remove(sidecar_path)
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable, eval_script,
        "--task-dir", abs_task,
        "--op-name", op_name,
        "--kernel-file", kernel_file,
        "--test-file", test_file,
        "--perf-file", perf_file,
        "--device-id", str(device_id),
        "--phases", "test,perf",
        "--timeout", str(timeout),
        "--output", sidecar_path,
    ]
    rc, stdout, stderr = _run_subprocess(
        cmd, cwd=task_dir, env=env, timeout=timeout * 2 + 30)
    log = (stdout + ("\n" + stderr if stderr else "")).strip()

    payload = _read_sidecar(sidecar_path)
    return _assemble_response(payload, rc, log)


def _assemble_response(payload: dict, rc: int, log: str
                       ) -> tuple[dict, dict]:
    """Build (verify_resp, profile_resp) from the single sidecar."""
    from .json_io import sanitize_floats

    verify_block = payload.get("verify") or {}
    gen_block = payload.get("profile_gen")
    base_block = payload.get("profile_base")

    verify_correct = bool(verify_block.get("correctness"))
    error_source = verify_block.get("error_source") if not verify_correct else None

    verify_resp = {
        "success": verify_correct,
        "log": log,
        "artifacts": {},
        "returncode": rc,
        "error_source": error_source,
        "verify_block": verify_block if isinstance(verify_block, dict) else {},
    }

    artifacts: dict[str, str] = {}
    if isinstance(base_block, dict):
        artifacts["base_profile_result.json"] = json.dumps(
            sanitize_floats(base_block))
    if isinstance(gen_block, dict):
        artifacts["generation_profile_result.json"] = json.dumps(
            sanitize_floats(gen_block))

    base_time = _avg_us(base_block)
    gen_time = _avg_us(gen_block)
    profile_resp = {
        "success": gen_time is not None or base_time is not None,
        "log": log,
        "artifacts": artifacts,
        "gen_time": gen_time,
        "base_time": base_time,
    }
    return verify_resp, profile_resp


# ---------------------------------------------------------------------------
# Async sibling (worker daemon)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import signal  # noqa: E402


def _killpg_quiet(proc) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        return


async def _run_subprocess_async(cmd: list[str], cwd: str, env: dict,
                                timeout: int) -> tuple[int, str, str]:
    preexec = os.setsid if hasattr(os, "setsid") else None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec,
        )
    except Exception as e:
        return 1, "", f"failed to launch eval: {e}"
    try:
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
            rc = proc.returncode or 0
            return (rc,
                    stdout_b.decode(errors="replace"),
                    stderr_b.decode(errors="replace"))
        except asyncio.TimeoutError:
            _killpg_quiet(proc)
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=5)
            except Exception:
                stdout_b, stderr_b = b"", b""
            return (124,
                    stdout_b.decode(errors="replace"),
                    stderr_b.decode(errors="replace")
                    + f"\n[eval_runner] eval timed out after {timeout}s")
    except asyncio.CancelledError:
        _killpg_quiet(proc)
        try:
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:
            pass
        raise


async def local_eval_async(task_dir: str, op_name: str,
                           kernel_file: str, test_file: str, perf_file: str,
                           timeout: int, device_id: int = 0,
                           **_legacy) -> tuple[dict, dict]:
    """Async sibling of `local_eval`. Same args/return; subprocess
    spawned via asyncio so worker cancellation tears it down promptly."""
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_script = os.path.join(scripts_dir, "engine", "eval_kernel.py")
    abs_task = os.path.abspath(task_dir)
    env = _build_env(device_id)

    sidecar_path = os.path.join(abs_task, ".eval_result.json")
    try:
        os.remove(sidecar_path)
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable, eval_script,
        "--task-dir", abs_task,
        "--op-name", op_name,
        "--kernel-file", kernel_file,
        "--test-file", test_file,
        "--perf-file", perf_file,
        "--device-id", str(device_id),
        "--phases", "test,perf",
        "--timeout", str(timeout),
        "--output", sidecar_path,
    ]
    rc, stdout, stderr = await _run_subprocess_async(
        cmd, cwd=task_dir, env=env, timeout=timeout * 2 + 30)
    log = (stdout + ("\n" + stderr if stderr else "")).strip()

    payload = _read_sidecar(sidecar_path)
    return _assemble_response(payload, rc, log)
