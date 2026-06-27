#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch (new workflow).

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - kernel.py (editable seed; written from the user's --kernel file)
  - <test_file> (pytest-style correctness script; user-supplied)
  - <perf_file> (perf script printing triton/cann timing; user-supplied)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

The new workflow drops the ref.py contract. Instead the user supplies
three files: a kernel under optimization, a pytest test script, and a
perf script that prints `triton:  median=X.XXms` + `cann:    median=X.XXms`.

Usage:
    python scripts/scaffold.py --kernel kernel.py --test test_op.py --perf perf_op.py --op-name my_op --devices <DEV>

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import yaml


from utils.settings import (  # noqa: E402
    default_max_rounds, default_eval_timeout, default_metric,
    default_code_checker_enabled,
)
from task_config import TEST_FILE_DEFAULT, PERF_FILE_DEFAULT  # noqa: E402


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    kernel_code: str,
    test_code: str,
    perf_code: str,
    op_name: str,
    desc: str = "",
    arch: str = "",
    devices: list | None = None,
    max_rounds: int | None = None,
    eval_timeout: int | None = None,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    test_filename: str | None = None,
    perf_filename: str | None = None,
    code_checker_enabled: bool | None = None,
    kernel_source_path: str | None = None,
    worker_url: str = "",
) -> str:
    """Create task directory with all files. Returns absolute path.

    New workflow: writes kernel + test + perf (no ref). The test/perf
    filenames are derived from the user's source filenames if provided,
    else fall back to TEST_FILE_DEFAULT / PERF_FILE_DEFAULT.
    """
    if max_rounds is None:
        max_rounds = default_max_rounds()
    if eval_timeout is None:
        eval_timeout = default_eval_timeout()
    if code_checker_enabled is None:
        code_checker_enabled = default_code_checker_enabled()
    if test_filename is None:
        test_filename = TEST_FILE_DEFAULT
    if perf_filename is None:
        perf_filename = PERF_FILE_DEFAULT

    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write kernel + test + perf from the user's files.
    _write(task_dir, editable_filename, kernel_code)
    _write(task_dir, test_filename, test_code)
    _write(task_dir, perf_filename, perf_code)

    # Sibling data files (e.g. .json shape lists, .pt caches) that the
    # kernel/test/perf scripts may read at runtime. Scan the kernel's
    # source directory for files matching the kernel's stem prefix.
    discovered_data_files: list[str] = []
    if kernel_source_path:
        try:
            import shutil as _shutil
            src_dir = os.path.dirname(os.path.abspath(kernel_source_path))
            kernel_stem = os.path.splitext(
                os.path.basename(kernel_source_path))[0]
            for fname in sorted(os.listdir(src_dir)):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".json", ".pt", ".npz"):
                    continue
                stem, _ = os.path.splitext(fname)
                # Match files whose stem is the kernel stem or starts
                # with "<kernel_stem>_". This avoids copying every
                # neighbouring op's data file.
                if stem != kernel_stem and not stem.startswith(kernel_stem + "_"):
                    continue
                src = os.path.join(src_dir, fname)
                if not os.path.isfile(src):
                    continue
                _shutil.copy(src, os.path.join(task_dir, fname))
                discovered_data_files.append(fname)
        except Exception as _e:
            print(f"[scaffold] WARNING: sidecar data file copy failed: {_e}",
                  file=sys.stderr)

    # Generate task.yaml. The new workflow puts test_file + perf_file
    # under the `eval` block (where loader expects them).
    eval_block = {
        "timeout": eval_timeout,
        "test_file": test_filename,
        "perf_file": perf_filename,
    }

    task_yaml = {
        "name": op_name,
        "description": desc or f"Optimize {op_name}",
        "arch": arch or None,
        "editable_files": [editable_filename],
        "eval": eval_block,
        "metric": {
            "primary": default_metric()["primary"],
            "lower_is_better": default_metric()["lower_is_better"],
            "improvement_threshold": default_metric()["improvement_threshold"],
        },
        "agent": {
            "max_rounds": max_rounds,
        },
    }
    if devices:
        task_yaml["devices"] = list(devices)
    if worker_url:
        worker_urls = [u.strip() for u in worker_url.split(",") if u.strip()]
        task_yaml["worker"] = {"urls": worker_urls}
    if discovered_data_files:
        task_yaml["data_files"] = discovered_data_files

    task_yaml["code_checker"] = {"enabled": bool(code_checker_enabled)}

    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write(task_dir, "task.yaml", yaml_content)

    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _write(task_dir: str, rel_path: str, content: str):
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit.

    The actual commit goes through git_utils.commit_in_task — same code
    path hooks use for round commits, so reliability is consistent.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.git_utils import commit_in_task

    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True, check=True)
    ok, info = commit_in_task(task_dir, ["."], "scaffold: baseline")
    if not ok:
        raise RuntimeError(f"scaffold baseline commit failed: {info}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    """Construct scaffold's argparse, with no side effects.

    Extracted out of main() so parse_args.py can reuse the exact same flag
    spec without duplicating it. Single source of truth for which flags
    /autoresearch accepts and how they're typed/defaulted.
    """
    parser = argparse.ArgumentParser(
        description="Scaffold a task directory for Claude Code autoresearch",
    )
    parser.add_argument("--kernel", required=True,
                        help="Path to seed kernel file (editable)")
    parser.add_argument("--test", required=True,
                        help="Path to pytest-style test script")
    parser.add_argument("--perf", required=True,
                        help="Path to perf script (prints triton/cann timing)")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (required)")
    # The repo is locked to triton_ascend on Ascend NPU + PyTorch by
    # construction. arch is derived from the picked --devices via npu-smi.
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Required.")
    parser.add_argument("--max-rounds", type=int, default=default_max_rounds())
    parser.add_argument("--eval-timeout", type=int, default=default_eval_timeout())
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    # Single flag, store_const so the absence of --no-code-checker yields
    # None (lets defaults.code_checker_enabled in config.yaml decide) and
    # presence yields False (pinned into task.yaml as enabled: false).
    parser.add_argument("--no-code-checker", dest="code_checker",
                        action="store_const", const=False, default=None,
                        help=("Disable the static Triton regression check "
                              "(validate_triton_impl) for this task. "
                              "Useful when the regression rules are too "
                              "strict for the chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    parser.add_argument("--worker-url", default="",
                        help="Remote worker URL(s) (host:port, comma-separated). "
                             "Routes eval through the remote HTTP worker "
                             "instead of local npu-smi.")
    return parser


def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.hw_detect import derive_arch

    # Hardware resolution: --devices is required unless --worker-url routes
    # eval to a remote machine. The repo is locked to triton_ascend / torch
    # / ascend by construction — those constants live in TaskConfig defaults
    # / generated templates, not on `args`.
    devices_list: list = []
    args.arch = None

    has_remote = bool(args.worker_url and args.worker_url.strip())

    if not args.devices and not has_remote:
        print(json.dumps({"status": "error",
                          "error": "--devices (local eval) or "
                                   "--worker-url (remote worker) is required."}))
        sys.exit(1)

    if args.devices:
        devices_list = [int(d.strip()) for d in args.devices.split(",")
                        if d.strip()]
    if not devices_list:
        # Remote-only: default to device 0 (the worker owns the real NPU).
        devices_list = [0]

    if not has_remote:
        args.arch = derive_arch(devices_list[0])
        if not args.arch:
            print(json.dumps({"status": "error",
                              "error": (f"could not derive arch from "
                                        f"device {devices_list[0]} "
                                        f"(is npu-smi on PATH?)")}))
            sys.exit(1)

    if not args.op_name:
        print(json.dumps({"status": "error",
                          "error": "--op-name is required"}))
        sys.exit(1)

    # Read kernel file
    if not os.path.isfile(args.kernel):
        print(json.dumps({"status": "error",
                          "error": f"Kernel file not found: {args.kernel}"}))
        sys.exit(1)
    with open(args.kernel, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # Read test file
    if not os.path.isfile(args.test):
        print(json.dumps({"status": "error",
                          "error": f"Test file not found: {args.test}"}))
        sys.exit(1)
    with open(args.test, "r", encoding="utf-8") as f:
        test_code = f.read()

    # Read perf file
    if not os.path.isfile(args.perf):
        print(json.dumps({"status": "error",
                          "error": f"Perf file not found: {args.perf}"}))
        sys.exit(1)
    with open(args.perf, "r", encoding="utf-8") as f:
        perf_code = f.read()

    # Preserve the user's test/perf filenames inside task_dir so the
    # scripts' own `from <kernel_module> import ...` statements resolve.
    test_filename = os.path.basename(args.test)
    perf_filename = os.path.basename(args.perf)

    print(f"[scaffold] Creating task directory for {args.op_name}...", file=sys.stderr)

    task_dir = scaffold_task_dir(
        kernel_code=kernel_code,
        test_code=test_code,
        perf_code=perf_code,
        op_name=args.op_name,
        devices=devices_list,
        arch=args.arch,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        test_filename=test_filename,
        perf_filename=perf_filename,
        code_checker_enabled=args.code_checker,
        kernel_source_path=args.kernel,
        worker_url=args.worker_url,
    )

    print(f"[scaffold] Task directory created: {task_dir}", file=sys.stderr)
    print(f"[scaffold] Files:", file=sys.stderr)
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}", file=sys.stderr)

    # Write per-op pointer so batch/run.py picks the exact dir we just
    # made, not whichever <op>_* in ar_tasks/ happens to have the freshest
    # mtime (which races with concurrent runs and stale prior task_dirs).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase_machine import write_task_dir_pointer
    write_task_dir_pointer(args.op_name, task_dir)

    # Reference validation is now a single path through baseline.py: the
    # generated verify routine splits ref-side and kernel-side try/excepts
    # and tags error_source on failure. Scaffold reads the resulting
    # baseline exit code and decides:
    #   - exit 4 (INFRA_FAIL) → reject task (operator must fix --ref or env)
    #   - any other non-zero → kernel-side failure, task activates and
    #     hook routes to PLAN
    # AST symbol presence was already checked earlier (validate_ref on
    # the source --ref file before copying), so import errors / missing
    # symbols never reach this point.
    if args.run_baseline:
        print(f"[scaffold] Running baseline eval...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_cmd = [sys.executable,
                        os.path.join(script_dir, "engine", "baseline.py"),
                        task_dir]
        rc = subprocess.run(baseline_cmd).returncode
        # baseline exit codes are binary now (workflow.baseline._EXIT_FOR):
        #   0 = task activatable (OK or KERNEL_FAIL — hook routes to PLAN)
        #   4 = task NOT activatable (INFRA_FAIL — operator must intervene)
        # Anything else here is an unexpected baseline crash.
        if rc == 4:
            from phase_machine import task_summary  # noqa: E402
            summary = task_summary(task_dir) or {}
            err_source = summary.get("baseline_error_source")
            if err_source == "infra":
                hint = ("The test or perf script is broken (import / "
                        "collection error, missing file, or internal "
                        "crash). Fix the SOURCE script and re-run "
                        "/autoresearch from scratch. The task directory "
                        "is left for inspection but MUST NOT be activated "
                        "— test/perf scripts are not editable.")
            else:
                hint = ("INFRA_FAIL: no valid base timing — the perf "
                        "script didn't produce a `cann: median=X.XXms` "
                        "line. Fix env (device / eval.timeout / worker / "
                        "OOM) and re-run `/autoresearch --resume "
                        "<task_dir>`. Phase stays at BASELINE.")
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": ("eval pipeline broken during baseline — see "
                          "[baseline]/[eval] stderr above"),
                "hint": hint,
            }))
            sys.exit(4)
        if rc != 0:
            # Unexpected baseline crash (not the 0/4 we know about).
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": (f"baseline crashed unexpectedly (exit {rc}); "
                          f"see [baseline]/[eval] stderr above"),
                "hint": ("This is not a classified outcome. Inspect the "
                         "baseline / eval stderr above and file a bug if "
                         "the exit code isn't in _EXIT_FOR."),
            }))
            sys.exit(rc)

    # Output
    # Surface baseline_outcome so callers can distinguish OK from
    # KERNEL_FAIL without rereading state. Both are activatable
    # (status=ok, rc=0); the difference is whether the seed kernel
    # produced valid timings or the first PLAN cycle has to rewrite it.
    # When --run-baseline wasn't passed, summary is None (no state.json
    # yet) → outcome stays None and the caller knows it's an
    # un-baselined task.
    from phase_machine import task_summary  # noqa: E402
    summary = task_summary(task_dir) or {}
    outcome = summary.get("baseline_outcome")
    print(json.dumps({"task_dir": task_dir, "status": "ok",
                      "baseline_outcome": outcome}))


if __name__ == "__main__":
    main()
