#!/usr/bin/env python3
"""验证钩子 — 封装验证脚本调用，输出带签名的结构化 JSON 结果。

设计目的：
  1. 将验证过程从大模型编排中解耦，大模型只需调用此脚本并读取结果
  2. 输出带 HMAC 签名的 JSON，防止大模型篡改验证结果
  3. 门控脚本 (phase_gate.py) 可验证签名，确保结果未被篡改

用法:
  python verify_hook.py verify \\
      --op_name <算子名> --verify_dir <验证目录> \\
      --output <结果输出路径> [--timeout 900] \\
      [--triton_impl_name triton_ascend_impl]

  python verify_hook.py ast_check \\
      --generated_code <代码文件路径> \\
      --output <结果输出路径>

  python verify_hook.py benchmark \\
      --op_name <算子名> --verify_dir <验证目录> \\
      --output <结果输出路径> \\
      [--warmup 5] [--repeats 50] \\
      [--triton_impl_name triton_ascend_impl] \\
      [--skip_framework] [--framework_latency_ms 0.0]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

SIGNING_KEY = "ascend_op_gen_verify_hook_v1"


def sign_result(result_dict):
    """对结果字典生成 HMAC 签名，防止篡改。"""
    payload = json.dumps(result_dict, sort_keys=True, ensure_ascii=False)
    signature = hashlib.sha256((payload + SIGNING_KEY).encode("utf-8")).hexdigest()
    return signature


def write_signed_result(result_dict, output_path):
    """写入带签名的结果文件。"""
    signature = sign_result(result_dict)
    signed = {"result": result_dict, "signature": signature}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(signed, f, ensure_ascii=False, indent=2)
    return signed


def run_ast_check(args):
    """执行 AST 退化预检查。"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    validate_script = os.path.join(scripts_dir, "validate_triton_impl.py")

    if not os.path.exists(validate_script):
        result = {
            "step": "ast_check",
            "passed": False,
            "error": f"验证脚本不存在: {validate_script}",
            "timestamp": time.time(),
        }
        write_signed_result(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(1)

    cmd = [sys.executable, validate_script, args.generated_code, "--json"]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        ast_result = {
            "step": "ast_check",
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timestamp": time.time(),
        }

        if proc.returncode == 0:
            ast_result["error"] = ""
        else:
            try:
                detail = json.loads(proc.stdout)
                ast_result["regression_type"] = detail.get("regression_type")
                ast_result["suggestion"] = detail.get("suggestion", "")
                ast_result["checks"] = detail.get("checks", {})
            except json.JSONDecodeError:
                ast_result["error"] = proc.stderr or proc.stdout

    except subprocess.TimeoutExpired:
        ast_result = {
            "step": "ast_check",
            "passed": False,
            "exit_code": -1,
            "error": "AST 检查超时",
            "timestamp": time.time(),
        }
    except Exception as e:
        ast_result = {
            "step": "ast_check",
            "passed": False,
            "exit_code": -1,
            "error": str(e),
            "timestamp": time.time(),
        }

    signed = write_signed_result(ast_result, args.output)
    print(json.dumps(ast_result, ensure_ascii=False, indent=2))
    sys.exit(0 if ast_result["passed"] else 1)


def run_verify(args):
    """执行功能验证。"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    verify_script = os.path.join(scripts_dir, "verify.py")

    if not os.path.exists(verify_script):
        result = {
            "step": "verify",
            "passed": False,
            "error": f"验证脚本不存在: {verify_script}",
            "timestamp": time.time(),
        }
        write_signed_result(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(1)

    cmd = [
        sys.executable, verify_script,
        "--op_name", args.op_name,
        "--verify_dir", args.verify_dir,
        "--triton_impl_name", args.triton_impl_name,
        "--timeout", str(args.timeout),
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout + 30
        )
        verify_result = {
            "step": "verify",
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0 and "验证成功" in proc.stdout,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timestamp": time.time(),
        }

        if verify_result["passed"]:
            verify_result["error"] = ""
        else:
            verify_result["error"] = proc.stderr or proc.stdout

    except subprocess.TimeoutExpired:
        verify_result = {
            "step": "verify",
            "passed": False,
            "exit_code": -1,
            "error": f"验证超时（{args.timeout}秒）",
            "timestamp": time.time(),
        }
    except Exception as e:
        verify_result = {
            "step": "verify",
            "passed": False,
            "exit_code": -1,
            "error": str(e),
            "timestamp": time.time(),
        }

    signed = write_signed_result(verify_result, args.output)
    print(json.dumps(verify_result, ensure_ascii=False, indent=2))
    sys.exit(0 if verify_result["passed"] else 1)


def run_benchmark(args):
    """执行性能测试。"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(scripts_dir, "benchmark.py")

    if not os.path.exists(benchmark_script):
        result = {
            "step": "benchmark",
            "passed": False,
            "error": f"性能测试脚本不存在: {benchmark_script}",
            "timestamp": time.time(),
        }
        write_signed_result(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(1)

    cmd = [
        sys.executable, benchmark_script,
        "--op_name", args.op_name,
        "--verify_dir", args.verify_dir,
        "--triton_impl_name", args.triton_impl_name,
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--output", args.output,
    ]

    if args.skip_framework:
        cmd.append("--skip_framework")
        cmd.extend(["--framework_latency_ms", str(args.framework_latency_ms)])

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )

        benchmark_result = {
            "step": "benchmark",
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timestamp": time.time(),
        }

        if benchmark_result["passed"] and os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
            benchmark_result["perf_data"] = perf_data
            benchmark_result["error"] = ""
        else:
            benchmark_result["error"] = proc.stderr or proc.stdout

    except subprocess.TimeoutExpired:
        benchmark_result = {
            "step": "benchmark",
            "passed": False,
            "exit_code": -1,
            "error": "性能测试超时",
            "timestamp": time.time(),
        }
    except Exception as e:
        benchmark_result = {
            "step": "benchmark",
            "passed": False,
            "exit_code": -1,
            "error": str(e),
            "timestamp": time.time(),
        }

    signed = write_signed_result(benchmark_result, args.output)
    print(json.dumps(benchmark_result, ensure_ascii=False, indent=2))
    sys.exit(0 if benchmark_result["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description="验证钩子 — 封装验证脚本调用")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ast_check 子命令
    ast_parser = subparsers.add_parser("ast_check", help="AST 退化预检查")
    ast_parser.add_argument("--generated_code", required=True, help="生成代码文件路径")
    ast_parser.add_argument("--output", required=True, help="结果输出路径")

    # verify 子命令
    verify_parser = subparsers.add_parser("verify", help="功能验证")
    verify_parser.add_argument("--op_name", required=True, help="算子名称")
    verify_parser.add_argument("--verify_dir", required=True, help="验证目录路径")
    verify_parser.add_argument("--output", required=True, help="结果输出路径")
    verify_parser.add_argument("--timeout", type=int, default=900, help="超时秒数")
    verify_parser.add_argument("--triton_impl_name", default="triton_ascend_impl",
                               help="Triton 实现模块名")

    # benchmark 子命令
    bench_parser = subparsers.add_parser("benchmark", help="性能测试")
    bench_parser.add_argument("--op_name", required=True, help="算子名称")
    bench_parser.add_argument("--verify_dir", required=True, help="验证目录路径")
    bench_parser.add_argument("--output", required=True, help="结果输出路径")
    bench_parser.add_argument("--warmup", type=int, default=5, help="warmup 次数")
    bench_parser.add_argument("--repeats", type=int, default=50, help="正式测试次数")
    bench_parser.add_argument("--triton_impl_name", default="triton_ascend_impl",
                               help="Triton 实现模块名")
    bench_parser.add_argument("--skip_framework", action="store_true",
                               help="跳过 framework 性能测试")
    bench_parser.add_argument("--framework_latency_ms", type=float, default=0.0,
                               help="预设 framework 参考延迟")

    args = parser.parse_args()

    if args.command == "ast_check":
        run_ast_check(args)
    elif args.command == "verify":
        run_verify(args)
    elif args.command == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()
