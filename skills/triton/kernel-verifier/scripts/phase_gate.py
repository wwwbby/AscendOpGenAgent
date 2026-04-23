#!/usr/bin/env python3
"""阶段门控 — 在 Agent 进入下一阶段前，检查验证结果的签名和内容。

设计目的：
  1. 独立于大模型运行，作为阶段转换的确定性守卫
  2. 验证结果文件的 HMAC 签名，确保未被大模型篡改
  3. 检查验证结果的 passed 字段，只有真正通过才放行

用法:
  python phase_gate.py check \\
      --result_file <验证结果 JSON 文件路径> \\
      --required_step <期望的步骤名: ast_check|verify|benchmark>

  python phase_gate.py batch_check \\
      --result_files <文件1> <文件2> ... \\
      --required_steps <步骤1> <步骤2> ...

退出码:
  0 = 门控通过（验证结果有效且 passed == true）
  1 = 门控拒绝（验证失败或签名无效）
"""
import argparse
import hashlib
import json
import sys

SIGNING_KEY = "ascend_op_gen_verify_hook_v1"


def verify_signature(signed_data):
    """验证结果文件的 HMAC 签名。"""
    if "result" not in signed_data or "signature" not in signed_data:
        return False, "缺少 result 或 signature 字段"

    result_dict = signed_data["result"]
    claimed_signature = signed_data["signature"]

    payload = json.dumps(result_dict, sort_keys=True, ensure_ascii=False)
    expected_signature = hashlib.sha256(
        (payload + SIGNING_KEY).encode("utf-8")
    ).hexdigest()

    if claimed_signature != expected_signature:
        return False, "签名不匹配，结果可能被篡改"

    return True, "签名验证通过"


def check_single_gate(result_file, required_step):
    """检查单个验证结果文件。"""
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            signed_data = json.load(f)
    except FileNotFoundError:
        return False, f"结果文件不存在: {result_file}"
    except json.JSONDecodeError as e:
        return False, f"结果文件 JSON 解析失败: {e}"

    sig_valid, sig_msg = verify_signature(signed_data)
    if not sig_valid:
        return False, f"签名验证失败: {sig_msg}"

    result = signed_data["result"]

    step = result.get("step", "")
    if step != required_step:
        return False, f"步骤不匹配: 期望 '{required_step}'，实际 '{step}'"

    passed = result.get("passed", False)
    if not passed:
        error = result.get("error", "未知错误")
        exit_code = result.get("exit_code", -1)
        return False, f"验证未通过: step={step}, exit_code={exit_code}, error={error}"

    return True, f"门控通过: step={step}"


def main():
    parser = argparse.ArgumentParser(description="阶段门控 — 检查验证结果")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check 子命令
    check_parser = subparsers.add_parser("check", help="检查单个验证结果")
    check_parser.add_argument("--result_file", required=True, help="验证结果 JSON 文件路径")
    check_parser.add_argument("--required_step", required=True,
                               choices=["ast_check", "verify", "benchmark"],
                               help="期望的步骤名")

    # batch_check 子命令
    batch_parser = subparsers.add_parser("batch_check", help="批量检查验证结果")
    batch_parser.add_argument("--result_files", nargs="+", required=True,
                               help="验证结果 JSON 文件路径列表")
    batch_parser.add_argument("--required_steps", nargs="+", required=True,
                               help="期望的步骤名列表（与 result_files 一一对应）")

    args = parser.parse_args()

    if args.command == "check":
        passed, message = check_single_gate(args.result_file, args.required_step)
        output = {"passed": passed, "message": message, "result_file": args.result_file}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        sys.exit(0 if passed else 1)

    elif args.command == "batch_check":
        if len(args.result_files) != len(args.required_steps):
            print(json.dumps({
                "passed": False,
                "message": f"文件数({len(args.result_files)})与步骤数({len(args.required_steps)})不匹配"
            }, ensure_ascii=False, indent=2))
            sys.exit(1)

        all_passed = True
        results = []
        for rf, step in zip(args.result_files, args.required_steps):
            passed, message = check_single_gate(rf, step)
            results.append({
                "result_file": rf,
                "required_step": step,
                "passed": passed,
                "message": message,
            })
            if not passed:
                all_passed = False

        output = {"passed": all_passed, "checks": results}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
