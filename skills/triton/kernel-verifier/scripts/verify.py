#!/usr/bin/env python3
"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

多 shape 模式下：每个 shape 独立 try/except，全部跑完后落盘 verify_result.json。
策略 A：passed < total 即整体判失败（exit 1），同时失败清单记录在 JSON 的 `failures` 字段。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>]
"""
import argparse
import gc
import json
import os
import sys
import subprocess
import traceback


ERROR_MSG_LIMIT = 2000


def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs):
    """输入列表的结构化描述（用于 JSON）。"""
    try:
        import torch
    except Exception:
        torch = None
    descs = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


def cleanup_npu_memory():
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception:
        pass
    gc.collect()


def get_limit(data_type):
    """根据数据类型获取精度阈值"""
    import torch
    if data_type == torch.float16:
        return 0.004
    elif data_type == torch.bfloat16:
        return 0.03
    elif data_type == torch.int8:
        return 0.01
    else:
        return 0.02


def resolve_input_provider(torch_module):
    """解析任务文件的输入提供方式。"""
    if hasattr(torch_module, "get_input_groups"):
        groups = torch_module.get_input_groups()
        return groups, len(groups)
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()], 1
    else:
        raise AttributeError(
            f"模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def compare(fw_out, impl_out, limit, data_type):
    """对比框架输出和实现输出"""
    import torch
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    abs_diff = torch.abs(fw_finite.float() - impl_finite.float())
    abs_ref = torch.abs(fw_finite.float())
    eps = 1e-8
    relative_error = torch.where(abs_ref > eps, abs_diff / abs_ref, abs_diff)

    err_cnt = (relative_error > limit).sum().item()
    limit_cnt = int(finite_count * limit)

    if err_cnt > limit_cnt:
        max_error = relative_error.max().item()
        mean_error = relative_error.mean().item()
        mismatch_mask = relative_error > limit
        mismatch_indices = torch.where(mismatch_mask)[0]
        num_to_show = min(10, len(mismatch_indices))

        error_msg = (
            f"验证失败，输出不一致(误差数/最大容忍误差数): "
            f"err_cnt={err_cnt} / {limit_cnt}, dtype={data_type}, limit={limit}\n"
        )
        error_msg += f"最大相对误差: {max_error:.6e}, 平均相对误差: {mean_error:.6e}\n"
        error_msg += f"前 {num_to_show} 个不一致的值:\n"
        for i in range(num_to_show):
            idx = mismatch_indices[i].item()
            error_msg += (
                f"  位置[{idx}]: framework={fw_finite[idx]:.6e}, "
                f"impl={impl_finite[idx]:.6e}, "
                f"相对误差={relative_error[idx]:.6e}\n"
            )
        raise AssertionError(error_msg)


def run_single_case(
    framework_model,
    impl_model,
    inputs,
    device,
    case_idx,
    total_cases
):
    """验证单组输入。失败时抛出 AssertionError。"""
    import torch

    print(f"  测试第 {case_idx}/{total_cases} 组输入...", file=sys.stderr)

    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        impl_output = impl_model(*inputs_for_impl)
        framework_output = framework_model(*inputs_for_framework)

    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: "
                f"framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                data_type = fw_out.dtype
                limit = get_limit(data_type)
                compare(fw_out, impl_out, limit, data_type)
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e


def verify_implementations(op_name, verify_dir, triton_impl_name="triton_ascend_impl", output_path=None):
    """验证框架实现和生成实现的结果一致性。

    每个 shape 独立 try/except，全部跑完后写 verify_result.json。

    Returns:
        (passed_cases, total_cases)
    """
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, verify_dir)

    torch_module = __import__(f"{op_name}_torch")
    impl_module = __import__(f"{op_name}_{triton_impl_name}")

    FrameworkModel = torch_module.Model
    ModelNew = impl_module.ModelNew
    get_init_inputs = torch_module.get_init_inputs

    input_groups, total_cases = resolve_input_provider(torch_module)

    device = torch.device("npu")

    failures = []
    passed_cases = 0

    for case_idx, inputs in enumerate(input_groups, start=1):
        input_desc = describe_input(inputs)
        framework_model = None
        impl_model = None
        try:
            init_params = get_init_inputs()
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            framework_model = FrameworkModel(*init_params).to(device)

            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            impl_model = ModelNew(*init_params).to(device)

            run_single_case(
                framework_model, impl_model, inputs, device, case_idx, total_cases
            )
            passed_cases += 1
        except Exception as e:
            err_detail = traceback.format_exc()
            print(f"  [用例 {case_idx}/{total_cases}] 失败: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append({
                "case_idx": case_idx,
                "input_desc": input_desc,
                "error_type": type(e).__name__,
                "error_msg": truncate_error(err_detail),
            })
        finally:
            del framework_model
            del impl_model
            cleanup_npu_memory()

    failed_cases = total_cases - passed_cases

    # 落盘 verify_result.json
    if output_path is None:
        output_path = os.path.join(verify_dir, "verify_result.json")
    result = {
        "op_name": op_name,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "failures": failures,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"验证结果已保存到: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"警告: 无法写入 verify_result.json: {e}", file=sys.stderr)

    if failed_cases == 0:
        print(f"验证成功：共 {total_cases} 组测试用例全部通过")
    else:
        print(
            f"验证失败：{passed_cases}/{total_cases} 组通过，"
            f"{failed_cases} 组失败（详见 {output_path}）",
            file=sys.stderr,
        )

    return passed_cases, total_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}_torch.py 和 {op_name}_triton_ascend_impl.py（默认当前目录）",
    )
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（默认 900）")
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--output", default=None,
        help="验证结果 JSON 输出路径（默认 {verify_dir}/verify_result.json）",
    )
    parser.add_argument(
        "--_run", action="store_true",
        help=argparse.SUPPRESS,  # 内部参数：子进程模式，直接执行验证
    )
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    if args._run:
        # 子进程模式：直接执行验证逻辑
        try:
            passed, total = verify_implementations(
                args.op_name, verify_dir, args.triton_impl_name, args.output
            )
        except Exception as e:
            print(f"{e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        # 策略 A：passed < total → exit 1
        sys.exit(0 if passed == total and total > 0 else 1)
    else:
        # 主进程模式：启动子进程执行验证，超时后 kill 整个进程树
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--op_name", args.op_name,
            "--verify_dir", verify_dir,
            "--triton_impl_name", args.triton_impl_name,
            "--_run",
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(timeout=args.timeout)

            sys.stdout.buffer.write(stdout)
            sys.stdout.buffer.flush()
            sys.stderr.buffer.write(stderr)
            sys.stderr.buffer.flush()
            sys.exit(proc.returncode)

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"验证超时（{args.timeout}秒），已终止子进程", file=sys.stderr)
            sys.exit(1)
