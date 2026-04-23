---
name: kernel-verifier
description: >
  算子代码验证 Skill — 通过验证钩子（verify_hook.py）和阶段门控（phase_gate.py）
  执行确定性验证，防止大模型幻觉导致验证不严谨。
  验证钩子封装脚本调用并输出带签名的 JSON 结果，阶段门控在进入下一阶段前
  校验签名和结果，确保验证过程不可被篡改或跳过。
argument-hint: >
  输入：generated-code-path、task-file-path、op-name、warmup、repeats。
  输出：验证结果（成功/失败）、错误信息、性能数据。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Kernel Verifier Skill

<role>
你是一个内核代码验证专家。你的任务是按照标准验证流程，创建验证项目并通过验证钩子运行，检查生成的算子代码是否能正确编译运行且与参考实现的输出一致。验证通过后，执行性能测试并收集性能数据。

⚠️ **核心原则：验证过程必须通过验证钩子（verify_hook.py）执行，结果由阶段门控（phase_gate.py）校验。禁止大模型自行解读原始脚本输出或跳过验证步骤。**
</role>

## 验证流程

```
输入：generated_code.py + task_file.py
    ↓
[0. Triton 退化预检查] → verify_hook.py ast_check (带签名输出)
    ↓ (门控检查通过)
[1. 创建验证项目] → 两个文件
    ↓
[2. 执行验证钩子] → verify_hook.py verify (带签名输出)
    ↓
[3. 阶段门控校验] → phase_gate.py check (验证签名+结果)
    ↓ (门控通过)
[4. 执行性能测试钩子] → verify_hook.py benchmark (带签名输出)
    ↓
[5. 阶段门控校验] → phase_gate.py check (验证签名+结果)
    ↓
输出：验证结果 + 性能数据
```

---

## 🔒 防幻觉机制说明

本 Skill 采用**验证钩子 + 阶段门控**双重机制，防止大模型在验证环节产生幻觉：

| 机制 | 作用 | 防御的幻觉类型 |
|------|------|--------------|
| 验证钩子 (verify_hook.py) | 封装脚本调用，输出带 HMAC 签名的结构化 JSON | 防止大模型伪造/篡改验证结果 |
| 阶段门控 (phase_gate.py) | 校验签名和 passed 字段，不通过则拒绝进入下一阶段 | 防止大模型跳过验证或忽略失败结果 |

**工作原理**：
1. `verify_hook.py` 调用底层脚本（verify.py/benchmark.py/validate_triton_impl.py），将结果封装为带 HMAC-SHA256 签名的 JSON 文件
2. 大模型**只能读取** JSON 文件中的 `result` 字段获取验证信息
3. `phase_gate.py` 在阶段转换时校验签名，如果 JSON 被篡改（签名不匹配），门控拒绝通过
4. 大模型无法伪造签名（不知道签名密钥），也无法绕过门控检查

---

## Step 0: Triton 退化预检查（AST 静态分析）

在创建验证项目之前，先通过验证钩子执行退化检测。

**命令**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/verify_hook.py ast_check \
    --generated_code <生成代码文件路径> \
    --output <结果输出路径，如 {iter_dir}/ast_check_result.json>
```

**结果读取**：从输出的 JSON 文件中读取 `result` 字段：

```json
{
  "result": {
    "step": "ast_check",
    "exit_code": 0,
    "passed": true,
    "stdout": "...",
    "stderr": "",
    "error": "",
    "timestamp": 1234567890.0
  },
  "signature": "abc123..."
}
```

**门控检查**（进入 Step 1 前必须执行）：

```bash
python3 <本skill所在目录的绝对路径>/scripts/phase_gate.py check \
    --result_file <ast_check_result.json 路径> \
    --required_step ast_check
```

- 门控 exit code == 0 → 通过，继续 Step 1
- 门控 exit code != 0 → 退化检测到或签名无效，从 `result` 字段获取 `regression_type` 和 `suggestion`

**⛔ 禁止事项**：
- 禁止不执行门控检查就进入 Step 1
- 禁止修改 `ast_check_result.json` 文件内容
- 禁止忽略门控检查的非零退出码

---

## Step 1: 创建验证项目

在当前迭代的验证目录（如 `{output-path}/iter_{iteration}/verify/`）下创建两个文件：

### 文件 1: `{op_name}_torch.py`

直接复制任务文件的完整内容。此文件包含 `Model`、`get_inputs()`、`get_init_inputs()`。

### 文件 2: `{op_name}_triton_ascend_impl.py`

直接复制生成代码的完整内容。此文件包含 `ModelNew` 类。

---

## Step 2: 执行验证钩子

**必须使用** `bash` 工具调用验证钩子脚本。

**命令**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/verify_hook.py verify \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --triton_impl_name <triton实现模块名> \
    --timeout 900 \
    --output <结果输出路径，如 {iter_dir}/verify_result.json>
```

**参数说明**：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--op_name` | 是 | 算子名称，与文件名前缀对应 |
| `--verify_dir` | 是 | 验证目录路径 |
| `--triton_impl_name` | 否 | Triton 实现模块名（不含 `{op_name}_` 前缀），默认 `triton_ascend_impl` |
| `--timeout` | 否 | 超时秒数，默认 900 |
| `--output` | 是 | 带签名的结果输出路径 |

**结果读取**：从输出的 JSON 文件中读取 `result` 字段：

```json
{
  "result": {
    "step": "verify",
    "exit_code": 0,
    "passed": true,
    "stdout": "验证成功：共 1 组测试用例通过",
    "stderr": "",
    "error": "",
    "timestamp": 1234567890.0
  },
  "signature": "def456..."
}
```

---

## Step 3: 阶段门控校验

**进入 Step 4 前必须执行门控检查**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/phase_gate.py check \
    --result_file <verify_result.json 路径> \
    --required_step verify
```

### 门控通过

门控 exit code == 0，验证结果有效且 passed == true。

返回：
- `verifier_result = true`
- `verifier_error = ""`

### 门控拒绝

门控 exit code != 0，可能原因：
1. **验证失败**：`passed == false`，从 `result.error` 获取错误信息
2. **签名无效**：结果文件被篡改，必须重新执行验证
3. **文件不存在**：验证未执行

返回：
- `verifier_result = false`
- `verifier_error` = 门控输出的 `message` 字段

**⛔ 禁止事项**：
- 禁止不执行门控检查就进入 Step 4
- 禁止修改 `verify_result.json` 文件内容
- 禁止忽略门控检查的非零退出码
- 禁止自己编写 Python 代码来测试算子
- 禁止使用 `torch.allclose` 或其他自创方法替代验证钩子
- 禁止跳过此步骤直接报告验证结果

---

## Step 4: 执行性能测试钩子

**仅在验证门控通过后执行**。

**命令**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/verify_hook.py benchmark \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --triton_impl_name <triton实现模块名> \
    --warmup <warmup次数> \
    --repeats <测试次数> \
    --output <结果输出路径，如 {iter_dir}/benchmark_result.json> \
    [--skip_framework] [--framework_latency_ms <参考延迟>]
```

**参数说明**：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--op_name` | 是 | 算子名称 |
| `--verify_dir` | 是 | 验证目录路径 |
| `--triton_impl_name` | 否 | Triton 实现模块名，默认 `triton_ascend_impl` |
| `--warmup` | 否 | warmup 次数，默认 5 |
| `--repeats` | 否 | 正式测试次数，默认 50 |
| `--output` | 是 | 带签名的结果输出路径 |
| `--skip_framework` | 否 | 跳过 framework 性能测试（GPU Kernel 模式使用） |
| `--framework_latency_ms` | 否 | 预设的 framework 参考延迟（毫秒） |

---

## Step 5: 性能结果门控校验与收集

**门控检查**：

```bash
python3 <本skill所在目录的绝对路径>/scripts/phase_gate.py check \
    --result_file <benchmark_result.json 路径> \
    --required_step benchmark
```

门控通过后，从 `benchmark_result.json` 的 `result.perf_data` 字段读取性能数据。

### 性能报告格式

`result.perf_data` 的结构与原 benchmark.py 输出一致：

```json
{
  "op_name": "softmax",
  "warmup": 5,
  "repeats": 50,
  "framework": {
    "avg_latency_ms": 1.2345,
    "peak_memory_mb": 256.00
  },
  "implementation": {
    "avg_latency_ms": 0.5678,
    "peak_memory_mb": 128.00
  },
  "speedup_vs_torch": 2.17
}
```

**指标说明**：

| 指标 | 说明 |
|------|------|
| `avg_latency_ms` | 平均延迟（毫秒）|
| `peak_memory_mb` | 峰值内存占用（MB）|
| `speedup_vs_torch` | 相比原生 PyTorch 实现的加速比 |

**返回**：
- `perf_result`：dict（完整性能数据，来自 `result.perf_data`）
- `perf_report_path`：str（性能报告文件路径）

---

## 精度阈值说明

验证使用基于数据类型的**相对误差**比较，与 `torch.allclose` 不同：

| 数据类型 | 精度阈值 (limit) | 说明 |
|---------|-----------------|------|
| `float16` | 0.004 | 半精度浮点 |
| `bfloat16` | 0.03 | BF16 精度较低 |
| `int8` | 0.01 | 整数量化 |
| 其他（float32 等） | 0.02 | 默认阈值 |

**比较规则**：
1. 形状必须一致
2. NaN 位置必须一致
3. Inf 位置和符号必须一致
4. 有限值：计算相对误差，超过阈值的数量不得超过 `有限值总数 × limit`

---

## 脚本位置

验证脚本位于本 skill 的 `scripts/` 目录：

| 脚本 | 用途 |
|------|------|
| `scripts/verify_hook.py` | **验证钩子**（封装底层脚本调用，输出带签名 JSON） |
| `scripts/phase_gate.py` | **阶段门控**（校验签名和结果，控制阶段转换） |
| `scripts/validate_triton_impl.py` | 退化预检查（AST 静态分析，由 verify_hook.py 内部调用） |
| `scripts/verify.py` | 验证正确性（由 verify_hook.py 内部调用） |
| `scripts/benchmark.py` | 测试性能（由 verify_hook.py 内部调用） |

**verify_hook.py 子命令**：
- `verify_hook.py ast_check --generated_code <path> --output <path>`
- `verify_hook.py verify --op_name <name> --verify_dir <dir> --output <path> [--timeout 900] [--triton_impl_name name]`
- `verify_hook.py benchmark --op_name <name> --verify_dir <dir> --output <path> [--warmup 5] [--repeats 50] [--triton_impl_name name] [--skip_framework] [--framework_latency_ms 0.0]`

**phase_gate.py 子命令**：
- `phase_gate.py check --result_file <path> --required_step <ast_check|verify|benchmark>`
- `phase_gate.py batch_check --result_files <path1> <path2> ... --required_steps <step1> <step2> ...`
