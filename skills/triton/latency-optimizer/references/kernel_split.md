# Kernel 分裂优化

## 概述

当一个 kernel 在经过所有常规优化点（1-12）后，整体性能仍不达标（speedup\_vs\_torch < 0.8 倍 baseline）时，往往是因为某些特定 shape 模式下的性能极差拖累了整体平均。此时将 kernel 分裂为两个：一个通用 kernel 处理正常 shape，一个专用 kernel 针对性能瓶颈 shape 重新生成和优化。

## 触发条件

所有优化点 1-12 均已尝试完毕（无更多优化点命中），且最终 `speedup_vs_torch < 0.8`。

## 完整流程

```
while split_iteration < max_split_iterations:

    ── Step 1: 收集性能瓶颈 shape ──────────────────────
    从 perf_result.json 的 per_shape_results 中提取所有 shape 的 speedup_vs_torch。

    筛选条件：
      speedup_vs_torch < 0.8 的 shape → 瓶颈 shape 列表 bottleneck_shapes

    若 bottleneck_shapes 为空（所有 shape 均 ≥ 0.8 但几何平均 < 0.8）：
      放宽筛选条件为 speedup_vs_torch < 几何平均值 → 重新收集

    若仍为空 → 无法分裂，终止

    ── Step 2: 规律挖掘 ────────────────────────────────
    对 bottleneck_shapes 逐个分析其 shape 特征，尝试归纳共性规律。
    必须从以下维度逐一检查（详见下方「规律挖掘维度」）。

    对每个维度，总结出候选规律，例如：
      "reduce 轴为 axis=0 且 reduce 轴长度 > 4096 的 shape 性能差"
      "总元素量 < 1024 的 shape 性能差"

    ── Step 3: 规律验证 ────────────────────────────────
    对每个候选规律，统计验证：

    设 R 为某候选规律：
      R_shapes = 满足规律 R 的所有 shape
      non_R_shapes = 不满足规律 R 的所有 shape

    计算：
      R_avg_speedup = R_shapes 的 speedup_vs_torch 算术平均
      non_R_avg_speedup = non_R_shapes 的 speedup_vs_torch 算术平均

    判定条件（需同时满足）：
      1. R_avg_speedup < non_R_avg_speedup * 0.7
         （满足规律的 shape 性能显著差于不满足的，至少差 30%）
      2. len(R_shapes) >= 1
      3. len(non_R_shapes) >= 1

    通过验证 → 有效规律
    无有效规律 → 终止分裂

    若有多个有效规律，选择 R_avg_speedup 最低（性能最差）的那个作为本轮分裂依据。

    ── Step 4: Kernel 分裂 ─────────────────────────────
    根据选定的有效规律 R，将 kernel 分裂为两个：

    通用 kernel（kernel_general）：
      - 处理不满足规律 R 的 shape
      - 代码来源：当前最终代码，原样保留

    专用 kernel（kernel_specialized）：
      - 处理满足规律 R 的 shape
      - 需要从头重新生成，针对规律 R 描述的 shape 特征进行专门优化
      - 在 user_requirements 中追加规律描述，指导生成器针对该类 shape 特征
        调整 BLOCK_SIZE、grid 配置、内存访问模式等

    构建 dispatch wrapper：
      - 规律 R 的判定函数 matches_pattern(input_shapes) -> bool
      - forward() 函数：根据判定结果选择调用 kernel_general 或 kernel_specialized
      - 判定逻辑必须基于 shape 属性（维度、大小、stride 等），不得硬编码具体 shape 值

    ── Step 5: 验证分裂后代码 ──────────────────────────
    对 dispatch_wrapper.py 执行完整验证流程（功能验证 + 性能测试）

    ── Step 6: 分裂效果判定 ────────────────────────────
    split_speedup >= 0.8:
      → 分裂成功，终止

    split_speedup >= final_speedup * 1.05（有 5% 以上提升）:
      → 分裂有改善但未达标
      → 更新 final_speedup = split_speedup
      → 检查是否仍有 shape 的 speedup < 0.8
        是 → split_iteration++，continue
        否 → 终止

    split_speedup < final_speedup * 1.05:
      → 分裂无效，放弃本轮
      → split_iteration++，continue

达到 max_split_iterations → 以当前最佳结果终止
```

## 规律挖掘维度

对瓶颈 shape 进行分析时，必须从以下维度逐一检查，归纳共性规律。

### 维度 1: Shape 大小相关

- 总元素量（numel）是否过大（> 2^20）或过小（< 2^10）
- 某个维度是否特别大或特别小
- shape 是否高度不均衡（如 \[1, 4096] vs \[1024, 1024]）

**候选规律示例**：

- "总元素量 > 1048576 的 shape 性能差"
- "shape 最大维度 > 8192 的 shape 性能差"
- "shape 最小维度 = 1 且最大维度 > 4096 的 shape 性能差"

### 维度 2: 归约轴相关（适用于 reduction 类算子）

- reduce 的轴是哪一个（axis=0 vs axis=-1 vs 中间轴）
- reduce 轴的长度是否特别大或特别小
- 是否沿非连续内存方向做 reduce

**候选规律示例**：

- "reduce 轴为 axis=0 且轴长度 > 4096 的 shape 性能差"
- "沿非连续内存方向做 reduce 的 shape 性能差"
- "reduce 轴长度 < 64 的 shape 性能差"

### 维度 3: 内存布局相关

- 是否涉及广播（broadcast），广播的维度和倍数
- 是否涉及转置/permute 操作
- 输入 tensor 是否非连续（non-contiguous）
- stride 模式是否特殊（如间隔 stride、逆序 stride）

**候选规律示例**：

- "输入需要 permute 的 shape 性能差"
- "输入 tensor 非连续的 shape 性能差"
- "广播倍数 > 16 的 shape 性能差"

### 维度 4: 数据类型相关

- 是否涉及特定 dtype（如 float16 vs float32 vs bfloat16）
- 是否涉及类型转换（cast）

**候选规律示例**：

- "使用 bfloat16 的 shape 性能差"
- "涉及 float16 → float32 类型转换的 shape 性能差"

### 维度 5: 计算模式相关

- 是否涉及 mask 操作（如 causal mask、padding mask）
- 是否涉及动态 shape（shape 依赖运行时参数）
- 是否涉及条件分支（如 where 条件随 shape 变化）

**候选规律示例**：

- "涉及 causal mask 且序列长度 > 2048 的 shape 性能差"
- "涉及 padding mask 的 shape 性能差"

### 维度 6: 并行度相关

- 可用的并行 program 数量是否过少（如 shape 过小导致 grid 仅为 1-2）
- BLOCK\_SIZE 是否与 shape 不匹配（如 shape 远小于 BLOCK\_SIZE 导致大量 padding）

**候选规律示例**：

- "grid 大小 < 4 的 shape 性能差"
- "shape 最内层维度 < BLOCK\_SIZE/2 导致 padding 超过 50% 的 shape 性能差"

### 维度 7: 访存模式相关

- 是否跨步访问（strided access）严重
- 是否存在 bank conflict 风险的访问模式
- 是否内存对齐不佳

**候选规律示例**：

- "stride 不为 2 的幂次的 shape 性能差"
- "跨步访问步长 > 256 的 shape 性能差"

### 维度 8: Shape 对齐与整除性相关

- shape 各维度是否为 2 的幂次（power-of-two）
- shape 是否为 BLOCK\_SIZE 的整数倍（非整除时 padding 开销大）
- 是否存在奇数维度（影响向量化对齐）
- shape 各维度是否有公共因子（影响分块策略）

**候选规律示例**：

- "shape 非 BLOCK\_SIZE 整数倍导致 padding 开销大的 shape 性能差"
- "存在奇数维度的 shape 性能差"
- "shape 各维度无公共因子的 shape 性能差"

### 维度 9: 输入输出元素比例相关

- reduction 类算子的压缩比（输入元素数 / 输出元素数）
- 是否为高压缩比（reduce 长轴，输出远小于输入）或低压缩比
- broadcast 类算子的扩展比（输出元素数 / 实际计算元素数）
- 是否存在大量冗余计算（broadcast 导致的有效计算密度低）

**候选规律示例**：

- "输入输出元素比 > 1024（高压缩比 reduce）的 shape 性能差"
- "broadcast 扩展比 > 32 的 shape 性能差"
- "有效计算密度 < 0.1 的 shape 性能差"

### 维度 10: 维度数与结构相关

- tensor 的维度数（2D vs 3D vs 4D vs 5D+）
- 是否存在 size-1 维度（squeeze/unsqueeze 需求）
- 维度排列是否为标准顺序（如 NCHW vs NHWC）
- 是否为低维但大 shape（如 \[1, 1000000]）或高维但小 shape（如 \[2,2,2,2,2]）

**候选规律示例**：

- "维度数 >= 4 且存在 size-1 维度的 shape 性能差"
- "5D 及以上 tensor 的 shape 性能差"
- "低维但单维度 > 100000 的 shape 性能差"

### 维度 11: 多输入关系相关

- 多个输入 tensor 的 shape 是否完全相同
- 多个输入之间是否存在倍数关系（如 \[1, 512] 和 \[256, 512]）
- 是否需要隐式扩展（implicit broadcasting）
- 输入之间的 stride 模式是否兼容

**候选规律示例**：

- "多输入 shape 不一致需隐式广播的 shape 性能差"
- "输入之间 stride 模式不兼容的 shape 性能差"
- "输入 shape 存在倍数 > 8 的关系的 shape 性能差"

### 维度 12: 计算密度相关

- 算术强度（FLOPs / bytes accessed）是否过低（memory-bound）或过高（compute-bound）
- 每个元素的计算操作数是否随 shape 变化
- 是否涉及计算密集型操作（exp/log/sin/cos/sqrt 等）
- 是否为纯元素级操作（无数据复用）vs 有数据复用的操作（conv/gemm）

**候选规律示例**：

- "算术强度 < 1（memory-bound）的 shape 性能差"
- "涉及 exp/log 等计算密集型操作且 shape 较大的 shape 性能差"
- "纯元素级操作无数据复用的 shape 性能差"

### 维度 13: 边界与阈值相关

- shape 是否刚好超过 shared memory / L2 缓存容量（导致频繁换入换出）
- 单个 block 处理的数据量是否刚好超过硬件资源限制
- shape 是否处于某些性能关键区间（如 \[256, 256] 附近、\[1024, 1024] 附近）
- 总数据量是否接近 NPU 内存带宽饱和点

**候选规律示例**：

- "总数据量 > 4MB（超过 L2 缓存）的 shape 性能差"
- "shape 处于 \[256, 256] 附近的性能关键区间的 shape 性能差"

### 维度 14: 输出特征相关

- 输出是否为标量（全 reduce）或低维 tensor
- 输出元素量与输入元素量的比值
- 输出 tensor 是否需要动态分配大量内存
- 输出是否涉及非标准 layout（如 channel-last）

**候选规律示例**：

- "输出为标量（全 reduce）的 shape 性能差"
- "输出元素量 < 输入元素量 \* 0.001 的 shape 性能差"

### 维度 15: 算子组合模式相关

- 是否涉及多个算子的组合（如 reduce + broadcast + elementwise）
- 组合算子之间是否存在中间结果的重复计算
- 是否可以通过 kernel fusion 减少中间结果的 global memory 读写

**候选规律示例**：

- "涉及 reduce + broadcast 组合的 shape 性能差"
- "组合算子中间结果需要大量 global memory 写回的 shape 性能差"

### 维度 16: 数据局部性相关

- 同一数据是否被多次访问（working set 是否超出 cache）
- 是否存在时间局部性差的数据访问模式
- 是否存在空间局部性差的数据访问模式（如跳跃式访问）

**候选规律示例**：

- "working set > L2 缓存大小导致数据被反复换入换出的 shape 性能差"
- "同一数据被访问次数 > 4 且 working set 较大的 shape 性能差"

### 维度 17: 顺序依赖相关

- 是否有序列依赖（如 cumulative sum、scan 操作、prefix sum）
- 顺序依赖的长度是否过长
- 是否可以并行化顺序依赖的计算

**候选规律示例**：

- "涉及 cumulative 操作且操作轴长度 > 4096 的 shape 性能差"
- "存在长距离序列依赖的 shape 性能差"

### 维度 18: 条件计算密度相关

- 是否有大量条件分支导致实际计算量远小于名义计算量
- 条件分支的比例是否随 shape 变化
- 是否存在 warp divergence 导致的效率下降

**候选规律示例**：

- "条件分支导致实际计算量 < 名义计算量 \* 0.3 的 shape 性能差"
- "mask 覆盖率 < 30% 导致大量无效计算的 shape 性能差"

### 维度 19: 输入输出重叠相关

- 输入和输出 tensor 是否有内存重叠（in-place 操作）
- 重叠模式是否影响并行化
- 是否需要额外的同步操作

**候选规律示例**：

- "输入输出内存重叠的 shape 性能差"
- "in-place 操作且 shape 较大的 shape 性能差"

### 维度 20: Kernel 调度开销相关

- shape 过小导致 kernel launch 开销占比过高
- 是否存在大量小 shape 导致的调度开销累积
- 单次 kernel 执行时间是否过短（< 调度开销）

**候选规律示例**：

- "单次 kernel 执行时间 < 10us 的 shape 性能差"
- "shape 总元素量 < 256 的 shape 性能差"

## 规律验证方法

对每个候选规律，按以下步骤验证：

### 1. 统计验证

```
R_shapes = 满足规律 R 的所有 shape
non_R_shapes = 不满足规律 R 的所有 shape

R_avg_speedup = mean([s.speedup_vs_torch for s in R_shapes])
non_R_avg_speedup = mean([s.speedup_vs_torch for s in non_R_shapes])

判定条件（需同时满足）：
  1. R_avg_speedup < non_R_avg_speedup * 0.7
  2. len(R_shapes) >= 1
  3. len(non_R_shapes) >= 1
```

### 2. 交叉验证（可选，当 shape 数量较多时）

将 shape 随机分为两组，分别在两组上验证规律是否一致。
如果规律在两组上均成立，则可信度更高。

### 3. 规律选择

若有多个有效规律，选择 R\_avg\_speedup 最低（性能最差）的那个作为本轮分裂依据。
这样可以优先解决最严重的性能瓶颈。

## Dispatch Wrapper 模板

```python
import torch
import triton
import triton_ascend


def matches_pattern(*args, **kwargs):
    """
    根据输入 shape 特征判断是否应走专用 kernel。
    返回 True → 使用 kernel_specialized
    返回 False → 使用 kernel_general

    判定逻辑必须基于 shape 属性（维度、大小、stride 等），
    不得硬编码具体 shape 值。
    """
    # 示例：reduce 轴为 axis=0 且轴长度 > 4096
    # input_tensor = args[0]
    # if len(input_tensor.shape) >= 2 and input_tensor.shape[0] > 4096:
    #     return True
    return False


def forward(*args, **kwargs):
    if matches_pattern(*args, **kwargs):
        return kernel_specialized.forward(*args, **kwargs)
    else:
        return kernel_general.forward(*args, **kwargs)
```

## Dispatch Wrapper 完整 Demo

以 Softmax 算子为例：优化点 1-12 穷尽后 speedup\_vs\_torch = 0.52，规律挖掘发现 **reduce 轴长度 > 4096 时 avg speedup = 0.25，≤ 4096 时 avg speedup = 0.85**，验证通过（0.25 < 0.85 × 0.7）。分裂为通用 kernel + 专用 kernel。

```python
import torch
import torch.nn as nn
import triton
import triton_ascend
import tl.triton as tl


# 通用 kernel — 原 kernel 原样保留
@triton.jit
def softmax_kernel_general(input_ptr, output_ptr, M, N,
                           stride_im, stride_in, stride_om, stride_on,
                           BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = input_ptr + pid * stride_im
    out_row_start = output_ptr + pid * stride_om

    row_max = float('-inf')
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        mask = n_offs < N
        x = tl.load(row_start + n_offs * stride_in, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = 0.0
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        mask = n_offs < N
        x = tl.load(row_start + n_offs * stride_in, mask=mask, other=float('-inf'))
        x = tl.exp(x - row_max)
        row_sum += tl.sum(x, axis=0)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        mask = n_offs < N
        x = tl.load(row_start + n_offs * stride_in, mask=mask, other=float('-inf'))
        tl.store(out_row_start + n_offs * stride_on, tl.exp(x - row_max) / row_sum, mask=mask)


# 专用 kernel — 针对大 N 重新生成（更大 BLOCK_SIZE + 分核策略）
@triton.jit
def softmax_kernel_specialized(input_ptr, output_ptr, M, N,
                               stride_im, stride_in, stride_om, stride_on,
                               num_cores: tl.constexpr, SUB_M: tl.constexpr,
                               BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)

    for task_idx in range(pid, tl.cdiv(M, SUB_M), num_cores):
        for row in range(task_idx * SUB_M, tl.minimum((task_idx + 1) * SUB_M, M)):
            row_base = input_ptr + row * stride_im
            out_base = output_ptr + row * stride_om

            row_max = float('-inf')
            for n_start in range(0, N, BLOCK_N):
                n_offs = n_start + offs_n
                mask = n_offs < N
                row_max = tl.maximum(row_max, tl.max(
                    tl.load(row_base + n_offs * stride_in, mask=mask, other=float('-inf')), axis=0))

            row_sum = 0.0
            for n_start in range(0, N, BLOCK_N):
                n_offs = n_start + offs_n
                mask = n_offs < N
                x = tl.load(row_base + n_offs * stride_in, mask=mask, other=float('-inf'))
                row_sum += tl.sum(tl.exp(x - row_max), axis=0)

            for n_start in range(0, N, BLOCK_N):
                n_offs = n_start + offs_n
                mask = n_offs < N
                x = tl.load(row_base + n_offs * stride_in, mask=mask, other=float('-inf'))
                tl.store(out_base + n_offs * stride_on, tl.exp(x - row_max) / row_sum, mask=mask)


# Dispatch Wrapper — 规律判定 + 分发
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def matches_pattern(self, x: torch.Tensor) -> bool:
        return x.shape[-1] > 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        M, N = x_c.shape[0], x_c.shape[-1]
        output = torch.empty_like(x_c)

        if self.matches_pattern(x_c):
            SUB_M = min(512, M)
            grid = (min(triton.cdiv(M, SUB_M), self.VEC_CORE_NUM),)
            softmax_kernel_specialized[grid](
                x_c, output, M, N,
                x_c.stride(0), x_c.stride(1), output.stride(0), output.stride(1),
                num_cores=self.VEC_CORE_NUM, SUB_M=SUB_M, BLOCK_N=2048)
        else:
            grid = (M,)
            softmax_kernel_general[grid](
                x_c, output, M, N,
                x_c.stride(0), x_c.stride(1), output.stride(0), output.stride(1),
                BLOCK_N=1024)

        return output
```

### 关键点

| 要素                           | 说明                                                      |
| ------------------------------ | --------------------------------------------------------- |
| `matches_pattern`              | 基于 shape 属性判断，**不硬编码**具体 shape 值            |
| `if self.matches_pattern(x_c)` | forward 分发入口，True → 专用 kernel，False → 通用 kernel |
| 通用 kernel                    | 原 kernel 原样保留，`BLOCK_N=1024`，`grid=(M,)`           |
| 专用 kernel                    | 针对瓶颈 shape 重新生成，`BLOCK_N=2048`，分核策略 `SUB_M` |

### 常见规律的 matches\_pattern 写法

```python
# 规律：reduce 轴长度 > 阈值
def matches_pattern(self, x: torch.Tensor) -> bool:
    return x.shape[-1] > 4096

# 规律：总元素量过小
def matches_pattern(self, x: torch.Tensor) -> bool:
    return x.numel() < 1024

# 规律：输入非连续
def matches_pattern(self, x: torch.Tensor) -> bool:
    return not x.is_contiguous()

# 规律：高压缩比 reduce（输入/输出元素比 > 阈值）
def matches_pattern(self, x: torch.Tensor, dim: int = -1) -> bool:
    output_elements = 1
    for i, s in enumerate(x.shape):
        if i != (dim % x.dim()):
            output_elements *= s
    return x.numel() / max(output_elements, 1) > 1024

# 规律：多输入 shape 不一致需广播
def matches_pattern(self, a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape != b.shape
```

## 专用 Kernel 生成指导

生成专用 kernel 时，在 user\_requirements 中追加以下信息：

```
本 kernel 仅用于满足以下规律的 shape：{规律 R 的自然语言描述}。
请针对此规律进行优化设计，例如：
1. 调整 BLOCK_SIZE 以适配该类 shape 特征
2. 调整 grid 配置以充分利用硬件并行度
3. 调整内存访问模式以适配该类 shape 的访存特征
4. 针对特定轴/维度进行专门的 tiling 策略
5. 针对特定数据类型进行专门的计算优化
6. 针对特定计算模式进行专门的算法优化
```

## 终止条件

满足以下任一条件时，终止 Kernel 分裂流程：

1. 分裂后 speedup >= 0.8 → 分裂成功
2. 无法找到有效规律（所有候选规律均未通过验证）→ 无法分裂
3. 分裂后性能无明显提升（< 5%）→ 分裂无效
4. 达到最大分裂迭代次数（3 次）→ 以当前最佳结果终止
5. 瓶颈 shape 列表为空 → 无法分裂

## 注意事项

1. **判定函数必须基于 shape 属性**：matches\_pattern 函数中的判定逻辑必须基于
   shape 的通用属性（维度、大小、stride 等），不得硬编码具体的 shape 值。
   这确保了 dispatch wrapper 对未见过的 shape 也能正确分发。
2. **专用 kernel 需要从头生成**：专用 kernel 不是在通用 kernel 基础上修改，
   而是针对特定规律从头生成，确保可以采用完全不同的算法策略。
3. **每轮分裂只选择一个规律**：每轮分裂只基于一个最优规律（R\_avg\_speedup 最低），
   避免同时分裂过多导致 kernel 碎片化。
4. **规律验证的严格性**：R\_avg\_speedup < non\_R\_avg\_speedup \* 0.7 的条件确保
   只有真正显著的性能差异才会触发分裂，避免因微小差异导致不必要的分裂。
5. **分裂后仍需完整验证**：分裂后的 dispatch\_wrapper.py 必须通过完整的功能
   验证和性能测试，确保分裂没有引入精度问题。
6. **多轮分裂的累积效果**：每轮分裂后，如果仍有 shape 性能不达标，可以继续
   分裂。但每轮分裂的规律必须基于当前最新的性能数据，而非历史数据。

