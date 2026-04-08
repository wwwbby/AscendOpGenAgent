# Core Partition 分核优化模式

## 概述

在 Triton NPU kernel 中，**分核策略直接影响硬件利用率和性能**。NPU设备有多个AI Core，选择合适的核数、分核维度和任务分配方式是性能优化的关键。

## 触发条件

**当 Triton 代码中存在以下情况时，应考虑优化分核策略**：

1. **发射核数不合理**：grid 大小与数据规模不匹配（过多或过少）,npu 设备的物理核数一般为40或48，当grid 大小远大于物理核时或者远小于物理核时将会使得性能极大地弱化。
2. **tiling大小不合理**：在For循环中调用Vector计算单元时，运算的tile数据量远小于当前设备的UB大小（通常是192KB），导致无法充分使用算力单元。

## 优化方法

### 直接固定发射核数等于设备核数

#### 原始代码（发射核数过多）

```python
@triton.jit
def gather_dim1_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # 1 block per batch row
    pid_k = tl.program_id(1)  # 1 block per K-tile
    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k_off < K
    idx = tl.load(idx_ptr + pid_b * stride_ib + k_off * stride_ik, mask=mask)  # [BLOCK_K]
    x_val = tl.load(x_ptr + pid_b * stride_xb + idx * stride_xc, mask=mask)
    tl.store(out_ptr + pid_b * stride_ob + k_off * stride_ok, x_val, mask=mask)

# 调用
B = 128  # batch dim
K = 64  

BLOCK_B = 4
BLOCK_K = 128

grid = (B, triton.cdiv(K, BLOCK_K))

gather_dim1_kernel[grid](
    x, idx, out,
    x.stride(0), x.stride(1),
    idx.stride(0), idx.stride(1),
    out.stride(0), out.stride(1),
    B, K,
    BLOCK_B=BLOCK_B,
    BLOCK_K=BLOCK_K,
)
```

#### 优化后代码（合理核数）

```python
@triton.jit
def gather_dim1_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # 1 block per batch row
-   # 原始实现
-   pid_k = tl.program_id(1)  # 1 block per K-tile

-   k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
-   mask = k_off < K

-   idx = tl.load(idx_ptr + pid_b * stride_ib + k_off * stride_ik, mask=mask)  # [BLOCK_K]

-   x_val = tl.load(x_ptr + pid_b * stride_xb + idx * stride_xc, mask=mask)

-   tl.store(out_ptr + pid_b * stride_ob + k_off * stride_ok, x_val, mask=mask)

+   # 优化后实现使用向量化处理，一次处理一整个BLOCK_B，因此这里的得到的是一个向量
+   b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
+   b_mask = b_idx < B # 需要判断是否越界

+   # 对 K 维进行循环，向量化处理BLOCK_B * BLOCK_K个数据
+   for k_start in range(0, K, BLOCK_K):
+       ks = tl.arange(0, BLOCK_K)
+       k_mask = ks < K - k_start

+       idx_off = (b_idx[:, None] * stride_ib +
+                  (k_start + ks)[None, :] * stride_ik)
+       col_idx = tl.load(idx_ptr + idx_off, mask=b_mask[:, None] & k_mask)

+       x_off = (b_idx[:, None] * stride_xb +
+                col_idx * stride_xc)
+       x_val = tl.load(x_ptr + x_off, mask=b_mask[:, None] & k_mask)

+       out_off = (b_idx[:, None] * stride_ob +
+                  (k_start + ks)[None, :] * stride_ok)
+       tl.store(out_ptr + out_off, x_val, mask=b_mask[:, None] & k_mask)

# 调用
B = 128  # batch dim
K = 64  

BLOCK_B = 4
BLOCK_K = 128

— # 原始grid较大，每个核心处理BLOCK_K个数据，分核数=B*K/BLOCK_K
- grid = (B, triton.cdiv(K, BLOCK_K))
+ # 优化后grid变小，每个核心处理BLOCK_B*K个数据，分核数=B/BLOCK_B，内部展开循环处理BLOCK_K个数据
+ grid = (triton.cdiv(B, BLOCK_B),)

gather_dim1_kernel[grid](
    x, idx, out,
    x.stride(0), x.stride(1),
    idx.stride(0), idx.stride(1),
    out.stride(0), out.stride(1),
    B, K,
    BLOCK_B=BLOCK_B,
    BLOCK_K=BLOCK_K,
)
```

### 使用Triton-Ascend autotune搜索最佳分核参数
Triton-Ascend autotune是一个Triton-Ascend提供的tiling超参数性能调优工具，遍历搜索空间，尝试不同参数组合，展示每个组合的运行耗时与最优组合。使用Triton-Ascend autotune需要遵循以下工作流程：
1. 识别出 triton kernel 中哪些 `tl.constexpr` 参数是自由可调的 tiling 参数，包括影响分核（split）和切块（tiling）大小的参数，这里的分核指的是影响 grid 大小，而切块指的是影响 tile 大小，即影响 `tl.load` 或是 `tl.make_block_ptr` 产生的数据大小。
2. 如果这些参数能从 `tl.program_id`、`tl.arange`、`tl.range/range`、`mask/bounds` 表达式中被唯一识别出来，就尝试使用自动生成tiling `configs=[]`
3. 如果 kernel 语义上适合自动 tiling，但 DSL 写法让 parser 解析不出来，就显式传 `hints`
4. 如果某些 tiling 参数不可自由调整，例如某 kernel dsl 写法要求 grid 第一维必须固定为 `batch_size` 大小，或者根本没有暴露出可调的 tiling 参数，此时建议直接手写 triton.Config。

#### 使用方法
@triton.autotune 入参列表
| 参数名       | 类型                             | 必填    | 说明                                                              |
| --------- | ------------------------------ | ----- | --------------------------------------------------------------- |
| `configs` | `list[Config]`                 | 否     | 用户自定义的调优配置列表，为空时自动生成                                            |
| `key`     | `dict[str, str]` / `list[str]` | **是** | 缓存键，指定哪些参数变化时需要重新调优。使用 `hints` 时必须用**字典形式**，如 `{"x": "n_rows"}` |
| `hints`   | `dict`                         | 否     | **Ascend 扩展参数**，用于显式指定轴与 tiling 参数的映射关系                         |

hints 字典内部字段
| 字段名               | 类型               | 必填    | 说明                                                     |
| ----------------- | ---------------- | ----- | ------------------------------------------------------ |
| `split_params`    | `dict[str, str]` | **是** | 分核参数映射，如 `{"x": "BLOCK_M"}` 表示沿 `x` 轴切 program         |
| `tiling_params`   | `dict[str, str]` | **是** | 切块参数映射，如 `{"y": "BLOCK_N"}` 表示沿 `y` 轴切 block           |
| `low_dim_axes`    | `list[str]`      | **是** | 低维轴列表，如 `["y"]`，用于优化 tiling 效果                         |
| `reduction_axes`  | `list[str]`      | **是** | 规约轴列表，如 `[]`，用于优化 tiling 效果                            |
| `auto_gen_config` | `bool`           | 否     | 是否自动生成 tiling 配置，默认 `True`；当 `configs` 非空时默认变为 `False` |

有三种使用方法，以下是这三种方法的使用模板

##### 自动解析模板
详情见 Step 2：判断是否能用 configs=[] 自动生成 tiling
```python
@triton.autotune(
    configs=[],
    key=["n_rows"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)
```
##### 显式给 hints
详情见 Step 3：自动解析出错时，显式传 hints
```python
@triton.autotune(
    configs=[],
    key={"x": "n_rows", "y": "n_cols"},
    hints={
        "split_params": {"x": "BLOCK_M"},
        "tiling_params": {"y": "BLOCK_N"},
        "low_dim_axes": ["y"],
        "reduction_axes": [],
    },
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ...
```

##### 完全手写 triton.Config
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "multibuffer": True}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "multibuffer": True}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "multibuffer": False}),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ...
```

#### 注意事项

1. `@triton.autotune` 需要直接包在 `@triton.jit` 外层，示例如下：
```python
@triton.autotune(...)
@triton.jit
def kernel(...):
    ...
```

2. 自动生成 tiling 功能主要面向 vector kernel：当前 Triton-Ascend 这套自动解析/自动生成 tiling 面向的是 vector 类 kernel，且该 kernel 中所有分核和分块参数均可调。Cube 类算子目前还不支持自动 tiling 生成。

#### 详细步骤
##### Step 1：先识别哪些参数真的是可调参数
###### 1.1 先看“哪些 tl.constexpr 没有在 launch 时显式传入”
Triton-Ascend autotuner 在自动解析 split/tiling 参数时，首先会看 kernel 调用时 哪些参数没有传入，把这些“缺省的参数”当成候选项。

简单理解：

* Tensor 参数不可能是自动解析候选项；
* 普通运行时 shape 参数（如 n_rows、n_cols）通常属于 `key`；
* 真正的候选项通常是没有在 launch 处显式传值的 `tl.constexpr`；
* 如果某个 tl.constexpr 已经在 launch 时手动写死，它就不会再被当成自动解析候选项。
例如：
```python
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, # BLOCK_N是自动解析候选项
):
    ...

# BLOCK_M 已显式传入，不再是自动解析候选项
kernel[grid](x, y, n_rows, BLOCK_M=128)
```

###### 1.2 如何识别 split 参数

split 参数控制“一个 program 负责多大的一块数据”，它最常见的写法特征是：
1. 和 `tl.program_id(...)` 有直接关系；
2. 参与构造 block 起始位置；
3. 最后能通过 mask/bounds 表达式对应回某个 shape 轴。
例如：
```python
# 一维切分
pid = tl.program_id(0)
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)# 可以知道BLOCK_M 是 split 参数
mask_m = offs_m < n_rows

# 二维切分
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]# 可以知道BLOCK_M 是 split 参数
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]# 可以知道BLOCK_N 是 split 参数
mask_m = offs_m < n_rows
mask_n = offs_n < n_cols
```

###### 1.3 如何识别 tiling 参数
tiling 参数控制“在一个大的 split block 内，再按多大的子块去迭代”，它最常见的写法特征是：

1. 出现在 `tl.arange(0, PARAM)` 中；
2. 同时还出现在 for 循环的步长或循环次数推导中；
3. 最后能通过 mask/bounds 对应回某个轴长度参数。

例如：
```python
# 典型形态 1：步长是 tiling 参数
for k0 in tl.range(0, BLOCK_K, BLOCK_K_SUB):# 可以知道BLOCK_K_SUB 是 tiling 参数
    offs_k = k0 + tl.arange(0, BLOCK_K_SUB)# 可以知道BLOCK_K_SUB 是 tiling 参数
    mask_k = offs_k < k_size

# 典型形态 2：先计算循环次数
num_k_tiles = (k_size + BLOCK_K_SUB - 1) // BLOCK_K_SUB# 可以知道BLOCK_K_SUB 是 tiling 参数
for tile_id in range(num_k_tiles):
    offs_k = tile_id * BLOCK_K_SUB + tl.arange(0, BLOCK_K_SUB)# 可以知道BLOCK_K_SUB 是 tiling 参数
    mask_k = offs_k < k_size
```

##### Step 2：判断是否能用 configs=[] 自动生成 tiling

当你已经找到了候选参数后，可以按下面的检查表判断。
###### 2.1 可以优先尝试 configs=[] 的情况
一般同时满足下面几条时，`configs=[]` 成功率比较高：

* split 参数能从 `tl.program_id` 路径判断出来；
* tiling 参数能从 `tl.arange + for(range/tl.range)` 路径判断出来；
* 每个轴都有比较清晰的 mask/bounds 表达式，例如：
    * `offs < n`
    * `offs < min(block_end, n)`
* `key` 能和运行时 shape 参数一一对应；

###### 2.2 不适合直接用 configs=[] 的常见信号

下面这些情况，直接走自动 tiling 生成可能会出现解析失败的情况：

* 没有和轴长度直接绑定的 mask/bounds
* 某个参数必须覆盖完整语义维度
    * 例如 `BLOCK_SIZE >= hidden_dim`
* grid 某一维被业务语义固定，不允许自由切块
* 一个参数同时影响两个轴，或者同时影响“核数 + tile 形状”
* kernel 没暴露出可调 `tl.constexpr`

###### 2.3 如果出现错误建议打开调试日志
排查问题时，建议先启动环境变量debug：
```
export TRITON_PRINT_AUTOTUNING=1
```

日志中可以直接看到：

* 识别出的 split axes；
* 识别出的 tiling axes；
* 识别出的 low-dimensional axes；
* 识别出的 reduction axes；
* 生成的 config 数量。
小 shape 算子如果 benchmark 抖动大，也可以按需开启：
```
export TRITON_BENCH_METHOD=npu
```
这会测试的时间更准确，但 autotune 时间也会明显变长。

##### Step 3：自动解析出错时，显式传 hints

如果你已经确认 自动 tiling 适用于该 triton kernel，只是因为 DSL 写法不能够被当前 parser 识别，这时可以尝试显式传 `hints`。
`hints` 是 Triton-Ascend 在 `autotune` 装饰器中新增的一个参数，类型为 `dict`，用于给 Triton-Ascend autotune 提供该 triton kernel 的一些关键信息，帮助 autotune 更好的生成 tiling 配置。

###### 3.1 什么情况下可以考虑 hints

推荐显式传 `hints` 的场景：

* 你能人工判断哪个参数属于 `split`，哪个属于 `tiling`；
* 你知道每个参数对应轴的长度参数；

###### 3.2 hints 参数说明
* 当前 `hints` 参数中可以识别的字段有：
    * `split_params`: dict[str, str]，分核参数的映射关系，例如 `{"x": "BLOCK_M"}` 表示 `BLOCK_M` 是沿 `x` 轴切 program
    * `tiling_params`：dict[str, str]，切块参数的映射关系，例如 `{"y": "BLOCK_N"}` 表示 `BLOCK_N` 是沿 `y` 轴切 block
    * `low_dim_axes`：list[str]，低维轴的列表，例如 `["y"]` 表示 `y` 轴是低维轴
    * `reduction_axes`：list[str]，规约轴的列表，例如 `[]` 表示没有规约轴
    * `auto_gen_config`：bool，是否自动生成 tiling 配置，默认值为 `True`
* 注意：
    * 通过 `hints` 来显示指定轴关系时，autotune 中原本的参数 `key` 必须改为字典形式传入，因为后续 `split_params`、`tiling_params` 等参数都是按轴名来填写，需要和 `key` 里的轴名对应起来
    * 通过 `hints` 来显示指定轴关系时，`split_params`、`tiling_params`、`low_dim_axes`、`reduction_axes` 必须传入，即使某些参数为空
    * 合法的轴名称是 `x/y/z/w/v/t`，仅仅用做关系映射
    * `split_params` 和 `tiling_params` 为自动生成 tiling 算法必须的输入，`low_dim_axes` 和 `reduction_axes` 为 tiling 算法的可选输入，用于优化 tiling 效果，留空时 tiling 也能够自动生成，但可能会影响生成的候选 tiling 数量和质量
    * 当用户传入的 configs 不为空时，`auto_gen_config` 默认值为 `False`，如果希望此时也希望自动生成 tiling 配置并与用户传入的 configs 合并，需要显式在 `hints` 中传如入 `"auto_gen_config": True`

使用示例：
```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime

@triton.autotune(
    # configs 为空列表，表示不传入自定义配置
    # 此时 auto_gen_config 默认为 True，会自动生成 tiling 配置
    configs=[],
    
    # key 使用字典形式，轴名必须与 hints 中的轴名对应
    # "x" 对应 n_rows（行数），"y" 对应 n_cols（列数）
    # autotune 会根据这些维度值来缓存和选择最佳配置
    key={"x": "n_rows", "y": "n_cols"},
    
    # hints 参数：显式指定轴与 tiling 参数的映射关系
    hints={
        # split_params: 分核参数映射，指定沿哪个轴切分 program（任务）
        # "x": "BLOCK_M" 表示 BLOCK_M 沿 x 轴切分，即按行方向分核
        # 每个 program 处理 BLOCK_M 行数据
        "split_params": {"x": "BLOCK_M"},
        
        # tiling_params: 切块参数映射，指定沿哪个轴切分 block（数据块）
        # "y": "BLOCK_N" 表示 BLOCK_N 沿 y 轴切分，即按列方向切块
        # 每行数据在列方向上被切分为 BLOCK_N 大小的块，通过 for 循环处理
        "tiling_params": {"y": "BLOCK_N"},
        
        # low_dim_axes: 低维轴列表，用于优化 tiling 效果
        # ["y"] 表示 y 轴（列方向）是低维轴，访问连续性更好，适合作为内层循环
        "low_dim_axes": ["y"],
        
        # reduction_axes: 规约轴列表，本 kernel 无规约操作（如 sum/max 等）
        # 为空列表表示没有规约轴
        "reduction_axes": [],
        
        # auto_gen_config: 默认为 True，表示自动生成 tiling 配置
        # 由于 configs 为空，此处使用默认值 True 即可，无需显式传入
    },
)
@triton.jit
def kernel_with_hints(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    for n0 in range(0, n_cols, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)[None, :]
        mask_m = offs_m < n_rows
        mask_n = offs_n < n_cols
        mask = mask_m & mask_n

        x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
        tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

##### Step 4：手写 triton.Config
如果 Step 2 判断该 triton kernel 不适合或者无法使用自动 tiling 生成，那么可以使用社区 triton autotune 的基本功功能：手写一组 `triton.Config` 传入参数 `configs` 中。
######  手写 triton.Config 的总体原则
1. 对于影响 grid 发射核数的参数，一般我们尽量让其能够等于物理核数，如果数据量较小，也可能发射较少核数的时候能获得最优性能；对于影响 tile 块大小的参数，我们尽量在不产生 UB overflow 的情况下让其尽可能大，同时避免尾块的产生
2. 影响 grid 发射核数的参数：可以按照总长度从高到低设置为 X, X/2, X/4 等等的值，如果输入 shape 较大，可以设置为让 grid 发射核数正好等于物理核数的大小，例如 (X + num_cores - 1) // num_cores；这里是以一个切分轴为例，如果存在多个切分轴，那么就需要按照乘积来计算
3. 影响 tile 块大小的参数：起始值为切分轴参数（如果存在）或者轴长度，注意当该轴长度特别大的时候，我们可以直接从 16384 这样一个经验值开始取，然后按照 X / 2, X / 4 这样去取值
4. 上述按 2 的幂次方下降的值采样较为粗粒度，如果用户想要得到极致的最优性能，尤其是在输入大小不规则的情况下，需要在可能的最优区间内细粒度撒点，可以通过粗粒度采样后确认性能最优的大致区间后再进一步细分来实现。
5. 对于 vector 类算子，在设置了上述 tiling 大小的配置候选集后，可以加上 multibuffer 编译选项的调优。

示例：
```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


def get_configs():
    return [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "multibuffer": MB})
        for BM in [256, 128, 64, 32]
        for BN in [128, 64, 32, 16]
        for MB in [True, False]
    ]


@triton.autotune(
    configs=get_configs(),
    key=["n_rows", "n_cols"],
)
@triton.jit
def manual_config_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < n_rows) & (offs_n < n_cols)

    x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
    tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

##### 常见失败速查
下面这张表可以直接用来决定你下一步该怎么做。

| 现象                              | 更可能的原因                       | 建议动作                      |
| ------------------------------- | ---------------------------- | ------------------------- |
| configs=[] 直接解析失败              | split/tiling 轴没有从 DSL 唯一识别出来 | 先补 hints，再试             |
| parser 能识别一部分，但总差一个参数           | 某个参数没有和轴长度 mask 建立联系         | 改 DSL 写法或改手写 config      |
| kernel 完全没有合适的 tl.constexpr 可调项 | DSL 没暴露调参接口                  | 先改 kernel dsl，再谈 autotune |
| 自动生成能跑，但候选质量明显差                 | 当前算法不适合该 kernel 的参数耦合方式      | 手动构造 config 传入      |


## 性能收益

- **核数优化**：可提升 2-5x 性能

## 注意事项

1. **UB 容量约束**：确保 tile_size * dtype_size * buffers <= 192KB
2. **原子操作开销**：原子操作有额外开销，核数过多时可能成为瓶颈
