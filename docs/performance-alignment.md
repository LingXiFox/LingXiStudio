# Runtime 性能对齐报告：LingXiStudio vs MFLUX

> 历史记录：本文的 LoRA 是烘焙进权重的。Phase 1.5 已移除烘焙，改为动态 LoRA，见 [phase-1.5-report.md](phase-1.5-report.md)。

日期：2026-10-17 · 设备：MacBook Air M5 / 32 GB（无风扇）· macOS 27.2 · MLX 0.32.3（两边同版本）

## 结论

| 指标 | MFLUX 0.22.0 | LingXiStudio（优化前） | LingXiStudio（优化后） |
|---|---|---|---|
| 稳态去噪 s/step（步 2–6） | 6.02 / 6.086（均值 **6.053**） | 6.364 | 5.584 / 5.584 / 5.632（均值 **5.600**） |
| 相对 MFLUX 速度 | 100% | 95.1% | **108.1%** |
| 首步（含编译/预热） | 10.03 / 9.89 s | 6.36 s | 5.52–5.60 s |
| 峰值 MLX 内存 | 30.68 GB | 14.91 GB | 14.91 GB |
| 峰值进程 footprint | 27.38 GB | 25.57 GB | **17.80 GB** |
| 单次运行 Swap 变化 | +4.9 GB / +3.5 GB | +1.8 GB | **无增长** |

目标「稳态去噪性能达到 MFLUX 的 95%～100%」已达成且超出。精度、步数、采样方式都没有降低，也没有启用任何有损缓存。

## 测试配置（两边严格一致）

- 模型：`~/Models/Qwen-Image-2.1-BF16`，BF16 原始权重，不量化
- LoRA：`Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors`，scale 1.0，烘焙进权重（两边都是 bake，单步计算量与原版 40 步模型相同）
- 调度：viggle_turbo 6 节点 `[1, .9375, .875, .75, .5, .25]`，叠加分辨率时间偏移，不做 terminal 拉伸
- 1024×1024，seed 42，guidance 1.0（不开 CFG），同一提示词
- 每次运行前冷却 150 秒（无风扇机身的热状态会显著影响单步耗时：此前一次连续 40 步运行，单步从 6.3 s 降频到 9.5 s）
- 脚本：`Scripts/benchmark.sh <lingxi|mflux> <out-dir> [cooldown]`。LingXi 的逐步耗时由运行时直接打印（每步都 `eval` 同步）；MFLUX 的逐步耗时由 tqdm 累计值 `n × s/it` 差分得到

## 差异分析

| 方面 | MFLUX | LingXi 优化前 | 结论 |
|---|---|---|---|
| 矩阵乘法 | MLX `matmul` BF16 | 同 | 一致，是公共的耗时下限（每步约 57 TFLOP） |
| Attention | `mx.fast.scaled_dot_product_attention`，无 mask，K/V = [文本前缀缓存 \| 图像] | 同 | 一致 |
| 文本前缀 KV 缓存 | 每个 prompt 只算一次 | 同 | 一致 |
| Q/K RMSNorm + RoPE | 自定义 Metal kernel，单次读写 | 组合算子：fp32 转换、norm、拆分实/虚部、旋转、stack、回转，每个 64 MB 级张量要读写约 6 次 | **差异 1** |
| 计算图调度 | `mx.compile` 整步，逐元素算子融合 | eager，LayerNorm·(1+scale)、tanh 门控、残差、SiLU·proj 各自独立成 kernel | **差异 2** |
| 内存驻留 | 文本编码器全程驻留（峰值 30.7 GB，32 GB 机器上持续 Swap） | 分阶段加载/释放 | LingXi 占优 |
| VAE 解码 | — | Winograd 卷积的临时缓冲被 MLX 缓存留住，footprint 冲到 23.8 GB | **差异 3**（内存，不影响去噪速度） |

## 优化项与实测

每项都单独用开关做 A/B（`LINGXI_COMPILE=0`、`LINGXI_FUSED_QK=0`），条件与冷却时间相同：

| 配置 | 稳态 s/step | 相对优化前 | 出图 vs 优化前 |
|---|---|---|---|
| 优化前 | 6.364 | — | — |
| 只开 `compile` 整步 | 5.832 | −8.4% | 逐位一致（PSNR ∞） |
| 只开融合 QK-Norm+RoPE kernel | 5.922 | −6.9% | PSNR 31.33 dB，SSIM 0.9883 |
| 两项都开（默认） | 5.584 | **−12.3%** | PSNR 31.33 dB，SSIM 0.9883 |

1. **整步 `compile`**（`Transformer.swift`）：整个图像步（32 个 block + norm_out）只 trace 一次，MLX 会融合其中的逐元素算子。分辨率或提示词长度变化时，`compile` 会自动重新 trace。数值与 eager 逐位一致。
2. **融合 Q/K RMSNorm + RoPE Metal kernel**（`FusedKernels.swift`）：每个 `(token, head)` 行分配一个 threadgroup，每个线程负责一个 `(2i, 2i+1)` 对，全程在 fp32 寄存器里计算，写回时只舍入一次。和 MFLUX 的 kernel 思路相同，用的是 MLX 自带的 `MLXFast.metalKernel`，没有引入新依赖。
   - 隔离单测（`qwen-image debug-kernel`）：与 fp32 组合算子相比，最大误差 1.0 bf16 ulp，即只差一次舍入
   - 对 diffusers BF16 参考的单步 DiT：余弦 0.99983（优化前 0.99993）。差异来自参考实现在 norm 之后多做的一次 bf16 舍入，不是精度下降：kernel 的中间结果比参考更精确
   - 整图回归：与优化前 PSNR 31.3 dB、SSIM 0.988，肉眼无差别（构图、文字「QWEN IMAGE 2.1」、反光均一致）
3. **VAE 解码内存**（`VAE.swift`）：解码期间把 MLX 缓存上限临时设为 256 MB（可用 `LINGXI_VAE_CACHE_MB` 调），并在每个 resnet 之后 `eval`。仅解码的单测：footprint 从 25.5 GB 降到 14.0 GB，输出逐位一致；整条流水线中解码阶段从 23.8 GB 降到 12.7 GB，耗时约 +0.3 s。
   - 试过但没有采用：用 `MLX_CONV_WINOGRAD_WORKING_SET` 限制 Winograd 工作集。降到 1 GB 时 MLX 会改用其他卷积算法，数值发生变化（最大误差 0.0035）且更慢（7.6 s 对 5.1 s），不符合「不改变计算」的要求。

## 质量回归

- 优化后三次运行（`both` / `both2` / `final`）的出图两两逐位一致：运行是确定性的
- MFLUX 两次运行彼此逐位一致
- LingXi 与 MFLUX 的出图不同（PSNR 13.4 dB），构图也不同，这是**数值实现差异，不是配置差异**：
  - MFLUX 文生图路径（`Qwen21TextEncoder`，`text_mode=True`）取的是文本编码器**经过最终 RMSNorm** 的输出；diffusers 参考 pipeline 明确要求取 norm **之前**的最后一层输出，并注释说取错会先在文字渲染上出问题。LingXi 按参考实现
  - MFLUX 在 bf16 中烘焙 LoRA，它自己的日志提示只保留了 66% 的 LoRA 更新量；LingXi 在 fp32 中计算 `W + BA`，只舍入一次
  - 本组样例里，MFLUX 出图把「IMAGE」写成了「MAGE」，LingXi 的文字是完整的

## 稳定性（32 GB）

- 去噪阶段进程 footprint 16.6 GB，MLX 活跃内存 13.3 GB，与优化前相同（compile 和融合 kernel 都没有增加常驻内存）
- 整条流水线的进程 footprint 峰值是 17.8 GB，运行前后 Swap 没有增长
- 已知限制：运行中仍然存在之前进程遗留的系统 Swap（约 5 GB）。本次优化没有增加它，但也没有回收它

## 未做 / 后续

- 去噪的剩余开销主要是 BF16 矩阵乘法（两边共用同一实现）。继续提速只能靠量化或 Step Cache 一类有损手段，超出本阶段范围
- 只测了 1024² 一个分辨率。其他分辨率和长提示词由 `compile` 自动重新 trace，功能正确，但没有单独测速
- 标准 40 步配置按主人的说明（单步性能与 turbo 一致）没有重复测速
