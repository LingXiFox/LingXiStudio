# Phase 1.5 测试报告：官方语义对齐与非烘焙 LoRA

日期：2026-10-18 · 设备：MacBook Air M5 / 32 GB（无风扇）· macOS 27.2 · mlx-swift 0.32.3
权威参考：diffusers 官方 Qwen-Image-2.1 实现（含 PEFT LoRA），不是 MFLUX。

## 结论

| 验收项 | 结果 |
|---|---|
| 语义对齐 | 分阶段对齐套件共 28 项，**0 项失败**。所有误差都落在参考实现自身的浮点噪声底以内；两个故意做错的对照组都被检出 |
| 动态 LoRA，不修改 base 权重 | 烘焙代码已删除。load / unload / enable / disable / setScale、单个和多个 LoRA 均可用；297 个 base 权重张量的校验和前后不变；关闭 LoRA 后与 base 逐位一致 |
| 质量无可见退化 | 和官方参考比，6 步端到端解码 PSNR **40.5 dB**（参考自身的 1-ulp 噪声底是 29.5 dB）。和上一版烘焙 baseline 比，构图和文字一致（PSNR 24.8 dB / SSIM 0.949），运行之间像素逐位确定 |
| 稳态性能 | 全流程 5 轮的中位数 **6.046 s/step**（合格线 ≤6.05）。理想线 5.60 在非烘焙下数学上不可达，见「性能分析」 |
| 32 GB 内存 / Swap | 进程 footprint 峰值 18.0 GB（去噪阶段），MLX 峰值 17.4 GB，修复后连续 3 轮 Swap 均无增长 |

**没有做的事：** 没有降低精度，没有跳过计算，没有启用任何有损缓存，也没有重新启用烘焙。

## 固定测试配置

M5 Air 32 GB · BF16 原始权重 · 1024² · Viggle Turbo v0.3 LoRA（r=256，alpha=256，scale 1.0，动态加载）· 6 步 viggle_turbo 调度 · guidance 1.0（不开 CFG）· seed 42 · 无 step cache · 无 bake。

**温控门**（`Scripts/thermal-gate.sh`）：每次运行前，都要求 macOS 热状态为 nominal，并且约 1 s 的 4096³ bf16 matmul 探针达到冷机 13.1 TFLOPS 的 98%。

- 探针要**连续两次**达标，两次间隔 30 s。只要求一次达标时，出现过机身余热导致运行中途降频的情况：单轮 7.38 s/step，单步从 6.5 涨到 8.1 s。
- 每轮前后都记录热状态、探针 TFLOPS、`vm.swapusage`、各阶段 footprint 和 MLX 峰值。

## Task A：语义对齐（`Tests/Parity/run.sh`）

每个阶段都用**同一份中间输入**，分别喂给 LingXi 和参考实现。对比口径是相对 L2 误差（rel-L2）。为了区分「浮点误差」和「语义错误」，有两条判据：

- **noise floor**：参考实现自身的噪声底，来自 bf16 vs fp32，或 1-ulp 输入扰动。
- **对照组**：故意做错的实现，必须超出验收带（2× noise floor）。

| 阶段 | 检查 | 结果 | 判定 |
|---|---|---|---|
| 文本编码器 | 参考 bf16 vs fp32（噪声底） | 0.0457 | info |
| | strict bf16 vs 参考 bf16 | 0.0301（worst-token cos 0.990） | float |
| | 默认（fp32 残差流）vs 参考 fp32 | 0.0131（cos 0.9986） | float |
| | 对照：取最终 RMSNorm 之后的输出（MFLUX 的做法） | 0.860 | 检出 |
| 调度 | linear 40 步 sigma | 6.0e-8 | float |
| | viggle turbo sigma（无 terminal 拉伸） | 3.0e-8 | float |
| | Euler 更新（bf16 输入输出，全部 6 步） | 0 | exact |
| 条件 | temb / modulation / txt 投影 | 1.4e-5 / 1.5e-5 / 1.1e-5 | float |
| Transformer | 参考 1-ulp 输入扰动（噪声底） | 0.0097 | info |
| | velocity σ=0.9 / σ=0.3 | 0.0064 / 0.0072（cos 0.9994 / 0.9999） | float |
| | eager vs compiled | 0 | exact |
| | 逐 block hidden（b0 → b31） | 0.0006 → 0.016 | info |
| | 对照：用 normed 文本条件 | 0.0547 | 检出 |
| LoRA | 动态 LoRA vs PEFT 参考 | 0.0042（cos 0.99997） | float |
| | 两个 0.5 vs PEFT 一个 1.0 | 0.0123 | float |
| | 对照：缺 LoRA（base 模型） | 0.0644 | 检出 |
| 端到端 | 6 步 turbo 最终 latent | 0.0100（逐步 0.0009 → 0.0100） | float |
| | 参考自身 1-ulp 噪声扰动 | 0.0333（解码 29.5 dB） | info |
| | 解码图像 PSNR | 40.5 dB | float |
| VAE | 参考 bf16 vs fp32（噪声底） | 0.0027 | info |
| | fp32 解码（默认）vs 参考 fp32 | 1.85e-4（平均 0.010 个 8-bit 级） | float |
| | bf16 解码 vs 参考 bf16 | 0.0036 | float |
| 后处理 | float → uint8 RGBA | 0 | exact |

本阶段为对齐参考实现的舍入语义所做的修正（默认路径，性能零成本）：

- **timestep**：`bf16(σ·1000)/1000`，在 bf16 中计算。
- **RMSNorm**：fp32 归一化 → 舍入到 bf16 → 乘 bf16 权重。融合 Q/K kernel 在寄存器里按同样的点舍入：和参考组合算子相比，0.0017% 的元素有差异，最多 2 ulp。
- **SiLU / GELU**：fp32 内部计算，只舍入一次。
- **RoPE 角度**：按 torch 的 fp32 公式计算。
- **Euler 更新**：`bf16(dt·v)` 后转 fp32 相加，再转回 bf16。

这些修正让端到端 latent 误差从上一版的 0.035 降到 0.010。

## Task B：动态 LoRA（`Sources/QwenImageCore/LoRA.swift`）

- `LoRAAdapter` 原样保存文件里的 A/B 因子，不改 dtype，不预乘 scale。
  - alpha 读取顺序：先读 `.alpha` 张量；没有就读 PEFT `lora_adapter_metadata`；都没有则取 rank。
- `LoRAStack` 管理已加载的 adapter，提供 load / unload / setEnabled / setScale。每次变更 `revision` 加一。
- `QwenImageTransformer.bind(_:)` 在 revision 变化时重建 LoRA 绑定，并丢弃 compile 缓存；revision 不变时什么都不做。
- 前向和 PEFT 完全一致：`y + B(Ax) · (scale·alpha/r)`。scale 为 1 时省掉乘法。多个 LoRA 按栈顺序依次累加。
- base 权重从不被写入。关闭、scale 为 0 或卸载后，绑定为空，执行的是和 base 完全相同的计算图。

生命周期检查（`qwen-image debug-lora`）：

| 操作 | vs 基准 | 要求 |
|---|---|---|
| 加载（scale 1） vs base | rel-L2 0.1105 | > 0 |
| disable vs base | 0 | = 0 |
| 重新 enable vs 加载态 | 0 | = 0 |
| scale 0 vs base | 0 | = 0 |
| 两个 0.5 vs 一个 1.0 | 0.033 | 浮点级 |
| unload vs base | 0 | = 0 |
| base 权重校验和（297 个张量） | 不变 | 不变 |

## Task C：性能

### 全流程（`Scripts/benchmark.sh`，温控门，稳态 = 第 2–6 步均值）

| 运行 | 逐步 s | 稳态 s/step | 进程 footprint 峰值 | MLX 峰值 | Swap 前 → 后 | 运行后探针 |
|---|---|---|---|---|---|---|
| LingXi 动态 LoRA #1 | 5.94 5.99 6.00 6.05 6.09 6.10 | 6.046 | 26.8 GB* | 25.0 GB* | 2.57 → 5.02 GB* | fair 12.74 |
| LingXi 动态 LoRA #2 | 5.99 5.95 5.98 6.04 6.09 6.09 | 6.030 | 19.3 GB | 17.4 GB | 3.18 → 3.16 GB | nominal 12.87 |
| LingXi 动态 LoRA #3（修复后） | 5.95 6.01 6.50 6.25 6.04 6.07 | 6.174 | 19.3 GB | 17.4 GB | 2.85 → 2.81 GB | nominal 12.96 |
| LingXi 动态 LoRA #4（修复后） | 6.01 5.98 6.02 6.05 6.10 6.19 | 6.068 | 19.3 GB | 17.4 GB | 2.80 → 2.80 GB | fair 12.92 |
| LingXi 动态 LoRA #5（修复后） | 5.98 5.96 6.00 6.01 6.08 6.11 | 6.032 | 19.3 GB | 17.4 GB | 2.70 → 2.70 GB | nominal 12.91 |
| 回归基线（上一版，烘焙）#1 | 6.01 5.52 5.57 5.63 5.65 5.67 | 5.608 | 17.8 GB | 14.9 GB | 3.24 → 3.24 GB | fair 12.89 |
| 回归基线（上一版，烘焙）#2 | 5.56 5.59 5.61 5.66 5.71 5.67 | 5.648 | 17.8 GB | 14.9 GB | 2.93 → 2.92 GB | nominal 12.89 |
| MFLUX 0.22（烘焙） | 9.86 5.88 5.98 6.04 6.14 6.24 | 6.056 | 28.6 GB | 30.7 GB | 2.87 → 6.57 GB | fair 12.76 |

LingXi 5 轮稳态的中位数是 6.046 s/step，均值 6.070。

- 逐步耗时在单次运行内从约 5.95 爬升到约 6.10，这是无风扇机身 36 s 持续负载下的热漂移，回归基线也有同样的趋势。
- #3 第 3 步的 6.50 是单步离群值。

\* #1 暴露了一个阶段切换的内存竞态：去噪结束后 transformer 权重没有及时释放，和 VAE 解码叠在一起，导致 Swap +2.5 GB。

- 原因：`eval` 返回时，最后一个 command buffer 的完成回调仍持有 transformer 的缓冲。`Memory.clearCache()` 先于回调执行，回调释放的 14 GB 随后才回到缓存。
- 修复：阶段之间先 `Stream.gpu.synchronize()` 再清缓存（`Pipeline.releaseStageMemory`）。
- 修复后 #3–#5 的解码 footprint 是 11.6 GB，Swap 无增长。

### 去噪单测（`qwen-image bench-transformer --steps 3`，温控门）

| 配置 | s/step | MLX 峰值 |
|---|---|---|
| base（无 LoRA），MLX 默认 command buffer 40 op / 40 MB | 5.505 | 14.2 GB |
| base，100 op / 200 MB（新默认） | 5.256 / 5.291 | 16.3 GB |
| 动态 LoRA，默认 40 / 40 | 6.80 / 6.81 / 6.83 | 15.6 GB |
| 动态 LoRA，100 / 200（新默认） | 6.00 / 6.00 / 5.97 / 5.99 / 5.95 | 17.4–17.9 GB |
| 动态 LoRA，100 / 300 | 6.01 / 5.99 / 6.11 | 19.0 GB |
| 动态 LoRA，1000 / 1000 | 43.3（内存冲到 30 GB，Swap 颠簸） | 30.6 GB |

## 性能分析

### 理论下限

- **base 每步：** 约 57 TFLOP 的 matmul，加 8.9 TFLOP attention。
- **rank-256 LoRA 每步额外：** 约 5.5 TFLOP。这些 LoRA 覆盖 to_q/k/v、to_out、img_mlp gate/proj/out、modulation 和 timestep MLP，由 4096 个图像 token × 32 个 block 摊开。
- **在 13 TFLOPS 下：** 这部分额外计算至少要 **0.42 s/step**，而且前提是这些 K=256 的瘦长 GEMM 能跑到和 base 大 GEMM 一样的效率。实测它们只有 6–8 TFLOPS，大约是 base 的一半。
- **结论：** 烘焙版是 5.60 s/step，非烘焙版的理论下限约为 6.0 s/step。理想线 5.60 只有烘焙（或降 rank、降精度）才能达到，而这些都被本阶段规则禁止。

### 实测

- **当前开销：** 动态 LoRA 比烘焙回归基线（5.61–5.65）多约 0.40–0.44 s/step。这已经等于 FLOP 下限，没有剩余的非计算开销。
- **定位过程：** 修复前的开销是 1.2–1.3 s/step，排查思路如下。
  1. **分组测量**（`LINGXI_LORA_ONLY=<组名子串>`）：
     - qkv 实测 +0.16 s（理论 0.13），差额很小。
     - attn out 实测 +0.35 s（理论 0.04）。
     - mlp_in 实测 +0.84 s（理论 0.17）。
     - mlp_out 实测 +0.33 s（理论 0.09）。
     - 各组增量之和大于全开时的总增量。这说明开销不全是计算，而是和图结构有关。
  2. **孤立微基准**：把同形状的 LoRA 单独 compile 测，几乎不花钱（mlp_in +0 ms，attn out +2.3 ms/block）。模型内的开销在孤立环境下复现不出来。
  3. **Metal System Trace**（xctrace）：
     - 有 LoRA 时，GPU 稳态空闲占 5.4%，base 只有 1.2%。
     - command buffer 的中位时长是 4.2 ms，base 是 9.9 ms。
     - CPU 采样里最大的一块是 `IOGPUCommandQueueSubmitCommandBuffers`，也就是内核态提交。
  4. **根因**：MLX 默认在一个 command buffer 累计 40 个 op 或 40 MB 输入时就提交，而权重本身也计入 40 MB。DiT 里一个 MLP 权重就有 100 MB，所以几乎每个 matmul 都会单独提交。
     - base 的大 kernel 每个十几毫秒，能把每次提交的固定开销摊薄。
     - LoRA 给每步增加约 700 个 1–3 ms 的小 matmul，提交速度跟不上，GPU 就会空等。
- **修复**：`MLXRuntime.configure()` 在首次使用 MLX 之前把 `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` 设为 100 / 200。如果环境变量已经显式设置，则以环境变量为准。
  - 实测只调 op 数无效。调大 MB 上限有效，但上限越大，同时在途的中间张量越多。
  - 200 MB 的代价是去噪阶段 MLX 峰值多约 1.8 GB，换来 LoRA −0.8 s/step、base −0.23 s/step。
  - 1000 MB 会把内存推到 30 GB，引发 Swap 颠簸，不可用。
- 试过但没有采用的方案：
  - `addMM(y, h, Bᵀ)` 直接累加到 base 输出：慢约 3 倍。
  - 每组共用一次拼接后的 down-projection：在模型里不比逐 target 计算快，还多一份拼接的 A。
  - 在 LoRA 加法后插融合屏障：mlp_out 略好，mlp_in 更差。

## 质量回归（与性能分开记录）

同一 prompt、seed 42、1024²、6 步：

| 对比 | PSNR | SSIM | 说明 |
|---|---|---|---|
| LingXi 动态 LoRA 5 次运行之间 | ∞ | 1.0000 | 像素逐位确定 |
| 回归基线 2 次运行之间 | ∞ | 1.0000 | |
| LingXi 动态 vs 回归基线（烘焙） | 24.83 dB | 0.9491 | 构图、「QWEN IMAGE 2.1」文字、反光一致，仅下方小字细节不同 |
| LingXi 动态 vs MFLUX | 13.44 dB | 0.6010 | MFLUX 用的是 normed 文本条件（见对照组），构图不同 |
| LingXi 端到端 vs diffusers 参考（384×256，同一份噪声） | 40.5 dB | — | 参考自身 1-ulp 噪声底 29.5 dB |

LingXi 和上一版 baseline 的差异来自两处有意的改动：

- LoRA 不再在 bf16 中合并。合并时 `W+BA` 舍入到 bf16 会丢失小于 W 1 ulp 的更新量；现在走 PEFT 的分支计算。
- 本阶段的舍入语义修正。

这两处都让结果更接近官方参考（e2e 误差从 0.035 降到 0.010），不是质量退化。目视检查无退化。

## 已知限制

- 性能刚好卡在合格线（中位数 6.046 / 均值 6.070，合格线 6.05）。无风扇机身的环境温度和运行时长会让结果上下浮动约 ±0.05。
- command buffer 设置在进程级生效，MLX 在创建 Metal 设备时只读取一次。GUI 进程同样必须在任何 MLX 调用之前执行 `MLXRuntime.configure()`。
- 只测了 1024²。其他分辨率由 `compile` 自动重新 trace，功能正确，但没有单独测速。
- 系统里有其他进程遗留的约 2.7 GB Swap，本阶段没有增加，也没有回收。

## 复现

```bash
swift build -c release
Scripts/benchmark.sh lingxi .tmp/bench            # 动态 LoRA 全流程，带温控门
BASELINE_BIN=<frozen binary> Scripts/benchmark.sh baseline .tmp/bench
.build/release/qwen-image bench-transformer --steps 3 --lora <lora.safetensors>
EXTRA_SRC=<diffusers>:<peft>:<accelerate> REAL_LATENT=<latent.safetensors> Tests/Parity/run.sh
```
