# 多 LoRA 压力测试与参考图优化

日期：2026-10-18；设备为 M5 MacBook Air / 32 GB，接交流电。阶段基线：`3ecb274`。

## 约束与验收

- Turbo v0.3 r256 始终占第一项，随后依次加入 Qwen Fix v2、Canon DSLR、Samsung Photo；性能矩阵统一各 1.0 强度，1024² / Seed 42 / CFG 1 / 完整 Turbo 6 步。
- 保持基础模型 BF16、TF32 关闭、动态 LoRA；不烘焙、不量化、不减少参考图 token 或分辨率，不使用有损缓存。
- 测速前须 thermal nominal，BF16 探针达到 13.1 TFLOPS 冷机基线的 98%，相隔 30 秒连续两次通过。运行内在 prefill 前、2/4/6 步后取四个探针；失去门限立即停止该轮，失败结果保留。
- 250 ms 原生资源采样记录进程物理内存、CPU、I/O、全机压缩及 Swap 事件与 GPU 瞬时利用率。GPU 利用率不能归属到单个进程，Swap/压缩事件不是净驻留内存；CPU 时间按 Mach timebase 换算。
- 性能矩阵与视觉检查分开；不把不同 LoRA 的风格差异当数值回归，也不把 profiler 下的耗时当无采集测速。
- 所有原始命令、日志、参数、输入与图像、资源计数及失败记录保留在 Git 忽略的 `.tmp/lora-reference-20261018/`。临时探针源代码和执行脚本仅置于 `/tmp`。

## 文件来源与兼容性

四个文件已在本机下载完整，本次未重复下载。逐文件检查 safetensors 头、dtype、形状、连续无重叠的数据偏移和文件长度；SHA-256 与作者仓库 LFS 记录匹配，Canon/Samsung 另与 Civitai 按哈希查询结果匹配。

| 适配器 | 作者来源及固定 revision | SHA-256 | Rank / 匹配目标 | 文件 dtype |
|---|---|---|---|---|
| Viggle Turbo v0.3 | [Viggle](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/tree/009a44a895ef85f7e643c80fdca9543795248867) | `f06c266e04438b5272bdfb99410421d52a65d7a37f6f42aabc3cb1faf0142644` | 256 / 227 | BF16 |
| Qwen Fix v2 | [e-n-v-y](https://huggingface.co/e-n-v-y/Qwen-Image-2.1-Fix-v2.0/tree/045f682888e1f32977d1d7358e5022228fbac4f8) | `3f00361145703a64c7b9812d7797079820a1f7b0dbb5768b9d3df1f3c7750dfa` | 32 / 45 | BF16、FP32 |
| Canon DSLR | [Danrisi](https://huggingface.co/Danrisi/canon_qwen2.1/tree/deaba7d4b338db928a6d11199aaa936125eff969) | `5d59914dfaf18dee544862f5c41b333005fdeb11233f04cc44607beac08f9b1a` | 16 / 224 | FP16 |
| Samsung Photo | [Danrisi](https://huggingface.co/Danrisi/samsung_qwen2.1/tree/299e8dbeca35f28fd41e14cdfa5e59fc24e50ae9) | `d5b75fbd0a78c7642f0af49f11834d7217e3b7441ac86f2da2f598d4e0b370ee` | 16 / 224 | FP16 |

Canon Civitai 版本为 3352035，Samsung 为 3354434。Canon 触发词 `c2n0n`；Samsung 作者示例采用 `s2msun9`。作者基础模型均指向 Qwen-Image-2.1。Viggle 和 Danrisi 声明 Qwen Research License；Qwen Fix 模型卡未声明独立许可证，本阶段仅本机测试，不分发权重。原创代码的 MIT 许可不覆盖这些适配器。

原加载器只去除 `transformer.`，后三个文件的 `diffusion_model.` 因此前缀不匹配；三者的 `img_mlp.gate_up` 还需映射为前半 B 行对应 gate_layer、后半对应 proj，共用原 A。映射顺序与本机已有 MFLUX 的 Qwen21LoRAMapping 及 Qwen 模型 SwiGLU 公式交叉检查。只做键名与张量视图映射，因子原始字节和 dtype 不变。Fix 文件中的 per-layer alpha 也保留（8 个为 83、22 个为 96），不会错误地统一当作 rank 32。缺失因子、非法形状、重复目标和非法 alpha 直接报错，不静默跳过。

## 失败与修正：混合 dtype 传播

第一轮单 Turbo 稳态 5.964 秒/步，峰值 19.460 GB；发生 47,036 页新增 swap-out。四点探针及结束探针 nominal，结束 12.89 TFLOPS。该记录保留为 `stress-1.*`。

首次加入 Fix 后，没有限制投影输出 dtype 的原实现将适配器分支的 FP32 传播进整个 DiT 主干，基础 BF16 权重也被临时提升用于后续乘法。第二步时平均约 27.92 秒/步、BF16 探针仅 1.334 TFLOPS，峰值约 23.658 GB、新增 swap-out 228,112 页；该轮被终止，不能算完整六步通过，也不能只凭 nominal 热状态认定没有资源退化。自动终止脚本漏导入 `re`，未及时执行，本轮由人工命令终止；脚本已修复并加解析器自检。证据为 `stress-2.log`、`stress-2-resources.jsonl`、`stress-2-abort.json`、`stress-native-sequence.log` 之前的两份序列日志。

修正为每个基础投影依次完成所有适配器叠加后，再返回基础投影原 dtype（BF16），与 [PEFT LoRA 层返回基础输出 dtype 的约定](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) 一致。原生 FP16/BF16 混合分支由 MLX 无损提升至 FP32 运算，避免把 BF16 激活转成范围更小的 FP16；适配器存储及基础权重没有转换或合并。单 Turbo 的 BF16 算术边界保持不变；新增混合格式以独立的按序叠加公式及 BF16 输出边界作回归基线，不声称与失败轮错误提升的 FP32 主干逐像素一致。

## 动态多 LoRA 性能首轮

下表均为 1024²，完整 Turbo 6 步，全部强度 1.0。`MB` 限制指 MLX 空闲缓冲缓存，既不限制活跃张量，也不改变权重/激活精度。页计数为全机计数；不同轮 OS 文件缓存和后台内存状态不能完全等同。表内 GB 为十进制 10⁹ 字节，峰值取 250 ms 外部采样；原始 sidecar 另保留阶段内 20 ms 采样。CLI 旧日志以“GB”显示的 MLX 内存实际除以 2³⁰，不能与表格直接混用。

| 配置 | 稳态 2–6（秒/步） | 进程峰值 GB | 新增 swap-out 页 | 运行内四点 | 结束探针 | 状态 |
|---|---:|---:|---:|---|---|---|
| Turbo | 5.934 | 19.462 | 0 | 全通过 | nominal / 12.96 | 完整通过 |
| Turbo + Fix | 6.074 | 21.085 | 0 | 全通过 | nominal / 13.00 | 完整通过 |
| 再加 Canon | — | 22.092（提前终止时） | 13,692 | 第二步 12.827 未过门限 | nominal / 13.04 | 自动终止，无完整图像 |
| 再加 Samsung | — | 23.375（提前终止时） | 25,228 | 第二步 12.770 未过门限 | nominal / 12.86 | 自动终止，无完整图像 |
| Turbo，512 MiB | 5.959 | 19.074 | 0 | 全通过 | nominal / 13.09 | 完整通过 |
| Turbo + Fix，512 MiB | 6.092 | 19.928 | 0 | 全通过 | nominal / 12.90 | 完整通过 |
| 再加 Canon，512 MiB | 7.117 | 21.122 | 0 | 全通过 | nominal / **10.40** | 六步及图像完成，结束探针异常，非全程稳定成绩 |
| 再加 Samsung，512 MiB | — | 22.200（提前终止时） | 5,792 | 第六步 12.663 未过门限 | nominal / 12.93 | 第六步后终止，未继续解码 |

三 LoRA 结束探针异常后，下一轮门限的首次探针已恢复为 nominal / 13.10，随后 13.04 / 13.08，不能单凭一个低值断言热降频原因。反过来，也不把 nominal 当作频率恒定证明。本机 IOAccelerator 可读取利用率和内存字段，未暴露 GPU MHz/温度字段。

512 MiB 对两 LoRA 的峰值降低约 1.16 GB，吞吐差异约 0.3%；本轮支持它的内存收益，不足以证明去噪提速。单、双 LoRA 开关缓冲限制前后，图像像素分别完全一致。三、四项未完成轮的峰值不能与完整轮直接计算严格的优化百分比。

采样器每轮自身 CPU 时间约 0.03–0.18 秒，未发现它成为主耗时的证据。完整逐步耗时、CPU、读写、压缩/解压、Swap 与四点探针保留于 `stress-native-*`、`stress-cache512-*`，不能只保留表内最佳值。

最终又以原来的四项全 1.0 强度重复，启动前连续 **五次**、每次间隔 30 秒的门限通过（累计等待 220 秒，最后五次为 13.03 / 12.99 / 13.23 / 12.89 / 12.88）。运行探针为 13.065 / 12.971 / **12.799**，在第四步后中止；该轮峰值 22.212 GB、新增 swap-out 0，结束探针恢复 nominal / 13.03。证据为 `stress-final-4-*` 和 `stress-final-4.log`，未生成完整 PNG，不列入完整六步吞吐排名。延长冷却仍未得到全强度四 LoRA 的稳定连续通过，不能用后面的 Samsung 0.7 视觉样本替代这项压力验收。

## 参考图 A/B 与撤回的融合原型

基线输入 PNG、原始参数及既有完整生成结果均已复制保留。832×1248、一个参考图、Seed 42、CFG 1、Turbo 完整六步；每轮开始前以三次、相隔 30 秒的门限检查保证恢复。四点探针仍位于 prefill 前及 2/4/6 步后。

| 方案 | 稳态秒/步 | 峰值 GB | 新增 swap-out 页 | 温控验收 |
|---|---:|---:|---:|---|
| 当前对照，原缓存上限 | 6.990 | 22.066 | 56,644 | 运行四点通过，结束 nominal / 12.27，非全程稳定 |
| DiT 空闲缓存限制 512 MiB | 7.010 | 21.361 | 0 | 四点及结束均通过，结束 nominal / 13.01 |
| 原型：QK norm/RoPE + 前缀 K/V 拼接融合 | 7.002 | 21.968 | 0 | 四点及结束均通过，结束 nominal / 13.05 |
| 原型融合 + 512 MiB | — | 21.292（提前终止时） | 0 | 第四步 12.814，自动终止，无完整结果 |

对照、512 MiB、融合原型三张完整图像均与修改前保留的参考图结果逐像素一致：
`89a1462a64d69cdddfc47a93d0d70767401ba9a0475353d5a3a4a1f167337753`。

融合原型另以 BF16/FP32、不同头维度和非连续前缀验证 Q/K/V 原始字节，全部一致。但宏观上没有可确认的提速，峰值收益约 0.10 GB；组合轮未过门限。该原型及其专用单测已从最终代码撤回，保留 `rejected-prefix-fusion.patch` 和全部实验记录，不用额外维护一个没有宏观收益的 Kernel。实验阶段曾误将 `MLXRandom` 当成模块 import，导致一次构建失败；它实际是 MLX 内的命名空间，修正后实验 21 项单测通过。最终常规测试不包含已撤回的原型专用用例。

最终保留每次 DiT 调用期间的 **512 MiB 空闲缓存上限**，退出及异常时恢复调用前值；既有 prefill 的 256 MiB 临时上限仍保留。活跃张量、K/V、模型权重不受此缓存上限裁剪。`LINGXI_DIT_CACHE=0` 仅供关闭上限作回归对照；默认开启。诊断 benchmark 使用同一策略。当前证据证明内存减压和像素一致，未证明约 7 秒/步的去噪吞吐得到稳定提升，也不能把某轮零 Swap 推广为所有负载零 Swap。

## Metal Kernel 与条件 Token 分析

本轮从本机 Xcode 的 Metal System Trace 模板派生临时模板，只开启 Shader Timeline；使用白名单环境启动独立 CLI，不连接 App 窗口共享，也不采集用户进程的环境。四组均使用完整 32 层、1024² target、31 个 text token、动态适配器、一次预热和一次计时迭代。带 profiler 的计时仅作算子诊断，不能替代完整六步性能矩阵。

| 动态 LoRA 数 | Shader Timeline 采样区间 | 带采集的单次计时步（秒） | 最长连续 Compute 簇的 encoder 数 | 同进程 encoder 间隙合计（秒） |
|---|---:|---:|---:|---:|
| 1 | 2,053 | 6.003 | 386 | 0.581 |
| 2 | 2,205 | 6.210 | 414 | 0.566 |
| 3 | 3,614 | 7.307 | 487 | 0.544 |
| 4 | 3,744 | 7.920 | 611 | 0.576 |

最长簇按相邻 Compute encoder 间隙不超过 100 ms 聚类；上述区间包括前缀、预热和计时步。同进程 encoder 间隙不等于整机 GPU 空闲：其他进程也可能占用 GPU。该间隙没有随适配器数呈现同等幅度增长，因此本轮没有采用“扩大 Command Buffer 就能消除多 LoRA 增量”的假设，仍为 **100 ops / 200 MB**。第二组带采集结束探针为 nominal / 12.82，低于门限，进一步说明这些带采集步耗时不能当严格性能排名。

[Apple 对 Shader Timeline 的说明](https://developer.apple.com/videos/play/wwdc2020/10603/)指出它在采样时刻识别运行中的 shader，给出近似 GPU 时间。因此这里的条数不是精确 dispatch 次数，时间不是每层逐 Kernel 精确时钟，也不能据此计算真实硬件占用率。四组的 QK/RoPE 采样区间只有 43–96 条，反映采样覆盖差异，不代表源码实际调用数随 LoRA 减少。所有原始 shader 名称、近似时间、encoder 区间及分类结果保留在 `kernel-stack-*` 和 `kernel-summary.json`。

可见的主要家族包括 BF16 NAX GEMM、FP32 GEMM、split-K reduction、Attention、QK norm/RoPE、copy/cast 与编译后的逐元素叠加。依次加入 Fix、Canon、Samsung 后出现更多 FP32 适配器分支和转换样本，与源码中保持原始 dtype 的动态计算一致。BF16 GEMM 的 FP32 split-K 累加单独分类，不能把标签中的 `float32` 一律误认成主干转为 FP32。硬件 GPU counter 表仍为零行，不能宣称已测得 ALU/带宽/占用率瓶颈，也不精确划分基础 GEMM 与同形状 Turbo LoRA GEMM 的份额。

实际参考图布局为：编码序列 **1,038**，其中 image slots **1,014**；替换为 **4,056** 个条件 latent token 后，DiT 前缀为 **4,080**，目标为 **4,056**。每步 target attention 读取 **8,136** 个 K/V token；全部 32 层前缀 K/V 合计 **2,139,095,040 字节**。这些数量与基线一致，没有裁剪。相比算子诊断的 4,096 target + 31 text，目标 attention 的 QK 规模约为 1.95 倍；这是 attention 部分的工作量比值，不是整张图的耗时比值。

Block-Causal Prefill 继续按文本 causal、图像块内双向及后续文本可见前缀分段，保留既有融合 SDPA，避免把整个联合序列改成稠密方形 mask。前缀只计算一次，target 每步读取缓存。512 MiB 只限制闲置缓冲，并不把 2.14 GB 的有效 K/V 变小。未经验证的 token 减少、跨步近似、降精度缓存和改变 softmax 归约顺序均未采用。

实际参考图的额外采集在 prefill 和前两步后未恢复 12.838 TFLOPS 门限，停留在质量检查的冷却等待，180 秒上限最终结束记录（exit 54）。没有完成图像，不算性能或质量通过；部分轨迹仍保留，包括 5,481 个 Shader Timeline 采样区间及该原始布局。证据为 `kernel-reference-*`。本阶段后续视觉检查关闭 profiler，不拿这份部分轨迹代替完整参考图基线。

## 数值回归与资源释放验收

最终常规 XCTest **20 项、零失败**，xUnit 中实际 testcase 数为 20；新增格式测试覆盖 packed gate_up 的 A/B 原始字节与顺序、per-layer alpha、前缀映射、缺失/非法/重复/不支持的张量、四适配器原始 dtype 与按序公式、启停/卸载及基础输出字节不变。单 Turbo 在兼容性改动、BF16 投影边界修正及 512 MiB 开关前后，完整 1024² 输出仍与 `3ecb274` 保留结果一致（`d3c4a24a92e61cd523d03ea87afd493083078d6de2947b652e7a0902b25750ef`）。

真实模型 `HardwareLoRAStackTests` **1 项、零失败，354.180 秒**：同一进程按 1→4 个动态 LoRA，各运行原缓存上限和 512 MiB 两次，共 **8 张完整六步图**。这些输出为 256²，仅用于原始像素/条件复用/跨任务释放验证，不作为 1024² 性能或视觉质量结论。每张前均独立等待两次温控门限；每组开关缓存限制的 RGBA 逐字节相同，每加入一个实际适配器后输出均发生变化。首张条件 miss，其后七张命中，证明改 LoRA 不错误地失效文本条件，也没有把旧 LoRA 的 DiT 前缀当成可复用文本条件。

八张都通过 VAE 入口 MLX 活跃数组低于 1 GiB、每次结束清空空闲缓冲后低于 1 MiB、调用后恢复旧缓存上限的断言。这比独立 CLI 退出后释放更严格，覆盖了常驻进程中的连续叠加。证据为 `hardware-lora-lifecycle.log`、`lifecycle/stack-*-cache*.json/png`。后续运行按 UUID 子目录存放，避免覆盖初轮记录；这项路径调整不改变数值或释放断言。串行 SwiftPM 的空 Swift Testing XML 不代表 XCTest，本次依据完整 XCTest 日志、实际输出及退出码验收；不把未重跑的其他 opt-in 硬件用例计作本轮通过。

```bash
swift test -c release --parallel --num-workers 2 --skip Hardware --xunit-output .tmp/unit.xml
LINGXI_RUN_LORA_TESTS=1 GATE_PASSES=2 \
  MLX_MAX_OPS_PER_BUFFER=100 MLX_MAX_MB_PER_BUFFER=200 MLX_ENABLE_TF32=0 \
  swift test -c release --skip-build --filter HardwareLoRAStackTests
```

CPU 计时以 `proc_pid_rusage` 的 Mach ticks 乘实际 timebase 换算。本阶段额外与 `getrusage` 交叉自检：原始 501,154 ticks 换算 0.020881 秒，与 getrusage 的 0.020889 秒一致，不能把该原始字段直接当纳秒。原始自检为 `cpu-timebase-selfcheck.json`。

## 独立视觉质量检查

固定 1024²、Seed 1049、CFG 1、Turbo 完整六步，提示包含成人女性、蓝色雨衣、双手捧白色咖啡杯、雨后咖啡馆街道和精确文字 `LINGXI CAFE`，同时加入作者触发词 `c2n0n, s2msun9`。累计强度为 **1 / 1 / 1 / 0.7**，Samsung 采用作者示例推荐强度；不同于压力矩阵全 1.0。完整原始提示词和配置以 `visual-*-command.json`、PNG sidecar 为准。

这些是视觉检查，开启 `--thermal-wait 1`：探针低于门限时暂停，直到相隔 30 秒连续两次恢复。冷却计入 totalSeconds，不算作连续运行速度成绩。没有降低分辨率、减少六步或缩减 Token。

| 累计 LoRA | 完整总耗时秒（含冷却） | 峰值 GB | 新增 swap-out 页 | 热状态说明 | 实际图像观察 |
|---|---:|---:|---:|---|---|
| Turbo | 108.962 | 19.174 | 0 | 第二步后冷却 60 秒，结束 13.00 | 主体、雨衣、双手白杯及湿街符合描述，画面较明亮 |
| + Fix | 49.041 | 20.152 | 0 | 四点及结束通过，无冷却 | 头发、雨衣高光与细节变化，主体和杯子保留 |
| + Canon | 114.712 | 21.265 | 13,110 | 第六步后冷却 60 秒；结束 12.79 未过门限 | 景深更浅，画面变柔和、偏暗 |
| + Samsung 0.7 | 59.714 | 22.237 | 0 | 四点及结束通过，无冷却 | 明显变暗、偏欠曝且模糊，面部细节减弱，不能判定叠加后质量提高 |

四张都完成生成，未出现空图或纯噪声崩溃；**四张均未正确呈现指定的 `LINGXI CAFE` 字样**，精确文字遵循没有通过。上述风格及细节变化是不同适配器组合的观察，不是“数值等价”的证明，也不掩盖四项组合的画质退化。原图、日志、CPU/I/O/Swap 记录和逐项观察保留为 `visual-*`、`visual-review.json`。没有为了凑出更好画面挑选其他 Seed。

## 本阶段交付与限制

最终保留原生 LoRA 格式兼容与严格校验、基础 BF16 投影输出边界、DiT 空闲缓存 512 MiB 上限，以及诊断和回归用例。参考图受控完整轮峰值从 22.066 降至 21.361 GB，新增 swap-out 从 56,644 页降至 0，像素与既有完整结果完全一致；这只证明本轮内存减压，不能保证任何系统状态下都零 Swap。约 **7 秒/步仍未取得稳定提速**。

最终公共回归使用 `unit-final.xml`，实际 20 项、零失败/错误；真实四适配器回归另有一项及八张完整输出。完整 1024² 四项全强度连续压力验收仍未通过，全部退化与终止记录保留。Shader Timeline 为采样近似，硬件 counter 未获取，不能声称完成逐层精确周期或硬件占用率归因。本阶段未重做 GUI 人工验收，也未声称四适配器全管线与外部 PyTorch 实现逐像素对齐。
