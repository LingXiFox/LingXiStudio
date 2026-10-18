# LingXiStudio

面向 Apple Silicon 的原生图像工作台，使用 Swift、SwiftUI 和 [MLX Swift](https://github.com/ml-explore/mlx-swift) 运行 Qwen-Image-2.1。
推理不需要 Python、MFLUX、Gradio 或本地 Web 服务。

本项目处于开发阶段，提供本地构建、临时签名的 macOS App，尚未发布经过公证的安装包。项目代码采用 [MIT](LICENSE)；
**模型和 LoRA 使用各自的许可证，MIT 不授予模型商用权利**，详见[许可证与致谢](#许可证与致谢)。

本轮 Runtime v1 候选冻结在 [`runtime-v1-rc1`](https://github.com/LingXiFox/LingXiStudio/tree/runtime-v1-rc1)（`4e1426c`）。
新版完善原生工作流、记录追溯和 CI；详见 [Runtime 冻结](docs/runtime-v1-candidate.md)、[GUI 与真实创作验收](docs/workbench-v1-acceptance.md)、[交付准备](docs/release-readiness.md)。候选与测试通过不代表所有场景均已验证。

## 环境要求

- Apple Silicon Mac；当前实际验收设备为 M5 MacBook Air、32 GB 统一内存。
- macOS 26.2 或更新版本。Intel Mac、Linux 和 Windows 不在当前支持范围内。
- 完整 Xcode，包含 Swift 6.3+、macOS SDK、Metal 工具链和支持 Icon Composer 的 `actool`。本机通过的工具链为 Swift 6.4 / macOS SDK 27.2。
- 独立准备 Qwen-Image-2.1 的完整 Diffusers 目录，包括 `processor`、`text_encoder`、`transformer`、`vae` 和 `scheduler`。暂不支持 GGUF 或单文件量化模型。
- 首次构建需要获取 `Package.resolved` 锁定的 MLX Swift 0.32.3 及其 Swift Numerics 依赖。模型、LoRA 和运行结果不随仓库提供。

## 构建与启动

```bash
git clone https://github.com/LingXiFox/LingXiStudio.git
cd LingXiStudio
Scripts/make-app.sh
open dist/LingXiStudio.app
```

更新已有 App 前先退出旧进程。脚本构建 Release 可执行文件、Metal 资源、原生分层图标和许可证文本，并进行临时签名。
首次打开后，在模型库选择本地模型目录，或使用内置 Hugging Face 下载器。默认模型位置为 `~/Models/Qwen-Image-2.1-BF16`。
下载前应阅读并接受相应模型的上游许可。

工作台左侧编辑输入图、提示词和参数，右上方查看作品，右下方管理 LoRA、历史和任务。分隔线可拖动，双击恢复布局。
全局路径、外观和性能选项在设置中；内存、Swap 和组件状态在按需打开的运行时检查器中。

1. 选择创作模式；图片模式先导入或粘贴输入图，局部重绘再编辑蒙版。
2. 填写提示词。正面与负面提示词并排；负面提示词只在 Guidance > 1 时参与推理。预设和历史在独立窗口中搜索、选择。
3. 选择尺寸、种子和采样方案。Turbo 需要启用兼容的加速 LoRA，固定为 6 步。
4. 点击生成或按 **⌘↩**。所有模式在任务开始时自动切到右侧“作品”页，显示当前阶段、步数、实测 s/step 和预计剩余去噪时间。
   输入图和蒙版页仍可手动打开；已有蒙版不会因切页丢失。
5. 点击停止或按 **⌘.** 后立即显示“正在停止…”，在安全计算边界取消。已提交的 GPU 运算需要先完成。
   取消后预览留空，提示 3 秒后消失；已保存的作品仍在下方图库中。图库上方不再重复显示胶片缩略栏。

输出默认保存至 `~/Pictures/LingXiStudio`。每张 PNG 有 JSON 侧车记录，包含提示词、种子、模式、尺寸、采样、LoRA、阶段耗时和峰值内存；
使用的输入图复制到输出目录的 `inputs/` 中。设置、预设、任务记录和日志位于 `~/Library/Application Support/LingXiStudio`。

## 功能与边界

| 模式 | 实现方式 |
|---|---|
| 文生图 | 官方 Qwen-Image-2.1 文本条件管线 |
| 参考图创作 | 官方多模态条件管线 |
| 图像编辑 | 官方图像条件与编辑指令管线 |
| 图生图 | 通用 SDEdit 方法，从加噪的初始图继续去噪 |
| 局部重绘 | 编辑条件、逐步潜空间蒙版混合及可选像素合成 |

图生图和局部重绘在界面上标注为通用方法。图生图强度为 1 时丢弃初始图；启用“保留未选区”时，蒙版外按羽化设置保留原图。

- 输入图导入、粘贴、排序、图库送入创作；蒙版画笔/橡皮、撤销/重做、反转、PNG 导入/导出和独立大画布。
- 动态 LoRA 多选、启用、强度、排序和组合预设；基础权重不合并修改，调整应用于后续任务。
- 串行任务队列、取消、重试、实际种子和跨启动记录；图库筛选、对比、批量操作、旧 WebUI 元数据导入。
- 作品适配/100%/缩放/平移、独立查看器、复制与另存为、PNG/JPEG/TIFF 导出及移至废纸篓。
- 暖奶白/暖炭黑主题、点击空白取消输入焦点、自定义键盘焦点指示、提示词与生成预设。

LoRA 支持 `transformer.`、`diffusion_model.` 及 packed `gate_up` 映射，保留原始因子 dtype 和 per-layer alpha；动态叠加后返回基础投影的 BF16 dtype，基础权重不烘焙。DiT 阶段将空闲缓冲缓存预算限制到 128 MiB，退出时恢复旧值；不裁剪活跃张量或参考图 Token。
App 的生成任务另将 MLX 内存调度水位设为不高于物理内存的 5/8（32 GiB 机器为 20 GiB），让在途分配过快时等待已提交计算；完成、失败或取消后恢复原水位。这是软调度阈值，不是活跃张量硬上限，也不承诺所有后台负载下零 Swap。相同四 LoRA 参考图的隔离验证逐像素一致；原 CLI 默认策略保持不变。

文本条件采用全精度逐层加载，并保留最近一组条件的无损 CPU Embedding 缓存（最多 128 MiB）。同一模型、提示词、参考图和文本计算配置下，只修改 Seed 会复用条件，跳过视觉/文本编码器。缓存不跨 App 重启；超限时重新计算。
缓存校验准备后的参考图实际像素与顺序、提示词、CFG 分支、模型文件身份及配置内容，不依赖输入路径或历史哈希。模型权重被替换或修改时失效；生成期间模型变化会报错。详见[加载、缓存与性能实测](docs/conditioning-performance.md)。

**尚未实现**：自适应量化、模型量化配置、去噪步缓存和自动语义蒙版。DiT 仍使用 BF16，视觉塔和 VAE 使用真正的 FP32；本次没有降低精度、减少步数或采用有损缓存。系统整体内存压力仍可能影响响应。

## CLI

```bash
swift build -c release
bin_dir="$(swift build -c release --show-bin-path)"
"$bin_dir/qwen-image" validate --model "$HOME/Models/Qwen-Image-2.1-BF16"
"$bin_dir/qwen-image" generate --prompt 'A neon shop sign that reads "QWEN IMAGE 2.1"' --seed 42
```

默认 1024×1024、40 步、true CFG 关闭，输出到 `outputs/`。Turbo 增加
`--lora /path/to/turbo.safetensors:1.0 --scheduler viggle_turbo --steps 6`；
多个 LoRA 使用逗号分隔：`--lora a.safetensors:0.8,b.safetensors`。

| 模式 | 附加参数 |
|---|---|
| 参考图 | `--mode reference --image reference.png` |
| 编辑 | `--mode edit --image main.png` |
| 图生图 | `--mode img2img --init initial.png --strength 0.6` |
| 重绘 | `--mode inpaint --image main.png --mask mask.png --feather 8` |

蒙版白色表示重绘区域。使用 `--width` / `--height` 覆盖输入图尺寸策略；尺寸须为 32 的倍数。

## 验证与性能

```bash
mkdir -p .tmp
swift test -c release --parallel --num-workers 2 --skip Hardware --xunit-output .tmp/tests.xml
```

常规 XCTest 现有 27 项，本机 GPU 路径通过；标准 CI 使用 CPU 执行其中 26 项，明确不覆盖 Metal 编译图生命周期用例。测试覆盖蒙版、输出元数据、点击空白、加载/层间取消、逐层编码字节对齐及缓存失效。另有文本/参考条件缓存 2 项、多 LoRA 1 项及完整工作负载连续稳定性 1 项显式启用的真实模型测试，另有 1 项全尺寸内存调度/像素一致性隔离用例；默认跳过不能代替实际通过。条件/像素回归串行等待温控门限，连续稳定性只在开始前校准，运行内吞吐下降只记录，不暂停或中止。多 LoRA 用例在同一进程验证八次完整生成的像素和资源释放；启用命令见[多 LoRA 实测](docs/multi-lora-reference-performance.md)及[条件缓存实测](docs/conditioning-performance.md)。
[分阶段数值对齐脚本](Tests/Parity/run.sh) 是开发用工具，需要单独准备 Python/PyTorch 参考环境；正常推理不需要这些依赖。

| 证据 | 已验证范围 |
|---|---|
| [Runtime 连续稳定性](docs/runtime-stability-acceptance.md) | 性能有效性与资源安全分离、四 LoRA＋参考图连续生成、响应延迟及释放记录 |
| [多 LoRA 与参考图优化](docs/multi-lora-reference-performance.md) | 格式校验、动态四 LoRA 压力、温控失败记录、Kernel 采样及像素/释放回归 |
| [工作台验收](docs/workbench-acceptance.md) | 五种模式、CLI/手动 App 测试及四点温控记录 |
| [GUI 验收](docs/gui-design-acceptance.md) | 界面交互、原生图标、取消和后续调整；明确列出尚未实测的项 |
| [加载与条件缓存](docs/conditioning-performance.md) | Batch 4/8/16、全精度逐层加载、缓存逐字节对齐及 Metal 轨迹 |
| [UI 性能排查](docs/ui-performance-investigation.md) | 空闲采样、窗口共享干扰、真实生成采样与内存压力观察 |
| [数值对齐](docs/phase-1.5-report.md) | 28 项分阶段对齐检查、LoRA 与官方实现比较 |
| [性能对齐](docs/performance-alignment.md) | 早期同配置性能对照，保留测试上下文 |

此前手动启动 App 的 1024² / BF16 / Turbo 6 步 / 动态 LoRA 测得稳态 **6.053 s/step**；
两次温控 CLI 测得 6.035 / 6.016 s/step，四点吞吐下降约 3–4%。这些是特定设备与配置下的历史结果，
不代表最新 UI 修订重新通过了测速。窗口共享采集会干扰 GPU 与 WindowServer 负载，相关结果单独记录。
此前全精度逐层加载构建的同配置完整回归为 **5.978 s/step**（稳态 2–6 步），四点吞吐从 13.179 到 12.890 TFLOPS，有约 2.2% 下降；输出像素与 `0ca7773` 一致。详细条件、失败回归与修复证据见[实测报告](docs/conditioning-performance.md)。
最终参考图手动验收未观察到卡顿；同配置 GUI、CLI 和此前版本共五张输出的解码像素完全一致。参考图约 7 s/step，按复杂负载单独记录，不使用纯文生图基准判定失败。
当前 128 MiB DiT 缓存预算的纯文生图完整回归为 6.023 s/step，像素一致；运行内吞吐探针未全过校验，不作为恒定冷机性能证明。
**5–6 s/step 仅作为冷机、1024²、纯文生图、单 Turbo LoRA 的基准目标。** 多 LoRA、参考图和复杂编辑允许合理增加耗时；验收先看系统响应、新增 Swap-out/持续内存压力、组件与缓冲释放、连续生成稳定性，再优化速度。GPU 吞吐门限只判定测速有效性，不作为实际生成的中止条件；资源安全独立记录。详见[当前 Runtime 验收](docs/runtime-stability-acceptance.md)。

运行时分阶段加载并释放组件，每阶段结束清理 MLX 缓冲缓存。文本编码器权重文件约 16 GB，逐层加载的纯文本实测进程峰值约 1.8 GB；参考图还需要视觉编码和更长的激活。BF16 DiT 权重约 14 GB、FP32 VAE 解码器约 1.3 GB。
峰值还包括激活、LoRA、条件图与系统内存，不能把权重文件大小当成实际峰值。

## 源码结构

| 路径 | 职责 |
|---|---|
| `Sources/LingXiStudio/` | SwiftUI 工作台、队列、输入/图库/模型管理、持久化 |
| `Sources/QwenImageCore/` | 文本/视觉编码器、DiT、VAE、LoRA、调度和输出记录 |
| `Sources/qwen-image/` | CLI 生成、校验与数值诊断 |
| `Scripts/` | App 打包、温控与基准测试 |
| `Tests/Workbench/` | XCTest 回归测试 |
| `Tests/Parity/` | 与官方参考实现的分阶段数值比较 |
| `docs/` | 已保留的验收与性能报告 |
| `ICONs/` | 原生 Icon Composer 图标源文件与导出资源 |

模型权重、生成结果、运行日志、缓存、构建产物和环境文件由 `.gitignore` 排除，不随 Git 上传。

## 许可证与致谢

- 本仓库原创代码与文档：**MIT**，Copyright © 2026 LingXiFox，完整文本见 [LICENSE](LICENSE)。
- MLX Swift、MLX、MLX C、Swift Numerics 及其原生组件：保留各自许可，版本锁定见 `Package.resolved`；
  完整许可文本位于 [LICENSES](LICENSES)，组件归属见 [NOTICE](NOTICE)，打包时一并放入 App 的 `Contents/Resources/Legal/`。
- [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 和
  [Viggle Turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) 不随项目分发。
  两者当前采用 **Qwen RESEARCH LICENSE AGREEMENT**，只授予研究/评估用途，商业使用需要向权利人取得单独许可；
  以[模型许可原文](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)和
  [LoRA 许可原文](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/LICENSE)为准。

LingXiStudio 是独立项目，与模型及依赖作者没有官方隶属或背书关系。
