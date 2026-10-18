# LingXiStudio

Native Qwen-Image-2.1 runtime for Apple Silicon, written in Swift on top of [MLX](https://github.com/ml-explore/mlx-swift).
No Python, MFLUX, Gradio or local web service is needed at runtime.

## Status

The native SwiftUI workbench runs all five creation modes on a 32 GB M5 MacBook Air: text-to-image, reference-guided
creation, image-to-image, image editing and masked inpainting. See [the workbench acceptance report](docs/workbench-acceptance.md)
for the tested configurations, measured performance and remaining limits. This is a locally built development app, not a notarized release.

Latest acceptance: manually launched App, 1024² BF16 + dynamic Turbo LoRA, **6.053 s/step** steady state. Two thermally
gated CLI runs measured 6.035 / 6.016 s/step; four probes showed a 3–4% throughput decline during the runs, with no swap
growth. GUI automation observation measured ~7.9 s/step and disturbed throughput, so those results are recorded separately.

First run (M5 Air 32 GB, 1024×1024, 40 steps, seed 42): 403 s total, of which denoising 388 s (6.3 s/step at the
start, ~9.5 s/step once the fanless chassis throttles), VAE decode 10 s, loading all weights ~4.5 s.
Peak MLX memory 14.3 GB, no swap growth during the run.

Performance alignment (see [docs/performance-alignment.md](docs/performance-alignment.md)): with a compiled denoising
step and a fused Q/K RMSNorm + RoPE Metal kernel, steady-state denoising runs at 5.60 s/step against MFLUX's
6.05 s/step on the same machine and configuration (108%), at a 17.8 GB peak process footprint.

Phase 1.5 (see [docs/phase-1.5-report.md](docs/phase-1.5-report.md)): every stage is checked against the official
diffusers implementation (28 staged parity checks, all within the reference's own float noise), and LoRA is applied
dynamically — base weights are never modified, adapters load, unload, enable, disable and rescale at runtime. With the
rank-256 turbo LoRA applied on the fly, steady-state denoising runs at 6.05 s/step (median of 5 thermally gated runs),
which is the baked speed plus the LoRA's own arithmetic.

## Workbench app

`Scripts/make-app.sh` builds `dist/LingXiStudio.app` (SwiftUI, no extra dependencies, ad-hoc signed). The main window
follows the product spec: parameters on the left 2/3; on the right, the image workspace (top) and the mode switcher
with LoRA / history / task tabs (bottom), split 1:1. Dividers drag; double-click one to restore it.

| Area | What works |
|---|---|
| Model | Folder selection, integrity check per component, live residency (text encoder → transformer → VAE) |
| Prompt | Positive / negative (negative only active with true CFG), prompt history |
| Parameters | Size presets and custom (multiples of 32), steps, guidance, seed (random / fixed, actual seed shown), batch count with seed rule, standard / Turbo schedule (Turbo needs an enabled acceleration LoRA, locks 6 steps), measured time estimate |
| Performance | 32 GB balanced profile; app footprint, MLX allocation, swap, memory pressure, thermal state, stage, s/step, ETA |
| LoRA | Import (file panel or drag), several at once, enable / scale / order / remove, compatibility check without loading the model; changes apply to the next queued task |
| Tasks | Serial queue, cancel (memory is released), retry with the same parameters, logs, records kept across launches |
| Images | Fit / 100 % / zoom / pan, standalone viewer window, save as, copy image / file / path / prompt, reveal in Finder, reuse all parameters, regenerate with the same seed, move to Trash |

Text-to-image, reference-guided creation and editing follow the official Qwen-Image-2.1 paths. Image-to-image uses
SDEdit; inpainting uses editing conditions with per-step latent blending and optional pixel compositing. The UI labels
these generic methods. At full image-to-image strength the start image is discarded; inpainting protects pixels outside
the feathered mask when “keep unmasked” is enabled.

Input images can be imported, pasted, reordered or sent from the gallery. The mask editor supports a separate large
canvas, brush / eraser, undo / redo, inversion and PNG import / export. Gallery details include filters, comparison, batch
export and legacy WebUI metadata import. Generation / prompt / LoRA presets, model and LoRA libraries, a Hugging Face
downloader, PNG / JPEG / TIFF export and diagnostics are available. Outputs record every active LoRA and copy their
input images alongside the result. Quantized profiles, step caching and automatic semantic masks remain unavailable.

## Layout

| Path | Purpose |
|---|---|
| `Sources/QwenImageCore/Tokenizer.swift` | Byte-level BPE tokenizer read from `processor/tokenizer.json` |
| `Sources/QwenImageCore/TextEncoder.swift` | Qwen3-VL text stack (36 layers), last decoder output before the final norm |
| `Sources/QwenImageCore/Transformer.swift` | 32-block single-stream block-causal DiT with cached text-prefix K/V |
| `Sources/QwenImageCore/VAE.swift` | FP32 single-frame encoder and decoder of `AutoencoderKLQwenImage21` |
| `Sources/QwenImageCore/VisionEncoder.swift` | FP32 Qwen3-VL vision tower with deepstack features |
| `Sources/QwenImageCore/Conditioning.swift` | Image preprocessing and multimodal prompt / M-RoPE positions |
| `Sources/QwenImageCore/ModePlan.swift` | Initial noise, strength, latent masks and pixel compositing |
| `Sources/QwenImageCore/Scheduler.swift` | Flow-match Euler schedule (exponential shift + terminal stretch) |
| `Sources/QwenImageCore/Pipeline.swift` | Staged pipeline: only one large component is resident at a time |
| `Sources/QwenImageCore/ImageWriter.swift` | RGBA PNG + JSON sidecar with the full generation record |
| `Sources/QwenImageCore/FusedKernels.swift` | Fused Q/K RMSNorm + RoPE Metal kernel |
| `Sources/QwenImageCore/LoRA.swift` | Dynamic PEFT LoRA: adapters, stack (load/unload/enable/scale), runtime groups |
| `Sources/QwenImageCore/Runtime.swift` | Process-wide MLX settings (command buffer size), call before any MLX use |
| `Sources/QwenImageCore/ProcessMemory.swift` | Process footprint sampling per stage |
| `Sources/qwen-image/main.swift` | CLI (`generate`, `validate`, numeric `debug-*` dumps) |
| `Sources/LingXiStudio/` | SwiftUI workbench: `AppModel` (state, queue, persistence), `GenerationEngine` (single MLX thread), views |
| `Scripts/make-app.sh` | Builds and ad-hoc signs `dist/LingXiStudio.app` |
| `Scripts/benchmark.sh` | Same-config benchmark (LingXi / frozen baseline / MFLUX), thermally gated |
| `Scripts/thermal-gate.sh` | Waits until the GPU probe is back to cold throughput |
| `Tests/Parity/` | Staged numerical parity suite against diffusers (development only, needs Python) |

## Build and run

```bash
swift build -c release
```

```bash
./.build/release/qwen-image generate --prompt "A neon shop sign that reads \"QWEN IMAGE 2.1\"" --seed 42
```

Defaults: model `~/Models/Qwen-Image-2.1-BF16`, 1024×1024, 40 steps, true CFG off (the model is sampled without guidance).
Turbo: `--lora <file>:1.0 --scheduler viggle_turbo --steps 6`. Several LoRAs: `--lora a.safetensors:0.8,b.safetensors`.
Results go to `outputs/` with a `.json` sidecar (prompt, seed, size, steps, scheduler, timings, peak memory).

For image modes, add `--mode reference --image reference.png`, `--mode edit --image main.png`,
`--mode img2img --init initial.png --strength 0.6`, or
`--mode inpaint --image main.png --mask mask.png --feather 8`. White mask pixels mark the repaint area.
Use `--width` / `--height` to override the input-based size policy.

## Memory strategy (32 GB unified memory)

| Stage | Resident weights |
|---|---|
| Prompt encoding | Text encoder language layers, BF16 (~16 GB); FP32 vision tower runs first when condition images exist; `lm_head` is never read |
| Denoising | Transformer BF16 (~14 GB) + per-block text K/V cache |
| Decoding | VAE decoder FP32 (~1.3 GB file, decoder only) |

Each component is released and the MLX buffer cache cleared before the next one loads. FP32 stages use true FP32
(TF32 is disabled by default); the transformer uses BF16. Additional condition images increase attention and K/V memory
and cost per step; the UI enforces an estimated memory budget.

## Numerical verification

`Tests/Parity/run.sh` runs each stage of LingXi and the official diffusers implementation on the same intermediate
inputs (text encoder, conditioning, transformer per block, LoRA vs PEFT, scheduler, Euler update, VAE, postprocess,
6-step end to end). Each result is compared with the reference's own noise floor, and deliberately wrong controls
(post-norm text conditioning, missing LoRA) must be detected. Results are in
[docs/phase-1.5-report.md](docs/phase-1.5-report.md).

The text encoder keeps its residual stream in FP32 (matmuls stay BF16), which is why it tracks the FP32 reference
closer than a pure BF16 run (`LINGXI_TE_RESIDUAL=bf16` gives the reference's strict BF16 stream).
