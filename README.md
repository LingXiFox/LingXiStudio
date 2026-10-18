# LingXiStudio

Native Qwen-Image-2.1 runtime for Apple Silicon, written in Swift on top of [MLX](https://github.com/ml-explore/mlx-swift).
No Python, MFLUX, Gradio or local web service is needed at runtime.

## Status

P0 milestone 1 — the native executor generates Qwen-Image-2.1 BF16 text-to-image results on a 32 GB M5 MacBook Air.
The SwiftUI workbench described in the product spec comes next.

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

## Workbench app (P0)

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

Reference images, image-to-image, editing and inpainting are shown in the mode switcher but disabled with the reason:
the native executor does not implement those model paths yet. Quantized profiles and step caching are listed as
unavailable for the same reason.

## Layout

| Path | Purpose |
|---|---|
| `Sources/QwenImageCore/Tokenizer.swift` | Byte-level BPE tokenizer read from `processor/tokenizer.json` |
| `Sources/QwenImageCore/TextEncoder.swift` | Qwen3-VL text stack (36 layers), last decoder output before the final norm |
| `Sources/QwenImageCore/Transformer.swift` | 32-block single-stream block-causal DiT with cached text-prefix K/V |
| `Sources/QwenImageCore/VAE.swift` | Single-frame decoder of `AutoencoderKLQwenImage21` |
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

## Memory strategy (32 GB unified memory)

| Stage | Resident weights |
|---|---|
| Prompt encoding | Text encoder language layers, BF16 (~16 GB; vision tower and `lm_head` are never read) |
| Denoising | Transformer BF16 (~14 GB) + per-block text K/V cache |
| Decoding | VAE decoder FP32 (~1.3 GB file, decoder only) |

Each component is released and the MLX buffer cache cleared before the next one loads.

## Numerical verification

`Tests/Parity/run.sh` runs each stage of LingXi and the official diffusers implementation on the same intermediate
inputs (text encoder, conditioning, transformer per block, LoRA vs PEFT, scheduler, Euler update, VAE, postprocess,
6-step end to end). Each result is compared with the reference's own noise floor, and deliberately wrong controls
(post-norm text conditioning, missing LoRA) must be detected. Results are in
[docs/phase-1.5-report.md](docs/phase-1.5-report.md).

The text encoder keeps its residual stream in FP32 (matmuls stay BF16), which is why it tracks the FP32 reference
closer than a pure BF16 run (`LINGXI_TE_RESIDUAL=bf16` gives the reference's strict BF16 stream).
