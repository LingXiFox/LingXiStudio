# LingXiStudio

Native Qwen-Image-2.1 runtime for Apple Silicon, written in Swift on top of [MLX](https://github.com/ml-explore/mlx-swift).
No Python, MFLUX, Gradio or local web service is needed at runtime.

## Status

P0 milestone 1 — the native executor generates Qwen-Image-2.1 BF16 text-to-image results on a 32 GB M5 MacBook Air.
The SwiftUI workbench described in the product spec comes next.

First run (M5 Air 32 GB, 1024×1024, 40 steps, seed 42): 403 s total, of which denoising 388 s (6.3 s/step at the
start, ~9.5 s/step once the fanless chassis throttles), VAE decode 10 s, loading all weights ~4.5 s.
Peak MLX memory 14.3 GB, no swap growth during the run.

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
| `Sources/qwen-image/main.swift` | CLI (`generate`, `validate`, numeric `debug-*` dumps) |

## Build and run

```bash
swift build -c release
```

```bash
./.build/release/qwen-image generate --prompt "A neon shop sign that reads \"QWEN IMAGE 2.1\"" --seed 42
```

Defaults: model `~/Models/Qwen-Image-2.1-BF16`, 1024×1024, 40 steps, true CFG off (the model is sampled without guidance).
Results go to `outputs/` with a `.json` sidecar (prompt, seed, size, steps, scheduler, timings, peak memory).

## Memory strategy (32 GB unified memory)

| Stage | Resident weights |
|---|---|
| Prompt encoding | Text encoder language layers, BF16 (~16 GB; vision tower and `lm_head` are never read) |
| Denoising | Transformer BF16 (~14 GB) + per-block text K/V cache |
| Decoding | VAE decoder FP32 (~1.3 GB file, decoder only) |

Each component is released and the MLX buffer cache cleared before the next one loads.

## Numerical verification

Each component was checked against the diffusers / transformers reference implementations:

| Component | Check | Result |
|---|---|---|
| Tokenizer | token ids vs HF `tokenizers`, 6 cases incl. CJK, emoji, special tokens | identical |
| Scheduler | 40-step sigmas vs diffusers `FlowMatchEulerDiscreteScheduler` | max abs diff 1.3e-7 |
| Text encoder | hidden states vs transformers FP32 | worst-token cosine 0.9986 (transformers BF16: 0.9535) |
| Transformer | one step at 384×256 vs diffusers BF16 | cosine 0.99993, rel-L2 1.2% |
| VAE decoder | decode of a real image latent vs diffusers FP32 | rel-L2 2e-4, mean 0.01 8-bit levels |

The text encoder keeps its residual stream in FP32 (matmuls stay BF16), which is why it tracks the FP32 reference
closer than a pure BF16 run.
