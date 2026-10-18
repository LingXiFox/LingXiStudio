#!/bin/zsh
# Staged numerical parity suite: LingXiStudio (Swift + MLX) vs the official Qwen-Image-2.1 diffusers implementation.
# Development-only: needs a Python with torch + transformers, and diffusers / peft / accelerate sources (EXTRA_SRC).
# Usage: Tests/Parity/run.sh [work-dir]
set -euo pipefail
root=${0:A:h:h:h}
work=${1:-$root/.tmp/parity}
py=${PYTHON:-$HOME/.local/share/uv/tools/mflux/bin/python}
bin=$root/.build/release/qwen-image
lora=${LORA:-$HOME/Models/Qwen-Image-2.1-LoRA/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors}
prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'
ref=$root/Tests/Parity/reference.py
mkdir -p "$work"
# Reference stages already recorded in $work are skipped; delete result-<stage>.json to rerun one.
stage() { [[ -f $work/result-$1.json ]] && { echo "skip $1 (cached)"; return; }; $py -I $ref "$@" --work $work; }
geo=(--width 384 --height 256)

$py -I $ref inputs --work $work
# Stage 1: text encoder (default fp32 residual, and the strict bf16 stream of the reference).
$bin debug-text --prompt "$prompt" --out $work/te_default.safetensors
LINGXI_TE_RESIDUAL=bf16 $bin debug-text --prompt "$prompt" --out $work/te_strict.safetensors
stage text
# Stage 2-3: conditioning and transformer on the reference's own text embeddings.
$py -I -c "
from safetensors.torch import load_file, save_file
e = load_file('$work/embeds.safetensors'); l = load_file('$work/latents.safetensors')
save_file({'embeds': e['embeds'], 'latents': l['latents']}, '$work/dit_in.safetensors')
save_file({'embeds': e['embeds'], 'noise': l['noise']}, '$work/denoise_in.safetensors')"
$bin debug-conditioning --in $work/dit_in.safetensors --sigma 0.9 --out $work/cond.safetensors
$bin debug-transformer --in $work/dit_in.safetensors --sigmas 0.9,0.3 $geo --out $work/dit.safetensors
$bin debug-transformer --in $work/dit_in.safetensors --sigmas 0.9 $geo --trace 1 --out $work/dit_trace.safetensors
$bin debug-transformer --in $work/dit_in.safetensors --sigmas 0.9 $geo --lora "$lora:1.0" --out $work/dit_lora.safetensors
$bin debug-transformer --in $work/dit_in.safetensors --sigmas 0.9 $geo --lora "$lora:0.5,$lora:0.5" \
  --out $work/dit_lora_two_halves.safetensors
$bin debug-denoise --in $work/denoise_in.safetensors $geo --steps 6 --scheduler viggle_turbo --lora "$lora:1.0" \
  --out $work/denoise.safetensors
# Stage 4: scheduler and Euler update.
$bin debug-euler --in $work/latents.safetensors --scheduler linear --steps 40 --out $work/sched_linear.safetensors
$bin debug-euler --in $work/latents.safetensors --scheduler viggle_turbo --steps 6 --out $work/sched_turbo.safetensors
cp $work/sched_turbo.safetensors $work/euler.safetensors
stage scheduler
# Stage 5: VAE decode of a real image latent, and 8-bit postprocessing.
$bin debug-vae --in ${REAL_LATENT:?set REAL_LATENT to a 256x384 image latent} $geo --dtype f32 --out $work/vae_f32.safetensors
$bin debug-vae --in $REAL_LATENT $geo --out $work/vae_bf16.safetensors
stage vae --real-latent $REAL_LATENT
$bin debug-vae-encode --in $work/vae_f32.safetensors --out $work/vae_enc.safetensors
stage vae-encode
$bin debug-text-image --prompt "Turn the scene into a watercolor painting" --image $work/cond.png --out $work/te_image.safetensors
stage text-image
[[ -f $work/result-dit-cond.json ]] || {
  $py -I $ref dit-cond-inputs --work $work
  $bin debug-transformer-cond --in $work/dit_cond_in.safetensors --sigmas 0.9 $geo --out $work/dit_cond.safetensors
}
stage dit-cond
$bin debug-postprocess --in $work/image.safetensors --out $work/post.safetensors
stage postprocess
# Reference transformer stages last: they hold the 14 GB model on the CPU for several minutes.
stage transformer --sigmas 0.9,0.3
stage e2e-decode
stage e2e-floor
$py -I -c "
import json, glob
rows = sum((json.load(open(f)) for f in sorted(glob.glob('$work/result-*.json'))), [])
json.dump(rows, open('$work/parity.json', 'w'), indent=1)
bad = [r for r in rows if r['verdict'] in ('SEMANTIC', 'MISSED')]
print(f'{len(rows)} checks, {len(bad)} failing')"
