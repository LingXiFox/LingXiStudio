#!/bin/zsh
# Same-config denoising benchmark: LingXiStudio vs MFLUX on Qwen-Image-2.1 BF16 + Viggle turbo LoRA.
# Usage: Scripts/benchmark.sh <lingxi|mflux> <out-dir> [cooldown-seconds]
# Each run starts after a cooldown so the fanless chassis begins from the same thermal state.
set -euo pipefail
which=$1; out=$2; cooldown=${3:-180}
model=${MODEL:-$HOME/Models/Qwen-Image-2.1-BF16}
lora=${LORA:-$HOME/Models/Qwen-Image-2.1-LoRA/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors}
prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'
seed=${SEED:-42}
mkdir -p "$out"
stamp=$(date +%H%M%S)
log="$out/$which-$stamp.log"
sleep "$cooldown"
{
  date; sysctl vm.swapusage
  if [[ $which == lingxi ]]; then
    /usr/bin/time -l "${0:A:h}/../.build/release/qwen-image" generate --model "$model" --prompt "$prompt" \
      --seed $seed --steps 6 --scheduler viggle_turbo --lora "$lora:1.0" --output "$out/$which-$stamp.png"
  else
    /usr/bin/time -l "$HOME/.local/share/uv/tools/mflux/bin/mflux-generate-qwen-2.1" --model "$model" --prompt "$prompt" \
      --seed $seed --steps 6 --width 1024 --height 1024 --scheduler viggle_turbo --lora "$lora" 1.0 \
      --output "$out/$which-$stamp.png"
  fi
  sysctl vm.swapusage; date
} > "$log" 2>&1
# Per-step seconds: LingXi prints them; for MFLUX derive them from tqdm's cumulative n * s/it.
if [[ $which == lingxi ]]; then
  steps=$(grep "^step seconds:" "$log" | sed 's/step seconds: //')
else
  steps=$(tr '\r' '\n' < "$log" | grep -E '^ *[0-9]+%.*\| *[0-9]+/6 \[' | grep -v ' 0/6' | awk '!seen[$0]++' \
    | sed -E 's/.*\| *([0-9]+)\/6 .*, *([0-9.]+)s\/it\].*/\1 \2/' | awk '!seen[$1]++ {c=$1*$2; printf "%.2f ", c-p; p=c}')
fi
echo "$which steps: $steps"
echo $steps | awk '{s=0; for(i=2;i<=NF;i++) s+=$i; printf "'$which' steady-state (2-%d): %.3f s/step\n", NF, s/(NF-1)}'
grep -E "Peak MLX memory|peak MLX memory|swapusage|peak memory footprint|footprint peak" "$log" | sed 's/^/  /'
