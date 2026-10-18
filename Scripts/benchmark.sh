#!/bin/zsh
# Same-config denoising benchmark on Qwen-Image-2.1 BF16 + Viggle turbo LoRA (1024², 6 steps, seed 42, no CFG).
# Usage: Scripts/benchmark.sh <lingxi|baseline|mflux> <out-dir>
#   lingxi    current build (.build/release/qwen-image), LoRA applied dynamically
#   baseline  frozen regression binary given by BASELINE_BIN (v0.1-baked-baseline, LoRA baked)
#   mflux     MFLUX 0.22 reference
# Thermal gating: before each run the GPU is probed (~1 s of 4096³ bf16 matmuls) until macOS reports a nominal
# thermal state and the probe reaches GATE_TFLOPS (default 98% of COLD_TFLOPS). A fixed cooldown alone is not trusted.
set -euo pipefail
which=$1; out=$2
root=${0:A:h:h}
probe=$root/.build/release/qwen-image
model=${MODEL:-$HOME/Models/Qwen-Image-2.1-BF16}
lora=${LORA:-$HOME/Models/Qwen-Image-2.1-LoRA/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors}
prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'
seed=${SEED:-42}
mkdir -p "$out"
stamp=$(date +%H%M%S)
log="$out/$which-$stamp.log"

line=$("${0:A:h}/thermal-gate.sh")
{
  echo "$line"
  date; sysctl vm.swapusage
  case $which in
    lingxi|baseline)
      bin=$probe; [[ $which == baseline ]] && bin=${BASELINE_BIN:?set BASELINE_BIN}
      /usr/bin/time -l "$bin" generate --model "$model" --prompt "$prompt" --seed $seed --steps 6 \
        --scheduler viggle_turbo --lora "$lora:1.0" --output "$out/$which-$stamp.png" ;;
    mflux)
      /usr/bin/time -l "$HOME/.local/share/uv/tools/mflux/bin/mflux-generate-qwen-2.1" --model "$model" \
        --prompt "$prompt" --seed $seed --steps 6 --width 1024 --height 1024 --scheduler viggle_turbo \
        --lora "$lora" 1.0 --output "$out/$which-$stamp.png" ;;
  esac
  sysctl vm.swapusage; date
  echo "post-run probe: $($probe probe)"
} > "$log" 2>&1
if [[ $which == mflux ]]; then
  steps=$(tr '\r' '\n' < "$log" | grep -E '^ *[0-9]+%.*\| *[0-9]+/6 \[' | grep -v ' 0/6' | awk '!seen[$0]++' \
    | sed -E 's/.*\| *([0-9]+)\/6 .*, *([0-9.]+)s\/it\].*/\1 \2/' | awk '!seen[$1]++ {c=$1*$2; printf "%.2f ", c-p; p=c}')
else
  steps=$(grep "^step seconds:" "$log" | sed 's/step seconds: //')
fi
echo "$which steps: $steps"
echo $steps | awk '{s=0; for(i=2;i<=NF;i++) s+=$i; printf "'$which' steady-state (2-%d): %.3f s/step\n", NF, s/(NF-1)}'
grep -E "^gate passed|post-run probe|Peak MLX memory|peak MLX memory|swapusage|peak memory footprint|footprint peak" "$log" | sed 's/^/  /'
