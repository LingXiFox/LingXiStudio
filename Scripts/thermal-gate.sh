#!/bin/zsh
# Blocks until the GPU is back to its cold throughput: macOS thermal state nominal and a ~1 s bf16 matmul probe at
# GATE_TFLOPS (default 98% of COLD_TFLOPS = 13.1 on the M5 Air reference machine) on GATE_PASSES consecutive probes
# GATE_SPACING seconds apart. A single probe can pass while the chassis is still warm, which then throttles a longer
# run; requiring the probe to hold across a quiet interval rejects that state. Prints the last passing probe line.
set -euo pipefail
probe=${0:A:h:h}/.build/release/qwen-image
cold=${COLD_TFLOPS:-13.1}
gate=${GATE_TFLOPS:-$(echo "$cold * 0.98" | bc -l)}
passes=${GATE_PASSES:-2}
spacing=${GATE_SPACING:-30}
waited=0; streak=0
while true; do
  line=$($probe probe)
  state=$(echo "$line" | awk '{print $2}'); tflops=$(echo "$line" | awk '{print $4}')
  if [[ $state == nominal ]] && (( $(echo "$tflops >= $gate" | bc -l) )); then
    streak=$((streak + 1))
    (( streak >= passes )) && break
    sleep $spacing; waited=$((waited + spacing))
  else
    streak=0; sleep 20; waited=$((waited + 20))
  fi
done
echo "gate passed after ${waited}s (${passes} probes): $line"
