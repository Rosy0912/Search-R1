#!/bin/bash
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
RUN=$R/runs/bapo-fp32-cpu-parity-20261003
$R/env/bin/python $R/tools/bapo-retrieval-fp32-parity.py --mode compare --left "$R/runs/bapo-hardware-2855/retrieval-gpu.json" --right "$RUN/parity-cpu-fp32.json" --output "$RUN/comparison-original-gpu-fp32.json" > "$RUN/compare-gpu.log"
$R/env/bin/python $R/tools/bapo-retrieval-fp32-parity.py --mode compare --left "$R/runs/bapo-hardware-2855/retrieval-cpu.json" --right "$RUN/parity-cpu-fp32.json" --output "$RUN/comparison-cpu-half-fp32.json" > "$RUN/compare-cpu.log"
touch "$RUN/MEASUREMENTS_DONE"
cd "$RUN"
tar -cf - parity-cpu-fp32.json comparison-original-gpu-fp32.json comparison-cpu-half-fp32.json encoder-fp32-512-long.json time.log parity.log compare-gpu.log compare-cpu.log MEASUREMENTS_DONE
