#!/bin/bash
set -eu
run=$1
python /dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/tools/owned_processes.py monitor --run "$run" --retriever "$2" --ray "$3" --output "$run/cpu-process-monitor.jsonl" &
observer=$!

while true; do
 date -u +%FT%TZ >> "$run/gpu-monitor.csv"
 nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$run/gpu-monitor.csv"
 date -u +%FT%TZ >> "$run/process-monitor.csv"
 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$run/process-monitor.csv"

 sleep 10
done
