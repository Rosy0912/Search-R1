#!/bin/bash
#SBATCH --job-name=topd-data-prepare
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/data-prepare-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
test -f "$AGENT_ROOT/logs/full-copy-verification.ready"
export CUDA_VISIBLE_DEVICES="" SEARCH_R1_ASSETS="$AGENT_ROOT/assets-root" SEARCH_R1_REPO="$AGENT_ROOT/repo"
export SEARCH_R1_BASELINE_OUTPUT="$AGENT_ROOT/outputs"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/repo"
"$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/prepare-topd-data.py"
wiki="$AGENT_ROOT/assets-root/assets/wiki"
if [ ! -f "$wiki/e5_Flat.index" ]; then
  cat "$wiki/part_aa" "$wiki/part_ab" > "$wiki/e5_Flat.index.partial"
  mv "$wiki/e5_Flat.index.partial" "$wiki/e5_Flat.index"
fi
if [ ! -f "$wiki/wiki-18.jsonl" ]; then
  gzip -dc "$wiki/wiki-18.jsonl.gz" > "$wiki/wiki-18.jsonl.partial"
  mv "$wiki/wiki-18.jsonl.partial" "$wiki/wiki-18.jsonl"
fi
sha256sum "$wiki/e5_Flat.index" "$wiki/wiki-18.jsonl" > "$AGENT_ROOT/logs/derived-retrieval-assets.sha256"
touch "$AGENT_ROOT/logs/data-preparation.ready"
printf 'TOPD_DATA_PREPARATION_PASS\n'
