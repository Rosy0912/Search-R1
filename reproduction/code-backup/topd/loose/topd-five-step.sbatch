#!/bin/bash
#SBATCH --job-name=topd-reviewed-short-run
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-short-%j.log
set -euo pipefail
# Only this wrapper supplies execution flags. Its first invocation is CPU read-only.
for argument in "$@"; do
  case "$argument" in
    --run|--check-execution-proof) echo "Do not supply execution flags to SBATCH wrapper" >&2; exit 64 ;;
  esac
done
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
cd "$AGENT_ROOT/repo"
# Operator first runs dependency-only gate without --run; no waiting allocation
# for missing teacher/common fix. This file never submits another Slurm job.
export PATH="$AGENT_ROOT/env/bin:$PATH"
"$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/topd-short-run.py" --check-execution-proof "$@"
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/topd-short-run.py" --run "$@"
