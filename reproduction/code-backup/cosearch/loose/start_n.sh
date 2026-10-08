set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
test -z "$(squeue -h --name=cosearch-local-five -o %i)"
test ! -d "$R/runs/cosearch-local-gpu-five-20261003-n"
test "$(git -C "$R/repo" rev-parse --short HEAD)" = e2e401c
tmux new-session -d -s cosearch-local-gpu-five-n-20261003 "srun --immediate=60 -p gpu -w h100-16 -N1 --ntasks=1 --gres=gpu:h100:8 --exclusive --cpus-per-task=128 --mem=768G --job-name=cosearch-local-five env PYTHONPATH='$R/repo' COSEARCH_RUN_ID=cosearch-local-gpu-five-20261003-n '$R/env/bin/python' -m search_r1.baselines.preflight -- '$R/env/bin/python' '$R/tools/five_iteration_job_n.py' > '$R/logs/local-five-n.log' 2>&1"
sleep 3
squeue -h --name=cosearch-local-five -o '%i %T %N'
tail -4 "$R/logs/local-five-n.log"
