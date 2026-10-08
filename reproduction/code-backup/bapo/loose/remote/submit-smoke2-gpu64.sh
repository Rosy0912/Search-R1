set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
R=/dataset/shaoshitong/search-r1-rl/bapo
exec 9> "$D/smoke2-gpu64-submit.lock"
flock -n 9
if [ -f "$D/smoke2-gpu64-job-id.txt" ];then cat "$D/smoke2-gpu64-job-id.txt";exit 0;fi
existing=$(squeue -h -n bapo-smoke2-gpu64 -o '%i')
if [ -n "$existing" ];then printf '%s\n' "$existing" > "$D/smoke2-gpu64-job-id.txt";scontrol show job "$existing";exit 0;fi
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $R/env/bin/python $R/tools/verify_hardware_receipt.py"
id=$(sbatch --parsable "$D/smoke2-gpu64.sbatch")
[[ "$id" =~ ^[0-9]+$ ]]
printf '%s\n' "$id" > "$D/smoke2-gpu64-job-id.txt"
scontrol show job "$id"
