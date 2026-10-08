set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
exec 9> "$D/parity-gpu64-submit.lock"
flock -n 9
if [ -f "$D/parity-gpu64-job-id.txt" ];then cat "$D/parity-gpu64-job-id.txt";exit 0;fi
existing=$(squeue -h -n bapo-hw-gpu64 -o '%i')
if [ -n "$existing" ];then printf '%s\n' "$existing" > "$D/parity-gpu64-job-id.txt";scontrol show job "$existing";exit 0;fi
id=$(sbatch --parsable "$D/parity-gpu64.sbatch")
[[ "$id" =~ ^[0-9]+$ ]]
printf '%s\n' "$id" > "$D/parity-gpu64-job-id.txt"
scontrol show job "$id"
