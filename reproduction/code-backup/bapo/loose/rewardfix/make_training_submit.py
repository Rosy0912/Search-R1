from pathlib import Path
r=Path('work/rewardfix/training-tools')
base=Path('work/rewardfix/remote/parity-gpu64.sbatch').read_text()
for phase in ['smoke2','accept5','formal500']:
 s=base.replace('bapo-rfix-hw-b097','bapo-rfix-'+phase+'-b097').replace('slurm-parity-', 'slurm-'+phase+'-').replace('stage.sh hardware', 'stage.sh '+phase)
 (r/(phase+'.sbatch')).write_text(s)
script='''#!/bin/bash
set -euo pipefail
D=/dev/shm/bapo-rewardfix-b097-scheduler-20261004
R=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097
phase=${1:?smoke2|accept5|formal500}
case "$phase" in smoke2|accept5|formal500) ;; *) exit 2;; esac
exec 9> "$D/$phase-submit.lock"
flock -n 9
if [ -f "$D/$phase-job-id.txt" ]; then
 id=$(cat "$D/$phase-job-id.txt")
 case "$id" in ''|*[!0-9]*) exit 2;; esac
 printf '%s\\n' "$id";exit 0
fi
if [ -f "$D/$phase-submit-intent.txt" ]; then
 echo 'Prior submission outcome unknown: reconcile without resubmitting' >&2;exit 3
fi
if [ "$phase" = accept5 ]; then
 id=${SMOKE2_JOB_ID:?Independent corrected smoke receipt required}
 case "$id" in ''|*[!0-9]*) exit 2;; esac
 gate=" $R/env/bin/python $R/tools/verify_smoke_receipt-gpu64.py $id"
elif [ "$phase" = formal500 ]; then
 id=${ACCEPT5_JOB_ID:?Independent corrected five receipt required}
 case "$id" in ''|*[!0-9]*) exit 2;; esac
 gate=" $R/env/bin/python $R/tools/verify_five_receipt-gpu64.py $id"
else
 gate=" $R/env/bin/python $R/tools/verify_hardware_receipt.py"
fi
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 \
 "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONDONTWRITEBYTECODE=1 $gate"
active=$(squeue -h -w h100-5 -o '%j' | sed -n '/^bapo-rfix-/p')
if [ -n "$active" ]; then
 echo 'Corrected-source task still active; do not start another instance' >&2;exit 3
fi
printf 'source=b09736d9d1b487a5c134e7a358761e04613cea32\\nphase=%s\\n' "$phase" > "$D/$phase-submit-intent.txt"
if ! id=$(sbatch --parsable "$D/$phase.sbatch"); then
 echo 'Unknown/failed submission: intent retained' >&2;exit 3
fi
case "$id" in ''|*[!0-9]*) echo 'Invalid response: intent retained' >&2;exit 3;; esac
printf '%s\\n' "$id" > "$D/$phase-job-id.txt"
scontrol show job "$id"
'''
(r/'submit-training.sh').write_text(script)
