#!/usr/bin/env bash
set -euo pipefail
base=/dataset/shaoshitong/recursive-bran-a-h100-20261012
mkdir -p "$base/control" "$base/logs"
export RBA_H100_ALLOCATION_ID="$SLURM_JOB_ID"
python3 - "$base" <<'PY'
import json,os,pathlib,time,subprocess, socket
b=pathlib.Path(__import__('sys').argv[1]);d=dict(time=time.time(),hostname=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader'],text=True),phase='MIGRATION_PREPARATION_NOT_TRAINING')
(b/'control/allocation-live.json').write_text(json.dumps(d,indent=2))
PY
deadline=$((SECONDS+7200))
while [[ ! -f "$base/control/launch-ready.sh" ]]; do
  if ((SECONDS>=deadline)); then echo 'Migration preparation deadline reached; no training launched'; exit 2; fi
  sleep 5
done
exec bash "$base/control/launch-ready.sh"
