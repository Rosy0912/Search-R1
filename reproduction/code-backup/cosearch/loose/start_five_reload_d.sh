set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
RUN=$R/runs/cosearch-local-gpu-five-20261003-d
if [ ! -f "$RUN/driver-exit.json" ]; then echo WAITING_FIVE_DRIVER; exit 0; fi
if [ -f "$RUN/acceptance.json" ]; then cat "$RUN/acceptance.json"; exit 0; fi
if [ -f "$RUN/reload.queued" ]; then echo RELOAD_ALREADY_QUEUED; exit 0; fi
test -z "$(squeue -h --name=cosearch-local-five -o '%i')"
test -z "$(squeue -h --name=cosearch-five-reload -o '%i')"
bash "$R/tools/capture_worker_d.sh"
"$R/env/bin/python" - "$RUN" <<'CHECK'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);assert json.loads((r/'driver-exit.json').read_text())['returncode']==0
text=(r/'training.log').read_text(errors='replace')
if 'COSEARCH_OUTER_METRICS_JSON=' not in text:text=(r/'worker-training.log').read_text(errors='replace')
records=[]
for line in text.splitlines():
 if 'COSEARCH_OUTER_METRICS_JSON=' in line:
  records.append(json.JSONDecoder().raw_decode(line.split('COSEARCH_OUTER_METRICS_JSON=',1)[1])[0])
assert [x['step'] for x in records]==[1,2,3,4,5]
assert records[-1]['metrics']['training/actual_outer_updates']==5
assert records[-1]['metrics']['val/qa_count/all']==64
CHECK
# Claim this single reload launch atomically. Do not repeat an incomplete/failed queued run.
(set -o noclobber; echo "$(date -u +%FT%TZ)" > "$RUN/reload.queued")
tmux new-session -d -s cosearch-five-reload-d-20261003 "srun --immediate=60 -p gpu -w h100-16 -N1 --ntasks=1 --cpus-per-task=8 --mem=128G --job-name=cosearch-five-reload '$R/env/bin/python' '$R/tools/check_five_acceptance.py' --run '$RUN' > '$RUN/reload.log' 2>&1"
echo RELOAD_SUBMITTED
