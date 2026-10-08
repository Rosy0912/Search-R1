#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005
BASE=$ROOT/scheduler-head
D=$ROOT/scheduler-recovery3237
exec 8>"$BASE/submit-global.lock"
flock -n 8
if [ -f "$D/hardware-job-id.txt" ]; then
 id=$(cat "$D/hardware-job-id.txt");case "$id" in ''|*[!0-9]*) exit 2;; esac
 printf '%s\n' "$id";exit 0
fi
if [ -f "$D/hardware-submit-intent.txt" ]; then
 echo 'Unknown retry submission outcome; reconcile same intent, never repeat' >&2;exit 3
fi
python3 - "$BASE" "$D" <<'PY'
import pathlib,json,hashlib,re,sys
base,d=map(pathlib.Path,sys.argv[1:]);assert d==d.resolve()and not d.is_symlink()
m=json.loads((d/'approved-retry-hashes.json').read_text())
assert base==base.resolve()and not base.is_symlink()
for p,k in [(d/'submit-hardware-after3237.sh','retry_script'),(base/'hardware.sbatch','sbatch'),(base/'hardware-3237-terminal.txt','prior_terminal')]:assert hashlib.sha256(p.read_bytes()).hexdigest()==m[k]
assert (base/'hardware-job-id.txt').read_text().strip()=='3237'
t=(base/'hardware-3237-terminal.txt').read_text()
for field,val in [('JobId','3237'),('JobState','FAILED'),('ExitCode','1:0'),('Restarts','0'),('NodeList','h100-5')]:assert re.search(r'(?:^|\s)'+field+'='+re.escape(val)+r'(?:\s|$)',t)
review=d/'RETRY3237-INDEPENDENT-REVIEW.md';assert hashlib.sha256(review.read_bytes()).hexdigest()==m['independent_review']
PY
# The original worker tools/source and all original submit records remain fixed.
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "$ROOT/env/bin/python $ROOT/tools/verify_launch.py hardware"
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "$ROOT/env/bin/python $ROOT/tools/verify_shared_inputs.py"
queue=$(squeue -h -w h100-5 -o '%i|%j')
QUEUE_SNAPSHOT="$queue" python3 - <<'PYQUEUE'
import os
for line in os.environ['QUEUE_SNAPSHOT'].splitlines():
 fields=line.split('|');assert len(fields)==2,'Malformed scheduler snapshot'
 job,name=(x.strip()for x in fields)
 assert job!='3237'and 'bapo'not in name.lower(),'Prior/owned BAPO job still active; no second instance'
PYQUEUE
printf '%s\n' hardware-after-failed3237 > "$D/hardware-submit-intent.txt"
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "python3 -c 'import pathlib,json; r=pathlib.Path(\"$ROOT\"); assert not list((r/\"runs\").iterdir()); assert (r/\"source-expected.txt\").read_text().strip()==\"7de6494bff08f35ffeefbf23ed9a9e1b73e5612e\"; p=r/\"logs/submission-intent-hardware-after3237.json\"; f=p.open(\"x\"); json.dump({\"phase\":\"hardware\",\"prior_terminal_job\":3237,\"source_commit\":\"7de6494bff08f35ffeefbf23ed9a9e1b73e5612e\"},f); f.close()'"
if ! id=$(sbatch --parsable "$BASE/hardware.sbatch");then echo 'Unknown retry outcome; preserve intent' >&2;exit 3;fi
case "$id" in ''|0|*[!0-9]*) exit 3;; esac
printf '%s\n' "$id" > "$D/hardware-job-id.txt"
scontrol show job "$id"
