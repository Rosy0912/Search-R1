#!/bin/bash
set -euo pipefail
CONTROL=/dev/shm/shaoshitong-tree-grpo-control-20261003
exec 9>"$CONTROL/five-hybrid-versioned-dispatch.lock"
flock -n 9 || exit 1
test ! -e "$CONTROL/five-hybrid-versioned-dispatched.marker"
# Immutable gate receipt is inspected before allocating training GPUs.
# The stage performs the complete checker with its fresh local CPU instance.
ACCEPTANCE_SHA=$(python3 - "$CONTROL/hybrid-retrieval-acceptance-five-2879.json" "$CONTROL/hybrid-retrieval-acceptance-five-2879.sha256" <<'CHECK'
import pathlib,sys,hashlib,json,subprocess,shlex,time
raw=pathlib.Path(sys.argv[1]).read_bytes();assert hashlib.sha256(raw).hexdigest()==pathlib.Path(sys.argv[2]).read_text().strip()
r=json.loads(raw);assert r['status']=='ACCEPTED_AFTER_HYBRID_BINARY_HTTP_DIFFERENCE_REVIEW' and 'five' in r['permitted_stages']
assert r['index_deadline_epoch']-time.time()>=7500, 'Insufficient accepted auxiliary lifetime for 2h allocation plus queue/start margin'
remote_code="""import pathlib,hashlib,json,urllib.request,time
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/hybrid-retrieval-acceptance-five-2879.json');raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()==%r
r=json.loads(raw);assert 'five' in r['permitted_stages']
assert r['index_deadline_epoch']-time.time()>=7500
a=json.loads(pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/two-stage-acceptance.json').read_text());assert a['status']=='REAL_TREE_STAGE_ACCEPTED' and a['actual_outer_updates']==2 and a['stage']=='two'
for file,sha in a['evidence_file_sha256'].items():assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==sha
for file,sha in r['evidence_file_sha256'].items():assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==sha
with urllib.request.urlopen('http://127.0.0.1:18083/provenance',timeout=30) as f:assert json.load(f)==r['index_instance']
""" % hashlib.sha256(raw).hexdigest()
subprocess.run(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','root@h100-9','python3 -c '+shlex.quote(remote_code)],check=True,timeout=120)
print(hashlib.sha256(raw).hexdigest())
CHECK
)
date -u +%FT%TZ > "$CONTROL/five-hybrid-versioned-dispatched.marker"
set +e
srun -p gpu -w h100-9 -N1 --gres=gpu:h100:8 --exclusive --ntasks=1 --cpus-per-task=32 --mem=256G --time=02:00:00 --job-name=tree-five-hybrid-versioned env TREE_STAGE_MAX_SECONDS=7200 TREE_RETRIEVER_BACKEND=hybrid TREE_HYBRID_ACCEPTANCE_FILE=/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/hybrid-retrieval-acceptance-five-2879.json TREE_HYBRID_ACCEPTANCE_SHA256="$ACCEPTANCE_SHA" bash /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/train_tree_stage_versioned.sh five
rc=$?
echo "$rc" > "$CONTROL/five-hybrid-versioned-job.exit"
exit "$rc"
