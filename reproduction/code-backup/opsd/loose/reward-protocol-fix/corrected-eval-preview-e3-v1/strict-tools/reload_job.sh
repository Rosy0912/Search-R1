#!/usr/bin/env bash
set -euo pipefail
echo "STAGED PREVIEW: independent review required" >&2
exit 78
: "${SLURM_JOB_ID:?Must run in own allocation}"
: "${OPSD_FORMAL_COMPLETION_RECEIPT:?Actual successful new formal job completion receipt required}"
root=/mnt/data/shaoshitong/search-r1-rl/opsd
run="$root/runs/opsd-formal-hf-reload-$SLURM_JOB_ID"
[[ ! -e "$run" && ! -L "$run" ]]
mkdir "$run"
trap 'rc=$?; printf "%s\n" "$rc" > "$run/exit.txt"' EXIT
export OPSD_EVIDENCE="$run" OPSD_TEACHER="$root/assets-root/assets/qwen2.5-3b"
export OPSD_CHECKPOINT="$root/runs/opsd-formal-3007/output/checkpoints/opsd-gpu-formal500/actor/global_step_500"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source "$root/env/bin/activate"
cd "$root/repo"
python "$root/tools/hf500-runtime-8e60976d364a-formal-3007-activation-v5/source_bundle_check.py"
exec 9>"$root/gpu-job.lock"
flock -n 9
[[ $(git rev-parse HEAD) == 8e60976d364ab7d7ae51714415813c1359dafe9a ]]
python "$root/tools/hf500-runtime-8e60976d364a-formal-3007-activation-v5/formal_completion_check.py"
[[ -d "$OPSD_CHECKPOINT" ]]
sha256sum "$0" > "$run/reload-job-script.sha256"
python - <<'PY'
import datetime,json,os,re,subprocess
from pathlib import Path
job=os.environ['SLURM_JOB_ID']
node=os.environ.get('SLURMD_NODENAME',os.environ['SLURM_JOB_NODELIST'])
assert node=='h100-10',('Cleanup must run on previous training node',node)
assert re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text(),re.M)
remaining=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try: cgroup=(path/'cgroup').read_text()
    except OSError: continue
    if re.search(r'(?:^|/)job_3007(?:/|$)',cgroup,re.M): remaining.append(int(path.name))
assert not remaining,('Previous training job still has processes',remaining)
listeners=subprocess.check_output(['ss','-H','-ltnp'],text=True)
blocked=[line for line in listeners.splitlines() if len(line.split())>=4 and line.split()[3].rsplit(':',1)[-1] in ('16384','18084')]
assert not blocked,'Previous Ray/retrieval ports still listening'
result=dict(job_id=job,node_scope=node,previous_training_job='3007',previous_job_processes=remaining,previous_ports_listening=False,
            sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
Path(os.environ['OPSD_EVIDENCE'],'previous-job-cleanup.json').write_text(json.dumps(result,indent=2)+'\n')
print('PREVIOUS_TRAINING_JOB_CLEANUP_VERIFIED',result,flush=True)
PY
scontrol show job "$SLURM_JOB_ID" > "$run/slurm-job.txt"
git rev-parse HEAD > "$run/commit.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$run/pre-gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$run/pre-processes.csv"
python - <<'PY'
import os
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
inv,proc=gpu_snapshot()
records=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inv,proc)
assert len(records)==1,records
print('FRESH_RELOAD_GPU_THRESHOLD_OK',records,flush=True)
PY
python -u "$root/tools/hf500-runtime-8e60976d364a-formal-3007-activation-v5/resource_sample.py" "$run/resources.ndjson" $$ &
python -m search_r1.baselines.preflight -- python -u "$root/tools/hf500-runtime-8e60976d364a-formal-3007-activation-v5/hf_launcher.py" --checker "$root/tools/hf500-runtime-8e60976d364a-formal-3007-activation-v5/checkpoint_reload.py" --evidence "$run" --root "$root" --checker-sha256 5586dc36635d333bc2379de689b1128e38e498924649fcfcb00edbde960ff8fb
