#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Owned allocation required}"
: "${E1_EXPECTED_SOURCE:?Externally fixed source commit required}"
: "${SEARCH_R1_REPO:?Fixed fresh checkout required}"
: "${E1_HARDWARE_TOOLS:?Immutable task tool directory required}"
: "${E1_HARDWARE_MANIFEST_SHA256:?Externally fixed tool manifest SHA required}"
[[ $(hostname -s) == h100-10 ]]
[[ ${SLURM_CPUS_PER_TASK:-0} -ge 32 ]]
source /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/activate
export PYTHONPATH="$SEARCH_R1_REPO"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=4
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
run=/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/state-hardware-$SLURM_JOB_ID
[[ ! -e $run && ! -L $run ]]
mkdir -p "$(dirname "$run")"
mkdir "$run"
export E1_STATE_JOB_RUN=$run E1_STATE_FIXTURE_OUTPUT="$run/fixture"
export E1_ENTRY_SCRIPT=$0
trap 'rc=$?; printf "%s\n" "$rc" > "$run/exit.txt"' EXIT
cd "$SEARCH_R1_REPO"
python - <<'PY'
import os,pathlib,subprocess,socket,json,re,hashlib,csv,io
repo=pathlib.Path(os.environ['SEARCH_R1_REPO']);job=os.environ['SLURM_JOB_ID']
assert repo.is_absolute() and repo.resolve()==repo and socket.gethostname()=='h100-10'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==os.environ['E1_EXPECTED_SOURCE']
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
assert re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',pathlib.Path('/proc/self/cgroup').read_text(),re.MULTILINE)
assert os.environ['SLURM_JOB_NODELIST']=='h100-10'
selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(selected)==8 and len(set(selected))==8
raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
assert 'JobId='+job+' ' in raw and 'JobState=RUNNING' in raw and 'NodeList=h100-10' in raw
assert re.search(r'UserId=\S+\('+str(os.getuid())+r'\)',raw)
tres=re.search(r'(?:^| )(?:AllocTRES|TRES)=([^ ]+)',raw)
assert re.search(r'(?:^| )TresPerNode=gres:gpu:h100:8(?: |$)',raw),raw
if tres:
 gpu_counts=[v for v in tres.group(1).split(',') if v.startswith('gres/gpu=')]
 assert not gpu_counts or gpu_counts==['gres/gpu=8'],raw
p=pathlib.Path(os.environ['E1_STATE_JOB_RUN']);(p/'actual-allocation.txt').write_text(raw)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
(p/'pre-compute-processes.csv').write_text(processes)
assert not processes.strip(),processes
inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
(p/'pre-gpus.csv').write_text(inv)
rows=list(csv.reader(io.StringIO(inv)))
assert len(rows)==8 and all('H100' in v[2] for v in rows) and len({v[1].strip() for v in rows})==8
tools=pathlib.Path(os.environ['E1_HARDWARE_TOOLS'])
base=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/tools')
assert tools.is_absolute() and tools.resolve()==tools and tools.parent==base
assert tools.stat().st_uid==os.getuid()
assert pathlib.Path(os.environ['E1_ENTRY_SCRIPT']).resolve()==tools/'state_job.sh'
man=tools/'manifest.json';b=man.read_bytes()
assert man.stat().st_uid==os.getuid()
assert hashlib.sha256(b).hexdigest()==os.environ['E1_HARDWARE_MANIFEST_SHA256']
m=json.loads(b)
assert m['source_commit']==os.environ['E1_EXPECTED_SOURCE'] and set(m['tools'])=={'state_job.sh','state_entry.sh'}
assert m['fixture']['path']=='reproduction/baselines/e1-300/native_opsd_state_fixture.py'
for name,sha in m['tools'].items():
 q=tools/name
 assert q.resolve()==q and q.stat().st_uid==os.getuid() and hashlib.sha256(q.read_bytes()).hexdigest()==sha
q=repo/m['fixture']['path']
assert q.resolve()==q and hashlib.sha256(q.read_bytes()).hexdigest()==m['fixture']['sha256']
(p/'actual-tools-manifest.json').write_bytes(b)
print('FRESH_OWN_EIGHT_GPU_ALLOCATION_AND_SOURCE_CHECKED',flush=True)
PY
python -m search_r1.baselines.preflight -- bash "$E1_HARDWARE_TOOLS/state_entry.sh" 2>&1 | tee "$run/hardware.log"
python - <<'PY'
import os,pathlib,json,hashlib,subprocess
p=pathlib.Path(os.environ['E1_STATE_JOB_RUN'])
m=json.loads((p/'actual-tools-manifest.json').read_bytes())
tools=pathlib.Path(os.environ['E1_HARDWARE_TOOLS'])
for name,sha in m['tools'].items():
 assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==sha
repo=pathlib.Path(os.environ['SEARCH_R1_REPO'])
assert hashlib.sha256((repo/m['fixture']['path']).read_bytes()).hexdigest()==m['fixture']['sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==m['source_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
print('POST_HARDWARE_SOURCE_AND_TOOL_BYTES_UNCHANGED',flush=True)
PY
