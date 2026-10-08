import sys,json,subprocess,shlex,uuid,base64,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8-device-diagnostic-'+uuid.uuid4().hex[:12]
wrapper=r'''import os,json
from pathlib import Path
import torch
import verl.utils.torch_functional as functional
original=functional.logprobs_from_logits
count=0
def observe(logits,labels):
 global count
 count+=1
 print('COMMON8_DEVICE_DIAGNOSTIC='+json.dumps(dict(rank=int(os.environ['RANK']),call=count,logits_device=str(logits.device),labels_device=str(labels.device),logits_dtype=str(logits.dtype),labels_dtype=str(labels.dtype),logits_shape=list(logits.shape),labels_shape=list(labels.shape),labels_pointer=labels.data_ptr(),cuda_current_device=torch.cuda.current_device())),flush=True)
 return original(logits,labels)
functional.logprobs_from_logits=observe
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
def batchmeta(stage,batch):
 print('COMMON8_BATCH_DEVICE_DIAGNOSTIC='+json.dumps(dict(stage=stage,rank=int(os.environ['RANK']),cuda_current_device=torch.cuda.current_device(),tensors={k:dict(device=str(v.device),pointer=v.data_ptr(),storage_pointer=v.untyped_storage().data_ptr(),shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch.items()})),flush=True)
forward=DataParallelPPOActor._forward_micro_batch
def micro(self,batch,temperature):
 batchmeta('actor_microbatch_before_forward',batch)
 return forward(self,batch,temperature)
DataParallelPPOActor._forward_micro_batch=micro
generate=vLLMRollout.generate_sequences
def rollout(self,*args,**kwargs):
 out=generate(self,*args,**kwargs);batchmeta('rollout_return',out.batch);return out
vLLMRollout.generate_sequences=rollout
post=FSDPVLLMShardingManager.postprocess_data
def postprocess(self,data):
 out=post(self,data);batchmeta('sharding_postprocess_return',out.batch);return out
FSDPVLLMShardingManager.postprocess_data=postprocess
from reproduction.common8.model_smoke import main
main(os.environ['SEARCH_G1_BASE'],'initial')
'''
script='''#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?}"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export SEARCH_G1_PROFILE=common8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" XDG_CACHE_HOME="$BASE/cache" HF_HOME="$BASE/cache/hf"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS
cd "$BASE/repo"
python -m reproduction.common8.prepare_run --base "$BASE" --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725" --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit 4b1a06243e915c73b439fca8c62ba5f928ca812b
python - <<'ADMIT'
import os,json
from pathlib import Path
from reproduction.common8.native_allocation import admit
b=Path(os.environ['SEARCH_G1_BASE']);run=json.loads((b/'logs/run-manifest.json').read_text());print(admit(b))
ADMIT
exec python -m torch.distributed.run --standalone --nproc_per_node=8 "$BASE/preparation/device_diagnostic.py"
'''
code='import os,json,shutil,subprocess,time,hashlib,base64;from pathlib import Path\nb=Path('+repr(base)+');wrapper='+repr(base64.b64encode(wrapper.encode()).decode())+';script='+repr(base64.b64encode(script.encode()).decode())+'\n'+r'''
old=Path('/dataset/shaoshitong/Search-G1-common8-attempt2-39e7968c7831');assert json.loads((old/'logs/pipeline-status.json').read_text())['stage']=='FAILED'
ledger=json.loads((old/'logs/process-ledger.json').read_text())
for job in ledger['jobs'].values():
 assert job['state']=='STOPPED'
 for e in job['ledger'].values():
  p=Path('/proc')/str(e['pid'])
  assert not p.exists() or int((p/'stat').read_text().rsplit(')',1)[1].split()[19])!=e['start_ticks']
b.mkdir(exist_ok=False)
for n in ('data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/n).mkdir(parents=True)
for n in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):(b/n).symlink_to(old/n,target_is_directory=True)
shutil.copytree(old/'repo',b/'repo',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
run=json.loads((old/'logs/run-manifest.json').read_text())
for n,h in run['source_sha256'].items():assert hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest()==h
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):os.link(old/'data'/n,b/'data'/n)
shutil.copy2(old/'preparation/official-model-manifest-original.json',b/'preparation/official-model-manifest-original.json')
(b/'preparation/device_diagnostic.py').write_bytes(base64.b64decode(wrapper));(b/'preparation/job.sh').write_bytes(base64.b64decode(script))
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=01:00:00','--job-name=search-g1-device-diagnostic','--chdir='+str(b/'repo'),'--output='+str(b/'logs/slurm-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b),str(b/'preparation/job.sh')]
with (b/'preparation/submit-claim.json').open('x') as f:json.dump(dict(scope='native actual model device observation only; zero GRPO, original recipe and kernel unchanged',unix=time.time()),f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr
out=dict(base=str(b),job_id=r.stdout.strip().split(';')[0],source_commit=run['source_commit'],wrapper_sha256=hashlib.sha256((b/'preparation/device_diagnostic.py').read_bytes()).hexdigest(),argv=args,actual_grpo_updates=0,scope='Read device/shape metadata immediately before original CE kernel; no values/rawQA, no optimizer writes',submitted_unix=time.time());(b/'preparation/launch.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
compile(code,'remote','exec');r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'model-device-diagnostic-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
