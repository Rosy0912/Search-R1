import sys,json,subprocess,shlex,uuid,base64,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8-rope-causal-'+uuid.uuid4().hex[:12]
wrapper='"""Disposable native-only, same-token reference buffer intervention; no trainer."""\nimport os,json,hashlib\nfrom pathlib import Path\nimport torch\nfrom verl.workers.fsdp_workers import ActorRolloutRefWorker\ninstances={}\ninit=ActorRolloutRefWorker.init_model\n\ndef record(name,**data):\n print(\'COMMON8_ROPE_CAUSAL=\'+json.dumps(dict(kind=name,rank=int(os.environ[\'RANK\']),**data)),flush=True)\n\ndef hash_tensor(t):\n t=t.detach().cpu().contiguous()\n return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()\n\ndef initialize(self):\n result=init(self)\n if self._is_actor:instances[\'actor\']=self\n if self._is_ref:instances[\'reference\']=self\n return result\nActorRolloutRefWorker.init_model=initialize\nreference_call=ActorRolloutRefWorker.compute_ref_log_prob\n\ndef summarize(old,value,mask):\n delta=(old[mask].float()-value[mask].float()).abs()\n assert torch.isfinite(delta).all()\n return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),q99_abs=float(torch.quantile(delta,.99)),valid_tokens=int(mask.sum()))\n\ndef intervene(self,data):\n actor=instances[\'actor\'];am=actor.actor_module_fsdp;rm=self.ref_module_fsdp\n old=data.batch[\'old_log_probs\'].detach().cpu().clone();mask=data.batch[\'attention_mask\'][:,-old.shape[-1]:].detach().cpu().bool()\n ap=dict(am.named_parameters());rp=dict(rm.named_parameters());assert set(ap)==set(rp)\n parameter_equal={}\n for name,a in ap.items():\n  r=rp[name];assert a.shape==r.shape\n  parameter_equal[name]=bool(torch.equal(a.detach().cpu().to(torch.bfloat16),r.detach().cpu().to(torch.bfloat16)))\n assert all(parameter_equal.values()), \'Fresh actor/reference BF16 parameter shards differ\'\n record(\'canonical_bf16_shard_comparison\',equal=True,shards=len(parameter_equal),source=\'actual FSDP parameter shards at each rank, dtype-normalized only for comparison\')\n ab=dict(am.named_buffers());rb=dict(rm.named_buffers());names=sorted(n for n in ab if n.endswith(\'inv_freq\'))\n assert names and all(n in rb for n in names)\n original_buffers={};buffers=[]\n for n in names:\n  a,r=ab[n],rb[n];assert a.shape==r.shape\n  assert a.dtype==torch.float32 and torch.isfinite(a).all() and torch.isfinite(r).all()\n  parent,key=n.rsplit(\'.\',1);module=rm.get_submodule(parent)\n  assert module._buffers[key] is r\n  original_buffers[n]=(module,key,r)\n  buffers.append(dict(name=n,actor_dtype=str(a.dtype),reference_dtype=str(r.dtype),shape=list(a.shape),actor_sha256=hash_tensor(a),reference_sha256=hash_tensor(r),max_abs=float((a.detach().cpu().float()-r.detach().cpu().float()).abs().max())))\n record(\'rope_buffers_before\',buffers=buffers)\n # Hash all teacher-forced IDs/masks/positions before either reference call.\n keys=(\'input_ids\',\'responses\',\'attention_mask\',\'position_ids\');input_hash={k:hash_tensor(data.batch[k]) for k in keys}\n before=reference_call(self,data)\n assert {k:hash_tensor(data.batch[k]) for k in keys}==input_hash\n record(\'reference_original\',delta=summarize(old,before.batch[\'ref_log_prob\'].cpu(),mask),input_sha256=input_hash)\n try:\n  for n,(module,key,r) in original_buffers.items():module._buffers[key]=ab[n].detach().clone().to(device=r.device)\n  after=reference_call(self,data)\n  assert {k:hash_tensor(data.batch[k]) for k in keys}==input_hash\n  record(\'reference_fp32_actor_rope_intervention\',delta=summarize(old,after.batch[\'ref_log_prob\'].cpu(),mask),input_sha256=input_hash,buffers=[dict(name=n,dtype=str(module._buffers[key].dtype),sha256=hash_tensor(module._buffers[key])) for n,(module,key,r) in original_buffers.items()],semantics=\'Only reference inv_freq replaced by actual unrounded actor FP32 buffer; official parameters/tokens/masks/positions unchanged\')\n finally:\n  for n,(module,key,r) in original_buffers.items():module._buffers[key]=r\n  assert all(hash_tensor(rb[n])==hash_tensor(module._buffers[key]) for n,(module,key,r) in original_buffers.items())\n  record(\'reference_restored\',restored=True,optimizer_writes=getattr(actor.actor,\'_optimizer_writes\',0),optimizer_state_entries=len(actor.actor_optimizer.state))\n # Return the original reference result to the original smoke gate; do not make\n # the causal diagnostic impersonate a production parity acceptance.\n return before\nActorRolloutRefWorker.compute_ref_log_prob=intervene\nfrom reproduction.common8.model_smoke import main\nmain(os.environ[\'SEARCH_G1_BASE\'],\'initial\')\n'
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
python -m reproduction.common8.prepare_run --base "$BASE" --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725" --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit 1e19d6d60e69d273aef53d72874f8d13da8d25f5
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
shutil.copytree(Path('/dataset/shaoshitong/g1-common8-device-fix-cpu-d5a7a2d82f65'),b/'repo',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
run=json.loads((old/'logs/run-manifest.json').read_text())
expected=dict(run['source_sha256']);expected.update({'verl/workers/fsdp_workers.py': '34add4fa0f63eeda732cd0c813095730a825128f75087005b0304f782470abcf', 'verl/workers/actor/dp_actor.py': '7e42710cbe6b9f86bb2f337628b32dbaf4930ca03228c306d4365958e3b36161', 'reproduction/common8/model_smoke.py': 'dfc6dd95a59277faf1590c2018643a931fbb587ab2cf2da8f5b0b1fec7541b3f', 'reproduction/common8/test_model_smoke.py': '0d78717e1915e5dc176c35ae5d93781603fdc64ca39028482ecee507a04c7859'})
for n,h in expected.items():assert hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest()==h
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):os.link(old/'data'/n,b/'data'/n)
shutil.copy2(old/'preparation/official-model-manifest-original.json',b/'preparation/official-model-manifest-original.json')
(b/'preparation/device_diagnostic.py').write_bytes(base64.b64decode(wrapper));(b/'preparation/job.sh').write_bytes(base64.b64decode(script))
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=01:00:00','--job-name=search-g1-device-diagnostic','--chdir='+str(b/'repo'),'--output='+str(b/'logs/slurm-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b),str(b/'preparation/job.sh')]
with (b/'preparation/submit-claim.json').open('x') as f:json.dump(dict(scope='native actual model device observation only; zero GRPO, original recipe and kernel unchanged',unix=time.time()),f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr
out=dict(base=str(b),job_id=r.stdout.strip().split(';')[0],source_commit='1e19d6d60e69d273aef53d72874f8d13da8d25f5',wrapper_sha256=hashlib.sha256((b/'preparation/device_diagnostic.py').read_bytes()).hexdigest(),argv=args,actual_grpo_updates=0,scope='Same-input reference-only original-FP32 RoPE intervention and restoration; original gate result retained; no trainer/optimizer writes',submitted_unix=time.time());(b/'preparation/launch.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
compile(code,'remote','exec');r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'rope-causal-native-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
