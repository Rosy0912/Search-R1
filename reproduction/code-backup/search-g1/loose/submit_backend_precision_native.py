import sys,json,subprocess,shlex,uuid,base64,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8-backend-precision-'+uuid.uuid4().hex[:12]
wrapper='"""Native-only frozen-source precision observation, no training writes."""\nimport os,json,hashlib\nimport torch\nfrom verl.workers.fsdp_workers import ActorRolloutRefWorker\nimport reproduction.common8.model_smoke as smoke\ninstances={}\nimport verl.utils.torch_functional as functional\noriginal_ce=functional.logprobs_from_logits\nce_calls=0\ndef observe_ce(logits,labels):\n global ce_calls\n result=original_ce(logits,labels);ce_calls+=1\n flat=logits.reshape(-1,logits.shape[-1]);ys=labels.reshape(-1)\n # Compare actual selected-label CE on the same logits; bounded 64 rows only.\n indices=torch.linspace(0,flat.shape[0]-1,min(64,flat.shape[0]),device=flat.device).long()\n selected=flat.index_select(0,indices).float();target=ys.index_select(0,indices)\n fp32=torch.log_softmax(selected,dim=-1).gather(1,target[:,None]).squeeze(1)\n delta=(result.reshape(-1).index_select(0,indices).float()-fp32).abs()\n emit(\'same_logits_ce_fp32\',call=ce_calls,rows=len(indices),logits_dtype=str(logits.dtype),max_abs=float(delta.max()),mean_abs=float(delta.mean()),finite=bool(torch.isfinite(delta).all()))\n return result\nfunctional.logprobs_from_logits=observe_ce\ndef emit(kind,**data):\n record=dict(kind=kind,rank=int(os.environ[\'RANK\']),**data)\n from pathlib import Path\n root=Path(os.environ[\'SEARCH_G1_BASE\'])/\'logs\'\n with (root/(\'precision-rank-\'+os.environ[\'RANK\']+\'.jsonl\')).open(\'a\') as f:\n  f.write(json.dumps(record)+\'\\n\');f.flush();os.fsync(f.fileno())\ndef signature(module):\n result={}\n for n,b in module.named_buffers():\n  if n.endswith(\'inv_freq\'):\n   assert b.dtype==torch.float32 and torch.isfinite(b).all()\n   t=b.detach().cpu().contiguous()\n   result[n]=dict(dtype=str(b.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())\n assert result\n return result\ninit=ActorRolloutRefWorker.init_model\ndef initialize(self):\n result=init(self)\n if self._is_actor:instances[\'actor\']=self;emit(\'actor_initialized\',buffers=signature(self.actor_module_fsdp),optimizer_state_entries=len(self.actor_optimizer.state))\n if self._is_ref:instances[\'reference\']=self;emit(\'reference_initialized\',buffers=signature(self.ref_module_fsdp))\n if self._is_actor:\n  vm=self.rollout.inference_engine.llm_engine.model_executor.worker.model_runner.model\n  caches=[]\n  for name,child in vm.named_modules():\n   cache=getattr(child,\'cos_sin_cache\',None)\n   if cache is None:continue\n   positions=torch.tensor([x for x in (0,1,16,128,1024,4095) if x<cache.shape[0]],dtype=torch.long)\n   dim=child.rotary_dim;base=child.base\n   inv=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))\n   freqs=torch.outer(positions.float(),inv)\n   cpu=torch.cat((freqs.cos(),freqs.sin()),dim=-1).to(cache.dtype)\n   observed=cache.detach().index_select(0,positions.to(cache.device)).cpu()\n   delta=(observed.float()-cpu.float()).abs()\n   caches.append(dict(name=name,base=base,rotary_dim=dim,dtype=str(cache.dtype),positions=positions.tolist(),max_abs_vs_cpu=float(delta.max()),mean_abs_vs_cpu=float(delta.mean())))\n  emit(\'vllm_rotary_cache_vs_cpu\',caches=caches)\n return result\nActorRolloutRefWorker.init_model=initialize\ncompare=smoke.compare\ndef observed(old,reference,mask,tokens,logs,responses):\n valid=mask.bool().cpu();o=old.cpu().float();r=reference.cpu().float();raw=logs.cpu().float();v=valid[:,:raw.shape[1]]\n ar=(o[valid]-r[valid]).abs();av=(o[:,:raw.shape[1]][v]-raw[v]).abs()\n emit(\'actual_parity\',actor_ref_max=float(ar.max()),actor_vllm_max=float(av.max()),actor_vllm_mean=float(av.mean()),actor_vllm_q99=float(torch.quantile(av,.99)),valid_tokens=int(v.sum()),actor_buffers=signature(instances[\'actor\'].actor_module_fsdp),reference_buffers=signature(instances[\'reference\'].ref_module_fsdp),optimizer_state_entries=len(instances[\'actor\'].actor_optimizer.state),optimizer_writes=getattr(instances[\'actor\'].actor,\'_optimizer_writes\',0))\n return compare(old,reference,mask,tokens,logs,responses)\nsmoke.compare=observed\nsmoke.main(os.environ[\'SEARCH_G1_BASE\'],\'initial\')\n'
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
python -m reproduction.common8.prepare_run --base "$BASE" --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725" --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit 638a063
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
shutil.copytree(Path('/dataset/shaoshitong/g1-common8-rope-fix-cpu-2f7413107aa6'),b/'repo',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
run=json.loads((old/'logs/run-manifest.json').read_text())
expected=dict(run['source_sha256']);expected.update({'verl/workers/fsdp_workers.py': '0ea49eb3b169bfab4de2c31d557459604e65f1ec01d3b1f104c62509c2ea3a11', 'verl/workers/actor/dp_actor.py': '7e42710cbe6b9f86bb2f337628b32dbaf4930ca03228c306d4365958e3b36161', 'reproduction/common8/model_smoke.py': 'dfc6dd95a59277faf1590c2018643a931fbb587ab2cf2da8f5b0b1fec7541b3f', 'reproduction/common8/test_model_smoke.py': '0d78717e1915e5dc176c35ae5d93781603fdc64ca39028482ecee507a04c7859'})
expected['verl/utils/fsdp_utils.py']='5126d6f195486fb58996f9ede8906e788c2d4ec2bffb7b4902fa21950f26b20b'
expected['verl/utils/rope_precision.py']='c43b4b489de5cdf71d322b7a27e9044f5448d05412fdc6588a3d4f64c03d300c'
expected['reproduction/common8/test_rope_precision.py']='0c17c477d10080b93a96d1de8c475d0d387cd3d02e986d06e838085877f1c780'
for n,h in expected.items():assert hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest()==h
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):os.link(old/'data'/n,b/'data'/n)
shutil.copy2(old/'preparation/official-model-manifest-original.json',b/'preparation/official-model-manifest-original.json')
(b/'preparation/device_diagnostic.py').write_bytes(base64.b64decode(wrapper));(b/'preparation/job.sh').write_bytes(base64.b64decode(script))
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=01:00:00','--job-name=search-g1-device-diagnostic','--chdir='+str(b/'repo'),'--output='+str(b/'logs/slurm-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b),str(b/'preparation/job.sh')]
with (b/'preparation/submit-claim.json').open('x') as f:json.dump(dict(scope='native actual model device observation only; zero GRPO, original recipe and kernel unchanged',unix=time.time()),f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr
out=dict(base=str(b),job_id=r.stdout.strip().split(';')[0],source_commit='638a063',wrapper_sha256=hashlib.sha256((b/'preparation/device_diagnostic.py').read_bytes()).hexdigest(),argv=args,actual_grpo_updates=0,scope='Observe repaired actual FP32 rotary hashes and original model parity; no rawQA or optimizer writes',submitted_unix=time.time());(b/'preparation/launch.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
compile(code,'remote','exec');r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'backend-precision-native-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
