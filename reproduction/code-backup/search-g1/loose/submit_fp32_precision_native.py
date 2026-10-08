import sys,json,subprocess,shlex,uuid,base64,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8-fp32-precision-'+uuid.uuid4().hex[:12]
wrapper='"""Isolated complete-FP32 candidate; original parity tolerance unchanged."""\nimport os,json\nfrom pathlib import Path\nimport torch\nfrom omegaconf import open_dict\nimport reproduction.common8.model_smoke as smoke\noriginal_resolved=smoke.resolved\n\n\ndef resolved(base):\n run,c,args=original_resolved(base)\n with open_dict(c):\n  c.actor_rollout_ref.actor.fsdp_config.model_dtype=\'fp32\'\n  c.actor_rollout_ref.actor.fsdp_config.mixed_precision=dict(param_dtype=\'fp32\',reduce_dtype=\'fp32\',buffer_dtype=\'fp32\')\n  c.actor_rollout_ref.ref.fsdp_config.model_dtype=\'fp32\'\n  c.actor_rollout_ref.rollout.dtype=\'float32\'\n return run,c,args\nsmoke.resolved=resolved\noriginal_autocast=torch.autocast\n\n\ndef autocast(*args,**kwargs):\n device=args[0] if args else kwargs.get(\'device_type\')\n if device==\'cuda\':kwargs[\'enabled\']=False\n return original_autocast(*args,**kwargs)\ntorch.autocast=autocast\n# TF32/reduced-mantissa math is not part of this full-FP32 candidate.\ntorch.backends.cuda.matmul.allow_tf32=False\ntorch.backends.cudnn.allow_tf32=False\nfrom verl.workers.fsdp_workers import ActorRolloutRefWorker\noriginal_init=ActorRolloutRefWorker.init_model\n\n\ndef initialize(self):\n result=original_init(self)\n role=\'actor\' if self._is_actor else \'reference\'\n module=self.actor_module_fsdp if self._is_actor else self.ref_module_fsdp\n p=Path(os.environ[\'SEARCH_G1_BASE\'])/\'logs\'/(\'precision-rank-\'+os.environ[\'RANK\']+\'.jsonl\')\n row=dict(kind=\'actual_fp32_initialization\',rank=int(os.environ[\'RANK\']),role=role,\n  parameter_dtypes=sorted({str(v.dtype) for v in module.parameters()}),\n  buffer_dtypes=sorted({str(v.dtype) for v in module.buffers()}),\n  fsdp_mixed_param_dtype=str(module.mixed_precision.param_dtype),\n  autocast_cuda_enabled=False,allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,\n  requested_rollout_dtype=self.config.rollout.dtype,actual_grpo_updates=0,\n  semantics=\'Isolated precision candidate; official same model/data/recipe budgets, actual resolved dtype overrides\')\n if self._is_rollout:\n  model=self.rollout.inference_engine.llm_engine.model_executor.worker.model_runner.model\n  row[\'actual_rollout_parameter_dtypes\']=sorted({str(v.dtype) for v in model.parameters()})\n with p.open(\'a\') as f:f.write(json.dumps(row)+\'\\n\');f.flush();os.fsync(f.fileno())\n return result\nActorRolloutRefWorker.init_model=initialize\noriginal_compare=smoke.compare\n\n\ndef compare(*args,**kwargs):\n old,ref,mask,tokens,raw,responses=args;valid=mask.cpu().bool();raw_valid=valid[:,:raw.shape[1]]\n row=dict(kind=\'actual_fp32_parity\',rank=int(os.environ[\'RANK\']),actor_reference_max_abs=float((old.cpu()[valid].float()-ref.cpu()[valid].float()).abs().max()),\n  actor_vllm_max_abs=float((old.cpu()[:,:raw.shape[1]][raw_valid].float()-raw.cpu()[raw_valid].float()).abs().max()),tolerance=smoke.LOGPROB_ATOL,\n  parameter_writes=0,precision_candidate_only=True)\n p=Path(os.environ[\'SEARCH_G1_BASE\'])/\'logs\'/(\'precision-rank-\'+os.environ[\'RANK\']+\'.jsonl\')\n with p.open(\'a\') as f:f.write(json.dumps(row)+\'\\n\');f.flush();os.fsync(f.fileno())\n return original_compare(*args,**kwargs)\nsmoke.compare=compare\nsmoke.main(os.environ[\'SEARCH_G1_BASE\'],\'initial\')\n'
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
p=json.loads(r.stdout);(prep/'fp32-precision-native-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
