set -euo pipefail
python3 - <<'PY'
import json,hashlib,shutil,subprocess,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');run='author-native-checkpoint-fixed-five-p';m=r/'runs/author-native-phase-aware-warm-five-m';src=m/'tools';stage=r/'checkpoint-fixed-tools-p'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip(),'Node allocation must be empty'
assert json.loads((m/'driver-exit.json').read_text())['returncode']==1
assert not (r/'runs'/run).exists() and not (r/(run+'-launch-claim.json')).exists()
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True).strip()
ready=json.loads((src/'preparation-ready.json').read_text())
for n,h in ready['tools_sha256'].items():assert sha(src/n)==h,n
assert sha(src/'checkpoint_observer.py')=='867d76fd0c5872ae7f398685a92e8f89128e8d2d5a49b02758c6bef070b518a2'
fixed=r/'checkpoint-digest-v2/checkpoint_observer.py';assert sha(fixed)=='fb518c886f1a3c57dffb9b60653f5b3129bdb95735e241cc292e3fd84c8382fe'
identity=json.loads((src/'warm-initialization.json').read_text())
assert identity['cumulative_extra_supervised_updates']==112 and identity['fresh_RL_optimizers_required'] and identity['sft_optimizer_not_loaded_into_RL']
for role,v in identity['roles'].items():
 for n,h in v['files_sha256'].items():assert sha(Path(v['hf_path'])/n)==h,(role,n)
for n,h in identity['warm_manifest_sha256'].items():assert sha(r/'runs'/n/'warm-start/manifest.json')==h,n
manifest=json.loads((src/'source-manifest.json').read_text())
for n,h in manifest['sha256'].items():assert sha(r/'repo'/n)==h,n
cli=json.loads((src/'author-cli.json').read_text());assert len(cli)==102
assert sha(src/'author-cli.json')=='7b8ccb02f43454832afc07f3e16d137ea5aeeb8b6ffd0638e3bb487bbaa90f7a'
assert not stage.exists(),'Read existing preparation before retry; never overwrite frozen stage'
shutil.copytree(src,stage)
shutil.copy2(fixed,stage/'checkpoint_observer.py')
correction=dict(run=run,created_at=time.time(),previous_run=m.name,previous_driver_returncode=1,previous_checkpoint_files=0,new_run_fresh_RL_Adam=True,resume_previous_run=False,correction='Read OmegaConf ListConfig and DictConfig in existing full-state digest recursion',old_sha256=sha(src/'checkpoint_observer.py'),new_sha256=sha(fixed),training_math_budget_sampling_unchanged_from_M=True,actual_GPU_FSDP_save_load_proven=False,strict_accepted=False)
(stage/'checkpoint-digest-correction.json').write_text(json.dumps(correction,indent=2)+'\n')
identity['run']=run;identity['checked_at']=time.time();identity['checkpoint_digest_correction']=correction
(stage/'warm-initialization.json').write_text(json.dumps(identity,indent=2)+'\n')
# Deploy only after all original frozen files and warm identities passed; M is untouched.
for p in stage.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.sh','.yaml']:shutil.copy2(p,r/'tools'/p.name)
print(json.dumps(dict(stage=str(stage),original_frozen_tools_verified=True,full_HF_identity_verified=True,run=run,accepted=False)),flush=True)
PY
ROOT=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT/tools:$ROOT/repo:$ROOT/repo/verl" COSEARCH_NATIVE_RUN_ID=author-native-checkpoint-fixed-five-p "$ROOT/env/bin/python" "$ROOT/tools/cpu_launch_gate.py"
python3 - <<'PY'
import json,hashlib,time,shutil
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');t=r/'tools';stage=r/'checkpoint-fixed-tools-p';old=r/'runs/author-native-phase-aware-warm-five-m/tools'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ready=json.loads((old/'preparation-ready.json').read_text());gate=json.loads((t/'cpu-launch-gate.json').read_text())
assert gate['passed'] and gate['argv_count']==102 and gate['data']['train']['rows']==166222 and gate['data']['validation']['rows']==64
changed={'checkpoint_observer.py','warm-initialization.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert sha(t/n)==h,n
for n in changed|{'checkpoint-digest-correction.json'}:ready['tools_sha256'][n]=sha(t/n)
ready.update(created_at=time.time(),run='author-native-checkpoint-fixed-five-p',initialization='M same actorv4/rankerv2/112SFT/freshRLAdam; reviewed checkpoint digest OmegaConf compatibility only')
assert ready['cpu_preparation_passed'] and not ready['actual_gpu_resources_tested']
(t/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
for n in ['preparation-ready.json','cpu-launch-gate.json','resolved-author.yaml']:shutil.copy2(t/n,stage/n)
print(json.dumps(dict(run=ready['run'],ready_sha256=sha(t/'preparation-ready.json'),tools_sha256=ready['tools_sha256'],cpu_launch_gate=gate,accepted=False)),flush=True)
PY
