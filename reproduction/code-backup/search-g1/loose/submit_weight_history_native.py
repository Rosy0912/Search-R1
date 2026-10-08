import sys,json,subprocess,shlex,uuid,base64,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8-weight-history-'+uuid.uuid4().hex[:12]
wrapper='"""Disposable native audit at actual sync; original computation unchanged."""\nimport os,json,hashlib\nfrom pathlib import Path\nimport torch\nfrom verl.third_party.vllm.vllm_v_0_6_3.llm import LLM\nfrom verl.workers.fsdp_workers import ActorRolloutRefWorker\nfrom verl.third_party.vllm.vllm_v_0_6_3.dtensor_weight_loaders import redistribute_dtensor\nimport reproduction.common8.model_smoke as smoke\n\ndef emit(kind,**kw):\n p=Path(os.environ[\'SEARCH_G1_BASE\'])/\'logs\'/(\'precision-rank-\'+os.environ[\'RANK\']+\'.jsonl\')\n with p.open(\'a\') as f:\n  f.write(json.dumps(dict(kind=kind,rank=int(os.environ[\'RANK\']),**kw))+\'\\n\');f.flush();os.fsync(f.fileno())\noriginal_sync=LLM.sync_model_weights\n\ndef sync(self,actor_weights,load_format):\n original_sync(self,actor_weights,load_format)\n assert load_format==\'dtensor\'\n model=self.llm_engine.model_executor.worker.model_runner.model\n params=dict(model.named_parameters(remove_duplicate=False));rows=[]\n cfg=model.config;hd=cfg.hidden_size//cfg.num_attention_heads\n q=cfg.num_attention_heads*hd;k=cfg.num_key_value_heads*hd;inter=cfg.intermediate_size\n for source in sorted(actor_weights):\n  if source.endswith(\'inv_freq\'):continue\n  if cfg.tie_word_embeddings and source==\'lm_head.weight\':continue\n  target=source;offset=0;length=None\n  for src,dst,start,size in ((\'q_proj\',\'qkv_proj\',0,q),(\'k_proj\',\'qkv_proj\',q,k),(\'v_proj\',\'qkv_proj\',q+k,k),(\'gate_proj\',\'gate_up_proj\',0,inter),(\'up_proj\',\'gate_up_proj\',inter,inter)):\n   if \'.\'+src+\'.\' in source:\n    target=source.replace(\'.\'+src+\'.\',\'.\'+dst+\'.\');offset=start;length=size;break\n  assert target in params,(source,target)\n  loaded=redistribute_dtensor(target,actor_weights[source])\n  actual=params[target].detach()\n  if length is not None:actual=actual.narrow(0,offset,length)\n  expected=loaded.to(device=actual.device,dtype=actual.dtype)\n  assert actual.shape==expected.shape,(source,actual.shape,expected.shape)\n  equal=bool(torch.equal(actual,expected))\n  rows.append(dict(source=source,target=target,shape=list(actual.shape),equal=equal,max_abs=0.0 if equal else float((actual.float()-expected.float()).abs().max())))\n  del loaded,expected\n tied=dict(enabled=bool(cfg.tie_word_embeddings))\n if cfg.tie_word_embeddings:\n  a=params[\'model.embed_tokens.weight\'];b=params[\'lm_head.weight\'];tied.update(equal=bool(torch.equal(a,b)),same_storage=a.data_ptr()==b.data_ptr())\n emit(\'actual_sync_weight_audit\',rows=rows,tied=tied,scope=\'actual DTensor sync inputs after original sync before any offload; TP1 fused QKV/gateup mapped explicitly\')\nLLM.sync_model_weights=sync\noriginal_generate=LLM.generate\ncaptured={}\ndef generate(self,*args,**kw):\n captured[\'prompts\']=kw[\'prompt_token_ids\']\n out=original_generate(self,*args,**kw);captured[\'tokens\']=out[0].detach().cpu().clone();captured[\'logs\']=out[1].detach().cpu().clone();return out\nLLM.generate=generate\noriginal_ref=ActorRolloutRefWorker.compute_ref_log_prob\n\ndef reference(self,data):\n result=original_ref(self,data)\n ids=data.batch[\'input_ids\'].detach().cpu();mask=data.batch[\'attention_mask\'].detach().cpu().bool();pos=data.batch[\'position_ids\'].detach().cpu();responses=data.batch[\'responses\'].detach().cpu();old=data.batch[\'old_log_probs\'].detach().cpu()\n width=responses.shape[1];start=ids.shape[1]-width;raw=captured[\'logs\'];rows=[]\n for i,prompt in enumerate(captured[\'prompts\']):\n  pm=mask[i,:start];actual=ids[i,:start][pm];positions=pos[i,:start][pm];count=int(mask[i,start:].sum())\n  match=actual.tolist()==prompt;position_match=positions.tolist()==list(range(len(prompt))) and pos[i,start:start+count].tolist()==list(range(len(prompt),len(prompt)+count))\n  delta=(old[i,:count].float()-raw[i,:count].float()).abs();maximum=int(delta.argmax())\n  rows.append(dict(prompt_tokens=len(prompt),response_tokens=count,prompt_history_equal=match,positions_contiguous=position_match,max_abs=float(delta.max()),mean_abs=float(delta.mean()),max_at_response_position=maximum,last_token_abs=float(delta[-1]),last_token_is_eos=bool(responses[i,count-1]==self.tokenizer.eos_token_id),prompt_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest()))\n emit(\'actual_history_audit\',rows=rows)\n return result\nActorRolloutRefWorker.compute_ref_log_prob=reference\nsmoke.main(os.environ[\'SEARCH_G1_BASE\'],\'initial\')\n'
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
p=json.loads(r.stdout);(prep/'weight-history-native-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
