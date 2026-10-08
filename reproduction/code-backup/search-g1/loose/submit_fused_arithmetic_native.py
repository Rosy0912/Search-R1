import sys,json,subprocess,shlex,uuid,base64,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8-fused-arithmetic-'+uuid.uuid4().hex[:12]
wrapper='"""Actual same-operand fused arithmetic observations; original outputs preserved."""\nimport os,json\nfrom pathlib import Path\nimport torch\nfrom torch.nn import functional as F\nfrom vllm.distributed import get_tensor_model_parallel_world_size\nfrom vllm.model_executor.layers.linear import QKVParallelLinear,MergedColumnParallelLinear,UnquantizedLinearMethod\nfrom vllm.model_executor.layers.layernorm import RMSNorm\nfrom vllm.model_executor.layers.activation import SiluAndMul\nimport reproduction.common8.model_smoke as smoke\nfrom verl.third_party.vllm.vllm_v_0_6_3.llm import LLM\nrecords=[];active=False;sync_count=0\noriginal_sync=LLM.sync_model_weights\n\n\ndef sync(self,*args,**kwargs):\n global active,sync_count\n result=original_sync(self,*args,**kwargs)\n assert kwargs.get(\'load_format\',args[1] if len(args)>1 else None)==\'dtensor\'\n sync_count+=1;active=True\n return result\nLLM.sync_model_weights=sync\n\n\ndef compare(a,b):\n d=(a.float()-b.float()).abs();return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),different_values=int((d!=0).sum()),value_count=d.numel())\n\n\ndef linear_wrapper(original,kind):\n def forward(self,x):\n  if not active:return original(self,x)\n  output=original(self,x);count=getattr(self,\'_diagnostic_count\',0);self._diagnostic_count=count+1\n  if count<2:\n   assert get_tensor_model_parallel_world_size()==1 and type(self.quant_method) is UnquantizedLinearMethod\n   sizes=self.output_partition_sizes;assert sum(sizes)==self.weight.shape[0] and len(sizes)==(3 if kind==\'qkv\' else 2)\n   bias=self.bias if not self.skip_bias_add else None\n   def separate():\n    offset=0;chunks=[]\n    for size in sizes:\n     chunks.append(F.linear(x,self.weight.narrow(0,offset,size),None if bias is None else bias.narrow(0,offset,size)));offset+=size\n    return torch.cat(chunks,-1)\n   divided=separate();flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction\n   try:\n    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False\n    fused_full=F.linear(x,self.weight,bias);divided_full=separate()\n   finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag\n   records.append(dict(kind=kind,call=count,input_shape=list(x.shape),input_dtype=str(x.dtype),sizes=list(sizes),\n     original_vs_split_same_operand=compare(output[0],divided),original_vs_no_reduced_fused=compare(output[0],fused_full),\n     original_vs_no_reduced_split=compare(output[0],divided_full),default_reduced_bf16=flag,matmul_precision=torch.get_float32_matmul_precision()))\n  return output\n return forward\nQKVParallelLinear.forward=linear_wrapper(QKVParallelLinear.forward,\'qkv\')\nMergedColumnParallelLinear.forward=linear_wrapper(MergedColumnParallelLinear.forward,\'gate_up\')\noriginal_norm=RMSNorm.forward_cuda\n\ndef norm(self,x,residual=None):\n if not active:return original_norm(self,x,residual)\n count=getattr(self,\'_diagnostic_count\',0);self._diagnostic_count=count+1\n if count<2:\n  combined=x.clone() if residual is None else x+residual\n  v=combined.float().pow(2).mean(-1,keepdim=True)\n  expected=(combined.float()*torch.rsqrt(v+self.variance_epsilon)).to(combined.dtype)*self.weight\n out=original_norm(self,x,residual)\n if count<2:\n  normalized=out if residual is None else out[0]\n  row=dict(kind=\'rmsnorm\',call=count,residual=residual is not None,input_shape=list(x.shape),original_vs_HF_same_operand=compare(normalized,expected))\n  if residual is not None:row[\'residual_vs_HF\']=compare(out[1],combined)\n  records.append(row)\n return out\nRMSNorm.forward_cuda=norm\noriginal_silu=SiluAndMul.forward_cuda\n\ndef silu(self,x):\n if not active:return original_silu(self,x)\n count=getattr(self,\'_diagnostic_count\',0);self._diagnostic_count=count+1\n if count<2:\n  gate,up=x.chunk(2,-1);expected=F.silu(gate)*up\n out=original_silu(self,x)\n if count<2:records.append(dict(kind=\'silu\',call=count,input_shape=list(x.shape),original_vs_HF_same_operand=compare(out,expected)))\n return out\nSiluAndMul.forward_cuda=silu\noriginal_compare=smoke.compare\n\ndef gate(*args,**kwargs):\n assert active and sync_count==1\n p=Path(os.environ[\'SEARCH_G1_BASE\'])/\'logs\'/(\'precision-rank-\'+os.environ[\'RANK\']+\'.jsonl\')\n with p.open(\'x\') as f:\n  for row in records:f.write(json.dumps(dict(rank=int(os.environ[\'RANK\']),phase=\'after_actual_dtensor_sync\',sync_count=sync_count,**row))+\'\\n\')\n  f.flush();os.fsync(f.fileno())\n return original_compare(*args,**kwargs)\nsmoke.compare=gate\nsmoke.main(os.environ[\'SEARCH_G1_BASE\'],\'initial\')\n'
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
p=json.loads(r.stdout);(prep/'fused-arithmetic-native-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
