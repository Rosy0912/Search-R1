"""Independent actual CPU read-only witness; never publish, modify or use GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
import pathlib,json,hashlib,subprocess,time,zipfile,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
RUN=ROOT/'runs/bapo-instruct300-formal300-3231'
EP=RUN/'logs/first5-extraction-3231-20261005T0435/evidence.json'
EXPECTED='7587fe09a066a63d34161771e2b47cf24c989c36700f42c8db5d9958a7f69803'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
start=time.monotonic();cpu=time.process_time()
assert sha(EP)==EXPECTED;e=json.loads(EP.read_text())
assert e['extraction_complete']is True and not e['issues'] and e['expected_driver_pid']==40564
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='39cdcdd1404213b897bb0de99cc52d72707682f0'
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
checked=[];zipreports=[]
for item in e['checkpoint_members']:
 p=pathlib.Path(item['path']);assert RUN in p.resolve().parents and p.is_file()and not p.is_symlink()
 assert sha(p)==item['sha256'];checked.append(dict(path=str(p),sha256=item['sha256'],bytes=p.stat().st_size))
 if p.suffix=='.pt'and p.name.startswith('rank_'):
  with zipfile.ZipFile(p)as z:
   assert z.testzip()is None
   keys=z.namelist();storage=[n for n in keys if '/data/'in n]
   assert storage and len(set(keys))==len(keys)
   meta=next(n for n in keys if n.endswith('/data.pkl'));data=z.read(meta)
   assert all(x.encode()in data for x in ('model','optimizer','scheduler','rng'))
   zipreports.append(dict(path=str(p),tensor_storage_entries=len(storage),crc_all_entries_pass=True,full_state_keys_present=True))
assert len(zipreports)==8
sys.path.insert(0,str(ROOT/'repo'))
import torch
torch.set_num_threads(8)
from search_r1.baselines.full_state import state_digest
driverpath=RUN/'checkpoints/full_state/global_step_5/driver.pt'
driver=torch.load(driverpath,map_location='cpu',weights_only=False)
assert driver['driver_pid']==40564 and driver['global_step']==5 and driver['next_global_step']==6 and driver['actual_outer_updates']==5
assert state_digest(driver)==e['driver_state']['state_digest']
sampler=driver['sampler'];assert sampler['size']==169028 and sampler['cursor']==2560 and sampler['epoch']==0
assert len(sampler['order'])==169028 and set(sampler['order'])==set(range(169028))
assert driver['rng']['cuda']==[]and driver['bapo']==e['expected_bapo']
base=ROOT/'assets/qwen2.5-3b-instruct-aa8e725';ck=RUN/'checkpoints/actor/global_step_5'
from transformers import AutoTokenizer,AutoModelForCausalLM
t0=AutoTokenizer.from_pretrained(base,local_files_only=True);t1=AutoTokenizer.from_pretrained(ck,local_files_only=True)
assert t0.get_vocab()==t1.get_vocab()and t0.chat_template==t1.chat_template and t0.special_tokens_map==t1.special_tokens_map
from safetensors import safe_open
a=json.loads((base/'model.safetensors.index.json').read_text())['weight_map'];b=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
for item in e['hf_files']+e['official_base_files']:assert sha(pathlib.Path(item['path']))==item['sha256']
with safe_open(base/a['model.embed_tokens.weight'],framework='pt',device='cpu')as f:x=f.get_slice('model.embed_tokens.weight')[0:1024]
with safe_open(ck/b['lm_head.weight'],framework='pt',device='cpu')as f:y=f.get_slice('lm_head.weight')[0:1024]
delta=y.double()-x.to(y.dtype).double();assert torch.isfinite(delta).all()and delta.abs().max()>0
weight=dict(base_dtype=str(x.dtype),checkpoint_dtype=str(y.dtype),max_abs=float(delta.abs().max()),l1=float(delta.abs().sum()),comparison='base_cast_to_checkpoint_dtype_then_fp64')
del x,y,delta
with safe_open(ck/b['lm_head.weight'],framework='pt',device='cpu')as f, safe_open(ck/b['model.embed_tokens.weight'],framework='pt',device='cpu')as g:
 left=f.get_slice('lm_head.weight');right=g.get_slice('model.embed_tokens.weight')
 assert left.get_shape()==right.get_shape()==[151936,2048]
 for lo in range(0,151936,1024):assert torch.equal(left[lo:lo+1024],right[lo:lo+1024])
model,loading=AutoModelForCausalLM.from_pretrained(ck,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
assert all(not loading.get(k)for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
assert next(model.parameters()).device.type=='cpu'
n=0
for _,p in model.named_parameters():assert torch.isfinite(p).all();n+=p.numel()
assert n==3085938688 and model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
with torch.inference_mode():out=model(torch.tensor([[1,2,3,4,5]],dtype=torch.long)).logits
assert torch.isfinite(out).all()and out.device.type=='cpu'
def proc(pid):
 p=pathlib.Path('/proc')/str(pid);s=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
 env=dict(x.split('=',1)for x in p.joinpath('environ').read_bytes().decode().split('\0')if '='in x)
 return dict(pid=pid,start_ticks=int(s[19]),state=s[0],run=env.get('BAPO_RUN_ID'),job=env.get('SLURM_JOB_ID'))
processes=[]
for old in e['actual_processes_after_cpu']:
 now=proc(old['pid']);assert now['start_ticks']==old['start_ticks']and now['state']not in ('Z','X')and now['run']==str(RUN)and now['job']=='3231';processes.append(now)
text=(RUN/'train.log').read_text();assert text.count('BAPO_TRAIN_METRICS_JSON=')==5 and 'E1_STEP5_RELEASED_JSON='not in text and not (RUN/'logs/step5-independent-release.json').exists()
with (RUN/'train.log').open('rb')as f:assert hashlib.sha256(f.read(e['train_log_prefix']['bytes'])).hexdigest()==e['train_log_prefix']['sha256']
print(json.dumps(dict(status='PASS_INDEPENDENT_ACTUAL_CPU_FIRST5_WITNESS',source=e['source_expected'],job=3231,driver_pid=40564,evidence_sha256=EXPECTED,checkpoint_members=checked,native_zip=zipreports,driver_digest=state_digest(driver),sampler_digest=state_digest(sampler),weight_delta=weight,tied_elements_compared=311164928,finite_model_parameters=n,tokenizer_vocab_size=len(t0.get_vocab()),loading_info=loading,finite_forward=True,forward_shape=list(out.shape),processes=processes,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],sys_prefix=sys.prefix,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu),indent=2),flush=True)
