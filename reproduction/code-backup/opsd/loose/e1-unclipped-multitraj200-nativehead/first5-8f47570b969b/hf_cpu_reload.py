"""Read-only full CPU HF5 reload. No CUDA, optimizer, generation or release operation."""
import argparse,hashlib,json,os,socket,sys,time
from pathlib import Path
SOURCE='b611864d3f6ad048189b11b9b9613854e789da5d'
COUNT=3085938688
MODEL_MANIFEST_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'

def fail(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):
 def pairs(items):
  result={}
  for k,v in items:
   fail(k not in result,'Duplicate JSON key');result[k]=v
  return result
 return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def physical(path,root=None):
 p=Path(path);fail(p.is_absolute() and p.resolve(strict=True)==p and p.is_file() and p.stat().st_uid==os.getuid(),'Physical own file required')
 if root is not None:p.relative_to(root)
 return p

def verify_manifest_inputs(manifest,root):
 for rec in manifest['files'].values():fail(sha(physical(rec['path'],root))==rec['sha256'],'Checkpoint/launch evidence differs')
 config=read(physical(manifest['files']['resolved_config']['path'],root))
 fail(hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==manifest['config_sha256'],'Resolved configuration digest differs')
 return config

def tensor_map(directory):
 from safetensors import safe_open
 index=physical(directory/'model.safetensors.index.json');declared=read(index)['weight_map'];actual={};shapes={}
 fail(bool(declared),'Empty HF tensor map')
 for filename in set(declared.values()):
  fail(Path(filename).name==filename,'Escaped HF shard');path=physical(directory/filename)
  with safe_open(str(path),framework='pt',device='cpu') as f:
   for key in f.keys():
    fail(key not in actual,'Duplicate tensor key');actual[key]=filename;shapes[key]=f.get_slice(key).get_shape()
 fail(actual==declared,'Actual tensors do not match index');return declared,shapes

def canonical(keys,tied):return set(keys)-({'lm_head.weight'} if tied else set())
def empty_loading(info):
 fail(set(info)=={'missing_keys','unexpected_keys','mismatched_keys','error_msgs'} and all(v==[] for v in info.values()),'Missing/random/mismatched loaded model parameters')

def tokenizer_identity(a,b):
 fail(a.get_vocab()==b.get_vocab() and a.special_tokens_map==b.special_tokens_map and a.chat_template==b.chat_template and a.padding_side==b.padding_side and a.truncation_side==b.truncation_side,'Tokenizer identity changed')
 fail((a.bos_token_id,a.eos_token_id,a.pad_token_id)==(b.bos_token_id,b.eos_token_id,b.pad_token_id),'Tokenizer IDs changed')
 x=json.loads(a.backend_tokenizer.to_str());y=json.loads(b.backend_tokenizer.to_str())
 transient={name:{'base':x.pop(name,None),'hf':y.pop(name,None)} for name in ('padding','truncation')}
 fail(x==y,'Tokenizer rules changed');return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest(),transient

def load_compare(base,hf,expected_count=COUNT):
 import torch
 from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
 from safetensors import safe_open
 from search_r1.baselines.full_state import state_digest
 base_map,base_shapes=tensor_map(base);hf_map,hf_shapes=tensor_map(hf)
 bc=AutoConfig.from_pretrained(str(base),local_files_only=True);hc=AutoConfig.from_pretrained(str(hf),local_files_only=True)
 runtime={'_name_or_path','_commit_hash','transformers_version','torch_dtype','_attn_implementation_internal','_attn_implementation_autoset','bos_token_id','eos_token_id','pad_token_id'}
 fail({k:v for k,v in bc.to_dict().items() if k not in runtime}=={k:v for k,v in hc.to_dict().items() if k not in runtime},'Computation configuration changed')
 tied=bool(bc.tie_word_embeddings);fail(canonical(base_map,tied)==canonical(hf_map,tied),'Missing complete model tensors')
 for key in canonical(base_map,tied):fail(base_shapes[key]==hf_shapes[key],'Changed tensor geometry')
 a=AutoTokenizer.from_pretrained(str(base),local_files_only=True);b=AutoTokenizer.from_pretrained(str(hf),local_files_only=True);tokenizer_sha,transient=tokenizer_identity(a,b)
 for name in ('bos_token_id','eos_token_id','pad_token_id'):fail(getattr(hc,name)==getattr(b,name),'Saved model/tokenizer ID differs')
 models=[];loading=[]
 for directory,mapping in ((base,base_map),(hf,hf_map)):
  model,info=AutoModelForCausalLM.from_pretrained(str(directory),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=False);empty_loading(info)
  fail(model.config._attn_implementation=='eager' and canonical(model.state_dict(),tied)==canonical(mapping,tied),'Loaded model architecture differs')
  fail(sum(p.numel() for p in model.parameters())==expected_count,'Actual unique parameter count differs')
  fail(all(p.device.type=='cpu' and p.dtype==torch.float32 and torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite or non-CPU/FP32 loaded parameters')
  if tied:fail(model.get_input_embeddings().weight is model.get_output_embeddings().weight,'Loaded tied embeddings not same Parameter')
  # Compare each actually loaded state tensor to the corresponding serialized
  # value after the identical FP32 conversion, including optional tied alias.
  for key,tensor in model.state_dict().items():
   source_key='model.embed_tokens.weight' if key not in mapping and tied and key=='lm_head.weight' else key
   fail(source_key in mapping,'Loaded key inferred without an indexed weight')
   with safe_open(str(directory/mapping[source_key]),framework='pt',device='cpu') as handle:stored=handle.get_tensor(source_key).float()
   fail(torch.equal(tensor,stored),'Actual loaded value differs from exported tensor')
  model.eval();models.append(model);loading.append(info)
 changes={};base_params=dict(models[0].named_parameters());hf_params=dict(models[1].named_parameters());fail(base_params.keys()==hf_params.keys(),'Parameter names changed')
 for name,x in base_params.items():
  y=hf_params[name];delta=(x.detach()-y.detach()).abs();changes[name]={'elements':x.numel(),'changed_elements':int((delta!=0).sum()),'max_abs_delta':float(delta.max()),'mean_abs_delta':float(delta.mean())}
 fail(sum(v['changed_elements'] for v in changes.values())>0,'No real same-FP32 student parameter change')
 inputs=b.apply_chat_template([{'role':'user','content':'Return the word test.'}],add_generation_prompt=True,return_tensors='pt')
 with torch.no_grad():logits=models[1](input_ids=inputs,use_cache=False).logits
 fail(torch.isfinite(logits).all().item(),'CPU forward nonfinite')
 return {'loading_info':loading,'actual_unique_parameters':expected_count,'tied_embeddings':tied,'all_loaded_parameters_finite':True,'actual_cpu_forward_finite':True,'parameter_changes':changes,'total_changed_elements':sum(v['changed_elements'] for v in changes.values()),'base_named_parameter_digest':state_digest(base_params),'base_named_buffers_digest':state_digest(dict(models[0].named_buffers())),'base_computation_config':models[0].config.to_dict(),'hf_named_parameter_digest':state_digest(hf_params),'tokenizer_semantic_sha256':tokenizer_sha,'tokenizer_transient_state':transient,'canonical_state_keys':sorted(canonical(hf_map,tied)),'base_map':base_map,'hf_map':hf_map,'config_computation_equal':True,'runtime_config_fields':sorted(runtime),'cuda_is_initialized':torch.cuda.is_initialized()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--checkpoint-manifest',required=True);p.add_argument('--checkpoint-manifest-sha256',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 fail(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be explicitly disabled for this CPU inspection')
 script_path=physical(Path(__file__).resolve());script_sha=sha(script_path)
 root=Path(a.run_root);fail(root.is_absolute() and root.resolve(strict=True)==root,'Physical current run required');manifest_path=physical(a.checkpoint_manifest,root);fail(sha(manifest_path)==a.checkpoint_manifest_sha256,'External checkpoint SHA differs');manifest=read(manifest_path)
 verify_manifest_inputs(manifest,root)
 fail(manifest['source_commit']==SOURCE and manifest['run_root']==str(root) and manifest['global_step']==manifest['actual_outer_updates']==5 and manifest['next_global_step']==6 and manifest['world_size']==8 and manifest['uid']==os.getuid(),'Wrong actual checkpoint')
 repo=Path(a.repo);fail(repo.is_absolute() and repo.resolve(strict=True)==repo and repo.name=='repo-'+SOURCE[:12],'Wrong frozen repo')
 import subprocess
 fail(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Source not clean fixed HEAD')
 sys.path[:]=[str(repo)]+[x for x in sys.path if x and Path(x).resolve()!=repo]
 import torch
 fail(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'Native Torch2.4 CPU inspection required')
 from search_r1.baselines.checkpoint_gate import full_import_checker,verify_import_closure
 full_import_checker(manifest['files']['import_closure'],root,repo);verify_import_closure(manifest['files']['import_closure'],root,repo)
 config=verify_manifest_inputs(manifest,root)
 from omegaconf import OmegaConf
 from search_r1.baselines.e1_profile import validate_profile
 fail(config['baselines']['opsd_e1']['profile']=='multitraj200' and manifest['horizon']==200 and manifest['warmup_steps']==57,'Wrong actual multitraj200 profile')
 validate_profile(OmegaConf.create(config))
 base=Path(config['actor_rollout_ref']['model']['path']);hf=Path(manifest['files']['actor/model.safetensors.index.json']['path']).parent
 official_record=manifest['files']['model_manifest'];fail(sha(physical(official_record['path'],root))==official_record['sha256']==MODEL_MANIFEST_SHA,'Official model manifest differs')
 from search_r1.baselines.e1_profile import verify_official_assets
 original=physical(base/'VERIFIED-MODEL-MANIFEST.json');fail(sha(original)==MODEL_MANIFEST_SHA,'Original official model manifest differs');verify_official_assets(base,original)
 base_files={str(physical(path)):sha(path) for path in sorted(base.iterdir()) if path.is_file()}
 hashes={}
 for role,rec in manifest['files'].items():
  if role.startswith('actor/') and '/training_state/' not in role:
   path=physical(rec['path'],root);fail(sha(path)==rec['sha256'],'HF evidence changed');hashes[role]=rec
 start=time.monotonic();result=load_compare(base,hf)
 fail(not torch.cuda.is_initialized(),'CPU reload initialized CUDA')
 verify_manifest_inputs(manifest,root)
 fail(base_files=={str(physical(path)):sha(path) for path in sorted(base.iterdir()) if path.is_file()},'Official model content drifted during CPU load')
 fail(sha(original)==MODEL_MANIFEST_SHA,'Official manifest drifted')
 verify_official_assets(base,original)
 fail(sha(script_path)==script_sha,'CPU inspector source changed during load')
 fail(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Repo source drifted during CPU load')
 full_import_checker(manifest['files']['import_closure'],root,repo);verify_import_closure(manifest['files']['import_closure'],root,repo)
 result.update(inspector_source_sha256=script_sha,base_all_files=base_files,version=1,scope='ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA',source_commit=SOURCE,job_id=manifest['job_id'],run_root=str(root),checkpoint_manifest_sha256=a.checkpoint_manifest_sha256,checkpoint=str(hf),official_base=str(base),files=hashes,node=socket.gethostname(),uid=os.getuid(),pid=os.getpid(),torch_version=torch.__version__,seconds=time.monotonic()-start,passed=True)
 out=Path(a.output);fail(out.is_absolute() and out.parent.resolve(strict=True)==out.parent,'Physical output parent required');out.relative_to(root)
 with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
 print(json.dumps({'output':str(out),'sha256':sha(out),'scope':result['scope'],'cuda_is_initialized':False,'passed':True},sort_keys=True))
if __name__=='__main__':main()
