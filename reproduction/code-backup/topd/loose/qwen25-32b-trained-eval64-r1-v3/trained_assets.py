"""CPU-only Search-R1-100 export gate before original same64 evaluation; no approval writer."""
import hashlib,json,math,pathlib,subprocess
from checkpoint_inventory import inventory,tokenizer_compatible,sha
DATA_SHA='af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756'
SOURCE='0d8a9214dc6e19f10e21549b56f6c01b45d67100'
REPO_COMMIT='413a10c50f32febaafc83db8170e2b005e0e979f'
RUNTIME='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
GUARD='d340c77f54c8c744816da416c91016e59bd5a21bd42262b2ae672a3432f6588b'
def need(value,message):
 if not value:raise RuntimeError(message)
def local(path,root):
 p=pathlib.Path(path)
 need(not p.is_symlink() and root in p.resolve().parents,'Nonlocal or aliased evidence')
 return p.resolve()
def read_ref(ref,root):
 p=local(ref['path'],root);need(sha(p)==ref['sha256'],'Artifact bytes drift');return json.loads(p.read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def check_outer(outer,model):
 need(outer['actual_completed_outer_steps']==list(range(1,101)) and outer['export_step']==100 and pathlib.Path(outer['checkpoint_directory']).resolve()==model,'Need actual100 export, not profile2')
def semantic_changes(base,model,chunk_bytes=8*1024**2):
 """Mmap same-name tensors, FP32 semantic comparison; dtype-only change fails."""
 import torch
 from safetensors import safe_open
 def index(p):return json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
 bi,mi=index(base),index(model);need(set(bi)==set(mi),'Tensor key identity differs');changed=0;numel=0
 for key in sorted(bi):
  with safe_open(str(base/bi[key]),framework='pt',device='cpu') as a,safe_open(str(model/mi[key]),framework='pt',device='cpu') as b:
   x,y=a.get_slice(key),b.get_slice(key);shape=x.get_shape();need(shape==y.get_shape(),'Tensor shape differs')
   need(x.get_dtype() in ('BF16','F32') and y.get_dtype() in ('BF16','F32'),'Unsupported tensor dtype')
   stride=math.prod(shape[1:]) if shape else 1;rows=max(1,chunk_bytes//(4*stride))
   for start in range(0,shape[0] if shape else 1,rows):
    xx=(x[start:start+rows] if shape else a.get_tensor(key)).float();yy=(y[start:start+rows] if shape else b.get_tensor(key)).float()
    need(bool(torch.isfinite(xx).all()) and bool(torch.isfinite(yy).all()),'Nonfinite trained/base tensor')
    changed+=int(torch.count_nonzero(xx!=yy));numel+=xx.numel()
 need(changed>0,'No actual numerical parameter change; dtype/container change is insufficient')
 return {'scope':'ACTUAL_SAME_NAME_FP32_SEMANTIC_DIFFERENCE_NOT_QUALITY','changed_elements':changed,'compared_elements':numel,'raw_fp32_chunk_budget_bytes':chunk_bytes}
def check_special_ids(model,base):
 """Actual tokenizer IDs control EOS/PAD; official Base BOS override is legal."""
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 baseline=json.loads((base/'config.json').read_text())
 for name in ('config.json','generation_config.json'):
  config=json.loads((model/name).read_text())
  for key in ('bos_token_id','eos_token_id','pad_token_id'):
   value=config.get(key);actual=getattr(tok,key)
   if key=='bos_token_id':
    allowed={None,actual,baseline.get(key)}
    need(type(value) is int or value is None,'Invalid BOS config type')
    need(value in allowed,'Actual tokenizer/config BOS mismatch')
   elif key=='eos_token_id' and isinstance(value,list):
    need(bool(value) and all(type(x) is int and x==actual for x in value),'Actual tokenizer/config EOS list mismatch')
   else:
    need((value is None or type(value) is int) and value in (None,actual),'Actual tokenizer/config EOS/PAD mismatch')
 return {'bos':tok.bos_token_id,'eos':tok.eos_token_id,'pad':tok.pad_token_id}
def gate(binding_path,root,reused_sources):
 root=pathlib.Path(root).resolve();binding=json.loads(local(binding_path,root).read_text())
 need(binding['status']=='SEARCH_R1_100_EXPORT_BOUND_FOR_SAME64_UNREVIEWED_QUALITY' and binding['teacher_kind']=='SEARCH_R1_GRPO100_32B','Not actual Search-R1 trained evaluation binding')
 need(binding['data_sha256']==DATA_SHA and sha(local(binding['data_path'],root))==DATA_SHA,'Same64 data drift')
 model=local(binding['model_directory'],root);base=local(binding['base_model_directory'],root)
 original_ref=binding['original_eval_binding'];need(original_ref['sha256']=='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2','Original inventory binding differs')
 original=read_ref(original_ref,root);need(pathlib.Path(original['model_directory']).resolve()==base,'Original Base path differs')
 fresh_base=inventory(base);need({n:v['sha256'] for n,v in fresh_base['files'].items()}==original['model_files'],'Official Base bytes drift')
 need(binding['source_files']==original['source_files'] and binding['runtime_files']==original['runtime_files'],'Original evaluation/retrieval freeze differs')
 training=read_ref(binding['training_receipt'],root)
 need(training['status']=='INDEPENDENTLY_REVIEWED_ACTUAL_100_UPDATES' and training['actual_outer_updates']==100 and training['algorithm']=='SEARCH_R1_GRPO' and training['source_reference']==SOURCE,'Missing independently verified real100, not calibration/profile')
 inv=read_ref(binding['model_inventory'],root)
 need(training['hf_inventory_sha256']==binding['model_inventory']['sha256'] and inv['status']=='FULL_HF_FILES_VERIFIED' and pathlib.Path(inv['model_directory']).resolve()==model,'Actual export inventory mismatch')
 metadata={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','model.safetensors.index.json'}
 need(set(inv['shards'])|metadata<=set(inv['files']) and set(inv['files'])<=set(inv['shards'])|metadata|{'special_tokens_map.json','added_tokens.json','teacher-manifest.json'},'Incomplete/unexpected trained HF metadata')
 actual=inventory(model);need(actual['tensor_count']==771 and actual['files']==inv['files'] and actual['shards']==inv['shards'],'Fresh complete arbitrary-shard HF drift')
 result=read_ref(binding['train_result'],root);outer=read_ref(result['actual_outer_evidence'],root);check_outer(outer,model)
 import importlib.util
 helper_path=pathlib.Path(__file__).with_name('training_identity.py')
 need(not helper_path.is_symlink() and sha(helper_path)=='568ea251e12ae8bbe7620b2772e5521f97ed5c07b1cbe5d0aec9d687f6d30966','Standard training contract source drift')
 helper_spec=importlib.util.spec_from_file_location('bound_standard100_training_identity',helper_path)
 helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
 cfg_ref=binding['training_effective_config'];cfg=read_ref(cfg_ref,root)
 need(cfg_ref['sha256']==result['effective_config_sha256'] and pathlib.Path(cfg_ref['path']).resolve()==pathlib.Path(binding['train_result']['path']).resolve().parent/'effective-runtime-config.json','Actual standard config artifact differs')
 helper.check_standard100(training,result,outer,cfg,str(model))
 need(result['scope']=='ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD' and result['export']['files']==actual['files'] and pathlib.Path(result['export']['directory']).resolve()==model,'Actual trainer export differs')
 # Same source of raw count evidence is linked by the existing independently reviewed training receipt.
 need(training['actual_outer_evidence']==result['actual_outer_evidence'],'Training/raw outer provenance differs')
 tokenizer_compatible(model,base)
 check_special_ids(model,base)
 ignored={'_name_or_path','_commit_hash','torch_dtype','transformers_version','_attn_implementation','bos_token_id','eos_token_id','pad_token_id'}
 mc=json.loads((model/'config.json').read_text());bc=json.loads((base/'config.json').read_text())
 need({k:v for k,v in mc.items() if k not in ignored}=={k:v for k,v in bc.items() if k not in ignored},'Complete32B architecture configuration drift')
 semantic=semantic_changes(base,model)
 repo=root/'repo-terminal-fixed413';need(binding['code_commit']==REPO_COMMIT and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==REPO_COMMIT,'Frozen evaluation source HEAD drift')
 need(set(reused_sources)<=set(binding['source_files']),'Missing original evaluation source')
 for name,digest in binding['source_files'].items():need(sha(local(repo/name,root))==digest,'Original evaluation source drift')
 tools=pathlib.Path(__file__).parent.resolve()
 required={'eval64.py','supervisor.py','terminal.py','trained_assets.py','training_identity.py','checkpoint_inventory.py','runtime_safety.py','startup_guard.py'}
 need(required<=set(binding['tool_files']),'Incomplete trained evaluation tool binding')
 for name,digest in binding['tool_files'].items():
  p=(tools/name).resolve();need(tools in p.parents and sha(p)==digest,'Tool/source drift')
 need(sha(tools/'runtime_safety.py')==RUNTIME and sha(tools/'startup_guard.py')==GUARD,'WindowV4 source drift')
 need(sha(root/'tools/topd-short-run.py')==binding['owned_launcher_sha256'],'Own cleanup source drift')
 files=binding['runtime_files'];need({'assets-root/assets/wiki/e5_Flat.index','assets-root/assets/wiki/wiki-18.jsonl'}<=set(files),'Missing fullwiki retrieval binding')
 for directory in ('assets-root/assets/e5-base-v2','retriever-gpu-lib'):
  actual_files={str(p.relative_to(root)) for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts}
  need(actual_files=={n for n in files if n.startswith(directory+'/')},'Retrieval dependency exact set drift')
 for name,digest in files.items():need(sha(local(root/name,root))==digest,'Fresh local retrieval asset drift')
 return binding,model,local(binding['data_path'],root)
