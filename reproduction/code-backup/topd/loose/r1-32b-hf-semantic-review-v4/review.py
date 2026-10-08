"""Read-only full-HF semantic scan after actual own cleanup. No model forward."""
import argparse,hashlib,importlib.util,json,math,os,pathlib,resource,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
SOURCE='0d8a9214dc6e19f10e21549b56f6c01b45d67100'
OFFICIAL_BASE_INVENTORY_FILE_SHA='eedbfdd05bc06c08cb5e6f2e34022b36ebf95d6a29376514398786bb4a0f581a'
SOURCE_MANIFEST='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
PINS={'original_semantic_gate.py':'10ecd9ac072a93903e55acb870b4962844c1bcfd9bf5f2440355885e24e7b759','checkpoint_inventory.py':'ebd884a65e185b4ae55b48e8c2034e3da8a9e18f866de49719926565fdefee2f'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(name):
 p=pathlib.Path(__file__).with_name(name+'.py');need(not p.is_symlink() and sha(p)==PINS[p.name],'Bound original helper source drift')
 s=importlib.util.spec_from_file_location('bound_cpu_'+name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def local(p):
 p=pathlib.Path(p);need(not p.is_symlink() and ROOT in p.resolve().parents,'Only own local real evidence');return p.resolve()
def ref(r):
 p=local(r['path']);need(sha(p)==r['sha256'],'Raw/source artifact drift');return p,json.loads(p.read_text())
def job_id(value):
 if type(value) is int:result=value
 elif type(value) is str and value.isascii() and value.isdecimal():result=int(value)
 else:raise ValueError('Actual job must be positive integer or ASCII decimal string')
 need(result>0,'Actual job must be positive');return result
def binding_records(binding):
 need(binding['scope']=='REQUEST_BOUNDED_CPU_HF_SEMANTIC_SCAN_NOT_APPROVAL','No fabricated approval binding')
 _,result=ref(binding['train_result']);_,config=ref(binding['effective_config'])
 steps={'profile2':2,'train100':100}[binding['mode']]
 need(result['status']=='COMPLETE_UNREVIEWED' and result['cleanup_passed'] is True and result['error'] is None,'Complete own runtime/cleanup must precede scan')
 need(result['scope']==('ACTUAL_SEARCH_R1_PROFILE2_UNREVIEWED_NOT_100_NOT_TOPD' if steps==2 else 'ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD'),'Wrong actual algorithm/profile scope')
 need(job_id(result['job'])==job_id(binding['job_id'])==job_id(config['job']),'Actual job differs')
 need(sha(local(binding['effective_config']['path']))==result['effective_config_sha256'],'Actual config receipt mismatch')
 _,outer=ref(result['actual_outer_evidence']);model=local(result['export']['directory']);base=local(config['base_model'])
 need(config['mode']==binding['mode'] and config['expected_outer_steps']==steps and model==local(config['final_checkpoint']),'Actual expected/export/config path mismatch')
 need(outer['actual_completed_outer_steps']==list(range(1,steps+1)) and outer['export_step']==steps and model==local(outer['checkpoint_directory']),'Incomplete actual standard outer sequence')
 _,cleanup=ref(result['cleanup'])
 for key in ('own_processes_remaining','baseline_identity_missing','original_gpu_baseline_missing','own_gpu_records_remaining','parent_gpu_records','guardian_errors','sampler_errors'):
  need(cleanup[key]==[],'Actual cleanup/baseline/telemetry failed: '+key)
 need(cleanup['parent_cuda_initialized'] is False and cleanup['terminal_monitors_drained_before_authorized_exit'] is True and cleanup['cleanup']['actual_cgroup_empty'] is True,'Actual own cgroup/terminal not drained')
 need(cleanup['error'] is None and cleanup['primary_error'] is None and cleanup['failure_history']==[],'Actual cleanup primary/secondary failure retained')
 need(base==ROOT/'assets-root/assets/qwen2.5-32b' and config['base_model_inventory']['sha256']==OFFICIAL_BASE_INVENTORY_FILE_SHA,'Require reviewed official same32B Base path/inventory')
 source_path,_=ref(config['source_manifest']);need(config['source_manifest']['sha256']==SOURCE_MANIFEST and binding['source_reference']==SOURCE,'Original source manifest/algorithm differs')
 ref(config['tools_manifest'])
 data=[r for r in config['input_bindings'] if r['path'] in (config['train_file'],config['val_file'])]
 need(len(data)==2,'Missing actual dataset file identities')
 for r in data:need(sha(local(r['path']))==r['sha256'],'Actual training/eval data drift')
 _,base_inventory=ref(config['base_model_inventory']);_,fresh_base=ref(result['base_hf_fresh'])
 need(base_inventory['files']==fresh_base['files'],'Actual official Base inventory differs')
 return result,config,model,base,base_inventory,data,steps

def strict_inventory(actual,expected,official=False):
 need(actual['tensor_count']==771 and actual['files']==expected['files'] and actual['shards']==expected['shards'],'Complete771 HF files/schema drift')
 required={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json'}
 allowed=required|set(actual['shards'])|({'special_tokens_map.json','added_tokens.json','teacher-manifest.json'} if not official else set())
 need(required|set(actual['shards'])<=set(actual['files']) and set(actual['files'])<=allowed,'Unexpected/missing HF shard or metadata')
 need(set(actual['shards'])=={n for n in actual['files'] if n.endswith('.safetensors')},'Unindexed/extra model shard')
 if official:need(len(actual['shards'])==17 and len(actual['files'])==24,'Require original official17+7 Base')

def tokenizer_check(helper,model,base):
 answer=helper.tokenizer_compatible(model,base)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);vocab=tok.get_vocab();ids=list(vocab.values())
 need(len(vocab)==151665 and len(set(ids))==151665 and min(ids)==0 and max(ids)<152064,'Strict original151665 tokenizer/output dimensions')
 base_config=json.loads((base/'config.json').read_text())
 for name in ('config.json','generation_config.json'):
  check_special_ids(json.loads((model/name).read_text()),tok,base_config,name)
 answer['token_to_id_sha256']=hashlib.sha256(json.dumps(vocab,sort_keys=True).encode()).hexdigest();return answer

def check_special_ids(config,tok,base_config,name):
 eos=config.get('eos_token_id')
 need(type(eos) is int and eos==tok.eos_token_id or isinstance(eos,list) and bool(eos) and all(type(x) is int and x==tok.eos_token_id for x in eos),name+' actual teacher/tokenizer EOS mismatch')
 pad=config.get('pad_token_id');need(pad is None or type(pad) is int and pad==tok.pad_token_id,name+' actual teacher/tokenizer PAD mismatch')
 bos=config.get('bos_token_id');need(bos is None or type(bos) is int and bos in (tok.bos_token_id,base_config.get('bos_token_id')),name+' illegal BOS override')

def verify_tool_manifest(binding):
 path,manifest=ref(binding['review_tool_manifest']);directory=pathlib.Path(__file__).parent.resolve()
 need(path==directory/'files.json','Review tool manifest path differs from actual executed source')
 actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
 need(actual==set(manifest['files'])|{'files.json'},'Review source tree missing/extra/bytecode')
 for name,v in manifest['files'].items():
  p=directory/name;need(not p.is_symlink() and p.is_file() and directory in p.resolve().parents and p.stat().st_size==v['size'] and sha(p)==v['sha256'],'Actual review tool file drift')
 return binding['review_tool_manifest']
def run(binding):
 tool_manifest=verify_tool_manifest(binding)
 start=time.monotonic();usage=resource.getrusage(resource.RUSAGE_SELF)
 result,config,model,base,expected_base,data,steps=binding_records(binding)
 helper=load('checkpoint_inventory');semantic=load('original_semantic_gate')
 before={r['path']:sha(local(r['path'])) for r in (binding['train_result'],binding['effective_config'],result['cleanup'],result['actual_outer_evidence'],config['base_model_inventory'],config['source_manifest'],config['tools_manifest'],result['base_hf_fresh'],binding['review_tool_manifest'],*data)}
 bi,mi=helper.inventory(base),helper.inventory(model);strict_inventory(bi,expected_base,True);strict_inventory(mi,result['export'])
 tokenizer=tokenizer_check(helper,model,base)
 weights=semantic.compare_weights(base,model) # Original function only; old CLI/3393 selection never called.
 need(weights['compared_elements_including_tied_aliases']==32763876352 and weights['output_vocab_size']==152064,'Actual full32B semantic count/vocabulary differs')
 # Fresh full file SHA after mmap traversal detects concurrent drift, without model materialization.
 for directory,inventory in ((base,bi),(model,mi)):
  for name,v in inventory['files'].items():need(sha(directory/name)==v['sha256'],'HF changed during semantic scan')
 for p,h in before.items():need(sha(p)==h,'Runtime/source evidence changed during scan')
 end=resource.getrusage(resource.RUSAGE_SELF)
 return {'status':'FULL_HF_FINITE_SEMANTIC_CHANGED_UNREVIEWED','scope':'CPU_WEIGHT_SCAN_ONLY_NOT_TEACHER_QUALITY_NOT_TOPD_ACCEPTANCE','job_id':job_id(binding['job_id']),'outer_steps':steps,'source_reference':SOURCE,'source_manifest':config['source_manifest'],'tools_manifest':config['tools_manifest'],'train_result':binding['train_result'],'effective_config':binding['effective_config'],'outer_evidence':result['actual_outer_evidence'],'cleanup':result['cleanup'],'data_files':data,'base_inventory':bi,'actual_hf_inventory':mi,'tokenizer':tokenizer,'semantic':weights,'helper_source_sha256':PINS,'review_tool_manifest':tool_manifest,'cost':{'wall_seconds':time.monotonic()-start,'cpu_seconds':end.ru_utime+end.ru_stime-usage.ru_utime-usage.ru_stime,'ru_maxrss_native_units':end.ru_maxrss,'raw_fp32_chunk_elements':semantic.CHUNK_ELEMENTS,'not_total_rss_cap':True}}
def publish(path,value):
 path=pathlib.Path(path);need(not path.exists(),'Exclusive fresh own output required');tmp=path.with_name(path.name+'.private');need(not tmp.exists(),'Private pending output exists')
 with tmp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.link(tmp,path);tmp.unlink();fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True);a=p.parse_args();need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit no-GPU CPU scan required');need(int(os.environ.get('OMP_NUM_THREADS','0')) in (1,2),'Bounded CPU1/2 threads required');output=local(a.output);need(ROOT/'logs' in output.parents,'Ownlogs output only')
 need(not output.exists(),'Refuse existing output')
 try:r=run(json.loads(local(a.binding).read_text()))
 except BaseException as e:publish(output,{'status':'FAIL','scope':'CPU_WEIGHT_SCAN_ONLY_NOT_TEACHER_APPROVAL','error_type':type(e).__name__,'error':str(e)});raise
 publish(output,r);print(json.dumps({'status':r['status'],'output':str(output),'sha256':sha(output)}))
if __name__=='__main__':main()
