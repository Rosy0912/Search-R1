"""Independent local actual-byte audit. No signatures/GPU/native changes."""
import base64,csv,hashlib,importlib.util,io,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'formal-tools-89beac2'))
import formal_supervisor as f
REPO=ROOT.parent/'opsd-e1-300';SOURCE=f.SOURCE;RUN='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3198';REMOTE='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';closureSHA='1170bfd11f9df324506f64add056551248f96d2748cac25412b86e45d3846f22';keySHA='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
raw=(ROOT/'formal3198-pre-gpu-actual-collection.json').read_bytes();c=f.unique(raw);assert c['ready_exists'] is True and c['host']=='h100-10' and c['uid']==0 and c['source_head']==SOURCE and not c['source_status'] and c['gpu_compute'].strip()==''
f.slurm(c['slurm'],'3198',0)
def file(path):
 r=c['files'][path];b=base64.b64decode(r['base64']);assert r['physical'] is True and r['uid']==0 and len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];return b,r['sha256']
def value(path):return f.unique(file(path)[0])
ready=value(RUN+'/approval-ready.json');binding=value(RUN+'/bindings/manifest.json');m=value(REMOTE+'/tools/formal-209513e681cd/manifest.json')
assert file(REMOTE+'/tools/formal-209513e681cd/manifest.json')[1]=='209513e681cd5814a1aa7c1952e28a1e95038da5e724127e248275bfe54e4653'
assert ready['source_commit']==SOURCE and ready['job_id']=='3198' and ready['uid']==0 and ready['run_root']==RUN
assert isinstance(ready['pid'],int) and re.fullmatch('[0-9a-f]{64}',ready['nonce'])
assert ready['public_key_sha256']==keySHA and ready['binding_sha256']==file(RUN+'/bindings/manifest.json')[1]
actual=c['actual_ready_process'];assert actual['pid']==ready['pid'] and actual['start']==ready['process_start_ticks']==ready['process']['start']
assert re.search(r'(?:^|/)job_3198(?:/|$)',actual['cgroup'],re.M) and actual['cgroup']==ready['process']['cgroup']
assert set(map(int,re.search(r'^Uid:\s+(.*)',actual['status'],re.M).group(1).split()))=={0}
assert 'formal_supervisor.py' in actual['cmdline'] and '--prepare' in actual['cmdline']
assert set(binding['files'])==f.ROLES and binding['source_commit']==SOURCE and binding['run_root']==RUN
roles={}
for role,r in binding['files'].items():
 assert r['path']==RUN+'/bindings/'+role+'.json' and set(r)=={'path','sha256'}
 assert file(r['path'])[1]==r['sha256'];roles[role]=value(r['path'])
expected={'import_closure':closureSHA,'data_manifest':f.COMMON_SHA,'model_manifest':f.MODEL_SHA,'tool_manifest':f.ASSETS_SHA}
for role,pin in expected.items():assert binding['files'][role]['sha256']==pin
assert roles['source']['source_commit']==SOURCE and roles['source']['clean'] is True and roles['source']['source_files']==m['source_files']==c['source_file_sha256']
assert roles['source']['formal_manifest_sha256']=='209513e681cd5814a1aa7c1952e28a1e95038da5e724127e248275bfe54e4653'
a=roles['allocation'];assert a['job_id']=='3198' and a['uid']==0 and a['hostname']=='h100-10' and a['wrapper_process']==ready['process'];f.slurm(a['scontrol'],'3198',0)
inv=[]
for r in csv.reader(io.StringIO(c['gpu_inventory'])):inv.append(dict(index=int(r[0]),uuid=r[1].strip(),name=r[2].strip(),used=float(r[3]),total=float(r[4]),util=float(r[5])))
f.validate_snapshot(dict(inventory=inv,applications=[]),ready['nvml_uuids'],idle=True)
assert len(a['nvml_double_guard'])==2
for sample in a['nvml_double_guard']:f.validate_snapshot(sample,ready['nvml_uuids'],idle=True)
assert len(roles['import_closure']['modules'])==210 and set(roles['import_closure'])=={'modules'}
for name,r in roles['import_closure']['modules'].items():
 assert set(r)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',r['sha256'])
 if name.startswith(('search_r1','verl','reproduction.')):assert r['path'].startswith(REMOTE+'/repo-89beac273f78/')
for n in ['verl.workers.rollout.vllm_rollout','verl.workers.rollout.vllm_rollout.vllm_rollout','verl.workers.sharding_manager.fsdp_vllm']:assert n in roles['import_closure']['modules']
assert roles['tool_manifest']['status']=='FULL_TOOL_ASSET_CONTENT_VERIFIED' and roles['tool_manifest']['frozen_source_commit']=='04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
assert file(RUN+'/bindings/public.json')[1]==keySHA
assert value(RUN+'/bindings/public.json')['algorithm']=='RSA-SHA256'
proof=value(RUN+'/evidence/pre-gpu-import-check.json');assert proof['source_commit']==SOURCE and proof['job_id']=='3198' and proof['uid']==0 and proof['repo']==REMOTE+'/repo-89beac273f78' and proof['closure_sha256']==closureSHA and proof['cuda_is_initialized'] is False
assert proof['module_count']==len(proof['actual_loaded_modules']['modules']) and proof['function_code_count']>0
for name,r in proof['actual_loaded_modules']['modules'].items():assert roles['import_closure']['modules'][name]==r
profile=value(RUN+'/evidence/pre-gpu-profile.json')
sys.path.insert(0,str(REPO));from search_r1.baselines.e1_profile import validate_profile;from omegaconf import OmegaConf
validate_profile(OmegaConf.create(profile))
assert profile['baselines']['opsd_e1']['run_root']==RUN and profile['baselines']['opsd_e1']['binding_manifest_sha256']==ready['binding_sha256']
assert profile['actor_rollout_ref']['model']['path']==profile['baselines']['distillation']['teacher_path']==REMOTE+'/assets/qwen2.5-3b-instruct-aa8e725'
assert profile['retriever']['url']=='http://127.0.0.1:18084/retrieve'
assert profile['baselines']['opsd_e1']['public_key_sha256']==keySHA
assert not any(p in c['files'] for p in [RUN+'/bindings/gpu-init-approval.json',RUN+'/acceptance-5-receipt.json'])
summary=dict(status='ACTUAL_PRE_GPU_INDEPENDENT_REVIEW_PASS_INITIALIZATION_ONLY',source=SOURCE,job_id='3198',wrapper_pid=ready['pid'],process_start_ticks=actual['start'],nonce=ready['nonce'],ready_sha256=file(RUN+'/approval-ready.json')[1],binding_sha256=ready['binding_sha256'],public_key_sha256=keySHA,native_closure_sha256=closureSHA,modules=proof['module_count'],verified_function_count=proof['function_code_count'],uuids=ready['nvml_uuids'],all_seven_binding_hashes= {k:v['sha256'] for k,v in binding['files'].items()},collection_sha256=hashlib.sha256(raw).hexdigest(),scope='Actual current job CPU/asset/source/ready and emptyNVML evidence only; no GPU initialization, first5,300/effect acceptance')
(ROOT/'formal3198-pre-gpu-independent-audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,sort_keys=True))
