import copy, json
from pathlib import Path
from release.recursive_bran_a.ablation150_v1.contracts import sha,validate_config_dict
SCIENCE='e265dae594e8f0368f889815d7f6c4729257ce6c'
BASE=Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-Ablations-Resume-20261020-v1')
MODEL='/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1/models/base/a09a35458c702b33eeacc393d103063234e8bc28'
DATA='/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1/common-data'
GROUPS={'budget-5':'b5','C1-root-only':'c1','C2-random-final':'c2','budget-20':'b20','generation-2':'d2'}
STEPS={'budget-5':45,'C1-root-only':39,'C2-random-final':38,'budget-20':30,'generation-2':38}
def project_config(old,group,host):
 c=copy.deepcopy(old);oldmodel=c['actor_rollout_ref']['model']['path'];olddata=str(Path(c['data']['train_files']).parent)
 def walk(x):
  if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
  if isinstance(x,list):return [walk(v) for v in x]
  if isinstance(x,str):return x.replace(oldmodel,MODEL).replace(olddata,DATA)
  return x
 c=walk(c);c['trainer']['default_local_dir']=str(BASE/'runs'/group/'checkpoints');c['baselines']['local_runtime_host']=host
 validate_config_dict(c);return c
def validate_plan(p):
 g=p['group'];assert g in GROUPS and p['job']=='zhangjunming-7b-abl-'+GROUPS[g]+'-lresume1'
 assert p['root']==str(BASE/'runs'/g) and p['source_commit']==SCIENCE
 assert p['actual_update_limit']==150 and p['scheduler_scale']==200 and p['warmup_rounds']==57
 assert p['parent_step']==STEPS[g] and not p['fresh_from_common_initialization']
 assert p['allocation']=={'gpus':8,'cpus':64,'memory_gib':900,'partition':'ADA6000,L40S','min_gpu_memory_mib':44000}
 assert p['no_auto_retry'] and p['science_unchanged'] and p['world_size']==8
 assert p['model']==MODEL and p['data_root']==DATA
 return BASE/'runs'/g
def verify(p):
 root=validate_plan(p)
 for n,h in p['source_files'].items():assert sha(Path(p['source'])/n)==h,n
 for n,h in p['model_files'].items():assert sha(Path(MODEL)/n)==h,n
 data=json.loads((Path(DATA)/'manifest.json').read_text());assert sha(Path(DATA)/'manifest.json')==p['data_identity']
 for key,count in [('train',169028),('validation',512),('stage',64),('final',51713)]:
  r=data['splits'][key];assert r['rows']==count and sha(Path(DATA)/r['path'])==r['sha256']
 stage=BASE/'parents'/p['group'];manifest=stage/'manifest.json';assert sha(manifest)==p['parent_manifest_sha256']
 m=json.loads(manifest.read_text());assert m['identity']['completed_step']==p['parent_step'] and m['identity']['code_commit']==SCIENCE
 assert m['identity']['data_identity']==p['data_identity'] and m['identity']['job_id']==p['parent_job'] and len(m['worker_states'])==8
 assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and m['driver']['next_step']==p['parent_step']+1
 assert {r['identity']['rank'] for r in m['worker_states']}==set(range(8))
 proof=json.loads((stage/'cpu-accept.json').read_text());assert sha(stage/'cpu-accept.json')==p['cpu_acceptance_sha256']
 assert proof['status']=='PINNED_FULL_NATIVE_INDEPENDENT_CPU_ACCEPTED' and proof['parent_uid']==p['parent_uid'] and proof['manifest_sha256']==p['parent_manifest_sha256'] and proof['all_full_payload_sha_and_logical_state_verified']
 transfer=json.loads((stage/'transfer-complete.json').read_text());assert transfer['manifest_sha256']==p['parent_manifest_sha256'] and transfer['all_destination_physical_sha_verified']
 for r in m['worker_states']+[m['driver']]:
  f=stage/Path(r['path']).name;assert f.is_file() and not f.is_symlink() and transfer['files'][f.name]['sha256']==r['sha256']
 assert sha(stage/'resolved.yaml')==m['identity']['config_sha256']==p['parent_config_sha256']
 assert sha(p['retriever_receipt'])==p['retriever_receipt_sha256']
 return root
def mapped_manifest(p):
 stage=BASE/'parents'/p['group'];assert sha(stage/'manifest.json')==p['parent_manifest_sha256']
 m=json.loads((stage/'manifest.json').read_text())
 for r in m['worker_states']+[m['driver']]:r['path']=str(stage/Path(r['path']).name)
 return m
