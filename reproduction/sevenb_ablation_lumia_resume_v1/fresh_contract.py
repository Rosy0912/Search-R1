"""Separate full 150-round fresh trial; no old round or optimizer credit."""
from pathlib import Path
import copy,json
from .contract import BASE,MODEL,DATA,GROUPS,SCIENCE,sha
from .contract import project_config as _project
def project_config(old,g,host):
 c=_project(old,g,host);c['trainer']['default_local_dir']=str(BASE/'fresh-runs'/g/'checkpoints');return c
def validate_plan(p):
 g=p['group'];assert g in GROUPS and p['job']=='zhangjunming-7b-abl-'+GROUPS[g]+'-lfresh1'
 assert p['root']==str(BASE/'fresh-runs'/g) and p['source_commit']==SCIENCE and p['protocol']=='sevenb_ablation_lumia_fresh150_after_human_resume_v1'
 assert p['fresh_from_common_initialization'] and p['parent_step']==0 and p['parent_checkpoint'] is None
 assert p['actual_update_limit']==150 and p['scheduler_scale']==200 and p['warmup_rounds']==57 and p['world_size']==8
 assert p['allocation']=={'gpus':8,'cpus':64,'memory_gib':900,'partition':'ADA6000,L40S','min_gpu_memory_mib':44000}
 assert p['no_auto_retry'] and p['science_unchanged'] and p['model']==MODEL and p['data_root']==DATA
 assert p['superseded_condition_state']=='TERMINATED' and len(p['parent_uid'])==36 and len(p['run_id'])==32
 return BASE/'fresh-runs'/g
def verify(p):
 root=validate_plan(p)
 for n,h in p['source_files'].items():assert sha(Path(p['source'])/n)==h,n
 for n,h in p['model_files'].items():assert sha(Path(MODEL)/n)==h,n
 data=json.loads((Path(DATA)/'manifest.json').read_text());assert sha(Path(DATA)/'manifest.json')==p['data_identity']
 for key,count in [('train',169028),('validation',512),('stage',64),('final',51713)]:
  r=data['splits'][key];assert r['rows']==count and sha(Path(DATA)/r['path'])==r['sha256']
 old=BASE/'parent-small'/(p['group']+'-resolved.yaml');assert sha(old)==p['parent_config_sha256']
 assert sha(p['retriever_receipt'])==p['retriever_receipt_sha256']
 return root
