"""New Slurm identities and actual final150 execution; no tensor imports."""
from pathlib import Path
import re
SOURCE='b366989a90254813b5e790fd1f8c1fdfce7189d6'
SCIENCE='e265dae594e8f0368f889815d7f6c4729257ce6c'
BASE=Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-Ablations-Resume-20261020-v1')
GROUPS={'budget-5':'b5','C1-root-only':'c1','C2-random-final':'c2','budget-20':'b20','generation-2':'d2'}
def binding(sub,plan):
 g=sub['group'];assert g in GROUPS and sub['job']=='zhangjunming-7b-abl-'+GROUPS[g]+'-lfresh1'
 assert str(sub['job_id']).isdigit() and sub['fresh_full150'] and sub['source']==SOURCE and sub['science_source']==SCIENCE
 assert plan['source_commit']==SCIENCE and plan['entry_source_commit']==SOURCE and plan['fresh_from_common_initialization']
 assert plan['parent_checkpoint'] is None and plan['parent_step']==0 and plan['actual_update_limit']==150 and plan['scheduler_scale']==200 and plan['warmup_rounds']==57
 assert sub['root']==plan['root']==str(BASE/'fresh-runs'/g) and plan['job']==sub['job']
 assert re.fullmatch('[a-f0-9]{64}',sub['plan_sha256']) and re.fullmatch('[a-f0-9]{32}',plan['entry_run_id'])
 return {'group':g,'job_id':str(sub['job_id']),'job_name':sub['job'],'root':sub['root'],'plan_sha256':sub['plan_sha256'],'source':SOURCE,'science':SCIENCE,'run_id':plan['run_id'],'entry_run_id':plan['entry_run_id'],'output':str(Path(sub['root'])/'output'/('attempt-'+plan['entry_run_id'])),'native':str(Path(sub['root'])/'checkpoints/training_state/global_step_150/post-evaluation'),'hf':str(Path(sub['root'])/'checkpoints/actor/global_step_150'),'data_identity':plan['data_identity']}
def ready(s,b):
 for k in ['group','job_id','job_name','root','plan_sha256','source','science','output','native','hf','run_id']:
  assert s[k]==b[k],k
 assert s['resources']==dict(gpus=8,cpus=64,memory_gib=900)
 if s['state']!='COMPLETED':return False
 assert s['scheduler_exit']=='0:0' and s['worker_exit']=='0' and s['parent_wait_rc']==0 and s['entry_exit_rc']==0
 assert s['terminal_clock_order_verified'] and s['current_entry_bound'] and s['configuration_projection_verified'] and s['eight_uuid_entry']
 if not s['complete_exists'] or not s['manifest_exists']:return False
 c=s['training_complete'];assert c['status']=='COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV'
 assert c['completed_outer']==150 and c['group']==b['group'] and c['run_id']==b['run_id'] and c['source_commit']==SCIENCE
 assert c['evaluation_steps']==[5,50,100,150]
 m=s['manifest'];i=m['identity'];assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
 assert i['job_id']==b['job_id'] and i['code_commit']==SCIENCE and i['completed_step']==i['actual_update_limit']==150
 assert i['config_sha256']==s['resolved_sha256']==c['config_sha256'] and i['data_identity']==b['data_identity']
 assert len(m['worker_states'])==8 and {r['identity']['rank'] for r in m['worker_states']}==set(range(8))
 for r in m['worker_states']:
  rank=r['identity']['rank'];assert r['identity']['world_size']==8 and all(r['identity'][k]==v for k,v in i.items())
  assert r['path']==str(Path(b['native'])/('rank-%02d.pt'%rank)) and r['bytes']>0 and r['full_worker_state_finite']
  assert re.fullmatch('[a-f0-9]{64}',r['sha256'])
 assert m['driver']['path']==str(Path(b['native'])/'driver.pt') and m['driver']['next_step']==151 and m['hf_checkpoint']==b['hf']
 assert c['checkpoint']['path']==str(Path(b['native'])/'manifest.json') and c['checkpoint']['sha256']==s['manifest_sha256']
 assert c['actual_optimizer_calls_per_rank']==i['guard_continue_accounting']['completed_optimizer_calls_per_rank']
 return s['all_payload_files_present'] and s['stable_payload_metadata'] and s['cpu_headroom_bytes']>20*1024**3
