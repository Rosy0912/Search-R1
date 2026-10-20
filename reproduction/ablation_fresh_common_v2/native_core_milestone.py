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

def milestone_binding(sub,plan,step):
 assert type(step) is int and step in (50,100)
 b=binding(sub,plan);b.update(step=step,native=str(Path(b['root'])/'checkpoints/training_state'/('global_step_'+str(step))/'post-evaluation'),hf=str(Path(b['root'])/'checkpoints/actor'/('global_step_'+str(step))))
 return b

def ready(s,b):
 for k in ['group','job_id','job_name','root','plan_sha256','source','science','output','native','hf','run_id','step']:assert s[k]==b[k],k
 assert s['resources']==dict(gpus=8,cpus=64,memory_gib=900)
 if s['state'] not in ('RUNNING','COMPLETED'):return False
 assert s['scheduler_exit']=='0:0' and s['current_entry_bound'] and s['configuration_projection_verified'] and s['eight_uuid_entry']
 if s['state']=='RUNNING':assert s['worker_exit'] is s['parent_wait_rc'] is s['entry_exit_rc'] is None
 else:assert s['worker_exit']=='0' and s['parent_wait_rc']==s['entry_exit_rc']==0 and s['terminal_clock_order_verified']
 if not s['manifest_exists'] or not s.get('milestone_record_exists'):return False
 step=b['step'];assert step in (50,100) and s['completed_outer']>=step and s['payload_clock_order_verified']
 rec=s['milestone_record'];assert rec['completed_outer']==step and rec['group']==b['group'] and rec['run_id']==b['run_id'] and rec['source_commit']==SCIENCE and rec['config_sha256']==s['resolved_sha256']
 assert rec['checkpoint']['path']==str(Path(b['native'])/'manifest.json') and rec['checkpoint']['sha256']==s['manifest_sha256'] and not rec['checkpoint']['native_only']
 m=s['manifest'];i=m['identity'];assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and i['job_id']==b['job_id'] and i['code_commit']==SCIENCE and i['completed_step']==step and i['actual_update_limit']==150
 assert i['config_sha256']==s['resolved_sha256'] and i['data_identity']==b['data_identity']
 assert rec['optimizer_calls_per_rank']==i['guard_continue_accounting']['completed_optimizer_calls_per_rank']
 assert len(m['worker_states'])==8 and {r['identity']['rank'] for r in m['worker_states']}==set(range(8))
 for r in m['worker_states']:
  rank=r['identity']['rank'];assert r['identity']['world_size']==8 and all(r['identity'][k]==v for k,v in i.items())
  assert r['path']==str(Path(b['native'])/('rank-%02d.pt'%rank)) and r['bytes']>0 and r['full_worker_state_finite'] and re.fullmatch('[a-f0-9]{64}',r['sha256'])
 assert m['driver']['path']==str(Path(b['native'])/'driver.pt') and m['driver']['next_step']==step+1 and m['hf_checkpoint']==b['hf']
 return s['all_payload_files_present'] and s['stable_payload_metadata'] and s['cpu_headroom_bytes']>20*1024**3
