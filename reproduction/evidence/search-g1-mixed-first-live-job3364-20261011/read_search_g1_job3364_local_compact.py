from pathlib import Path
import json
p=max(Path('outputs/gigpo-e1-instruct200').glob('SEARCH-G1-JOB3364-ACTUAL-UPDATES*'),key=lambda p:p.stat().st_mtime)
d=json.loads(json.loads(json.loads(p.read_text())['stdout'])['stdout'])
print({'utc':d['utc'],'complete':[v['step'] for v in d['actual_updates']],'writes':[v['kind_counts'].get('OPTIMIZER_WRITE',0) for v in d['rank_write_counts'].values()],'monitors':[v['kind_counts'].get('POLICY_MONITOR',0) for v in d['rank_write_counts'].values()],'pools':d['candidate_pools'],'errors':[k for k,v in d['logs'].items() if any(x in v['tail'] for x in ['Traceback','RuntimeError:','CUDA out of memory','AssertionError:'])]})

import datetime,re
updates=d['actual_updates']
completed=[x['step'] for x in updates]
assert completed==list(range(1,len(completed)+1))
assert all(1 <= x['actual_optimizer_writes'] <= 20 and x['opportunities_processed']==20 for x in updates)
state_path=Path('/Users/shaoshitong/Documents/Search-G1/preparation/mixed-first-fresh200-state.json')
state=json.loads(state_path.read_text())
counts={k:v['kind_counts'].get('OPTIMIZER_WRITE',0) for k,v in d['rank_write_counts'].items()}
match=re.search(r'JobState=(\S+)',d['job']['stdout'])
state.update(updated_utc=d['utc'],last_observed_utc=d['utc'],completed_outer=len(completed),actual_adam_per_rank=min(counts.values()),actual_adam_per_rank_counts=counts,native_job_state=match.group(1) if match else 'UNKNOWN',actual_progress_evidence=str(p))
complete_writes=updates[-1]['optimizer_writes_after'] if updates else 0
assert min(counts.values()) >= complete_writes
backend_completed=max([a['completed_outer'] for a in d['actual_attempts'] if a.get('complete_outer')]+[len(completed)])
state['backend_completed_outer']=backend_completed
state['last_observed_stage']=('POST_UPDATE_EVALUATION_AND_CHECKPOINT' if backend_completed>len(completed) else 'ACTUAL_UPDATES' if min(counts.values())>complete_writes else 'NEXT_CANDIDATE_OR_UPDATE_PREPARATION')
state['latest_candidate_pools']=d['candidate_pools']
if d['candidate_pools']:
 latest_pool=max(d['candidate_pools'],key=lambda item:item['step'])
 state['current_attempt_id']=latest_pool['pool']
 state['completed_candidates']=latest_pool['completed']
 state['current_candidate_source_consumed']=latest_pool['source_consumed']
 state['candidate_statistics_scope']='latest observed candidate pool; mixed/fallback counts require its own selection audit'
 state['mixed_candidates']=None
 state['raw_completed_trajectories']=latest_pool['completed']*5
state['remaining'][0]=f'five complete and sixth start acceptance; currently {len(completed)} complete'
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
