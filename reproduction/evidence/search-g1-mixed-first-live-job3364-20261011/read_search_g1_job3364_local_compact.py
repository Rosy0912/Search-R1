from pathlib import Path
import json
p=max(Path('outputs/gigpo-e1-instruct200').glob('SEARCH-G1-JOB3364-ACTUAL-UPDATES*'),key=lambda p:p.stat().st_mtime)
d=json.loads(json.loads(json.loads(p.read_text())['stdout'])['stdout'])
print({'utc':d['utc'],'complete':[v['step'] for v in d['actual_updates']],'writes':[v['kind_counts'].get('OPTIMIZER_WRITE',0) for v in d['rank_write_counts'].values()],'monitors':[v['kind_counts'].get('POLICY_MONITOR',0) for v in d['rank_write_counts'].values()],'pools':d['candidate_pools'],'errors':[k for k,v in d['logs'].items() if any(x in v['tail'] for x in ['Traceback','RuntimeError:','CUDA out of memory','AssertionError:'])]})

import datetime,re
updates=d['actual_updates']
completed=[x['step'] for x in updates]
assert completed==list(range(1,len(completed)+1))
assert all(x['actual_optimizer_writes']==20 and x['opportunities_processed']==20 for x in updates)
state_path=Path('/Users/shaoshitong/Documents/Search-G1/preparation/mixed-first-fresh200-state.json')
state=json.loads(state_path.read_text())
counts={k:v['kind_counts'].get('OPTIMIZER_WRITE',0) for k,v in d['rank_write_counts'].items()}
match=re.search(r'JobState=(\S+)',d['job']['stdout'])
state.update(updated_utc=d['utc'],last_observed_utc=d['utc'],completed_outer=len(completed),actual_adam_per_rank=min(counts.values()),actual_adam_per_rank_counts=counts,native_job_state=match.group(1) if match else 'UNKNOWN',actual_progress_evidence=str(p))
state['last_observed_stage']='ACTUAL_UPDATES' if min(counts.values())>len(completed)*20 else 'NEXT_CANDIDATE_GENERATION'
state['latest_candidate_pools']=d['candidate_pools']
state['remaining'][0]=f'five complete and sixth start acceptance; currently {len(completed)} complete'
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
