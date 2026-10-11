"""Require actual four-rank optimizer evidence for completed outer iterations."""
import json,math
from pathlib import Path

def finite(value):
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,(list,tuple)):return all(finite(x) for x in value)
    if isinstance(value,(float,int)):return math.isfinite(value)
    return value is None or isinstance(value,(str,bool))

def verify(root,through,job,prior_job=None,prior_through=0,require_joint=True,prior_history=None):
    root=Path(root);events=[]
    for path in root.glob('main-*.jsonl'):
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    for path in root.glob('reranker-*.jsonl'):
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if prior_job is not None:
        if str(prior_job)==str(job) or not 0<prior_through<through:
            raise RuntimeError('Invalid recovery provenance interval')
        historical=verify(root,prior_through,prior_job,require_joint=False,**(prior_history or {}))
    elif prior_through:
        raise RuntimeError('Missing historical job identity')
    identities=[e for e in events if e['kind']=='initialized' and str(e['job'])==str(job)]
    latest={(e['role'],e['rank']):e for e in identities}
    if set(latest)!={(role,rank) for role in ('main','reranker') for rank in range(4)}:
        raise RuntimeError('Missing both-role four-rank GPU identities')
    main_uuids={latest[('main',rank)]['uuid'] for rank in range(4)}
    ranker_uuids={latest[('reranker',rank)]['uuid'] for rank in range(4)}
    if len(main_uuids)!=4 or main_uuids!=ranker_uuids:raise RuntimeError('Roles do not share exactly four GPUs')
    if any('job_'+str(job) not in e['cgroup'] for e in latest.values()):raise RuntimeError('Worker cgroup differs from allocated job')
    if any(e.get('kind')=='update' and str(e.get('job'))==str(job) and e['outer']>through for e in events):raise RuntimeError('Unexpected update beyond requested stopping boundary')
    updates=[e for e in events if e['kind']=='update' and str(e['job'])==str(job) and e['outer']<=through]
    if any(e['outer']<=prior_through for e in updates):raise RuntimeError('Recovery repeated completed iterations')
    if prior_job is not None:
        for segment in historical['provenance_intervals']:
            updates += [e for e in events if e['kind']=='update' and str(e['job'])==segment['job'] and segment['start']<=e['outer']<=segment['end']]
    main=[e for e in updates if e['role']=='main']
    expected={(step,rank) for step in range(1,through+1) for rank in range(4)}
    actual=[(e['outer'],e['rank']) for e in main]
    if set(actual)!=expected or len(actual)!=len(expected):raise RuntimeError('Missing or duplicate main updates')
    for e in updates:
        if e['world_size']!=4 or not finite(e['metrics']):raise RuntimeError('Invalid world size or nonfinite native metrics')
        if e['scheduler_after']-e['scheduler_before']!=1:raise RuntimeError('Incorrect scheduler advancement')
        if e['role']=='main' and (e['optimizer_calls']!=20 or e['local_rows']!=640):raise RuntimeError('Main batch or actual updates differ')
    reranker=[e for e in updates if e['role']=='reranker']
    joint=[]
    for step in range(1,through+1):
        records=[e for e in reranker if e['outer']==step]
        if records:
            if {e['rank'] for e in records}!=set(range(4)) or len(records)!=4:raise RuntimeError('Incomplete reranker rank participation')
            if len({e['optimizer_calls'] for e in records})!=1 or any(e['optimizer_calls']<=0 for e in records):raise RuntimeError('Inconsistent reranker updates')
            joint.append(step)
    if require_joint and not any(step>prior_through for step in joint):raise RuntimeError('No actual joint iteration: short-run joint acceptance is unproven')
    result={'status':'ACTUAL_UPDATE_SHORT_RUN_PASS','through_outer':through,'job':str(job),'main_updates_per_rank':through*20,'joint_iterations':joint,'gpu_memory_stability_still_requires_review':True,'checkpoint_resume_still_requires_verification':True}
    if not require_joint:result['status']='ACTUAL_MAIN_UPDATES_VERIFIED_JOINT_ACCEPTANCE_PENDING'
    if prior_job is not None:result['recovered_from_job']=str(prior_job);result['recovered_through_outer']=prior_through
    result['provenance_intervals']=(historical['provenance_intervals'] if prior_job is not None else [])+[{'job':str(job),'start':prior_through+1,'end':through}]
    return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('through',type=int);p.add_argument('job');a=p.parse_args()
    print(json.dumps(verify(a.root,a.through,a.job)))
