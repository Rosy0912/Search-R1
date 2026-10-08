"""Source60f/actual3315 terminal-only readonly evidence candidate; never a release."""
import argparse,json,math,os,re,sys
from pathlib import Path
from first5_helpers import *
SOURCE_EXPECTED='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
RUN_EXPECTED=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
HELPER_SHA='3ea5370c5812a9460d580eb1f94e2f1a848b8423a790f23ea06470e8dbabd967'
def strict_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'Duplicate JSON key');d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def records(text,marker):
    return [strict_json(line.split(marker,1)[1])for line in text.splitlines()if marker in line]
def terminal_gate(proof):
    require(SOURCE==SOURCE_EXPECTED and RUN==RUN_EXPECTED and JOB==3315,'Fixed actual source/job/run changed')
    require(proof.get('source_commit')==SOURCE and proof.get('run_root')==str(RUN)and type(proof.get('job_id'))is int and proof['job_id']==JOB,'Actual terminal source/job/run required')
    require(proof.get('scheduler_state')=='COMPLETED' and type(proof.get('exit_code'))is int and proof['exit_code']==0,'Terminal COMPLETED zero exit required')
    require(proof.get('native_acceptance')is False,'Terminal observation cannot assert acceptance')
    require(type(proof.get('restarts'))is int and proof['restarts']==0 and type(proof.get('requeue'))is int and proof['requeue']==0,'Unique non-requeued actual job required')
    p=owned(proof['raw_slurm_path']);require(p.is_file()and digest(p)==proof.get('raw_slurm_sha256'),'Exact terminal raw Slurm bytes required')
    raw=p.read_text()
    for k,v in dict(JobId=str(JOB),JobState='COMPLETED',ExitCode='0:0',Restarts='0',Requeue='0',NodeList='h100-5',Comment='BAPO_FRESH200:formal200:'+SOURCE+':bab8981d9fd14afc98923eb4ebb89e33').items():
        require(re.search(r'(?:^|\s)'+re.escape(k)+'='+re.escape(v)+r'(?:\s|$)',raw)is not None,'Raw terminal identity differs '+k)
    return binding(p)
def replay_attempts(attempts,completed,pid):
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    progress=Fresh200Progress();accepted=[]
    for a in attempts:
        outer=progress.completed_outers+1;aid=progress.begin_attempt(outer)
        require(a.get('source_commit')==SOURCE and type(a.get('driver_pid'))is int and a['driver_pid']==pid,'Source/same-process attempt identity differs')
        require(type(a.get('attempt_id'))is int and a['attempt_id']==aid and type(a.get('candidate_outer'))is int and a['candidate_outer']==outer,'Ordered acceptedouter/attempt required')
        complete=progress.record(a['rank_write_ledger'],outer)
        require(a.get('accepted_complete_outer')is complete and a.get('status')==a['rank_write_ledger'][0]['status'] and a.get('progress')==progress.state_dict(),'Actual ledger/progress differs')
        if complete:accepted.append(outer)
    require(accepted==list(range(1,201))and completed==progress.state_dict(),'Exactly200 accepted outers and final actual cumulative ledger required')
    return dict(progress=progress.state_dict(),accepted_outers=accepted,opportunities_per_accepted_outer=[20]*49+[10]*151,expected_no_skip_opportunities=2490,actual_writes_per_rank=progress.actual_adam_writes_per_rank,scope='Actual retained abort writes included; skipped/aborted attempts do not advance acceptedouter or scheduler')
def final_metrics(metrics,size):
    counts={k.removeprefix('val/evaluated_count/'):v for k,v in metrics.items()if k.startswith('val/evaluated_count/')}
    require(type(size)is int and size==51713 and counts and all(type(v)is int and v>0 for v in counts.values())and sum(counts.values())==51713,'Full final51713 counts required; stage64/debug512 excluded')
    scores={k:metrics.get('val/test_score/'+k)for k in counts}
    require(all(finite(v)for v in scores.values()),'Finite per-source final scores required')
    require(all(not isinstance(v,(int,float))or finite(v)for v in metrics.values()),'Nonfinite final metric')
    return dict(evaluated_count=51713,counts=counts,per_source_scores=scores,weighted_score=sum(scores[k]*n for k,n in counts.items())/51713,raw_metrics=metrics,scope='Terminal final only; no stage/debug pooling')
def collect(proof_path,output,hash_files=False,cpu_hf=False):
    # No heavy reads until independently supplied exact actual terminal evidence.
    proof_path=owned(proof_path);proof=strict_json(proof_path.read_text());terminal=terminal_gate(proof)
    bind_source();require(digest(Path(__file__).with_name('first5_helpers.py'))==HELPER_SHA,'Reviewed helper bytes changed')
    sys.path.insert(0,str(ROOT/'repo'))
    log=(RUN/'train.log').read_bytes();require(log.endswith(b'\n'),'Partial terminal train log')
    text=log.decode();done=records(text,'FRESH200_COMPLETED_RUN_JSON=');require(len(done)==1,'Exactly one completedrun marker')
    attempts=records(text,'FRESH200_ATTEMPT_JSON=');require(attempts,'Actual attempts absent');pid=attempts[0]['driver_pid'];ledger=replay_attempts(attempts,done[0],pid)
    final=records(text,'SEARCH_R1_FINAL_METRICS_JSON=');sizes=re.findall(r'^(?:\([^\n]*pid=\d+\)\s*)?Final validation dataset size: (\d+)$',re.sub(r'\x1b\[[0-9;]*m','',text),re.M)
    require(len(final)==1 and len(sizes)==1,'Unique terminal final aggregate/dataset marker required')
    metrics=final_metrics(final[0],int(sizes[0]))
    pauses=records(text,'FRESH200_STEP5_PAUSED_JSON=');releases=records(text,'FRESH200_STEP5_RELEASED_JSON=')
    require(len(pauses)==len(releases)==1 and pauses[0].get('pid')==releases[0].get('pid')==pid,'Original first5 pause/release/same-driver chronology required')
    train_metrics=records(text,'BAPO_TRAIN_METRICS_JSON=')
    require(len(train_metrics)==len(attempts),'Metrics required for every attempt, not only200 accepted')
    for a,m in zip(attempts,train_metrics):
        require(m.get('attempt_id')==a['attempt_id'] and m.get('step')==a['candidate_outer'] and m.get('accepted_complete_outer')is a['accepted_complete_outer'],'Attempt metrics identity differs')
        lp=owned(a['ledger_path']);require(RUN in lp.parents and strict_json(lp.read_text())==a,'Immutable perattempt raw ledger differs')
    state=RUN/'checkpoints/fresh200_state/outer_200_COMPLETE';status=strict_json((state/'status.json').read_text())
    require(status.get('status')=='COMPLETE' and status.get('actual_outer')==200 and status.get('completed_outer_updates')==200 and status.get('progress')==done[0],'Final200 status/counters differ')
    ranks=[]
    for rank in range(8):
        p=state/'ranks'/f'rank_{rank}.json';r=strict_json(p.read_text())
        require(r.get('rank')==rank and r.get('world_size')==8 and r.get('step')==200 and r.get('scheduler_last_epoch')==200 and r.get('status')=='FULL_STATE_SERIALIZED_OK','Final native saved rank/schema differs')
        require(r.get('roundtrip')is False,'Source200 save is not GPU reload; never fabricate roundtrip')
        require(type(r.get('pid'))is int and r['pid']>0 and r.get('model_state_dict_type')=='SHARDED_STATE_DICT' and r.get('model_dtensor_entries',0)>0 and r.get('optimizer_state_entries',0)>0,'Actual native200 model/Adam evidence required')
        ranks.append(r)
    require(len({r['pid']for r in ranks})==8,'Eight distinct native worker PIDs required')
    cleanup_path=RUN/'own-process-cleanup.json';cleanup=strict_json(cleanup_path.read_text())
    require(cleanup.get('own_marker')==str(RUN) and cleanup.get('remaining')==[],'Exact own-run terminal cleanup required')
    resources=[RUN/'resource-watch.jsonl',RUN/'gpu-monitor.csv',RUN/'cpu-process-monitor.jsonl',RUN/'process-monitor.csv',RUN/'driver-time.log',RUN/'phase.log']
    resource_bindings=[dict(path=str(owned(p)),bytes=p.stat().st_size,**({'sha256':digest(p)}if hash_files else {}))for p in resources]
    groups={'native_state':state,'hf_export':RUN/'checkpoints/actor/global_step_200','algorithm':RUN/'checkpoints/algorithm/global_step_200'}
    members=[]
    for role,d in groups.items():
        require(d.is_dir()and not d.is_symlink(),'Expected final directory absent '+role)
        for p in sorted(d.rglob('*')):
            require(not p.is_symlink(),'Final artifact symlink refused')
            if p.is_file():members.append(dict(role=role,path=str(owned(p)),bytes=p.stat().st_size,**({'sha256':digest(p)}if hash_files else {})))
    result=dict(status='FINAL200_READONLY_COLLECTION_REQUIRES_INDEPENDENT_REVIEW',source_commit=SOURCE,run_root=str(RUN),job_id=JOB,driver_pid=pid,terminal=terminal,train_log=binding(RUN/'train.log'),ledger=ledger,attempt_metrics=train_metrics,final_evaluation=metrics,ranks=ranks,files=members,resource_files=resource_bindings,own_cleanup=binding(cleanup_path),full_file_hashes_performed=hash_files,native_gpu_reload_verified=False,native_acceptance=False,production_release=False,unknown_or_pending=['Loss/gradient all-rank raw numerical replay','Guardian terminal journal/cleanup/cost review','Actualinitial official all-tensor/8rank Adam0/scheduler0 proof','First5 independent release receipt and same PID chronology','Independent full native200 CPU loading limitations (DTensor); never fake processgroup'])
    if hash_files:
        import torch
        from search_r1.baselines.full_state import state_digest
        driver=torch.load(state/'driver.pt',map_location='cpu',weights_only=False);reload=strict_json((state/'driver-reload.json').read_text())
        require(driver.get('source_commit')==SOURCE and driver.get('global_step')==200 and driver.get('next_global_step')==201 and driver.get('actual_outer_updates')==200 and driver.get('progress')==done[0] and driver.get('driver_pid')==pid,'Actual driver200 state identity differs')
        require(reload.get('status')=='DRIVER_SERIALIZED_OK' and reload.get('pid')==pid and reload.get('state_digest')==state_digest(driver),'Serialized driver200 digest differs; no roundtrip implied')
        for r in ranks:
            path=owned(r['path']);require(state in path.parents and digest(path)==r['sha256'],'Native200 complete saved bytes differ')
        result['driver_serialization_verified']=True
    if cpu_hf:
        require(hash_files,'Full file hashes before optional CPU HF')
        result['cpu_hf']=hf_proof(MODEL,groups['hf_export'])
    output=owned(output);require(RUN/'logs' in output.parents,'New run/logs report only');write_new(output,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--terminal-proof',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute-terminal',action='store_true');p.add_argument('--hash-files',action='store_true');p.add_argument('--cpu-hf',action='store_true');a=p.parse_args()
    if not a.execute_terminal:print(json.dumps(dict(status='CODE_ONLY_CANDIDATE_NO_EXECUTION',job_id=JOB,native_acceptance=False)))
    else:
        require(a.terminal_proof is not None and a.output is not None,'Explicit terminal proof/output required');collect(a.terminal_proof,a.output,a.hash_files,a.cpu_hf)
