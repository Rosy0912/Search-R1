"""CPU-only metadata binding for the actual3356 PARTIAL; never submit or release.

Rank .pt bytes are not read here: original producer SHA is carried for the exact
rank loader to verify later. Only the small driver payload is CPU-unpickled.
"""
from pathlib import Path
import hashlib,json,math,os,re,subprocess,sys
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009')
RUN=ROOT/'runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356'
SOURCE='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47'
PARTIAL=RUN/'checkpoints/fresh200_state/attempt_1_candidate_outer_1_PARTIAL'
LIMIT=64*1024*1024

def require(ok,message):
    if not ok:raise RuntimeError(message)
def ordinary(p):
    p=Path(p);require(p.is_file()and not p.is_symlink()and p.resolve()==p,'ordinary canonical source file required: '+str(p));return p

def small_binding(p):
    p=ordinary(p);require(p.stat().st_size<=LIMIT and p.suffix!='.pt','small metadata only: '+str(p))
    raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def json_file(p):
    binding=small_binding(p);return json.loads(Path(p).read_bytes()),binding

def exclusive(p,value):
    require(p.parent.is_dir()and p.parent.resolve()==p.parent and not p.is_symlink(),'canonical output required')
    with p.open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'optimized collector forbidden')
    require(sys.platform=='linux'and os.uname().nodename=='h100-11','actual node11 metadata collector required')
    require(ROOT.resolve()==ROOT and RUN.resolve()==RUN and RUN.is_dir(),'canonical original3356 required')
    output=ROOT/'evidence/partial-resume-bindings3356-v2.json';stage_path=ROOT/'evidence/partial-resume-stage3356-v2.json'
    for p in (output,stage_path):require(not p.exists()and not p.is_symlink(),'existing or unknown collector outcome: no overwrite/retry')
    require(output.parent.is_dir()and output.parent.resolve()==output.parent,'canonical existing evidence directory required')
    # Validate all original rank metadata, deferring large-state hashing/loading.
    ranks=[];reports=[]
    for rank in range(8):
        path=ordinary(PARTIAL/'ranks'/f'rank_{rank}.pt');side=path.with_suffix('.json');report,side_binding=json_file(side)
        require(report.get('status')=='PARTIAL_CURRENT_STATE_SAVED'and report.get('native_state_version')==3 and report.get('rank')==rank and report.get('world_size')==8 and report.get('step')==1 and report.get('scheduler_last_epoch')==0 and report.get('path')==str(path),'original rank partial metadata mismatch')
        require(isinstance(report.get('sha256'),str)and re.fullmatch('[0-9a-f]{64}',report['sha256'])is not None,'original producer native SHA required')
        require(report.get('latest_rank_event',{}).get('optimizer_steps')==20 and report.get('aligned_pending_outer',{}).get('scheduler_before')==0,'original twenty retained writes required')
        ranks.append(dict(rank=rank,state=dict(path=str(path),bytes=path.stat().st_size,sha256=report['sha256']),report=side_binding));reports.append(report)
    original_ledger,ledger_binding=json_file(PARTIAL/'all-rank-write-ledger.json')
    require(isinstance(original_ledger,list)and len(original_ledger)==8 and {r['rank']for r in original_ledger}==set(range(8)),'original all-rank ledger required')
    require(all(next(x for x in original_ledger if x['rank']==r['rank'])==r['latest_rank_event']for r in reports),'sidecar/original ledger mismatch')
    driver_report,driver_report_binding=json_file(PARTIAL/'driver-reload.json');driver=ordinary(PARTIAL/'driver.pt')
    require(driver.stat().st_size<=LIMIT and driver_report.get('status')=='PARTIAL_CURRENT_STATE_SAVED'and driver_report.get('step')==1 and driver_report.get('path')==str(driver),'small original partial driver required')
    driver_binding=dict(path=str(driver),bytes=driver.stat().st_size,sha256=driver_report['sha256'])
    log=ordinary(RUN/'train.log');log_binding=small_binding(log);raw=log.read_bytes();require(raw.endswith(b'\n'),'terminal train log must be complete-line')
    markers=[]
    for line in raw.splitlines():
        marker=b'BASELINE_EVALUATION_METRICS_JSON='
        if marker in line:
            value=json.loads(line.split(marker,1)[1]);counts=[v for k,v in value.items()if k.startswith('val/evaluated_count/')]
            if counts and sum(counts)==64 and value.get('val/qa_count/all')==64:markers.append((line,line.split(marker,1)[1],value))
    require(len(markers)>=1,'original stage64 metric marker required')
    line,marker_json,metrics=markers[-1];total=count=0
    for key,value in metrics.items():
        if key.startswith('val/test_score/'):
            n=metrics['val/evaluated_count/'+key.split('/')[-1]];require(type(n)is int and n>0,'stage source count must be integer')
            total+=float(value)*n;count+=n
    score=float(total/count)if count else float('nan')
    require(count==64 and math.isfinite(score)and score==14/64,'actual stage14/64 required')
    config_binding=small_binding(RUN/'resolved-config.yaml')
    # Inspect only driver.pt, with GPUs hidden before the isolated Python starts.
    probe=r'''
import os,sys,json,hashlib,torch
from pathlib import Path
from search_r1.baselines.full_state import state_digest
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()and not torch.cuda.is_initialized()
root=Path(sys.argv[1]);p=Path(sys.argv[2]);expected=json.loads(sys.argv[3]);assert sys.prefix==str(root/'env')
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
v=torch.load(p,map_location='cpu',weights_only=False)
assert state_digest(v)==expected['state_digest'] and v['status']=='PARTIAL'and v['source_commit']==sys.argv[4]
assert v['global_step']==1 and v['next_global_step']==1 and v['actual_outer_updates']==0 and v['attempt_id']==1
assert v['bapo']==dict(best=14/64,early=True,stalled=0,patience=5)
assert v['sampler']['cursor']==512 and set(v['sampler'])=={'epoch','cursor','order_sha256'}
assert v['progress']['attempts']==1 and v['progress']['completed_outers']==0 and v['progress']['recorded_attempts']==0
assert not torch.cuda.is_available()and not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_SMALL_DRIVER_CPU_BINDING_VERIFIED_NOT_NATIVE_RELOAD',driver_pid=v['driver_pid'],bapo=v['bapo'],sampler=v['sampler'],progress=v['progress'],state_digest=state_digest(v),source_commit=v['source_commit'],score_observed_once=True)))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
    command=[str(ROOT/'env/bin/python'),'-B','-c',probe,str(ROOT),str(driver),json.dumps(driver_report),SOURCE]
    result=subprocess.run(command,cwd=ROOT/'repo',env=env,capture_output=True,text=True,timeout=120)
    require(result.returncode==0,'actual CPU driver verification failed: '+result.stdout+'\n'+result.stderr);driver_proof=json.loads(result.stdout)
    # Archive bindings of small original proof records; never touch rank .pt.
    suffixes={'.json','.jsonl','.log','.yaml','.yml','.txt','.csv','.err','.out'}
    closure=[]
    for path in sorted(RUN.rglob('*')):
        if path.is_file()and path.suffix in suffixes:closure.append(small_binding(path))
    # Carry original producer hashes for large native files without reading them.
    closure.extend(entry['state']for entry in ranks)
    closure.append(driver_binding)
    require(any('/training-em-audit/'in b['path']for b in closure)and any('/aligned-outer-prepared/'in b['path']for b in closure),'original EM/prepared proof missing')
    initial_reports=[]
    for rank in range(8):
        matches=[b for b in closure if Path(b['path']).name==f'native-initial-rank_{rank}.json']
        require(len(matches)==1,'exact original native initial report required')
        initial=json.loads(Path(matches[0]['path']).read_bytes())
        require(initial.get('scheduler_last_epoch')==0 and initial.get('lr')==[1e-6/57] and initial.get('native_process',{}).get('rank')==rank,'original initial scheduler0 warmup LR differs')
        initial_reports.append(dict(rank=rank,report=matches[0],lr=initial['lr'],scheduler_last_epoch=0))
    stage=dict(schema='aligned-partial-original-stage3356-v1',old_run_root=str(RUN),old_source_commit=SOURCE,original_log=log_binding,
        marker_line_sha256=hashlib.sha256(marker_json).hexdigest(),original_full_line_sha256=hashlib.sha256(line).hexdigest(),metrics=metrics,score=score,evaluated_count=64,observed_in_saved_bapo=True,
        driver=driver_binding,driver_report=driver_report_binding,driver_state_digest=driver_report['state_digest'],driver_cpu_proof=driver_proof,
        scope='ORIGINAL_STAGE_ALREADY_OBSERVED_ONCE_NOT_A_SECOND_STAGE_EVALUATION')
    exclusive(stage_path,stage)
    template=dict(schema='aligned-partial3356-binding-template-v1',old_run_root=str(RUN),old_source_commit=SOURCE,ranks=ranks,rank_write_ledger=original_ledger,
        original_ledger=ledger_binding,driver=driver_binding,driver_report=driver_report_binding,original_config=config_binding,original_closure=closure,
        stage_metrics=small_binding(stage_path),original_log=log_binding,marker_line_sha256=stage['marker_line_sha256'],metrics=metrics,score=score,
        observed_in_saved_bapo=True,driver_cpu_proof=driver_proof,initial_scheduler_reports=initial_reports,scope='ORIGINAL_BINDINGS_TEMPLATE_NO_TARGET_JOB_NO_SUBMISSION_NO_NATIVE_RELOAD',
        deferred_checks=['all8_large_native_SHA_and_full_v3_load','actual_target_job_source_run_binding','cross_job_driver_native_roundtrip','pending_stage_commit_once','lineage_first5_acceptance'])
    # No target identity exists yet: callers must bind actual unique new job later.
    require(not any(k.startswith('target_')for k in template),'template cannot preauthorize target identity')
    producer_bindings={entry['state']['path']:entry['state']for entry in ranks}
    producer_bindings[driver_binding['path']]=driver_binding
    for b in closure:
        if Path(b['path']).suffix=='.pt':
            require(producer_bindings.get(b['path'])==b and ordinary(b['path']).stat().st_size==b['bytes'],'producer-bound native file stat changed during collection')
        else:require(small_binding(Path(b['path']))==b,'original small proof changed during collection')
    require(small_binding(log)==log_binding,'original log changed during collection')
    exclusive(output,template)
    print(json.dumps(dict(status='ORIGINAL3356_PARTIAL_BINDINGS_TEMPLATE_COLLECTED_NOT_RESTORED',path=str(output),sha256=small_binding(output)['sha256'],stage_metrics_sha256=small_binding(stage_path)['sha256'],scope=template['scope'])))
if __name__=='__main__':main()
