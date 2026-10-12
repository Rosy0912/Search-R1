"""Bind actual3357 COMPLETE5 and its original3356 chain, CPU metadata only.

Never submits, releases, loads rank states, hashes large checkpoints or assets.
Large checkpoint SHA values come only from retained producer sidecars; the real
native loader and first-five gate must still verify their bytes later.
"""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v8-20261009')
RUN=ROOT/'runs/formal-aligned-em200-ae20107aeead4a848b43e450d0c1ca11-3357'
SOURCE='e82ef0231a23eedfd68ebba1644966a5759a7575'
OLD_RUN=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356')
OLD_SOURCE='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47'
STATE=RUN/'checkpoints/fresh200_state/outer_5_COMPLETE'
LIMIT=64*1024*1024
SUFFIXES={'.json','.jsonl','.log','.yaml','.yml','.txt','.csv','.err','.out'}
def require(ok,message):
    if not ok:raise RuntimeError(message)
def ordinary(p):
    p=Path(p);require(p.is_file()and not p.is_symlink()and p.resolve()==p,'ordinary canonical file required: '+str(p));return p
def small(p):
    p=ordinary(p);require(p.suffix!='.pt'and p.stat().st_size<=LIMIT,'small metadata only: '+str(p))
    raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):
    b=small(p);raw=Path(p).read_bytes();require(hashlib.sha256(raw).hexdigest()==b['sha256'],'metadata changed while reading');return json.loads(raw),b
def producer(report):
    p=ordinary(report['path']);require(p.suffix=='.pt'and (RUN in p.parents or OLD_RUN in p.parents),'checkpoint outside exact original chain')
    require(re.fullmatch('[0-9a-f]{64}',report.get('sha256',''))is not None,'original producer SHA required')
    return dict(path=str(p),bytes=p.stat().st_size,sha256=report['sha256'])
def exclusive(p,value):
    require(p.parent.is_dir()and p.parent.resolve()==p.parent,'canonical existing output directory required')
    with p.open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def main():
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'optimized collector forbidden')
    require(sys.platform=='linux'and os.uname().nodename=='h100-11','actual node11 collector required')
    require(ROOT.resolve()==ROOT and RUN.resolve()==RUN and RUN.is_dir(),'canonical original3357 required')
    output=ROOT/'evidence/complete5-resume-bindings3357-v2.json'
    require(not output.exists()and not output.is_symlink(),'existing or unknown outcome: no overwrite/retry')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'repo',text=True).strip()==SOURCE,'original3357 source changed')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'repo',text=True).strip(),'original source dirty')
    lineage,lineage_binding=read(RUN/'partial-resume/lineage.json');nested=lineage['manifest']
    require(nested['old_run_root']==str(OLD_RUN)and nested['old_source_commit']==OLD_SOURCE and nested['target_run_root']==str(RUN)and nested['target_source_commit']==SOURCE and nested['target_job_id']=='3357','original nested3356 identity differs')
    require(len(lineage['rank_witnesses'])==8 and {w['rank']for w in lineage['rank_witnesses']}==set(range(8))and lineage['driver_witness']['status']=='EXACT_ALIGNED_PARTIAL_DRIVER_RELOAD_OK','original partial witnesses absent')
    closure={};producer_map={};reports=[];ranks=[]
    def add(b):
        prior=closure.get(b['path']);require(prior is None or prior==b,'conflicting original binding');closure[b['path']]=b
    # Bind every retained small3357 record, including empty diagnostic logs.
    for p in sorted(RUN.rglob('*')):
        if p.is_file()and p.suffix in SUFFIXES:add(small(p))
    # Cross-root files are admitted only by the retained3357 partial manifest.
    for b in nested['original_closure']:
        p=ordinary(b['path']);require(OLD_RUN in p.parents and p.stat().st_size==b['bytes'],'nested original path/stat differs')
        if p.suffix!='.pt':require(small(p)==b,'nested small original changed')
        add(b)
    # Producer sidecars prove deferred hashes, including all nine initial files.
    for b in list(closure.values()):
        if Path(b['path']).suffix=='.json':
            doc=json.loads(Path(b['path']).read_bytes())
            if isinstance(doc,dict)and isinstance(doc.get('path'),str)and Path(doc['path']).suffix=='.pt'and 'sha256'in doc:
                pb=producer(doc);prior=producer_map.get(pb['path']);require(prior is None or prior==pb,'conflicting producer SHA');producer_map[pb['path']]=pb
    for rank in range(8):
        report,rb=read(STATE/'ranks'/f'rank_{rank}.json');state=producer(report)
        require(state['path']==str(STATE/'ranks'/f'rank_{rank}.pt')and report.get('status')=='EXACT_FULL_STATE_RELOAD_OK'and report.get('roundtrip')is True and report.get('native_state_version')==3,'original native COMPLETE5 roundtrip missing')
        require(all(type(report.get(k))is int and report[k]==v for k,v in [('rank',rank),('world_size',8),('step',5),('scheduler_last_epoch',5)])and 'aligned_pending_outer'in report and report['aligned_pending_outer']is None,'original committed5 native counters differ')
        require(report.get('state_digest')==report.get('reloaded_state_digest')and re.fullmatch('[0-9a-f]{64}',report.get('state_digest',''))is not None,'original native digest mismatch')
        event=report['latest_rank_event'];require(event['rank']==rank and event['attempt_id']==5 and event['optimizer_steps']==20 and event['outer_commit_phase']=='STAGE_COMMITTED'and event['accepted_complete_outer']is True,'original fifth transaction not committed')
        ranks.append(dict(rank=rank,state=state,report=rb));reports.append(report);add(state);add(rb)
    ledger,lb=read(STATE/'all-rank-write-ledger.json')
    require(isinstance(ledger,list)and len(ledger)==8 and {e['rank']for e in ledger}==set(range(8))and all(next(e for e in ledger if e['rank']==r['rank'])==r['latest_rank_event']for r in reports),'original all-rank ledger differs')
    dr,drb=read(STATE/'driver-reload.json');driver=producer(dr)
    require(driver['path']==str(STATE/'driver.pt')and driver['bytes']<=LIMIT and dr.get('status')=='EXACT_DRIVER_RELOAD_OK'and dr.get('step')==5,'small exact COMPLETE5 driver required')
    add(driver);add(drb)
    for b in list(closure.values()):
        if Path(b['path']).suffix=='.pt':require(producer_map.get(b['path'])==b,'checkpoint lacks matching retained producer sidecar')
    initials=[b for b in producer_map.values()if Path(b['path']).name.startswith('native-initial-')and OLD_RUN in Path(b['path']).parents]
    require(len(initials)==9 and {Path(b['path']).name for b in initials}=={'native-initial-official-hf.pt'}|{f'native-initial-rank_{r}.pt'for r in range(8)},'exact original nine initial checkpoints required')
    for b in initials:require(OLD_RUN in Path(b['path']).parents,'initial official identity must remain original3356');add(b)
    config=small(RUN/'resolved-config.yaml');require(config['path']in closure,'original config missing')
    probe=r'''
import os,sys,json,hashlib,math,torch
from pathlib import Path
from search_r1.baselines.full_state import state_digest
from search_r1.baselines.fresh200_driver import Fresh200Progress
from search_r1.baselines.fresh200_contract import validate_rank_updates
root=Path(sys.argv[1]);p=Path(sys.argv[2]);report=json.loads(sys.argv[3]);ledger=json.loads(sys.argv[4])
assert sys.prefix==str(root/'env')and os.environ['CUDA_VISIBLE_DEVICES']==''and not torch.cuda.is_available()and not torch.cuda.is_initialized()
assert hashlib.sha256(p.read_bytes()).hexdigest()==report['sha256']
d=torch.load(p,map_location='cpu',weights_only=False)
assert state_digest(d)==report['state_digest']and report['pid']==d['driver_pid']
assert d['status']=='COMPLETE'and d['protocol']=='BAPO_FRESH200_V1'and d['source_commit']==sys.argv[5]
for k,v in dict(global_step=5,next_global_step=6,attempt_id=5,actual_outer_updates=5,attempted_outer=5,previous_completed_outer=4,epoch=0).items():assert type(d[k])is int and d[k]==v
assert d['rank_write_ledger']==ledger and type(d['sampler']['cursor'])is int and d['sampler']['cursor']==2560
assert d['progress']==Fresh200Progress(baseline_aligned=True,attempts=5,completed_outers=5,actual_adam_writes_per_rank=[100]*8,recorded_attempts=5).state_dict()
validate_rank_updates(ledger,5)
assert 'rng'in d and d['bapo']['patience']==5
b=d['bapo'];assert type(b['best'])in(int,float)and math.isfinite(b['best'])and 0<=b['best']<=1 and type(b['stalled'])is int and b['stalled']>=0 and type(b['early'])is bool
assert not torch.cuda.is_available()and not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_SMALL_COMPLETE5_DRIVER_CPU_BINDING_VERIFIED_NOT_NATIVE_RELOAD',driver_pid=d['driver_pid'],state_digest=state_digest(d),source_commit=d['source_commit'],component_digests={k:state_digest(d[k])for k in ('rng','sampler','bapo')},progress=d['progress'],sampler=d['sampler'])))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
    result=subprocess.run([str(ROOT/'env/bin/python'),'-B','-c',probe,str(ROOT),str(STATE/'driver.pt'),json.dumps(dr),json.dumps(ledger),SOURCE],cwd=ROOT/'repo',env=env,capture_output=True,text=True,timeout=120)
    require(result.returncode==0,'CPU driver verification failed: '+result.stdout+'\n'+result.stderr);proof=json.loads(result.stdout)
    value=dict(schema='aligned-complete5-3357-binding-template-v1',old_run_root=str(RUN),old_source_commit=SOURCE,ranks=ranks,rank_write_ledger=ledger,original_ledger=lb,driver=driver,driver_report=drb,original_config=config,original_closure=sorted(closure.values(),key=lambda b:b['path']),nested_partial_lineage=lineage_binding,driver_cpu_proof=proof,scope='ORIGINAL_COMPLETE5_BINDINGS_TEMPLATE_NO_TARGET_JOB_NO_SUBMISSION_NO_NATIVE_RELOAD',deferred_checks=['all_large_native_SHA_and_full_v3_load','actual_unique_target_identity','exact_8rank_and_driver_roundtrip','current_process_first5_CPUHF_independent_acceptance_and_release'])
    require(not any(k.startswith('target_')for k in value),'template cannot authorize a target')
    for b in value['original_closure']:
        if Path(b['path']).suffix=='.pt':require(ordinary(b['path']).stat().st_size==b['bytes']and producer_map.get(b['path'])==b,'producer-bound checkpoint stat changed')
        else:require(small(b['path'])==b,'original small record changed during collection')
    exclusive(output,value)
    print(json.dumps(dict(status='ORIGINAL3357_COMPLETE5_BINDINGS_TEMPLATE_COLLECTED_NOT_RESTORED',output=small(output),scope=value['scope'])))
if __name__=='__main__':main()
