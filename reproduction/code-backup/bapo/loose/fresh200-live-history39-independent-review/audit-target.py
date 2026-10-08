import hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
RUN=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
OUT=RUN/'logs/live-history-recovery-a1'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
sys.path.insert(0,str(ROOT/'first5-evidence-tools-3315-v4'))
from first5_helpers import bind_source,verify_snapshot_prefix,digest
assert digest(ROOT/'first5-evidence-tools-3315-v4/first5_helpers.py')=='716f9410465c2f642b5cf9af7cbf6d7f503ff43c58aeb4e195f2c82deae525df'
path=ROOT/'final-evidence-tools-3315-v3/history_validator.py'
assert digest(path)=='dc0b2810342a93ec5288a3a8fd23f886bb14399151196a9b790771ae60a8475b'
wall=time.perf_counter();cpu=time.process_time();bind_source()
import torch
assert not torch.cuda.is_initialized()
spec=importlib.util.spec_from_file_location('review_history',path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
raw_result=(OUT/'history-replay.json').read_bytes();assert sha(raw_result)=='0bbec1bbb652ad33bf81f8f79bcd317d9c1626251e0e0c456eeee8b76cccda28'
saved=h.strict_json(raw_result);verify_snapshot_prefix(saved['prefix'])
raw_index=(OUT/'ledger-index.json').read_bytes();assert sha(raw_index)=='86e9f1b7225358025b7b85bfcb2f0ba80aa624acf91ce91666f8dec28d8b0f2f'
index=h.strict_json(raw_index);assert len(index)==39
docs={}
for i,entry in enumerate(index,1):
    origin=RUN/('checkpoints/diagnostics/attempt_%d_candidate_outer_%d/all-rank-write-ledger.json'%(i,i))
    copy=OUT/('ledger-%d.json'%i)
    assert entry['original_path']==str(origin) and entry['local_path']==str(copy)
    for p in (origin,copy): assert p.is_file() and not p.is_symlink()
    raw=origin.read_bytes();assert raw==copy.read_bytes() and sha(raw)==entry['sha256']
    docs[str(origin)]=h.strict_json(raw)
raw=(OUT/'train-prefix.log').read_bytes();assert len(raw)==105620659 and sha(raw)==saved['prefix']['sha256']
replayed=h.replay(raw.decode(),docs)
assert all(saved[k]==v for k,v in replayed.items())
assert replayed['progress']['completed_outers']==39 and replayed['progress']['attempts']==39
ranks=[];current=[];counts={};extrema={};last=[]
for line in raw.decode().splitlines():
    if 'ACTOR_RANK_UPDATE_JSON=' in line:
        current.append(h.strict_json(line.split('ACTOR_RANK_UPDATE_JSON=',1)[1]))
    elif 'FRESH200_ATTEMPT_JSON=' in line:
        event=h.strict_json(line.split('FRESH200_ATTEMPT_JSON=',1)[1]);assert len(current)==8
        for rank in current:
            for key in ('actor/pg_loss','actor/grad_norm'):
                values=rank['metrics'][key];assert len(values)==20 and all(type(v) in (int,float) and math.isfinite(v) for v in values)
                counts[key]=counts.get(key,0)+len(values)
                lo,hi=extrema.get(key,(math.inf,-math.inf));extrema[key]=[min(lo,min(values)),max(hi,max(values))]
        if event['attempt_id']>=37:
            last.append({'attempt_id':event['attempt_id'],'rank_writes':[x['optimizer_steps'] for x in sorted(current,key=lambda x:x['rank'])],
                'all_rank_means':{key:math.fsum(v for x in current for v in x['metrics'][key])/sum(len(x['metrics'][key]) for x in current) for key in ('actor/pg_loss','actor/grad_norm')},
                'scheduler_epochs':[x['scheduler_last_epoch'] for x in sorted(current,key=lambda x:x['rank'])]})
        ranks.extend(current);current=[]
assert not current and len(ranks)==312 and counts=={'actor/pg_loss':6240,'actor/grad_norm':6240}
assert all(not x['numerical_coverage_gaps'] for x in replayed['attempts'])
verify_snapshot_prefix(saved['prefix']);assert not torch.cuda.is_initialized()
print(json.dumps({'status':'PASS_ACTUAL_READONLY_HISTORY_1_TO_39_NOT_FINAL_ACCEPTANCE','source_commit':h.SOURCE,'job_id':3315,'run_root':str(RUN),'driver_pid':replayed['driver_pid'],
 'progress':replayed['progress'],'prefix':saved['prefix'],'result_sha256':sha(raw_result),'ledger_index_sha256':sha(raw_index),'ledger_bindings':index,
 'raw_rank_record_count':len(ranks),'numeric_value_counts':counts,'numeric_extrema':extrema,'last_three':last,'gap_attempts':[],
 'validator_sha256':digest(path),'helper_sha256':digest(ROOT/'first5-evidence-tools-3315-v4/first5_helpers.py'),
 'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,'whole200_accepted':False,'release_authorized':False,'gpu_initialized':False},allow_nan=False))
