import hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
RUN=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
OUT=RUN/'logs/live-history-recovery-a4'
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
raw_result=(OUT/'history-replay.json').read_bytes();assert sha(raw_result)=='7464a7d0564e2742c78e3c3cd6d3bbb66b3de34a25fd31363105d9cabb861077'
saved=h.strict_json(raw_result);verify_snapshot_prefix(saved['prefix'])
raw_index=(OUT/'ledger-index.json').read_bytes();assert sha(raw_index)=='e605f2292aefca93993e42359a754435c2ea9543f46e78b0095c9190cfd5da01'
index=h.strict_json(raw_index);assert len(index)==126
docs={}
for i,entry in enumerate(index,1):
    origin=RUN/('checkpoints/diagnostics/attempt_%d_candidate_outer_%d/all-rank-write-ledger.json'%(i,i))
    copy=OUT/('ledger-%d.json'%i)
    assert entry['original_path']==str(origin) and entry['local_path']==str(copy)
    for p in (origin,copy): assert p.is_file() and not p.is_symlink()
    raw=origin.read_bytes();assert raw==copy.read_bytes() and sha(raw)==entry['sha256']
    docs[str(origin)]=h.strict_json(raw)
raw=(OUT/'train-prefix.log').read_bytes();assert len(raw)==237834792 and sha(raw)==saved['prefix']['sha256']
fulltext=raw.decode();lines=fulltext.splitlines(keepends=True);last=max(i for i,line in enumerate(lines) if 'BAPO_TRAIN_METRICS_JSON=' in line);text=''.join(lines[:last+1]);tail=''.join(lines[last+1:]);scope=h.strict_json((OUT/'live-scope.json').read_bytes());assert scope['last_full_line']==last+1 and scope['excluded_trailing_bytes']==len(tail.encode())==50 and scope['excluded_trailing_sha256']==sha(tail.encode());replayed=h.replay(text,docs)
assert all(saved[k]==v for k,v in replayed.items())
assert replayed['progress']['completed_outers']==126 and replayed['progress']['attempts']==126
ranks=[];current=[];counts={};extrema={};last=[]
for line in text.splitlines():
    if 'ACTOR_RANK_UPDATE_JSON=' in line:
        current.append(h.strict_json(line.split('ACTOR_RANK_UPDATE_JSON=',1)[1]))
    elif 'FRESH200_ATTEMPT_JSON=' in line:
        event=h.strict_json(line.split('FRESH200_ATTEMPT_JSON=',1)[1]);assert len(current)==8
        for rank in current:
            for key in ('actor/pg_loss','actor/grad_norm'):
                values=rank['metrics'][key];assert len(values)==(20 if event['attempt_id']<50 else 10) and all(type(v) in (int,float) and math.isfinite(v) for v in values)
                counts[key]=counts.get(key,0)+len(values)
                lo,hi=extrema.get(key,(math.inf,-math.inf));extrema[key]=[min(lo,min(values)),max(hi,max(values))]
        if event['attempt_id'] in (99,100,125,126):
            last.append({'attempt_id':event['attempt_id'],'rank_writes':[x['optimizer_steps'] for x in sorted(current,key=lambda x:x['rank'])],
                'all_rank_means':{key:math.fsum(v for x in current for v in x['metrics'][key])/sum(len(x['metrics'][key]) for x in current) for key in ('actor/pg_loss','actor/grad_norm')},
                'scheduler_epochs':[x['scheduler_last_epoch'] for x in sorted(current,key=lambda x:x['rank'])]})
        ranks.extend(current);current=[]
assert not current and len(ranks)==1008 and counts=={'actor/pg_loss':14000,'actor/grad_norm':14000}
assert all(not x['numerical_coverage_gaps'] for x in replayed['attempts'])
verify_snapshot_prefix(saved['prefix']);assert not torch.cuda.is_initialized()
debug=[h.strict_json(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]) for line in text.splitlines() if 'BAPO_TRAIN_METRICS_JSON=' in line]
debug100=[x for x in debug if x['step']==100];assert len(debug100)==1
assert debug100[0]['metrics']['val/qa_count/all']==512 and debug100[0]['metrics']['val/evaluated_count/nq']==debug100[0]['metrics']['val/evaluated_count/hotpotqa']==256
assert debug100[0]['metrics']['val/em/all']==148/512
assert hashlib.sha256(json.dumps([x for x in debug if x['step'] in (50,100)],sort_keys=True,allow_nan=False).encode()).hexdigest()=='425549890ddde2024048074c31a044ea6b759856ec6a86707fd9113660e4f457'
from pathlib import Path
import json,datetime,os,time
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
run=root/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3315,'run':str(run),'scope':'Small metadata and file presence only; no large hash, torch/HF/GPU load or acceptance','groups':{}}
for label,base in [('hf',run/'checkpoints/actor/global_step_100'),('native',run/'checkpoints/fresh200_state/outer_100_COMPLETE'),('algorithm',run/'checkpoints/algorithm/global_step_100')]:
 files=[];meta={}
 if base.is_dir():
  for p in sorted(base.rglob('*')):
   if p.is_file():
    st=p.stat();files.append({'relative':str(p.relative_to(base)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'symlink':p.is_symlink()})
    if (p.name in ('status.json','driver-reload.json','state.json','model.safetensors.index.json') or (p.parent.name=='ranks' and p.suffix=='.json')) and st.st_size<2000000:
     try:meta[str(p.relative_to(base))]=json.loads(p.read_text())
     except ValueError:meta[str(p.relative_to(base))]={'status':'WRITE_IN_PROGRESS_OR_MALFORMED'}
 result['groups'][label]={'directory_exists':base.is_dir(),'files':files,'small_metadata':meta}
st=os.statvfs(run);result['disk_available_bytes']=st.f_bavail*st.f_frsize

assert sha(json.dumps(result['groups'],sort_keys=True,allow_nan=False).encode())=='afe868ba6b3dff3c4a0e75a00642b1b4a62770b9f39bedb55b0e4d49f9594c6c'
native=result['groups']['native']['small_metadata'];event100=docs[str(RUN/'checkpoints/diagnostics/attempt_100_candidate_outer_100/all-rank-write-ledger.json')]
for i in range(8):
 v=native['ranks/rank_%d.json'%i];assert v['latest_rank_event']==event100['rank_write_ledger'][i] and v['step']==100 and v['scheduler_last_epoch']==100 and v['roundtrip'] is False and v['status']=='FULL_STATE_SERIALIZED_OK'
assert native['status.json']['progress']==event100['progress']
assert native['driver-reload.json']['pid']==760122 and native['driver-reload.json']['step']==100
assert result['groups']['algorithm']['small_metadata']['state.json']['algorithm_source_commit']==h.SOURCE
print(json.dumps(dict(status='PASS_INDEPENDENT_TARGET_COMPLETED126_PREFIX_AND100_METADATA_ONLY',progress=replayed['progress'],prefix=saved['prefix'],completed_text_sha256=sha(text.encode()),excluded_trailing_bytes=len(tail.encode()),excluded_trailing_sha256=sha(tail.encode()),ledger_index_sha256=sha(raw_index),result_sha256=sha(raw_result),rank_record_count=len(ranks),numeric_counts=counts,numeric_extrema=extrema,numerical_gap_attempts=[],selected_outers=last,debug100=debug100[0],checkpoint100groups_sha256=sha(json.dumps(result['groups'],sort_keys=True,allow_nan=False).encode()),checkpoint100group_counts={k:len(v['files']) for k,v in result['groups'].items()},checkpoint100progress=event100['progress'],cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,whole200_accepted=False),allow_nan=False))
