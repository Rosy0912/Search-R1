import datetime,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
RUN=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
sys.path.insert(0,str(ROOT/'first5-evidence-tools-3315-v4'))
from first5_helpers import bind_source,immutable_prefix,verify_snapshot_prefix,write_new,digest,owned
assert digest(ROOT/'first5-evidence-tools-3315-v4/first5_helpers.py')=='716f9410465c2f642b5cf9af7cbf6d7f503ff43c58aeb4e195f2c82deae525df'
modulepath=ROOT/'final-evidence-tools-3315-v3/history_validator.py'
assert digest(modulepath)=='dc0b2810342a93ec5288a3a8fd23f886bb14399151196a9b790771ae60a8475b'
bind_source();assert not __import__('torch').cuda.is_initialized()
spec=importlib.util.spec_from_file_location('actual_frozen_history_validator',modulepath);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
OUT=RUN/'logs/live-history-recovery-a4';assert not OUT.exists();OUT.mkdir(mode=0o700)
wall=time.perf_counter();cpu=time.process_time();prefix=immutable_prefix(RUN/'train.log',OUT/'train-prefix.log');snapshot_text=(OUT/'train-prefix.log').read_text();lines=snapshot_text.splitlines(keepends=True);last_full=max(i for i,line in enumerate(lines) if 'BAPO_TRAIN_METRICS_JSON=' in line);text=''.join(lines[:last_full+1]);excluded_tail=''.join(lines[last_full+1:]);write_new(OUT/'live-scope.json',{'last_full_line':last_full+1,'excluded_trailing_bytes':len(excluded_tail.encode()),'excluded_trailing_sha256':hashlib.sha256(excluded_tail.encode()).hexdigest(),'scope':'Replay only complete original prefix; preserved raw snapshot includes in-progress trailing attempt'});documents={};index=[]
for line in text.splitlines():
 if 'FRESH200_ATTEMPT_JSON='not in line:continue
 event=h.strict_json(line.split('FRESH200_ATTEMPT_JSON=',1)[1]);path=owned(Path(event['ledger_path']));assert path.name=='all-rank-write-ledger.json'and RUN in path.parents
 raw=path.read_bytes();assert str(path)not in documents;doc=h.strict_json(raw);assert doc==event
 copy=OUT/('ledger-'+str(event['attempt_id'])+'.json')
 with copy.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 documents[str(path)]=doc;index.append({'original_path':str(path),'local_path':str(copy),'sha256':hashlib.sha256(raw).hexdigest()})
result=h.replay(text,documents);verify_snapshot_prefix(prefix);assert not __import__('torch').cuda.is_initialized()
result.update(prefix=prefix,validator_sha256=digest(modulepath),helper_sha256=digest(ROOT/'first5-evidence-tools-3315-v4/first5_helpers.py'),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,scope='Full actual completed prefix plus copied durable ledgers; not final200 acceptance or generation-chain audit')
write_new(OUT/'ledger-index.json',index);write_new(OUT/'history-replay.json',result)
summary={k:result[k]for k in ['status','source_commit','job_id','run_root','driver_pid','progress','prefix','validator_sha256','helper_sha256','observed_utc','cpu_seconds','wall_seconds','scope']}
summary.update(output_path=str(OUT/'history-replay.json'),output_sha256=digest(OUT/'history-replay.json'),ledger_index_path=str(OUT/'ledger-index.json'),ledger_index_sha256=digest(OUT/'ledger-index.json'),attempts_replayed=len(result['attempts']),gap_attempts=[x['attempt_id']for x in result['attempts']if x['numerical_coverage_gaps']],whole200_accepted=False)
summary['excluded_trailing_bytes']=len(excluded_tail.encode());summary['debug512_records']=[h.strict_json(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]) for line in text.splitlines() if 'BAPO_TRAIN_METRICS_JSON=' in line and h.strict_json(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]).get('step') in [50,100]];write_new(OUT/'summary.json',summary);print(json.dumps(summary,allow_nan=False))
