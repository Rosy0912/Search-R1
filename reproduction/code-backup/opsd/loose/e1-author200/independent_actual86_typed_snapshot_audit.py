"""Independent local only full snapshot SHA + fixed-prefix join. No remote or training control."""
from pathlib import Path
import hashlib, json, os, stat, importlib.util
B=Path(__file__).resolve().parent
S=B/'actual-snapshots/3261-20261006T001926275981Z'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(raw):
 def pairs(xs):
  d={}
  for k,v in xs:
   if k in d: raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def read(p):
 assert p.is_absolute() and p.resolve(strict=True)==p
 with p.open('rb') as f:
  a=os.fstat(f.fileno());assert stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid()
  raw=f.read();z=os.fstat(f.fileno());n=p.stat()
  key=lambda q:(q.st_dev,q.st_ino,q.st_uid,q.st_size,q.st_mtime_ns,q.st_ctime_ns)
  assert key(a)==key(z)==key(n) and len(raw)==z.st_size
 return raw
summaryraw=read(S/'observation-summary.json');d=load(summaryraw)
assert d['job_id']=='3261' and d['node']=='h100-10'
assert len(d['files'])==352
checked=[]
for rel,rec in d['files'].items():
 p=S/'original-files'/rel
 assert Path(rel).as_posix()==rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts
 raw=read(p)
 assert sha(raw)==rec['sha256'] and len(raw)==rec['bytes']
 checked.append(dict(relative=rel,sha256=sha(raw),bytes=len(raw)))
assert d['errors']==[{'path':'/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261/runtime','reason':'symlink_not_followed'}]
callerraw=read(B/'review_progress_typed.py');assert sha(callerraw)=='d9c1e303b32e74c747bf9158aefdbc5bb173d22df49eae14ab44d12a69af2eca'
from types import ModuleType
m=ModuleType('independent_typed_caller86');m.__file__=str(B/'review_progress_typed.py');exec(compile(callerraw,m.__file__,'exec'),m.__dict__)
g,c,tm,tmr=m.runtime_helpers();ctx,anchors=m.source_context(S,d,g,c,tm,tmr)
pre=load(read(B/'ACTUAL75_NEW_FRAMING_TOOLS_DEPLOYMENT_AND_CPU_START_INDEPENDENT_REVIEW.json'))
prefixraw=[];joins={}
for label,rel,other in [('driver','ray-stdout-pid-1533312.out','driver'),('aggregate','train.log','training')]:
 raw=m.captured_member(S,d,rel)[0];q=pre['prelaunch_log_prefixes'][other];p=raw[:q['bytes']]
 assert len(p)==q['bytes'] and p.endswith(b'\n') and sha(p)==q['sha256']
 records=c.parse_records(p,ctx)
 assert len(records['outers'])==pre['prelaunch_complete_counts_as_recorded'][label]
 prefixraw.append(p);joins[label]=dict(bytes=len(p),sha256=sha(p),outer_count=len(records['outers']))
pp=m.review_logs(*prefixraw,ctx,c)
assert pp['actual_complete_outer']==86 and pp['aggregate_complete_outer']==85
frames=pp['log_join']['interleaved_frames']['aggregate']
assert len(frames)==1 and frames[0]['record_sha256']=='2653535246fd9c4d4fe193fbc58c4f5502822c1cf7b189f0d40021f6cafb92a2'
assert frames[0]['source_binding']['source_record']['sha256']=='ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
actual=load(read(B/'INDEPENDENT_ACTUAL_LATEST_TYPED_PROGRESS_001926.json'))
assert actual['actual_complete_outer']==86 and actual['aggregate_complete_outer']==85
assert actual['snapshot_summary']['sha256']==sha(summaryraw)
assert actual['same_process_adam_calls_per_rank']==1376
assert actual['job_id']=='3261' and actual['source_commit']==m.SOURCE and actual['driver_pid']==1533312
out=dict(status='PASS_LOCAL_FULL352_SHA_AND_ACTUAL86_FIXED_PRELAUNCH_PREFIX_JOIN',
 scope='Actual captured originals and typed logs/resources only; not native75, live restore, terminal, final200 or effect improvement',
 snapshot=str(S),snapshot_summary_sha256=sha(summaryraw),snapshot_finished_utc=d['finished_utc'],
 caller_sha256=sha(callerraw),actual_progress_output_sha256=sha(read(B/'INDEPENDENT_ACTUAL_LATEST_TYPED_PROGRESS_001926.json')),
 source_commit=m.SOURCE,config_sha256=m.CONFIG,job_id='3261',driver_pid=1533312,driver_start_ticks='1829085316',
 original_file_count=len(checked),all_original_files_sha_match=True,original_files=checked,
 prelaunch_fixed_prefixes=joins,frame_sha256=frames[0]['record_sha256'],additional_fsdp_source_sha256=frames[0]['source_binding']['source_record']['sha256'],
 actual_complete_outer=86,aggregate_complete_outer=85,same_process_adam_calls_per_rank=1376,
 latest_metrics=actual['latest_metrics'],gpu_gib_range=[min(actual['observed_gpu_used_gib']),max(actual['observed_gpu_used_gib'])],
 errors_preserved=d['errors'],limits=['Resources and logs are captured observations. Resource history may be a captured prefix, not final whole accounting.',
 '75 CPU pipeline and observer were running at capture; absence of native/exits is preserved.',
 'Training batch rewards are different cohorts; negative fixed512 dev0 to50 result remains unchanged.',
 'The known immutable formal terminal parser failure on preserved FSDP frame remains unresolved by this read-only caller.'])
p=B/'independent-actual86-full352-prefix-join-original.json'
with p.open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(status=out['status'],files=len(checked),actual_complete_outer=86,aggregate_complete_outer=85,output_sha256=sha(p.read_bytes()))))
