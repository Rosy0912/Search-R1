"""Local independent actual100 snapshot and fixed-set aggregate audit. No remote/control."""
from pathlib import Path
from types import ModuleType
from datetime import datetime,timezone
import hashlib,json,os,gzip,stat
B=Path(__file__).resolve().parent;S=B/'actual-snapshots/3261-20261006T005604410916Z'
def sha(r):return hashlib.sha256(r).hexdigest()
def loads(r):
 def pairs(xs):
  d={}
  for k,v in xs:
   assert k not in d,k;d[k]=v
  return d
 v=json.loads(r,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
 json.dumps(v,allow_nan=False);return v

def read(p,pin=None):
 assert p.resolve(strict=True)==p and p.is_file()
 with p.open('rb') as f:
  a=os.fstat(f.fileno());assert a.st_uid==os.getuid() and stat.S_ISREG(a.st_mode)
  r=f.read();z=os.fstat(f.fileno());n=p.stat()
  key=lambda x:(x.st_dev,x.st_ino,x.st_uid,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
  assert key(a)==key(z)==key(n) and len(r)==z.st_size
 if pin:assert sha(r)==pin
 return r
sr=read(S/'observation-summary.json');s=loads(sr);assert len(s['files'])==400
file_checks=[]
for relative,meta in s['files'].items():
 assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
 r=read(S/'original-files'/relative,meta['sha256']);assert len(r)==meta['bytes']
 file_checks.append({'relative':relative,'sha256':sha(r),'bytes':len(r)})
assert s['errors']==[{'path':'/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261/runtime','reason':'symlink_not_followed'}]
p=B/'review_progress_typed.py';r=read(p,'d9c1e303b32e74c747bf9158aefdbc5bb173d22df49eae14ab44d12a69af2eca')
m=ModuleType('independent_actual100_typed');m.__file__=str(p);exec(compile(r,str(p),'exec'),m.__dict__)
result=m.review(S);assert result['actual_complete_outer']==result['aggregate_complete_outer']==100
assert result['same_process_adam_calls_per_rank']==1600
root=loads(read(B/'actual3261-progress-20261006T0056-typed.json'));assert result==root
with (B/'INDEPENDENT_ACTUAL100_TYPED_PROGRESS_ORIGINAL.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
g,c,tm,tmeta=m.runtime_helpers();ctx,_=m.source_context(S,s,g,c,tm,tmeta)
streams={}
for label,rel in [('driver','ray-stdout-pid-1533312.out'),('aggregate','train.log')]:
 r=m.captured_member(S,s,rel)[0];streams[label]=c.parse_records(r[:r.rfind(b'\n')+1],ctx)
coverage=c.check_evaluation_sequence(streams['driver'],100);coverage_a=c.check_evaluation_sequence(streams['aggregate'],100)
assert [(v['kind'],v['step'],v['count']) for v in coverage['events']]==[('dev',0,512),('stage',5,64),('dev',50,512),('stage',50,64),('dev',100,512),('stage',100,64)]
assert [{k:v for k,v in e.items() if k!='console_line'} for e in coverage['events']]==[{k:v for k,v in e.items() if k!='console_line'} for e in coverage_a['events']]
metrics_keys=['em','f1','format_rate','answer_format','search_calls','generated_tokens']
fixed={kind:[{'step':e['step'],'count':e['count'],**{k:e['original_metrics']['val/'+k+'/all'] for k in metrics_keys}} for e in coverage['events'] if e['kind']==kind] for kind in ['dev','stage']}
resources=s['resource_history'];gr=read(S/'resources-prefix-original.ndjson.gz',resources['gzip_sha256']);assert len(gr)==resources['gzip_bytes']
res_sha=hashlib.sha256();res_bytes=0
with gzip.open(S/'resources-prefix-original.ndjson.gz','rb') as f:
 while True:
  chunk=f.read(4*1024*1024)
  if not chunk:break
  res_sha.update(chunk);res_bytes+=len(chunk)
assert res_bytes==resources['prefix_bytes'] and res_sha.hexdigest()==resources['sha256']
assert resources['complete_prefix_at_start'] is True and resources['before']['bytes']==res_bytes
pending_root='evidence/checkpoint-100-cpu-inspection-once-v1/'
assert pending_root+'native.json' not in s['files'] and pending_root+'inspection-completed-review-required.json' not in s['files']
for kind in ['observer','native','srun','controller']:assert pending_root+kind+'-exit-original.json' not in s['files']
clock_rows=streams['driver']['outers']
for step,row in enumerate(clock_rows,1):
 for rank in range(8):
  assert row['metrics']['actor/optimizer_steps/rank_'+str(rank)]==16 and row['metrics']['actor/samples/rank_'+str(rank)]==64
  assert row['metrics']['actor/opsd_loss_finite/rank_'+str(rank)]==row['metrics']['actor/grad_norm_finite/rank_'+str(rank)]==1
out={'version':1,'status':'PASS_ACTUAL100_PROGRESS_AND_FIXED_SET_AGGREGATE_PAIRING_WITH_NEGATIVE_EFFECT',
 'reviewer_role':'INDEPENDENT_REVIEWER','utc':datetime.now(timezone.utc).isoformat(),'snapshot':str(S),'snapshot_finished_utc':s['finished_utc'],
 'snapshot_summary_sha256':sha(sr),'actual400_original_SHA_verified':True,'originals':file_checks,
 'typed_caller_sha256':sha(read(p)),'typed_result_sha256':sha(read(B/'INDEPENDENT_ACTUAL100_TYPED_PROGRESS_ORIGINAL.json')),
 'root_existing_typed_result_SHA':sha(read(B/'actual3261-progress-20261006T0056-typed.json')),'exact_root_typed_result_match':True,
 'source_commit':m.SOURCE,'config_sha256':m.CONFIG,'job_id':'3261','driver_pid':1533312,'driver_start_ticks':'1829085316',
 'actual_complete_outer':100,'aggregate_complete_outer':100,'same_process_Adam_calls_per_rank':1600,
 'all1to100_8rank16_optimizer_calls_64samples_finite_and_teacher_clocks_from_frozen_gate':True,
 'latest_metrics':result['latest_metrics'],'gpu_gib_range':[min(result['observed_gpu_used_gib']),max(result['observed_gpu_used_gib'])],
 'fixed_evaluation_coverage':coverage,'aggregate_evaluation_pairing_equal':True,'same512_dev':fixed['dev'],'same64_stage':fixed['stage'],
 'effect_conclusion':'Negative fixed512 aggregate effect: dev EM/F1/format lower at100 than initialization and50. No effect-improvement claim. Same64 stage100 returns to EM of5 and below50.',
 'preserved_FSDP_frame_sha256':result['log_join']['interleaved_frames']['aggregate'][0]['record_sha256'],
 'resources_captured_prefix':{'bytes':res_bytes,'sha256':res_sha.hexdigest(),'gzip_sha256':resources['gzip_sha256'],'scope':'Complete prefix at capture start, not whole terminal accounting or writer closure.'},
 'CPU100_scope':'Launch/in-flight observer only at this snapshot; no native/exits/completion so no stored-content acceptance.',
 'limits':['Fixed-set aggregates and matching console/outer JSON, not per-question trajectory re-scoring or new checkpoint restoration.',
 'Changing training batches prevent using train reward differences as paired EM effects.',
 'Logged OPSD loss is clipped objective, not true KL. Gradient norm is before gradient clipping.',
 'Original frozen formal/c738 terminal parser still rejects the preserved known FSDP frame; its true terminal must be reported separately.',
 'Resources gzip is a captured prefix and may continue growing. No final cost, resource release or final200/51713 acceptance.']}
with (B/'ACTUAL100_PROGRESS_AND_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps({'status':out['status'],'report_sha256':sha(read(B/'ACTUAL100_PROGRESS_AND_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json')),'same512':fixed['dev'],'same64':fixed['stage']},ensure_ascii=False))
