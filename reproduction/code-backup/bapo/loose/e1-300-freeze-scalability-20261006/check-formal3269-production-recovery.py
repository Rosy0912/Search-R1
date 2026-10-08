from pathlib import Path
import json,hashlib,types,collections,math,base64
B=Path(__file__).resolve().parent;E=B.parent/'evidence';A=E/'freeze-scalability-hardware3263-raw-evidence';ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006');RUN=ROOT/'runs/bapo-instruct300-formal300-3269';SOURCE='b7119791b86951727cea36a1881e8a525c932494';RAW=E/'freeze-scalability-formal3269-resource-watch-production-snapshot.jsonl'
mapping={str(RUN/'resource-watch.jsonl'):RAW,str(ROOT/'tools/resource_watch.py'):A/'tools/resource_watch.py',str(ROOT/'tools/guard_journal.py'):A/'tools/guard_journal.py'}
class ReadArtifactPath(type(Path())):
 def local(self):return mapping.get(str(self),Path(str(self)))
 def read_bytes(self):return self.local().read_bytes()
 def read_text(self,*a,**kw):return self.local().read_text(*a,**kw)
 def resolve(self,*a,**kw):return self
 def is_file(self):return self.local().is_file()
 def is_symlink(self):return self.local().is_symlink()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert h(A/'tools/resource_watch.py')=='e56781fefcd42b3d97f06af824312ac69fd318eb480093a66c4194d1f0fcaf74' and h(A/'tools/guard_journal.py')=='2a714082c22675b975eff07dd060b13ca8eb9da9394d9b6f8bea605ffff46960'
m=types.ModuleType('bound_actual_helper');m.__file__=str(ROOT/'tools/guard_journal.py');exec(compile((A/'tools/guard_journal.py').read_bytes(),m.__file__,'exec'),m.__dict__);m.Path=ReadArtifactPath
summary=m.replay(ReadArtifactPath(str(RUN/'resource-watch.jsonl')),3269,str(RUN),SOURCE,ReadArtifactPath(str(ROOT/'tools/resource_watch.py')))
rows=[json.loads(x)for x in RAW.read_bytes().splitlines()];groups=[];group=None
for row in rows:
 if row.get('event')=='recovery_started':group=[row]
 elif group is not None:
  group.append(row)
  if row.get('event')=='recovery_finished':groups.append(group);group=None
assert group is None and len(groups)==2
foreign={2210497+i for i in range(8)};details=[]
for g in groups:
 stop=[x for x in g if x.get('event')=='owned_freeze_signal'];cont=[x for x in g if x.get('event')=='owned_resume_signal'];stable=[x for x in g if x.get('event')=='owned_freeze_stable'];samples=[x for x in g if x.get('event')=='recovery_sample'];epochs=lambda xs:{x['pid']:x['start_ticks']for x in xs}
 assert len(stop)==len(cont)==301 and epochs(stop)==epochs(cont)and not foreign.intersection(epochs(stop))and len(stable)==4 and all(len(x['pids'])==301 and set(x['pids'])==set(epochs(stop))and x['epochs']=={str(p):t for p,t in epochs(stop).items()}for x in stable)
 assert len(samples)==3 and all(x['pause']is False for x in samples)and all(x['reason']=='within_threshold'for x in samples)
 freeze=stable[0]['time']-g[0]['time'];assert 0<freeze<=5 and math.isclose(g[-1]['reserved_gpu_hours'],g[-1]['elapsed_seconds']*8/3600,rel_tol=1e-10)
 roles=collections.Counter(x['role']for x in stop);assert roles['phase_descendant']==273
 # Foreign GPU lifecycle remains original stable epoch both before and after queries.
 for s in samples:
  for p in foreign:
   tick=s['pid_start_ticks'][str(p)];assert tick['before']==tick['after']and tick['before']>0
 details.append({'start_time_unix':g[0]['time'],'first_stable_after_seconds':freeze,'recovery_total_seconds':g[-1]['elapsed_seconds'],'reserved_gpu_hours':g[-1]['reserved_gpu_hours'],'stop_count':301,'cont_count':301,'stable_snapshots':4,'safe_samples':3,'roles':dict(roles),'foreign_signals':0})
raw=json.loads((E/'freeze-scalability-formal3269-actual-recovery-journal-gzip-snapshot.json').read_text());metrics=[json.loads(x.split('BAPO_TRAIN_METRICS_JSON=',1)[1])for x in raw['files']['metrics']['lines']];assert [x['step']for x in metrics]==[26,27,28]
# Exhaustively reject nonfinite scalar training measurements in this snapshot.
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for y in x.values():finite(y)
 elif isinstance(x,list):
  for y in x:finite(y)
for x in metrics:finite(x)
print(json.dumps({'status':'PASS_INDEPENDENT_FORMAL3269_ACTUAL_PRODUCTION_RECOVERY_REPLAY','source_commit':SOURCE,'job_id':3269,'raw_sha256':h(RAW),'raw_bytes':RAW.stat().st_size,'journal_rows':len(rows),'guard_summary':summary,'recoveries':details,'completed_new_driver_steps':[x['step']for x in metrics],'post_second_recovery_completed_outer_in_snapshot':False,'host_SIGSTOP_not_instant_CUDA_quiescence_proof':True}))
