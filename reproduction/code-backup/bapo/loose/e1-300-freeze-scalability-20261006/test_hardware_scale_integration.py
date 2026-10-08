"""Synthetic raw-contract tests only: no CUDA, SSH, process signals or GPU acceptance."""
import ast,copy,hashlib,json,math,pathlib,tempfile,unittest
BASE=pathlib.Path(__file__).parent
WATCH='e56781fefcd42b3d97f06af824312ac69fd318eb480093a66c4194d1f0fcaf74'
FIX='2de29718f0a30fb90d0e1b814f146ee78a50e7680cc87640807fc89e9394ab80'
def sha(p):
 if p.name=='resource_watch.py':return WATCH
 if p.name=='freeze_scalability_gpu_fixture.py':return FIX
 return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(v):
 if type(v)is float:assert math.isfinite(v)
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
tree=ast.parse((BASE/'deployment-tools/verify_hardware_receipt.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='validate_freeze_scalability')
ns=dict(pathlib=pathlib,json=json,math=math,hashlib=hashlib,sha=sha,finite=finite);exec(compile(ast.Module(body=[fn],type_ignores=[]),'contract','exec'),ns);validate=ns['validate_freeze_scalability']
class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name).resolve();self.run=self.root/'runs/bapo-instruct300-hardware-99';self.d=self.run/'freeze-scalability-gpu';self.d.mkdir(parents=True);(self.root/'tools').mkdir();self.source='b'*40;self.job=99;self.u=['GPU-'+str(i)for i in range(8)]
  for name in ('resource_watch.py','freeze_scalability_gpu_fixture.py'):(self.root/'tools'/name).write_text('mock')
  self.ident=dict(source_commit=self.source,job_id=99,run_root=str(self.run),fixture_sha256=FIX,watch_sha256=WATCH,scope='FORCED_OWN_FREEZE_FIXTURE_NOT_PRODUCTION_UNSAFE_OBSERVATION')
  created={str(pid):dict(pid=pid,ppid=1 if pid==100 else 100,state='S',start_ticks=pid+1000,uid=1)for pid in range(100,410)};cpus=list(range(101,402));ranks=list(range(402,410))
  self.put('topology.json',dict(phase_pid=100,cpu_pids=cpus,rank_pids=ranks,created=created))
  self.rows('created-so-far.jsonl',[dict(source_commit=self.source,job_id=99,run_root=str(self.run),creator_pid=100,process=v)for v in created.values()])
  cert={p:dict(v,parent_start_ticks=1 if p=='100'else 1100,run=str(self.run),slurm_job_id='99',role='phase'if p=='100'else'phase_descendant',allocated_uuid_set=self.u,gpu_leases_inherited=True,inherited_gpu_lease_uuids=self.u,command=['python',str(self.root/'tools/freeze_scalability_gpu_fixture.py')])for p,v in created.items()}
  states={p:dict(v,state='T')for p,v in created.items()};counts={str(i):3 for i in range(8)}
  self.put('paused-proof.json',dict(before=counts,paused_first=counts,paused_second=counts,states=states,held_certificates=cert,foreign_pid=999,foreign_state=dict(pid=999,start_ticks=2000,uid=1,state='S'),freeze_seconds=.8))
  self.put('resumed-proof.json',dict(before=counts,after={str(i):6 for i in range(8)},finite_allreduce=36.,all_eight_progress=True))
  for rank,pid in enumerate(ranks):
   rows=[dict(self.ident,rank=rank,world_size=8,backend='nccl',pid=pid,start_ticks=pid+1000,uid=1,cuda_device=rank,nccl_socket_ifname='lo',nccl_ib_disable='1',completed_collectives=i,allreduce_value=36.,time_unix=100+i,cuda_uuid=self.u[rank],cuda_visible_devices='0,1,2,3,4,5,6,7')for i in range(1,7)]
   self.rows(f'rank-{rank}-raw.jsonl',rows);self.put(f'rank-{rank}-finished.json',dict(rows[-1],status='ACTUAL_NCCL_RANK_FINISHED'));self.put(f'heartbeat-{rank}.json',rows[-1]);(self.d/f'rank-{rank}.log').write_text('mock real rank stderr')
  journal=[dict(self.ident,event='owned_freeze_signal',pid=int(p),start_ticks=v['start_ticks'],uid=1,lease_uuids=self.u,certificate=v)for p,v in cert.items()]
  journal+=[dict(self.ident,event='owned_freeze_stable',pids=list(range(100,410)),epochs={p:v['start_ticks']for p,v in created.items()})]
  journal+=[dict(self.ident,event='owned_resume_signal',pid=int(p),start_ticks=v['start_ticks'],signal=18)for p,v in created.items()]
  journal+=[dict(self.ident,event='fixture_finished',forced_own_freeze=True,actual_unsafe_gpu_observation=False,wall_seconds=10.,reserved_gpu_hours=10*8/3600)]
  self.rows('forced-own-freeze-journal.jsonl',journal)
  slot=dict(time=100.,run=str(self.run),slurm_job_id='99',pause=False,reason='within_threshold');raw=json.dumps(slot).encode();(self.run/'resource-watch.jsonl').write_bytes(raw+b'\n');self.put('outer-sample-selection.json',dict(scope='SCHEDULING_ONLY_NOT_RESOURCE_PERMISSION',raw_row=slot,raw_line_sha256=hashlib.sha256(raw).hexdigest(),selection_time=101.))
  self.result=dict(self.ident,status='PASS_ACTUAL_NCCL8_301CPU_FORCED_OWN_FREEZE_ONLY',actual_unsafe_gpu_observation=False,all_own_stopped=True,all_rank_counters_stable=True,all_rank_counters_resumed=True,allocated_uuids=self.u,foreign_signals=0,actual_ranks=8,cpu_processes=301,foreign_pid=999,freeze_seconds=.8,wall_seconds=10.,reserved_gpu_hours=10*8/3600)
  self.rebind()
 def tearDown(self):self.tmp.cleanup()
 def put(self,n,v):(self.d/n).write_text(json.dumps(v))
 def get(self,n):return json.loads((self.d/n).read_text())
 def rows(self,n,v):(self.d/n).write_text(''.join(json.dumps(x)+'\n'for x in v))
 def rebind(self):
  self.result['raw_files']={p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))for p in self.d.iterdir()if p.name not in ('result.json','forced-own-freeze-journal.jsonl')}
  p=self.d/'forced-own-freeze-journal.jsonl';self.result['journal']=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p));self.put('result.json',self.result);(self.run/'hardware.log').write_text('ACTUAL_FORCED_OWN_NCCL8_CPU301_FREEZE_JSON='+json.dumps(self.result)+'\n')
  self.bound={str(p):sha(p)for p in self.root.rglob('*')if p.is_file()}
 def check(self):return validate(self.run,self.root,self.source,99,self.bound,self.u)
 def test_valid_mock_contract_not_gpu_acceptance(self):self.assertEqual(self.check()['own_certificates'],310)
 def test_identity_flags_rebound_rejected(self):
  for key,val in [('source_commit','old'),('job_id',True),('actual_ranks',True),('foreign_signals',1),('actual_unsafe_gpu_observation',True),('all_own_stopped',False),('fixture_sha256','0'*64),('allocated_uuids',self.u[:-1]),('reserved_gpu_hours',1)]:
   with self.subTest(key=key):
    old=copy.deepcopy(self.result);self.result[key]=val;self.rebind()
    with self.assertRaises(AssertionError):self.check()
    self.result=old;self.rebind()
 def test_raw_math_rebound_rejected(self):
  p=self.d/'rank-0-raw.jsonl';original=p.read_text()
  for key,val in [('allreduce_value',35.),('completed_collectives',False),('backend','gloo'),('cuda_uuid','GPU-1'),('pid',999),('uid',2),('source_commit','old'),('time_unix',float('nan'))]:
   with self.subTest(key=key):
    rows=[json.loads(x)for x in original.splitlines()];rows[0][key]=val;self.rows(p.name,rows);self.rebind()
    with self.assertRaises(AssertionError):self.check()
  p.write_text(original)
 def test_cert_pause_resume_foreign_rebound_rejected(self):
  for name,mutate in [('paused-proof.json',lambda x:x['states']['101'].update(state='S')),('paused-proof.json',lambda x:x['held_certificates']['101'].update(run='/foreign')),('paused-proof.json',lambda x:x['held_certificates']['101'].update(parent_start_ticks=999)),('paused-proof.json',lambda x:x['held_certificates']['101'].update(inherited_gpu_lease_uuids=[])),('paused-proof.json',lambda x:x['paused_second'].update({'0':4})),('paused-proof.json',lambda x:x['foreign_state'].update(state='T')),('resumed-proof.json',lambda x:x['after'].update({'0':5}))]:
   with self.subTest(name=name):
    old=self.get(name);bad=copy.deepcopy(old);mutate(bad);self.put(name,bad);self.rebind()
    with self.assertRaises(AssertionError):self.check()
    self.put(name,old)
 def test_journal_foreign_and_abort_rebound_rejected(self):
  old=(self.d/'forced-own-freeze-journal.jsonl').read_text()
  for key,val in [('pid',999),('event','owned_abort_signal')]:
   rows=[json.loads(x)for x in old.splitlines()];rows[0][key]=val;self.rows('forced-own-freeze-journal.jsonl',rows);self.rebind()
   with self.assertRaises((AssertionError,KeyError)):self.check()
  (self.d/'forced-own-freeze-journal.jsonl').write_text(old)
 def test_missing_hash_and_extra_file_refused(self):
  del self.bound[str(self.d/'rank-0-raw.jsonl')]
  with self.assertRaises(AssertionError):self.check()
  self.rebind();(self.d/'unbound.txt').touch()
  with self.assertRaises(AssertionError):self.check()
 def test_outer_slot_cannot_fake_resource_permission(self):
  row=self.get('outer-sample-selection.json');row['raw_row']['pause']=True;self.put('outer-sample-selection.json',row);self.rebind()
  with self.assertRaises(AssertionError):self.check()
 def test_freeze_deadline_exact_boundary_and_bad_values(self):
  original=self.get('paused-proof.json')
  for seconds in (5,5.0):
   self.result['freeze_seconds']=seconds;paused=copy.deepcopy(original);paused['freeze_seconds']=seconds;self.put('paused-proof.json',paused);self.rebind();self.check()
  for seconds in (6,5.000001,0,-1,True,float('nan'),float('inf')):
   with self.subTest(seconds=seconds):
    self.result['freeze_seconds']=seconds;paused=copy.deepcopy(original);paused['freeze_seconds']=seconds;self.put('paused-proof.json',paused);self.rebind()
    with self.assertRaises(AssertionError):self.check()
 def test_cost_boolean_and_nonfinite_rejected(self):
  for key,value in [('wall_seconds',True),('reserved_gpu_hours',True),('wall_seconds',float('inf')),('reserved_gpu_hours',float('nan'))]:
   old=copy.deepcopy(self.result);self.result[key]=value;self.rebind()
   with self.subTest(key=key),self.assertRaises(AssertionError):self.check()
   self.result=old
 def test_duplicate_json_raw_rejected_after_rebinding(self):
  p=self.d/'paused-proof.json';text=p.read_text();p.write_text(text[:-1]+',"freeze_seconds":0.8}')
  self.rebind()
  with self.assertRaisesRegex(AssertionError,'Duplicate'):self.check()
 def test_finished_rank_missing_and_creation_alias_rejected(self):
  old=self.get('rank-0-finished.json');bad=copy.deepcopy(old);bad['completed_collectives']=5;self.put('rank-0-finished.json',bad);self.rebind()
  with self.assertRaises(AssertionError):self.check()
  self.put('rank-0-finished.json',old)
  p=self.d/'created-so-far.jsonl';rows=[json.loads(x)for x in p.read_text().splitlines()];rows.pop();self.rows(p.name,rows);self.rebind()
  with self.assertRaises(AssertionError):self.check()
 def test_hardware_call_once_and_all_original_calls_preserved(self):
  text=(BASE/'deployment-tools/hardware.sh').read_text();self.assertEqual(text.count('python "$ROOT/tools/freeze_scalability_gpu_fixture.py" --run "$RUN" --source "$SOURCE"'),1);self.assertNotIn('--output "$RUN"',text)
  for name in ('native-count-fixture.py','native-state-fixture.py','native-actor-fixture.py','numerics-gpu-fixture.py','native-stability-fixture.py','flash_preflight.py','bapo-retrieval-parity.py','stage-gpu-inner.sh'):self.assertIn(name,text)
  self.assertLess(text.index('freeze_scalability_gpu_fixture.py'),text.index('bash "$ROOT/tools/stage-gpu-inner.sh" hardware'))
if __name__=='__main__':unittest.main(verbosity=2)
