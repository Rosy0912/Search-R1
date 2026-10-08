import base64,copy,hashlib,importlib.util,json,pathlib,tempfile,types,unittest
from unittest.mock import patch
D=pathlib.Path(__file__).parent/'deployment-tools'
spec=importlib.util.spec_from_file_location('actual_proof',D/'identity_evidence_validator.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name).resolve();self.job=17;self.run=self.root/'runs/bapo-instruct300-hardware-17';self.bound={};self.uuids=['GPU-'+str(i)for i in range(8)];self.source='s';self.exe=dict(resolved_path='/python',device=1,inode=2);self.anchor=dict(source_commit='s',executable=self.exe);self.title='ray::WorkerDict.actor_rollout_save_training_state';self.mockw=types.SimpleNamespace(load_runtime_anchor=lambda *a,**k:self.anchor,RAY_APPROVED_TITLES={self.title,'ray::IDLE'})
  self.write(self.root/'tools/resource_watch.py','watch');self.write(self.root/'tools/actual-title-dependency-preflight.py','sampler');self.write(self.run/'production-ray-identity.log','actual');self.write(self.run/'slurm-terminal.txt','UserId=user(1003) JobId=17')
  self.write(self.root/'logs/runtime-executable.json',dict(source_commit='s',root=str(self.root),executable_sha256='binary'))
  self.write(self.root/'logs/tools-manifest.json',dict(source_commit='s',files={n:self.bound[str(self.root/'tools'/n)]for n in ('resource_watch.py','actual-title-dependency-preflight.py')}))
  self.expected=dict(BAPO_RUN_ID=str(self.run),SLURM_JOB_ID='17',SPT_NOENV='1',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':u}for u in self.uuids]));marker=dict(values=self.expected,original_selected_environ_base64=base64.b64encode(b'\0'.join((k+'='+val).encode()for k,val in self.expected.items())+b'\0').decode());rows=[]
  for rank in range(8):
   pid=100+rank;epoch=500+rank;wp=self.run/'recovery-witness'/f'rank_{rank}.json';vp=self.run/'recovery-native-v3-roundtrip'/f'rank_{rank}.json';self.write(wp,dict(status='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK',rank=rank,world_size=8,new_pid=pid,process=dict(start_ticks=epoch)));self.write(vp,dict(status='EXACT_FULL_STATE_RELOAD_OK',rank=rank,world_size=8,pid=pid,roundtrip=True,native_state_version=3))
   def snap(title,t):
    raw=title.encode()+b'\0'*(100-len(title));return dict(pid=pid,start_ticks=epoch,uid=1003,executable=dict(self.exe,sha256='binary'),process_markers=copy.deepcopy(marker),command=raw.decode().split('\0'),cmdline_base64=base64.b64encode(raw).decode(),captured_monotonic_ns=t)
   active=snap(self.title,100);post=snap('ray::IDLE',200);rows.append(dict(rank=rank,witness_path=str(wp),witness_sha256=self.bound[str(wp)],v3_report_path=str(vp),v3_report_sha256=self.bound[str(vp)],v3_report_observed_monotonic_ns=150,active=active,post_v3=post,snapshots=[active,post]))
  self.record=dict(status='COMPLETE_ACTUAL_PRODUCTION_RAY8_PROC_IDENTITY_NOT_GPU_ACCEPTANCE',root=str(self.root),run_root=str(self.run),source_commit='s',job_id=17,tool_manifest_sha256=self.bound[str(self.root/'logs/tools-manifest.json')],sampler_sha256=self.bound[str(self.root/'tools/actual-title-dependency-preflight.py')],runtime_anchor=self.anchor,collector_uid=1003,expected_process_markers=self.expected,signals_sent=0,extra_ray_calls=0,extra_training_updates=0,ranks=rows)
 def tearDown(self):self.tmp.cleanup()
 def write(self,p,value):
  p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value if type(value)is str else json.dumps(value));self.bound[str(p)]=v.sha(p)
 def runproof(self,record=None):
  self.write(self.run/'production-ray-identity/result.json',record or self.record)
  with patch.object(v,'load_module',return_value=self.mockw):return v.validate_production_ray_identity(self.run,self.root,'s',17,self.bound,self.uuids)
 def reject(self,mutate):
  r=copy.deepcopy(self.record);mutate(r)
  with self.assertRaises((ValueError,KeyError)):self.runproof(r)
 def test_positive_actual_parser(self):self.assertEqual(self.runproof()['actual_ray_workers'],8)
 def test_missing_rank(self):self.reject(lambda r:r['ranks'].pop())
 def test_duplicate_pid(self):self.reject(lambda r:r['ranks'][1]['active'].update(pid=100))
 def test_epoch_reuse(self):self.reject(lambda r:r['ranks'][0]['post_v3'].update(start_ticks=501))
 def test_clobbered_original_env(self):self.reject(lambda r:r['ranks'][0]['post_v3']['process_markers'].update(original_selected_environ_base64='AA=='))
 def test_spurious_executable(self):self.reject(lambda r:r['ranks'][0]['active']['executable'].update(inode=99))
 def test_idle_not_active(self):
  def mutate(r):
   s=r['ranks'][0]['active'];raw=b'ray::IDLE'+b'\0'*91;s.update(command=raw.decode().split('\0'),cmdline_base64=base64.b64encode(raw).decode())
  self.reject(mutate)
 def test_post_before_active(self):self.reject(lambda r:r['ranks'][0]['post_v3'].update(captured_monotonic_ns=99))
 def test_raw_capacity_change(self):
  def mutate(r):
   s=r['ranks'][0]['post_v3'];raw=b'ray::IDLE'+b'\0'*90;s.update(command=raw.decode().split('\0'),cmdline_base64=base64.b64encode(raw).decode())
  self.reject(mutate)
 def test_report_hash_rebinding(self):self.reject(lambda r:r['ranks'][0].update(v3_report_sha256='bad'))
 def test_report_after_post(self):self.reject(lambda r:r['ranks'][0].update(v3_report_observed_monotonic_ns=201))
 def test_missing_post(self):self.reject(lambda r:r['ranks'][0].pop('post_v3'))
 def test_uid_foreign(self):self.reject(lambda r:r.update(collector_uid=2000))
 def test_signal_boolean_zero(self):self.reject(lambda r:r.update(signals_sent=False))
 def test_new_source(self):self.reject(lambda r:r.update(source_commit='other'))
 def test_unallocated_uuid(self):self.reject(lambda r:r['expected_process_markers'].update(SEARCH_R1_GPU_LEASES='[]'))
 def test_no_synthetic_mode(self):self.reject(lambda r:r.update(status='PASS_ACTUAL_CPU25_TITLES_NOT_GPU_OR_PERMISSION'))
 def test_stage_order_and_budget(self):
  s=(D/'stage-gpu-inner.sh').read_text();self.assertLess(s.index('--production-ray-watch'),s.index('/usr/bin/time -v python -m search_r1.baselines.main "${args[@]}" >'));self.assertIn('if [ "$PHASE" = hardware ]; then wait "$ray_identity_pid";fi',s)
if __name__=='__main__':unittest.main()
