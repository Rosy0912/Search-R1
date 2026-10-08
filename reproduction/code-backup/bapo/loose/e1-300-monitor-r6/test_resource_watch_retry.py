import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_watch',BASE/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
RAW=BASE.parent/'evidence/e1-300-precision-r5-hardware3171-capture/raw/runs/bapo-instruct300-hardware-3171/resource-watch.jsonl'
events=[json.loads(x) for x in RAW.read_text().splitlines()]
S=events[-1]['diagnostic']['samples'];RUN=events[-1]['run'];PID=3880656;TICK=1414362696
UUIDS={r['uuid'] for r in S[0]['allocated_gpus'].values()};JOB='3171'
ENV={'SLURM_JOB_ID':JOB,'SEARCH_R1_GPU_LEASES':json.dumps([{'uuid':v,'fd':i+20} for i,v in enumerate(sorted(UUIDS))])}
PROOF={'start_ticks':TICK,'run':RUN,'slurm_job_id':JOB,'uid':os.getuid(),'lease_uuids':sorted(UUIDS)}
def fake_stat(tick):return '7 (name) S '+ ' '.join(['1']+['0']*17+[str(tick)]+['0']*3)
class RetryTests(unittest.TestCase):
 def drive(self, samples=None, owners=True, starts=None, threshold=False, wrong_uuid=False, clear_final=True, sm_excess=False, stable_unknown=False):
  samples=samples or [S[0],S[1],S[1]]
  samples=json.loads(json.dumps(samples))
  if len(samples)>2 and clear_final:
   samples[-1]['raw_compute']='\n'.join(x for x in samples[-1]['raw_compute'].splitlines() if ', 3880656,' not in x)+'\n'
  if threshold:
   samples[0]['raw_compute']=samples[0]['raw_compute'].replace(', 2210497, 1122',', 2210497, 20000')
  if sm_excess:samples[0]['raw_pmon']=samples[0]['raw_pmon'].replace('2210497     C      -','2210497     C     16')
  if stable_unknown:
   for row in samples[0]['raw_nvml'].values():row['whole_gpu_sm_percent']=50
  if wrong_uuid:samples[0]['raw_inventory']=samples[0]['raw_inventory'].replace(next(iter(UUIDS)),'GPU-other')
  query=[];ticks=[];times=[];nvml=[];proofs=[]
  for i,s in enumerate(samples):
   query.extend([s['raw_inventory'],s['raw_compute'],s['raw_pmon']]);nvml.append(s['raw_nvml'])
   b={int(p):r['before'] for p,r in s['pid_start_ticks'].items() if r['before'] is not None};a={int(p):r['after'] for p,r in s['pid_start_ticks'].items() if r['after'] is not None}
   ticks.extend([b,a]);times.extend([s['window_start_us']*1000,s['window_end_us']*1000]);proofs.append({PID:PROOF} if i==0 and owners else {})
  if starts:ticks=starts
  with patch.dict(os.environ,ENV),patch.object(w,'query',side_effect=query),patch.object(w,'collect_nvml',side_effect=nvml),patch.object(w,'process_starts',side_effect=ticks),patch.object(w,'snapshot_owned',side_effect=proofs),patch.object(w,'proc_start_ticks',return_value=1377355135),patch.object(w,'own_pid',return_value=False),patch.object(w.time,'time_ns',side_effect=times),patch.object(w.time,'sleep') as sleep:
   try:return w.sample(RUN,[str(i) for i in range(8)]),sleep
   except w.SamplingError as e:raise e
 def test_actual_exit_ghost_then_fresh(self):
  result,sleep=self.drive();self.assertFalse(result['pause']);self.assertEqual(len(result['discarded_lifecycle_samples']),2);self.assertEqual(sleep.call_count,2)
  self.assertNotIn(PID,[r['pid'] for r in result['external_processes']]);self.assertEqual(len(result['external_processes']),8)
  self.assertIn('3880656',result['discarded_lifecycle_samples'][1]['raw_compute'])
 def test_unknown_exit_immediate(self):
  with self.assertRaisesRegex(w.SamplingError,'Uncertified'):self.drive(owners=False)
 def test_pid_reuse_rejected(self):
  b={int(p):r['before'] for p,r in S[0]['pid_start_ticks'].items()};a=dict(b);a[PID]=TICK+1
  with self.assertRaisesRegex(w.SamplingError,'reused'):self.drive(starts=[b,a])
 def test_memory_excess_not_retried(self):
  with self.assertRaisesRegex(w.SamplingError,'exceeds 15'):self.drive(threshold=True)
 def test_uuid_change(self):
  with self.assertRaisesRegex(w.SamplingError,'UUIDs changed'):self.drive(wrong_uuid=True)
 def test_exhaustion_stays_failed(self):
  with self.assertRaises(w.SamplingError) as c:self.drive(samples=[S[0],S[1],S[1],S[1]],clear_final=False)
  self.assertEqual(len(c.exception.diagnostic['samples']),4)
 def test_sm_excess_not_retried(self):
  with self.assertRaisesRegex(w.SamplingError,'exceeds 15'):self.drive(sm_excess=True)
 def test_stable_unknown_external_sm(self):
  with self.assertRaisesRegex(w.SamplingError,'unknown'):self.drive(stable_unknown=True)
 def test_ghost_uuid_association_change(self):
  samples=json.loads(json.dumps([S[0],S[1]]));old=S[0]['allocated_gpus']['0']['uuid'];new=S[0]['allocated_gpus']['1']['uuid']
  samples[1]['raw_compute']=samples[1]['raw_compute'].replace(old+', 3880656',new+', 3880656')
  with self.assertRaisesRegex(w.SamplingError,'association changed'):self.drive(samples=samples,clear_final=False)
 def test_exited_pid_reappears(self):
  starts=[]
  for i,ss in enumerate([S[0],S[1]]):
   b={int(p):r['before'] for p,r in ss['pid_start_ticks'].items() if r['before'] is not None};a={int(p):r['after'] for p,r in ss['pid_start_ticks'].items() if r['after'] is not None}
   if i==1:b[PID]=TICK+5;a[PID]=TICK+5
   starts.extend([b,a])
  with self.assertRaisesRegex(w.SamplingError,'reappeared'):self.drive(samples=[S[0],S[1]],starts=starts)
 def test_missing_uuid_leases(self):
  with patch.dict(os.environ,{'SEARCH_R1_GPU_LEASES':'[]'}),self.assertRaisesRegex(ValueError,'eight'):w.lease_uuids()
 def test_wrong_owned_uuid_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td)/str(PID);d.mkdir();(d/'stat').write_text(fake_stat(TICK));(d/'environ').write_bytes(b'\0'.join([('BAPO_RUN_ID='+RUN).encode(),b'SLURM_JOB_ID=3171',b'SEARCH_R1_GPU_LEASES=[]']))
   with patch.dict(os.environ,ENV):self.assertEqual(w.snapshot_owned({PID:TICK},RUN,UUIDS,Path(td)),{})
 def mixed_proc(self,bad_gpu=False):
  sample=S[0];ticks={int(p):r['before'] for p,r in sample['pid_start_ticks'].items()};ticks[500]=1234
  actual_ticks=w.proc_start_ticks;actual_snapshot=w.snapshot_owned;actual_owner=w.own_pid
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for pid,tick in ticks.items():
    d=root/str(pid);d.mkdir();(d/'stat').write_text(fake_stat(tick));env=b''
    if pid==PID:env=b'\0'.join([('BAPO_RUN_ID='+RUN).encode(),b'SLURM_JOB_ID=3171',('SEARCH_R1_GPU_LEASES='+('[]' if bad_gpu is True else ENV['SEARCH_R1_GPU_LEASES'])).encode()])
    if pid==PID and bad_gpu=='missing':env=b'\0'.join(x for x in env.split(b'\0') if not x.startswith(b'SEARCH_R1_GPU_LEASES='))
    if pid==PID and bad_gpu=='malformed':env=env.replace(ENV['SEARCH_R1_GPU_LEASES'].encode(),b'{badJSON')
    if pid==PID and bad_gpu=='slurm':env=env.replace(b'SLURM_JOB_ID=3171',b'SLURM_JOB_ID=wrong')
    if pid==PID and bad_gpu=='duplicate':env+=b'\0BAPO_RUN_ID=other'
    if pid==500:env=b'\0'.join([('BAPO_RUN_ID='+RUN).encode(),b'SLURM_JOB_ID=3171'])
    (d/'environ').write_bytes(env)
   with patch.dict(os.environ,ENV),patch.object(w,'process_starts',side_effect=[ticks,ticks]),patch.object(w,'proc_start_ticks',side_effect=lambda pid,proc_root=None:actual_ticks(pid,root)),patch.object(w,'snapshot_owned',side_effect=lambda starts,run,uuids:actual_snapshot(starts,run,uuids,root)),patch.object(w,'own_pid',side_effect=lambda pid,run:actual_owner(pid,run,root)),patch.object(w,'query',side_effect=[sample['raw_inventory'],sample['raw_compute'],sample['raw_pmon']]),patch.object(w,'collect_nvml',return_value=sample['raw_nvml']),patch.object(w.time,'time_ns',side_effect=[sample['window_start_us']*1000,sample['window_end_us']*1000]):
    return w.sample(RUN,[str(i) for i in range(8)])
 def test_same_run_unleased_cpu_sibling_does_not_kill_sample(self):
  result=self.mixed_proc();self.assertFalse(result['pause']);self.assertNotIn('500',result['owned_before_query']);self.assertIn(str(PID),result['owned_before_query'])
 def test_same_run_unleased_gpu_pid_fails_closed(self):
  with self.assertRaisesRegex(w.SamplingError,'lacks prequery'):self.mixed_proc(bad_gpu=True)
 def test_same_run_wrong_slurm_gpu_pid_fails_closed(self):
  with self.assertRaisesRegex(w.SamplingError,'lacks prequery'):self.mixed_proc(bad_gpu='slurm')
 def test_same_run_duplicate_identity_gpu_pid_fails_closed(self):
  with self.assertRaisesRegex(w.SamplingError,'lacks prequery'):self.mixed_proc(bad_gpu='duplicate')
 def test_same_run_missing_leases_gpu_pid_fails_closed(self):
  with self.assertRaisesRegex(w.SamplingError,'lacks prequery'):self.mixed_proc(bad_gpu='missing')
 def test_same_run_malformed_leases_gpu_pid_fails_closed(self):
  with self.assertRaisesRegex(w.SamplingError,'lacks prequery'):self.mixed_proc(bad_gpu='malformed')
 def test_same_run_wrong_uid_gpu_pid_fails_closed(self):
  with patch.object(w.os,'getuid',return_value=os.getuid()+1),self.assertRaisesRegex(w.SamplingError,'lacks prequery'):self.mixed_proc()
 def test_owned_epoch_snapshot(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td)/str(PID);d.mkdir();(d/'stat').write_text(fake_stat(TICK))
   (d/'environ').write_bytes(b'\0'.join([('BAPO_RUN_ID='+RUN).encode(),b'SLURM_JOB_ID=3171',('SEARCH_R1_GPU_LEASES='+ENV['SEARCH_R1_GPU_LEASES']).encode()]))
   with patch.dict(os.environ,ENV):self.assertEqual(w.snapshot_owned({PID:TICK},RUN,UUIDS,Path(td))[PID],PROOF)
 def test_duplicate_identity_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td)/str(PID);d.mkdir();(d/'stat').write_text(fake_stat(TICK));(d/'environ').write_bytes(b'\0'.join([('BAPO_RUN_ID='+RUN).encode(),b'BAPO_RUN_ID=other',b'SLURM_JOB_ID=3171',('SEARCH_R1_GPU_LEASES='+ENV['SEARCH_R1_GPU_LEASES']).encode()]))
   with patch.dict(os.environ,ENV):self.assertEqual(w.snapshot_owned({PID:TICK},RUN,UUIDS,Path(td)),{})
 def test_wrong_slurm_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td)/str(PID);d.mkdir();(d/'stat').write_text(fake_stat(TICK));(d/'environ').write_bytes(b'\0'.join([('BAPO_RUN_ID='+RUN).encode(),b'SLURM_JOB_ID=other',('SEARCH_R1_GPU_LEASES='+ENV['SEARCH_R1_GPU_LEASES']).encode()]))
   with patch.dict(os.environ,ENV):self.assertEqual(w.snapshot_owned({PID:TICK},RUN,UUIDS,Path(td)),{})
if __name__=='__main__':unittest.main()
