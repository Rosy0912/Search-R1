"""Local labelled CPU fixtures only; no Slurm submission or actual200 acceptance."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/'final200-runtime-bundle-f11580b6a7bc'


def load(path,name,expected=None):
 raw=path.read_bytes()
 if expected is not None:assert hashlib.sha256(raw).hexdigest()==expected
 module=types.ModuleType(name);module.__file__=str(path)
 exec(compile(raw,str(path),'exec'),module.__dict__);return module

s=load(HERE/'final200_postterminal_cpu.py','candidate_postterminal_cpu')
ctl=load(BUNDLE/'final200_cpu_once.py','unchanged_sealed_old_controller',s.CONTROLLER_SHA)


def request():
 return dict(version=1,kind=s.REQUEST_KIND,source_commit=s.SOURCE,config_sha256=s.CONFIG,
 stored_training_job_id='3304',accepted5_receipt_sha256=s.RECEIPT,bundle_manifest_sha256=s.BUNDLE_SHA,
 execution_job_id=None,output=str(s.OUTPUT),explicit_decision='ONE_NEW_CPU_ONLY_ALLOCATION_AFTER_ACTUAL3304_TERMINAL',
 logs=dict(driver={},aggregate={}),terminal=dict(training_child_exit={},wrapper_exit={},scheduler_original={}),
 previous_attempt=dict(directory=str(s.OLD_OUTPUT),state='NO_PRIOR_ATTEMPT',files={}),reuse_prefix={},reviews=dict(root={},independent={}))


def process(pid=4321,job='3304',step='7'):
 return dict(pid=pid,start_ticks=10000+pid,uid=0,uids=[0]*4,cgroup=f'0::/slurm/uid_0/job_{job}/step_{step}\n',node='h100-10')


def record(path):
 st=path.stat();return dict(physical_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,uid=st.st_uid,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


class FixtureCommon:
 """Owned temp-file reader; content scope is explicitly mocked, not native model."""
 def verify_record(self,rec):
  p=Path(rec['physical_path']);assert record(p)==rec;return p
 def read_json(self,path,expected=None,root=None):
  p=Path(path);r=record(p)
  if expected is not None:assert r['sha256']==expected
  return s.strict_json(p.read_bytes()),r
 def validate_inventory(self,value):
  assert value['kind']=='CPU_SYNTHETIC_OBSERVER_CONTENT_ONLY_FIXTURE_NOT_ACTUAL200'
 def canonical(self,v):return s.canonical(v)
 def recheck(self,records):
  for r in records.values():self.verify_record(r)


class RequestTests(unittest.TestCase):
 def test_current_exact_sealed_bundle_pin(self):
  self.assertEqual(hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest(),s.BUNDLE_SHA)
  manifest=json.loads((BUNDLE/'manifest.json').read_bytes())
  for name,pin in manifest['tools'].items():self.assertEqual(hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest(),pin)
 def test_strict_request_legacy_and_future_ids(self):
  s.request_headers(request())
  for key,value in [('version',True),('execution_job_id','9999'),('source_commit','old'),('stored_training_job_id','3007'),('unknown',None)]:
   with self.subTest(key=key):
    r=request();r[key]=value
    with self.assertRaises(ValueError):s.request_headers(r)
 def test_reuse_prefix_canonical_key_order_and_no_gaps_or_full_duplicate(self):
  r=request();r['reuse_prefix']={'hf':{},'native':{},'observe':{}};s.request_headers(s.strict_json(s.canonical(r)))
  for keys in [('hf',),('native',),('observe','hf'),('observe','native','hf','archives')]:
   r=request();r['reuse_prefix']={k:{} for k in keys}
   with self.assertRaises(ValueError):s.request_headers(r)
 def test_unknown_or_successful_previous_state_rejected(self):
  for state in ['UNKNOWN','RUNNING','FOUR_STORED_CONTENT_SCOPES_PASSED']:
   r=request();r['previous_attempt']['state']=state
   with self.assertRaises(ValueError):s.request_headers(r)
 def test_fingerprint_binds_terminal_reuse_and_prior_not_self_reviews(self):
  r=request();sha=s.fingerprint(r);r['reviews']['root']={'future-review':None};self.assertEqual(s.fingerprint(r),sha)
  for key in ['logs','terminal','previous_attempt','reuse_prefix']:
   changed=request();changed[key]={'different':None};self.assertNotEqual(s.fingerprint(changed),sha)
 def test_actual_root_peer_review_fingerprint_and_nonalias(self):
  with tempfile.TemporaryDirectory() as directory:
   r=request();c=FixtureCommon();fp=s.fingerprint(r)
   for role in ['root','independent']:
    path=Path(directory)/f'{role}.json';path.write_bytes(s.canonical(dict(version=1,kind='POSTTERMINAL_CPU_FALLBACK_INPUT_REVIEW',role='ROOT' if role=='root' else 'INDEPENDENT_REVIEWER',status=s.REVIEW_STATUS,source_commit=s.SOURCE,stored_training_job_id='3304',request_fingerprint_sha256=fp,prior_attempt_state='NO_PRIOR_ATTEMPT')));r['reviews'][role]=record(path)
   s.qualify_reviews(r,c,{})
   changed=copy.deepcopy(r);changed['terminal']={'changed':None}
   with self.assertRaises(ValueError):s.qualify_reviews(changed,c,{})
   r['reviews']['independent']=r['reviews']['root']
   with self.assertRaises(ValueError):s.qualify_reviews(r,c,{})


class ResourceAndPhysicalTests(unittest.TestCase):
 def allocation(self,**updates):
  fields=dict(JobId='901',JobState='RUNNING',UserId='root(0)',NodeList='h100-10',NumNodes='1',NumCPUs='4',ReqTRES='cpu=4,mem=128G,node=1',AllocTRES='cpu=4,mem=128G,node=1',TresPerNode='(null)');fields.update(updates)
  return ' '.join(k+'='+v for k,v in fields.items())
 def test_own_cpu_allocation_positive_no_gpu(self):
  self.assertEqual(s.allocation(self.allocation(),'901')['JobId'],'901')
  s.allocation(self.allocation(ReqTRES='cpu=4,mem=128G,node=1,gres/gpu=0'),'901')
 def test_gpu_wrong_job_node_missing_tres_low_memory_rejected(self):
  cases=[dict(JobId='3304'),dict(NodeList='a800'),dict(ReqTRES='cpu=4,mem=128G,gres/gpu:h100=8'),dict(AllocTRES='cpu=4,mem=128G,gres/gpu=1'),dict(TresPerNode='gres:gpu:h100:8'),dict(ReqTRES='cpu=4,mem=127G'),dict(AllocTRES='(null)'),dict(JobState='PENDING'),dict(NumCPUs='2')]
  for values in cases:
   with self.subTest(values=values):
    with self.assertRaises(ValueError):s.allocation(self.allocation(**values),'901')
 def test_unknown_gpu_token_and_duplicate_fields_rejected(self):
  with self.assertRaises(ValueError):s.allocation(self.allocation(ReqTRES='cpu=4,mem=128G,gpu=0'),'901')
  with self.assertRaises(ValueError):s.allocation(self.allocation()+' JobId=901','901')
 def test_submission_actual_id_and_unknown_not_inferred(self):
  self.assertEqual(s.parse_submission('901;local\n',0),'901')
  for stdout,rc in [('Submitted batch job901\n',0),('901\n902\n',0),('3304\n',0),('901\n',1),('901\n',False),('',0),('0\n',0)]:
   with self.subTest(stdout=stdout,rc=rc):
    with self.assertRaises(ValueError):s.parse_submission(stdout,rc)
 def test_real_read_write_nooverwrite_symlink_and_sha(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve();p=root/'raw';s.write(p,b'actual literal bytes\n');pin=hashlib.sha256(p.read_bytes()).hexdigest();self.assertEqual(s.read(p,pin),p.read_bytes())
   with self.assertRaises(FileExistsError):s.write(p,b'overwrite')
   with self.assertRaises(ValueError):s.read(p,'0'*64)
   alias=root/'alias';alias.symlink_to(p)
   with self.assertRaises(ValueError):s.read(alias)
 def test_same_fd_replacement_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory).resolve()/'raw';p.write_bytes(b'A');real=os.read;replaced=[False]
   def replace(fd,size):
    raw=real(fd,size)
    if not replaced[0]:
     replaced[0]=True;q=p.with_name('replacement');q.write_bytes(b'A');q.replace(p)
    return raw
   with mock.patch.object(s.os,'read',side_effect=replace):
    with self.assertRaises(ValueError):s.read(p)
 def test_json_duplicate_nonfinite_and_integer_bool_guard(self):
  for raw in [b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}']:
   with self.assertRaises(ValueError):s.strict_json(raw)
  with self.assertRaises(ValueError):s.typed(False,0,'typed zero')
 def test_batch_and_srun_explicit_cpu_environment_no_old_job_or_overlap(self):
  intent=s.OUTPUT/'intent.original.json';sha='1'*64;raw=s.batch_script(intent,sha)
  self.assertIn(b'CUDA_VISIBLE_DEVICES=',raw);self.assertNotIn(b'private',raw)
  argv=s.sbatch_command(s.OUTPUT/'batch.original.sh');self.assertFalse(any(arg.startswith(('--gres','--gpus','--exclusive')) for arg in argv));self.assertIn('--mem=128G',argv);self.assertNotIn('--overlap',argv)
  argv=s.srun_command('901',intent,sha);self.assertIn('--jobid=901',argv);self.assertIn('--gres=none',argv);self.assertNotIn('--jobid=3304',argv)


class HistoricalReuseAndPipelineTests(unittest.TestCase):
 def fixture(self,root,stage='observe'):
  c=FixtureCommon();parent=process(100);execution=dict(parent,execution_job_id='3304',execution_step_id='7',stored_training_job_id='3304',python=str(s.PYTHON));child=process(101)
  report=dict(version=1,kind='CPU_SYNTHETIC_OBSERVER_CONTENT_ONLY_FIXTURE_NOT_ACTUAL200',source_commit=s.SOURCE,config_sha256=s.CONFIG,job_id='3304',run_root=str(ctl.RUN),checkpoint_step=200,common_source_sha256=s.COMMON_SHA,accepted5=dict(receipt_sha256=s.RECEIPT,public_key_sha256=ctl.PUBLIC),observer_source_sha256=ctl.PINS['observe_final200.py'],execution_identity=dict(execution,pid=child['pid'],start_ticks=child['start_ticks'],cgroup=child['cgroup']),scope='CPU_FIXTURE_NO_ACTUAL200_CLAIM')
  entries=dict(report=report,child_start=dict(child=child,parent=parent,child_pid=child['pid'],parent_pid=parent['pid'],capture_errors=[]),exit=dict(returncode=0,waited=True,waited_by_pid=parent['pid'],child_pid=child['pid'],capture_errors=[]),cpu_step_start=dict(execution_identity=execution))
  records={}
  for role,value in entries.items():
   p=root/(s.FILENAMES[stage] if role=='report' else f'{stage}-{role}.json');p.write_bytes(s.canonical(value));records[role]=record(p)
  return c,records,entries
 def test_reuse_actual_wait_source_identity_without_old_pid_liveness(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve();c,entry,_=self.fixture(root)
   with mock.patch.object(s,'OLD_OUTPUT',root),mock.patch.object(ctl,'process',side_effect=AssertionError('Old process must not be queried')):
    got=s.reuse_report('observe',entry,c,ctl,{},{});self.assertEqual(got,entry['report'])
 def test_reuse_rejects_nonzero_missing_wait_wrong_source_and_pid(self):
  edits=[('exit','returncode',1),('exit','waited',False),('exit','child_pid',999),('report','observer_source_sha256','0'*64)]
  for role,key,value in edits:
   with tempfile.TemporaryDirectory() as directory:
    root=Path(directory).resolve();c,entry,raw=self.fixture(root);raw[role][key]=value;p=Path(entry[role]['physical_path']);p.write_bytes(s.canonical(raw[role]));entry[role]=record(p)
    with mock.patch.object(s,'OLD_OUTPUT',root):
     with self.assertRaises(ValueError):s.reuse_report('observe',entry,c,ctl,{},{})
 def test_historical_job_step_pair_not_cross_paths_bool_uid_wrong_node(self):
  valid=process();s.historical_cpu(valid,'3304','7')
  for patch in [dict(cgroup='0::/slurm/job_3304/step_8\n1::/slurm/job_901/step_7\n'),dict(cgroup='0::/slurm/job_33040/step_7\n'),dict(uid=False),dict(uids=[False]*4),dict(node='other'),dict(start_ticks=True)]:
   with self.subTest(patch=patch):
    value=dict(valid,**patch)
    with self.assertRaises(ValueError):s.historical_cpu(value,'3304','7')
 def test_reused_parent_identity_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve();c,entry,raw=self.fixture(root);raw['child_start']['parent']['cgroup']='0::/slurm/job_3304/step_8\n';p=Path(entry['child_start']['physical_path']);p.write_bytes(s.canonical(raw['child_start']));entry['child_start']=record(p)
   with mock.patch.object(s,'OLD_OUTPUT',root):
    with self.assertRaises(ValueError):s.reuse_report('observe',entry,c,ctl,{},{})
 def test_true_subprocess_exit_zero_nonzero_signal_and_metadata_failure_waited(self):
  for code,expected in [('pass',0),('raise SystemExit(6)',6),('import os,signal;os.kill(os.getpid(),signal.SIGTERM)',-15)]:
   with tempfile.TemporaryDirectory() as directory:
    root=Path(directory).resolve();identity=lambda pid:process(pid)
    result,_=ctl.wait_child([sys.executable,'-c',code],root,'child',identity)
    self.assertEqual(result['returncode'],expected);self.assertIs(result['waited'],True);self.assertEqual(json.loads((root/'child-exit.original.json').read_text())['returncode'],expected)
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve()
   def unavailable(pid):raise RuntimeError('metadata unavailable fixture')
   result,_=ctl.wait_child([sys.executable,'-c','raise SystemExit(5)'],root,'child',unavailable)
   self.assertEqual(result['returncode'],5);self.assertTrue(result['waited']);self.assertEqual(len(result['capture_errors']),2)
 def test_partial_prefix_skips_finished_helpers_preserves_old_files(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve();old=root/'old.json';old.write_bytes(b'{"actual_old":"unchanged"}\n');prior=record(old);execution=dict(process(100,job='901'),execution_job_id='901',execution_step_id='7')
   common=FixtureCommon();real=ctl.wait_child;seen=[]
   def true_wait(argv,out,prefix,identity_reader,cwd):
    seen.append(prefix);return real(argv,out,prefix,lambda pid:process(pid,'901','7'),None)
   def commands(stage,records):
    path=s.OUTPUT/s.FILENAMES[stage];code='from pathlib import Path;Path('+repr(str(path))+').write_text("{}")';return [sys.executable,'-c',code]
   with mock.patch.object(s,'OUTPUT',root/'new'),mock.patch.object(ctl,'wait_child',side_effect=true_wait),mock.patch.object(ctl,'qualify',return_value=None):
    s.OUTPUT.mkdir();rc=s.pipeline(ctl,common,execution,dict(observe=prior,native=prior,hf=prior),commands)
    self.assertEqual(rc,0);self.assertEqual(seen,['archives']);self.assertEqual(old.read_bytes(),b'{"actual_old":"unchanged"}\n')
    report=json.loads((s.OUTPUT/'pipeline-result.original.json').read_text());self.assertIsNone(report['actual_final_completion']);self.assertEqual(report['stages'][0]['status'],'REUSED_ORIGINAL_FINISHED_SCOPE_NOT_NEW_EXECUTION')
 def test_failed_new_stage_stops_real_wait_and_never_creates_hf_or_archives(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve();seen=[];real=ctl.wait_child
   def wait(argv,out,prefix,identity_reader,cwd):seen.append(prefix);return real(argv,out,prefix,lambda pid:process(pid,'901','7'),None)
   with mock.patch.object(s,'OUTPUT',root),mock.patch.object(ctl,'wait_child',side_effect=wait):
    rc=s.pipeline(ctl,FixtureCommon(),dict(process(100,job='901'),execution_job_id='901',execution_step_id='7'),{},lambda stage,records:[sys.executable,'-c','raise SystemExit(4)'])
    self.assertEqual(rc,1);self.assertEqual(seen,['observe']);self.assertEqual(json.loads((root/'observe-exit.original.json').read_text())['returncode'],4);self.assertIsNone(json.loads((root/'pipeline-result.original.json').read_text())['actual_final_completion'])
 def test_actual_submission_checks_raw_sha_true_exit_and_env(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve();stdout=b'901\n';stderr=b'';exit_=dict(returncode=0,waited=True,capture_errors=[])
   for name,raw in [('sbatch-stdout.original',stdout),('sbatch-stderr.original',stderr),('sbatch-exit.original.json',s.canonical(exit_))]: (root/name).write_bytes(raw)
   result=dict(status='ACTUAL_ID_BOUND_NOT_CONTENT_ACCEPTANCE',execution_job_id='901',wait=exit_,error=None,stdout_sha256=hashlib.sha256(stdout).hexdigest(),stderr_sha256=hashlib.sha256(stderr).hexdigest());(root/'submission-result.original.json').write_bytes(s.canonical(result))
   with mock.patch.object(s,'OUTPUT',root),mock.patch.dict(os.environ,{'SLURM_JOB_ID':'901'}):
    self.assertEqual(s.actual_submission(0)['execution_job_id'],'901')
    (root/'sbatch-stdout.original').write_bytes(b'902\n')
    with self.assertRaises(ValueError):s.actual_submission(0)
 def test_actual_submission_missing_outcome_and_unknown_never_inferred(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory).resolve()
   with mock.patch.object(s,'OUTPUT',root):
    with self.assertRaises(ValueError):s.actual_submission(0)
    (root/'submission-result.original.json').write_bytes(s.canonical(dict(status='FAILED_OR_UNKNOWN_NO_AUTOMATIC_RETRY',execution_job_id=None)))
    with self.assertRaises(ValueError):s.actual_submission(0)


if __name__=='__main__':unittest.main()
