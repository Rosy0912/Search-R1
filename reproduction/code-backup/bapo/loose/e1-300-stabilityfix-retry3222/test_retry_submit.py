"""Mock-only execution of actual wrapper; subprocess never reaches SSH/Slurm."""
import base64,copy,fcntl,hashlib,io,json,pathlib,subprocess,sys
import pytest
P=pathlib.Path(__file__).parent
REMOTE='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005'
SOURCE='39cdcdd1404213b897bb0de99cc52d72707682f0'
RAW=pathlib.Path(__file__).resolve().parents[1]/'evidence/formal3222-user-stop-terminal.json'
SBATCH=pathlib.Path(__file__).resolve().parents[1]/'e1-300-stabilityfix-r2-tools/formal300.sbatch'
def sha(b):return hashlib.sha256(b).hexdigest()

class Mock:
 def __init__(self,tmp,monkeypatch,case='ok'):
  self.root=tmp/'root';self.root.mkdir();(self.root/'scheduler-head').mkdir();(self.root/'logs').mkdir()
  self.d=self.root/'scheduler-retry-after3222';self.d.mkdir();self.case=case;self.submits=0;self.calls=[];self.queues=0
  self.raw=json.loads(RAW.read_text());self.raw['after']['stdout']=self.raw['after']['stdout'].replace(REMOTE,str(self.root))
  self.archived=json.dumps(self.raw)+'\n'
  sb=SBATCH.read_bytes();(self.root/'scheduler-head/formal300.sbatch').write_bytes(sb)
  (self.root/'scheduler-head/approved-launch-hashes.json').write_text(json.dumps({'formal300.sbatch':sha(sb)}))
  (self.d/'INDEPENDENT-REVIEW.md').write_text('MOCK ONLY, not operational approval\n')
  self.approval=dict(status='PASS_INDEPENDENT_RETRY_REVIEW',source_commit=SOURCE,old_job=3222,
   root=str(self.root),retry_dir=str(self.d),script_sha256=sha((P/'retry-submit.py').read_bytes()),
   report_sha256=sha((self.d/'INDEPENDENT-REVIEW.md').read_bytes()),original_sbatch_sha256=sha(sb),
   old_stop_record_sha256=sha(self.archived.rstrip('\n').encode()))
  (self.d/'retry-approval.json').write_text(json.dumps(self.approval))
  monkeypatch.setattr(subprocess,'run',self.run)
 def run(self,args,**kw):
  self.calls.append(args);out='';err='';code=0
  if args[0]=='scontrol':
   if args[-1]=='3222':
    if self.case=='purged':code=1;err='slurm_load_jobs error: Invalid job id specified\n'
    elif self.case=='mixedstderr':code=1;err='slurm_load_jobs error: Invalid job id specified\nRPC unavailable\n'
    elif self.case=='rpc':code=1;err='RPC unavailable\n'
    elif self.case=='running':out='JobId=3222 JobState=RUNNING\n'
    elif self.case=='cancelprefix':out='JobId=3222 JobState=CANCELLED_COMPLETING\n'
    elif self.case=='wrongjob':out='JobId=9999 JobState=CANCELLED\n'
    else:out='JobId=3222 JobState=CANCELLED\n'
   else:out='JobId=3223 JobState=PENDING\n'
  elif args[0]=='squeue':
   assert '-w' not in args,'Global queue required'
   self.queues+=1
   if self.case=='queuedold':out='3222|unrelated\n'
   elif self.case=='queuedbapo':out='9999| BAPO-other-node\n'
   elif self.case=='race'and self.queues>=2:out='9999|bapo-another-run\n'
   elif self.case=='badqueue':out='unexpected\n'
   else:out='7000|foreign-retained\n'
  elif args[0]=='sbatch':
   self.submits+=1
   assert args==['sbatch','--parsable',str(self.root/'scheduler-head/formal300.sbatch')]
   if self.case=='sbatchuncertain':code=1;err='Timeout with unknown outcome'
   else:out='3223\n'
  elif args[:3]==['env','-u','LD_LIBRARY_PATH']:
   assert args[-2]=='h100-5';script=args[-1]
   if script.startswith('cat '):out=self.archived if self.case!='archivechanged'else self.archived+' '
   elif 'verify_launch.py' in script:
    if self.case=='launchfail':code=1;err='Actual launch gate refused'
    else:out='ACTUAL_LAUNCH_GATE_MOCK\n'
   elif 'verify_hardware_receipt.py' in script:
    if self.case=='hardwarefail':code=1;err='Hardware receipt refused'
    else:out='ACTUAL_HARDWARE_GATE_MOCK\n'
   elif 'OLD_OWNED_PROCESSES_EMPTY' in script:
    if self.case=='oldproc':code=1;err='Old marked process found'
    else:out='OLD_OWNED_PROCESSES_EMPTY\n'
   elif 'FRESH_DURABLE_WORKER_INTENT' in script:
    target=self.root/'logs/submission-intent-formal300-retry-after3222.json'
    if target.exists():code=1;err='Already exists'
    else:
     target.write_text('MOCK worker intent\n')
     if self.case=='workeruncertain':code=255;err='SSH observation failed after write'
     else:out='FRESH_DURABLE_WORKER_INTENT\n'
   else:raise AssertionError('Unrecognized mocked worker command '+script)
  else:raise AssertionError('No real external command permitted '+repr(args))
  return subprocess.CompletedProcess(args,code,out,err)
 def execute(self):
  # Substitute only remote ROOT with private temporary root; other actual source
  # control flow executes unchanged, and __file__ binds actual reviewed bytes.
  code=(P/'retry-submit.py').read_text().replace(REMOTE,str(self.root))
  exec(compile(code,str(P/'retry-submit.py'),'exec'),{'__file__':str(P/'retry-submit.py')})

@pytest.mark.parametrize('case',['ok','purged'])
def test_fresh_only_after_all_gates_and_exact_terminal(tmp_path,monkeypatch,case):
 m=Mock(tmp_path,monkeypatch,case);m.execute()
 assert m.submits==1 and m.queues==2
 assert (m.d/'formal300-job-id.txt').read_text()=='3223\n'
 assert (m.d/'formal300-submit-intent.json').exists()
 assert (m.root/'logs/submission-intent-formal300-retry-after3222.json').exists()
 assert (m.root/'scheduler-head/formal300.sbatch').read_bytes()==SBATCH.read_bytes()
 # A known ID returns without another external command or submission.
 n=len(m.calls)
 with pytest.raises(SystemExit)as ex:m.execute()
 assert ex.value.code==0 and len(m.calls)==n and m.submits==1

@pytest.mark.parametrize('case',['mixedstderr','rpc','running','cancelprefix','wrongjob','queuedold','queuedbapo','badqueue','race','archivechanged','launchfail','hardwarefail','oldproc'])
def test_contradiction_or_failed_gate_never_submits(tmp_path,monkeypatch,case):
 m=Mock(tmp_path,monkeypatch,case)
 with pytest.raises((AssertionError,RuntimeError)):m.execute()
 assert m.submits==0
 assert not (m.d/'formal300-submit-intent.json').exists()

@pytest.mark.parametrize('case',['workeruncertain','sbatchuncertain'])
def test_uncertain_outcome_never_repeated(tmp_path,monkeypatch,case):
 m=Mock(tmp_path,monkeypatch,case)
 with pytest.raises(RuntimeError):m.execute()
 assert (m.d/'formal300-submit-intent.json').exists()
 old=m.submits;n=len(m.calls)
 with pytest.raises(AssertionError,match='Unknown prior outcome'):m.execute()
 assert m.submits==old and len(m.calls)==n

def test_global_lock_blocks_parallel_retry(tmp_path,monkeypatch):
 m=Mock(tmp_path,monkeypatch)
 with (m.root/'scheduler-head/submit-global.lock').open('a')as held:
  fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with pytest.raises(BlockingIOError):m.execute()
 assert m.submits==0 and m.calls==[]

@pytest.mark.parametrize('key',['script_sha256','report_sha256','original_sbatch_sha256','source_commit','old_stop_record_sha256'])
def test_identity_tampering_rejected(tmp_path,monkeypatch,key):
 m=Mock(tmp_path,monkeypatch);m.approval[key]='0'*64
 (m.d/'retry-approval.json').write_text(json.dumps(m.approval))
 with pytest.raises(AssertionError):m.execute()
 assert m.submits==0

def test_original_worker_intent_not_overwritten(tmp_path,monkeypatch):
 m=Mock(tmp_path,monkeypatch);p=m.root/'logs/submission-intent-formal300-retry-after3222.json';p.write_text('existing immutable intent')
 with pytest.raises(RuntimeError):m.execute()
 assert p.read_text()=='existing immutable intent'and m.submits==0

def test_stop_raw_exact_provenance():
 assert sha(RAW.read_bytes().rstrip(b'\n'))=='1f355220b7519025fb8deb3cea3404ec85eb0087f2080f6c7496d14cf5383050'
 record=json.loads(RAW.read_text());assert record['after']['returncode']==0
 assert 'JobId=3222 JobName=bapo-instruct300-formal300' in record['after']['stdout']
 assert 'JobState=CANCELLED ' in record['after']['stdout']

@pytest.mark.parametrize('case',['ok','hashbad','extra','intent','symlink'])
def test_deploy_only_whitelisted_exact_bytes(tmp_path,monkeypatch,case):
 root=tmp_path/'root';root.mkdir();d=root/'scheduler-retry-after3222'
 report=b'MOCK staging review';script=(P/'retry-submit.py').read_bytes()
 approval=dict(status='PASS_INDEPENDENT_RETRY_REVIEW',root=str(root),retry_dir=str(d),
               script_sha256=sha(script),report_sha256=sha(report))
 values={'retry-submit.py':script,'INDEPENDENT-REVIEW.md':report,
         'retry-approval.json':json.dumps(approval).encode()}
 payload={n:{'base64':base64.b64encode(b).decode(),'sha256':sha(b)}for n,b in values.items()}
 if case=='hashbad':payload['retry-submit.py']['sha256']='0'*64
 if case=='extra':payload['foreign.txt']=payload['retry-submit.py']
 if case=='intent':d.mkdir();(d/'formal300-submit-intent.json').write_text('preserve')
 if case=='symlink':root.rename(tmp_path/'real');root.symlink_to(tmp_path/'real',target_is_directory=True)
 monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(payload)))
 # No subprocess API may execute for this deployment helper.
 def refuse(*args,**kw):raise AssertionError('Deployment must never call subprocess')
 monkeypatch.setattr(subprocess,'run',refuse)
 code=(P/'deploy.py').read_text().replace(REMOTE,str(root))
 if case=='ok':
  exec(compile(code,str(P/'deploy.py'),'exec'),{})
  assert {p.name for p in d.iterdir()}==set(values)
  for n,b in values.items():assert (d/n).read_bytes()==b
 else:
  with pytest.raises(AssertionError):exec(compile(code,str(P/'deploy.py'),'exec'),{})
  if case=='intent':assert (d/'formal300-submit-intent.json').read_text()=='preserve'
