"""Local CPU collector fixtures; no SSH/GPU/job/acceptance."""
import base64,contextlib,copy,hashlib,io,json,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
H=Path(__file__).resolve().parent
transport=types.ModuleType('transport');transport.remote=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No actual remote allowed'))
M=types.ModuleType('collector_cpu_fixture');M.__file__=str(H/'collect_probe_originals.py')
with mock.patch.dict(sys.modules,{'transport':transport}):exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
REQUEST=M.local_request(H/M.BINDING_NAME)
REMOTE={};exec(compile(M.REMOTE_SOURCE,'original_remote_definitions_only','exec'),REMOTE)
REMOTE['PUBLIC_SHA']=M.PUBLIC_SHA;REMOTE['SCOPE']=M.SCOPE

def payload():
 source=H/('job'+REQUEST['job_id']+'-pre-gpu-actual-originals');rows={}
 for i,n in enumerate(M.REQUIRED,1):
  raw=(source/n).read_bytes();info=[1,i,0,len(raw),10,10]
  rows[n]=dict(base64=base64.b64encode(raw).decode(),sha256=M.sha(raw),bytes=len(raw),uid=0,physical=REQUEST['run_root']+'/'+n,device=1,inode=i,stat_before=info,stat_after=info,reading_scope='STABLE_FULL_BYTES_AT_OBSERVATION_NOT_TERMINAL_PROOF')
 return dict(version=1,kind=M.SCOPE,scope=M.SCOPE,job_id=REQUEST['job_id'],source_commit=REQUEST['source_commit'],run_root=REQUEST['run_root'],hostname='h100-10',uid=0,observed_utc='CPU_FIXTURE_NOT_ACTUAL_REMOTE_OBSERVATION',files=rows,missing_optional=list(M.OPTIONAL))
class Tests(unittest.TestCase):
 def decode(self,p):return M.decode_snapshot(json.dumps(p).encode(),REQUEST)
 def reject(self,change):
  p=payload();change(p)
  with self.assertRaises((ValueError,KeyError,TypeError)):self.decode(p)
 def test_current_local_positive_submission_join(self):self.assertEqual(REQUEST['source_commit'],M.SOURCE)
 def test_cpu_stub_content_only_decode(self):self.assertEqual(len(self.decode(payload())[1]),len(M.REQUIRED))
 def test_wrong_job(self):self.reject(lambda p:p.update(job_id='999999'))
 def test_wrong_source(self):self.reject(lambda p:p.update(source_commit='0'*40))
 def test_wrong_host(self):self.reject(lambda p:p.update(hostname='other-node'))
 def test_bool_uid(self):self.reject(lambda p:p.update(uid=False))
 def test_unknown_tensor_member(self):self.reject(lambda p:p['files'].update({'full_state/model.pt':copy.deepcopy(p['files']['approval-ready.json'])}))
 def test_missing_binding_member(self):self.reject(lambda p:p['files'].pop('bindings/source.json'))
 def test_sha_corruption(self):self.reject(lambda p:p['files']['approval-ready.json'].update(sha256='0'*64))
 def test_physical_alias(self):self.reject(lambda p:p['files']['approval-ready.json'].update(physical=REQUEST['run_root']+'/../other/approval-ready.json'))
 def test_bool_bytes(self):self.reject(lambda p:p['files']['approval-ready.json'].update(bytes=True))
 def test_inconsistent_stat(self):self.reject(lambda p:p['files']['approval-ready.json'].update(stat_after=[1,1,0,99,10,10]))
 def test_wrong_reading_scope(self):self.reject(lambda p:p['files']['approval-ready.json'].update(reading_scope='CLOSED_COMPLETE_LIFETIME'))
 def test_missing_optional_list_lies(self):self.reject(lambda p:p.update(missing_optional=[]))
 def test_role_body_identity_changed_even_rehashed(self):
  def mutate(p):
   r=p['files']['approval-ready.json'];j=M.unique(base64.b64decode(r['base64']));j['source_commit']='0'*40;raw=json.dumps(j).encode();r.update(base64=base64.b64encode(raw).decode(),sha256=M.sha(raw),bytes=len(raw));r['stat_before']=[1,1,0,len(raw),10,10];r['stat_after']=r['stat_before']
  self.reject(mutate)
 def test_remote_file_cpu_regular_json(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'x.json';p.write_bytes(b'{}');r=REMOTE['member'](p,uid=os.getuid());self.assertEqual(r['bytes'],2)
 def test_remote_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'x.json';p.write_bytes(b'{}');q=Path(d).resolve()/'alias.json';q.symlink_to(p)
   with self.assertRaises(ValueError):REMOTE['member'](q,uid=os.getuid())
 def test_remote_wrong_uid_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'x.json';p.write_bytes(b'{}')
   with self.assertRaises(ValueError):REMOTE['member'](p,uid=os.getuid()+1)
 def test_remote_size_equal_limit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'x.json'
   with p.open('wb') as f:f.truncate(M.LIMIT)
   with self.assertRaises(ValueError):REMOTE['member'](p,uid=os.getuid())
 def growth(self,suffix,mutate=False):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/('x'+suffix);p.write_bytes(b'{}');real=os.read;calls=[]
   def read(fd,n):
    b=real(fd,n)
    if not calls:
     if mutate:p.write_bytes(b'xx')
     else:
      with p.open('ab') as f:f.write(b'\n')
    calls.append(1);return b
   with mock.patch.object(os,'read',read):return REMOTE['member'](p,uid=os.getuid())
 def test_remote_json_growth_rejected(self):
  with self.assertRaises(ValueError):self.growth('.json')
 def test_remote_log_growth_prefix_only(self):
  r=self.growth('.log');self.assertEqual(r['bytes'],2);self.assertEqual(r['stat_after'][3],3);self.assertIn('NOT_CLOSED',r['reading_scope'])
 def test_remote_log_prefix_replacement_rejected(self):
  with self.assertRaises(ValueError):self.growth('.log',True)
 def test_write_once_nooverwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'x';M.write_once(p,b'old')
   with self.assertRaises(FileExistsError):M.write_once(p,b'new')
   self.assertEqual(p.read_bytes(),b'old')
 def test_transport_nonzero_originals_preserved_no_retry(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d).resolve()/'out';calls=[]
   def stub(*a,**k):calls.append(1);raise subprocess.CalledProcessError(255,['synthetic_transport'],output=b'partial',stderr=b'failure')
   with mock.patch.object(M,'remote',stub),mock.patch.object(sys,'argv',['collector','--binding',str(H/M.BINDING_NAME),'--output',str(out)]):
    with self.assertRaises(subprocess.CalledProcessError):M.main()
   self.assertEqual(calls,[1]);self.assertEqual((out/'transport.stdout.original').read_bytes(),b'partial');self.assertEqual(M.unique((out/'transport-exit.original.json').read_bytes())['returncode'],255);self.assertFalse((out/'inventory.json').exists())
 def test_old_terminal_requirement_absent_and_remote_no_cuda(self):
  text=Path(M.__file__).read_text();self.assertNotIn("job=='3303'",text);self.assertNotIn('JobState=FAILED',text);self.assertNotIn('rglob',M.REMOTE_SOURCE);self.assertNotIn('torch',M.REMOTE_SOURCE);self.assertNotIn('cuda',M.REMOTE_SOURCE);self.assertEqual(len(M.ALLOWED),52)
if __name__=='__main__':unittest.main(verbosity=2)
