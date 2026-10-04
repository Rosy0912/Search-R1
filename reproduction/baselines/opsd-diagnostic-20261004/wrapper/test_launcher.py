import json,tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from launcher import lease_fds,validate_allocation,wait_allocation,loads
class Tests(unittest.TestCase):
 def request(self):return dict(version=1,job_id='99999',uid=0,node='h100-10',selected_gpu_uuid='GPU-00000000-0000-0000-0000-000000000000')
 def test_actual_request_bound(self):r=self.request();self.assertEqual(validate_allocation(r,r),r)
 def test_wrong_job_uuid_uid_node_rejected(self):
  for key,v in [('job_id','3007'),('uid',1),('node','a800'),('selected_gpu_uuid','GPU-other')]:
   a=self.request();a[key]=v
   with self.subTest(key=key),self.assertRaises(ValueError):validate_allocation(a,self.request())
 def test_fd9_preserved_and_single_native_lease(self):self.assertEqual(lease_fds([dict(uuid=self.request()['selected_gpu_uuid'],fd=3)]),(9,3))
 def test_overwritten_fd_invalid_and_multi_gpu_rejected(self):
  for rows in ([dict(uuid=self.request()['selected_gpu_uuid'],fd=9)],[dict(uuid=self.request()['selected_gpu_uuid'],fd=2)],[],[dict(uuid=self.request()['selected_gpu_uuid'],fd=3)]*2):
   with self.assertRaises(ValueError):lease_fds(rows)
 def test_allocation_external_sha_and_exact_fields(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'allocation.json';p.write_text(json.dumps(self.request()));(root/'allocation.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
   self.assertEqual(wait_allocation(root,self.request(),timeout=1)[2],self.request())
   p.write_text(json.dumps({**self.request(),'job_id':'3007'}))
   with self.assertRaises(ValueError):wait_allocation(root,self.request(),timeout=1)
 def test_allocation_times_out_without_guessing_jobid(self):
  with tempfile.TemporaryDirectory() as d:
   ticks=[0]
   def clock():return ticks[0]
   def pause(n):ticks[0]+=n
   with self.assertRaises(TimeoutError):wait_allocation(Path(d),self.request(),timeout=300,clock=clock,pause=pause)
   self.assertEqual(ticks[0],300);self.assertEqual(list(Path(d).iterdir()),[])
 def test_duplicate_allocation_json_rejected(self):
  with self.assertRaises(ValueError):loads('{"job_id":"9","job_id":"10"}')
 def test_preexisting_output_not_touched(self):
  # Waiting never writes allocation or diagnostic output; collect itself owns creation.
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'collect';p.mkdir();(p/'saved.json').write_text('original');before=(p/'saved.json').read_bytes()
   ticks=[0]
   with self.assertRaises(TimeoutError):wait_allocation(Path(d),self.request(),timeout=1,clock=lambda:ticks[0],pause=lambda n:ticks.__setitem__(0,ticks[0]+n))
   self.assertEqual((p/'saved.json').read_bytes(),before);self.assertEqual({x.name for x in p.iterdir()},{'saved.json'})


class MoreTests(unittest.TestCase):
 def test_bool_uid_or_version_rejected(self):
  r=Tests().request()
  for key,v in (('uid',False),('version',True)):
   a=dict(r);a[key]=v
   with self.assertRaises(ValueError):validate_allocation(a,r)
 def test_collect_preserves_reviewed_binding_and_no_future_probe_pin(self):
  import launcher
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'tools'/'new-reviewed-bundle';root.mkdir(parents=True);probe=root/'probe.py';probe.write_text('# current actual reviewed probe')
   source=root/'distillation.py';source.write_text('# reviewed source')
   binding=root/'binding.json';binding.write_text(json.dumps({'probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}))
   env={'OPSD_DISTRIBUTION_BUNDLE':str(root),'OPSD_DISTRIBUTION_PROBE_SHA256':hashlib.sha256(probe.read_bytes()).hexdigest(),'OPSD_DISTRIBUTION_SOURCE_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'OPSD_DISTRIBUTION_PROBE':str(probe),'OPSD_DISTRIBUTION_BINDING':str(binding),'OPSD_DISTRIBUTION_BINDING_SHA256':hashlib.sha256(binding.read_bytes()).hexdigest()}
   def file(p):
    p=Path(p)
    if not p.is_absolute() or not p.is_file():raise ValueError('missing file')
    return p
   with patch.object(launcher,'owned_file',side_effect=file),patch.object(launcher,'ROOT',root.parent.parent):
    command=launcher.collect_command(env,root/'allocation.json','a'*64,root/'collect')
    self.assertIn('collect',command);self.assertNotIn('train',command)
    with self.assertRaises(ValueError):launcher.collect_command({**env,'OPSD_DISTRIBUTION_BUNDLE':str(root.parent/'initial-distribution-7ac-96cfe2326cfe')},root/'allocation.json','a'*64,root/'collect')
    probe.write_text('# different actual bytes')
    with self.assertRaises(ValueError):launcher.collect_command(env,root/'allocation.json','a'*64,root/'collect')


class WrapperPinTests(unittest.TestCase):
 def test_external_wrapper_pin_then_exact_bootstrap_and_launcher(self):
  import launcher
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);bootstrap=root/'bootstrap.sh';bootstrap.write_text('# reviewed bootstrap')
   actual=Path(launcher.__file__).resolve();manifest=root/'manifest.json'
   data={'files':{'bootstrap.sh':{'path':str(bootstrap),'sha256':hashlib.sha256(bootstrap.read_bytes()).hexdigest()},'launcher.py':{'path':str(actual),'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()}}}
   manifest.write_text(json.dumps(data));pin=hashlib.sha256(manifest.read_bytes()).hexdigest()
   env={'OPSD_DISTRIBUTION_WRAPPER_MANIFEST':str(manifest),'OPSD_DISTRIBUTION_WRAPPER_MANIFEST_SHA256':pin,'OPSD_DISTRIBUTION_LAUNCHER':str(actual),'OPSD_DISTRIBUTION_EXECUTING_BOOTSTRAP':str(bootstrap)}
   with patch.object(launcher,'owned_file',side_effect=lambda p:Path(p)):
    launcher.verify_wrapper(env)
    bootstrap.write_text('# changed bootstrap');data['files']['bootstrap.sh']['sha256']=hashlib.sha256(bootstrap.read_bytes()).hexdigest();manifest.write_text(json.dumps(data))
    with self.assertRaises(ValueError):launcher.verify_wrapper(env)
    with self.assertRaises(ValueError):launcher.verify_wrapper({**env,'OPSD_DISTRIBUTION_WRAPPER_MANIFEST_SHA256':'bad'})
if __name__=='__main__':unittest.main()
