"""Synthetic local original-file lineage faults; no payload or scheduler work."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .test_contract import fixture
from . import lineage as l

class OriginalBundle(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  _,self.n,self.m,self.plan,self.sub=fixture()
  self.raw={'driver.pt':b'synthetic-driver-only','resolved.yaml':b'synthetic-config-only'}
  self.h=lambda b:hashlib.sha256(b).hexdigest()
  config=self.h(self.raw['resolved.yaml']);driver=self.h(self.raw['driver.pt'])
  self.m['identity']['config_sha256']=config;self.m['driver']['sha256']=driver
  for w in self.m['worker_states']:w['identity']['config_sha256']=config
  self.planraw=json.dumps(self.plan,sort_keys=True).encode();self.planhash=self.h(self.planraw)
  self.sub['plan_sha256']=self.planhash;self.n['plan_sha256']=self.planhash
  self.raw.update({'plan.json':self.planraw,'submission.json':json.dumps(self.sub).encode(),
   'submission-intent.json':json.dumps(dict(plan_sha256=self.planhash)).encode(),
   'entry-once.json':json.dumps(dict(plan_sha256=self.planhash,job_id=l.JOB,gpus=['GPU-fixture-%d, model'%i for i in range(8)])).encode(),
   'manifest.json':json.dumps(self.m).encode(),'training-returned.json':json.dumps(self.n['training_returned']).encode(),
   'train-start.json':json.dumps(dict(job=l.JOB,at=1000.,config_sha256=config)).encode(),'training-exit.code':b'0\n'})
  self.n['manifest_sha256']=self.h(self.raw['manifest.json']);self.n['files']={}
  for name,data in self.raw.items():self.write(name,data)
  self.receipt=self.root/'native.json';self.save()
 def tearDown(self):self.temp.cleanup()
 def write(self,name,data):
  (self.root/name).write_bytes(data);self.n['files'][name]=dict(local_path=name,bytes=len(data),sha256=self.h(data))
 def save(self):self.receipt.write_text(json.dumps(self.n))
 def validate(self):
  self.save()
  with patch.object(l,'PLAN',self.planhash):return l.read_local_bundle(self.receipt,self.root)
 def reject(self):
  with self.assertRaises((ValueError,KeyError)):self.validate()
 def test_original_bundle_positive_all_files(self):self.assertEqual(self.validate()['repeat_count'],3)
 def test_changed_raw_sha(self):(self.root/'training-exit.code').write_bytes(b'1\n');self.reject()
 def test_self_consistent_nonzero_exit(self):self.write('training-exit.code',b'1\n');self.reject()
 def test_missing_return_file(self):self.n['files'].pop('training-returned.json');self.reject()
 def test_self_consistent_foreign_return(self):self.write('training-returned.json',json.dumps(dict(job='foreign',at=2000.)).encode());self.reject()
 def test_return_before_start(self):self.write('train-start.json',json.dumps(dict(job=l.JOB,at=3000.,config_sha256=self.m['identity']['config_sha256'])).encode());self.reject()
 def test_start_config_mismatch(self):self.write('train-start.json',json.dumps(dict(job=l.JOB,at=1000.,config_sha256='0'*64)).encode());self.reject()
 def test_safe_relative_path(self):self.n['files']['driver.pt']['local_path']='../escape';self.reject()
 def test_symlink(self):
  p=self.root/'driver.pt';p.unlink();p.symlink_to(self.root/'resolved.yaml');self.reject()
 def test_manifest_driver_sha(self):self.n['files']['driver.pt']['sha256']='0'*64;self.reject()

if __name__=='__main__':unittest.main(verbosity=2)
