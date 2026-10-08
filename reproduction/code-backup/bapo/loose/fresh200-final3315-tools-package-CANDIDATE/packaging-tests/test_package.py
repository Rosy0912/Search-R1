import hashlib,importlib.util,io,json,os,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('installer',B/'install-and-test-worker.CANDIDATE.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class Package(unittest.TestCase):
 def test_exact_payload_and_runtime_bytes(self):
  files,m=v.package(B/'final-readonly-tools.tar');self.assertEqual(len(files),11);self.assertEqual(m['target_fixture_tests'],36)
  origins={'terminal_resource.py':B.parent/'fresh200-terminal-resource-3315-v2/terminal_resource.py'}
  for n in ('final-evidence.py','milestones.py','final_json.py','first5_helpers.py','history_validator.py'):origins[n]=B.parent/'fresh200-final3315-readonly-v3-CANDIDATE'/n
  for n,p in origins.items():self.assertEqual(hashlib.sha256(files[n]).hexdigest(),hashlib.sha256(p.read_bytes()).hexdigest())
 def test_tampered_archive_and_hostile_member_refused(self):
  files,_=v.package(B/'final-readonly-tools.tar')
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'bad.tar';p.write_bytes((B/'final-readonly-tools.tar').read_bytes()+b'bad')
   with self.assertRaises(ValueError):v.package(p)
   for variant in ('duplicate','absolute','symlink','tamper'):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w')as t:
     for n,raw in files.items():
      name=('/escape'if variant=='absolute'and n=='milestones.py'else n);i=tarfile.TarInfo(name);i.size=len(raw)
      if variant=='symlink'and n=='milestones.py':i.type=tarfile.SYMTYPE;i.linkname='/escape';i.size=0;t.addfile(i);continue
      if variant=='tamper'and n=='milestones.py':raw=b'x'*len(raw)
      t.addfile(i,io.BytesIO(raw))
     if variant=='duplicate':i=tarfile.TarInfo('milestones.py');i.size=1;t.addfile(i,io.BytesIO(b'x'))
    p.write_bytes(stream.getvalue())
    with patch.object(v,'TAR_SHA',hashlib.sha256(stream.getvalue()).hexdigest()):
     with self.assertRaises(ValueError):v.package(p)
 def test_receipt_source_nonce_job_and_eight_uuid_binding(self):
  r=dict(phase='formal200',source=v.SOURCE,nonce=v.NONCE,job_id='3315',comment='BAPO_FRESH200:formal200:'+v.SOURCE+':'+v.NONCE);w=dict(r,run=str(v.RUN),host='h100-5',pid=123);e=dict(run=str(v.RUN),slurm_job_id='3315',allocated_gpus={str(i):{'uuid':'GPU-'+str(i)}for i in range(8)})
  self.assertEqual(len(v.identity(r,w,e)),8)
  for key,value in [('source','old'),('nonce','old'),('job_id',3315),('comment','E1')]:
   with self.assertRaises(ValueError):v.identity(dict(r,**{key:value}),w,e)
  for change in ({'slurm_job_id':'old'},{'allocated_gpus':{'0':{'uuid':'GPU-0'}}},{'allocated_gpus':{str(i):{'uuid':'GPU-same'}for i in range(8)}}):
   with self.assertRaises(ValueError):v.identity(r,w,dict(e,**change))
 def test_review_exact_binding_not_native_pass(self):
  files,m=v.package(B/'final-readonly-tools.tar');h=v.sha(B/'install-and-test-worker.CANDIDATE.py');r=dict(status='PASS_CODE_ONLY_FINAL3315_READONLY_PACKAGE_REVIEW',mustfix=[],native_acceptance=False,source_commit=v.SOURCE,job_id=3315,package_manifest_sha256=v.MANIFEST_SHA,package_sha256=v.TAR_SHA,installer_sha256=h,files=m['files']);v.review_document(r,files,m,h)
  for key,value in [('status','PASS_NATIVE'),('native_acceptance',True),('job_id',True),('installer_sha256','0'*64),('mustfix',['fault']),('files',{})]:
   with self.assertRaises(ValueError):v.review_document(dict(r,**{key:value}),files,m,h)
 def test_install_exclusive_old_destination_and_no_collector_command(self):
  with tempfile.TemporaryDirectory()as d:
   dest=Path(d)/'new';v.exclusive_files(dest,{'tests/x.py':b'x'});self.assertEqual((dest/'tests/x.py').read_bytes(),b'x')
   with self.assertRaises(ValueError):v.exclusive_files(dest,{'tests/x.py':b'changed'})
   self.assertEqual((dest/'tests/x.py').read_bytes(),b'x')
  source=(B/'install-and-test-worker.CANDIDATE.py').read_text();self.assertNotIn('--execute-terminal',source);self.assertIn("cwd=ROOT/'repo'",source);self.assertIn("str(ROOT/'env/bin/python'),'-B','-m','unittest'",source)
if __name__=='__main__':unittest.main()
