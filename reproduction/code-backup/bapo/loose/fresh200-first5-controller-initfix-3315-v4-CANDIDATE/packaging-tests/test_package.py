import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('installer',B/'install-and-test-worker.CANDIDATE.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class Package(unittest.TestCase):
 def test_exact14_members_and_unchanged_reviewed_other_runtime(self):
  files,m=v.package(B/'first5-v4-portable.tar');self.assertEqual(len(files),14);self.assertEqual(len(m['files']),13);self.assertEqual(m['target_fixture_tests'],33)
  old=B.parent/'fresh200-first5-controller-initfix-3315-v3-CANDIDATE'
  for n in ('first5_helpers.py','first5-hf.py','publish-first5-release.CANDIDATE.py','guard_snapshot.py'):
   self.assertEqual(hashlib.sha256(files[n]).hexdigest(),hashlib.sha256((old/n).read_bytes()).hexdigest())
 def test_tar_tamper_and_existing_dest_preserved(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'bad.tar';p.write_bytes((B/'first5-v4-portable.tar').read_bytes()+b'bad')
   with self.assertRaises(ValueError):v.package(p)
   dest=Path(d)/'new';v.exclusive_files(dest,{'tests/x.py':b'x'})
   with self.assertRaises(ValueError):v.exclusive_files(dest,{'tests/x.py':b'new'})
   self.assertEqual((dest/'tests/x.py').read_bytes(),b'x')
 def test_package_review_all13_installer_binding_native_false(self):
  files,m=v.package(B/'first5-v4-portable.tar');h=v.sha(B/'install-and-test-worker.CANDIDATE.py');r=dict(status='PASS_CODE_ONLY_3315_FIRST5_V4_NATIVE_FORMAT_PACKAGE_REVIEW',mustfix=[],native_acceptance=False,source_commit=v.SOURCE,job_id=3315,package_manifest_sha256=v.MANIFEST_SHA,package_sha256=v.TAR_SHA,installer_sha256=h,files=m['files']);v.review_document(r,files,m,h)
  for key,value in [('status','PASS_NATIVE'),('native_acceptance',True),('installer_sha256','0'*64),('files',{}),('job_id',True)]:
   with self.assertRaises(ValueError):v.review_document(dict(r,**{key:value}),files,m,h)
 def test_installer_only_cpu33_lexical_env_cwd_repo(self):
  s=(B/'install-and-test-worker.CANDIDATE.py').read_text();self.assertNotIn('hf_proof(',s);self.assertNotIn('collect(',s);self.assertIn("str(ROOT/'env/bin/python'),'-B','-m','unittest'",s);self.assertIn("cwd=ROOT/'repo'",s);self.assertIn('Ran 33 tests',s)
if __name__=='__main__':unittest.main()
