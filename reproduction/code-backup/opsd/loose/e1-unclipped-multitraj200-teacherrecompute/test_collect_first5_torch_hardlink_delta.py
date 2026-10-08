"""Real local hardlinks of captured ca839 source; fixture UID/path are explicitly local."""
from pathlib import Path
import types,os,tempfile,unittest,hashlib,json
H=Path(__file__).resolve().parent
p=H/'collect_first5_originals_candidate.py';m=types.ModuleType('candidate');m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
oldp=H/'collect_first5_originals_candidate-v3-c960-failed-hardlink.original.py';old=types.ModuleType('old');old.__file__=str(oldp);exec(compile(oldp.read_bytes(),str(oldp),'exec'),old.__dict__)
raw=(H.parent/'e1-author200/actual-fsdp-warning-source-20261005T235824197037Z/fully_sharded_data_parallel.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==m.NATIVE_WARNING_SHA

def fixture(module):
 ns={};exec(compile(module.reader_constants()+module.REMOTE_LIBRARY,'generated_actual_reader_fixture','exec'),ns);return ns
class Delta(unittest.TestCase):
 def with_file(self,fn):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p=root/'distributed/fsdp/fully_sharded_data_parallel.py';p.parent.mkdir(parents=True);p.write_bytes(raw);alias=root/'package-cache-source-alias';os.link(p,alias);self.assertEqual(p.stat().st_nlink,2);return fn(root,p)
 def test_old_c960_real_two_links_counterexample_preserved(self):
  def run(root,p):
   ns=fixture(old);ns['TORCH_PATH']=str(root)
   with self.assertRaises(ValueError):ns['member'](p,uid=os.getuid(),expected=m.NATIVE_WARNING_SHA)
  self.with_file(run)
 def test_new_exact_source_two_links_full_sha_verified(self):
  def run(root,p):
   ns=fixture(m);ns['TORCH_PATH']=str(root);r=ns['member'](p,uid=os.getuid(),expected=m.NATIVE_WARNING_SHA,source_hardlink_pin=m.NATIVE_WARNING_SHA)
   self.assertEqual(r['sha256'],m.NATIVE_WARNING_SHA);self.assertEqual(r['link_count'],2);self.assertEqual(r['link_count_after'],2);self.assertEqual(r['third_party_hardlink_source_pin'],m.NATIVE_WARNING_SHA)
  self.with_file(run)
 def test_exception_never_relaxes_unknown_path_pin_metadata_prefix(self):
  def run(root,p):
   ns=fixture(m);ns['TORCH_PATH']=str(root)
   for extra in ({'expected':'0'*64,'source_hardlink_pin':'0'*64},{'metadata':True,'expected':m.NATIVE_WARNING_SHA,'source_hardlink_pin':m.NATIVE_WARNING_SHA},{'prefix':True,'expected':m.NATIVE_WARNING_SHA,'source_hardlink_pin':m.NATIVE_WARNING_SHA}):
    with self.assertRaises(ValueError):ns['member'](p,uid=os.getuid(),**extra)
   alias=root/'package-cache-source-alias'
   with self.assertRaises(ValueError):ns['member'](alias,uid=os.getuid(),expected=m.NATIVE_WARNING_SHA,source_hardlink_pin=m.NATIVE_WARNING_SHA)
   with self.assertRaises(ValueError):ns['member'](p,uid=os.getuid())
  self.with_file(run)
 def test_source_hash_and_three_links_still_reject(self):
  def run(root,p):
   ns=fixture(m);ns['TORCH_PATH']=str(root);p.write_bytes(b'x'*len(raw))
   with self.assertRaises(ValueError):ns['member'](p,uid=os.getuid(),expected=m.NATIVE_WARNING_SHA,source_hardlink_pin=m.NATIVE_WARNING_SHA)
   p.write_bytes(raw);os.link(p,root/'third-alias')
   with self.assertRaises(ValueError):ns['member'](p,uid=os.getuid(),expected=m.NATIVE_WARNING_SHA,source_hardlink_pin=m.NATIVE_WARNING_SHA)
  self.with_file(run)
 def test_actual_failure_and_stat_originals_match_only_proven_exception(self):
  failed=json.loads((H/'job3304-first5-complete-originals-v1/transport-exit.original.json').read_bytes());self.assertEqual(failed['returncode'],1)
  actual=json.loads((H/'job3304-installed-torch-fsdp-stat-original/stdout.original').read_bytes());self.assertEqual(actual['lstat'],actual['stat']);self.assertEqual(actual['sha256'],m.NATIVE_WARNING_SHA);self.assertEqual(actual['lstat']['st_nlink'],2);self.assertEqual(actual['lstat']['st_uid'],0);self.assertEqual(actual['lstat']['st_size'],len(raw));self.assertTrue(actual['is_regular']);self.assertFalse(actual['is_symlink']);self.assertEqual(actual['path'],m.TORCH_PATH+'/distributed/fsdp/fully_sharded_data_parallel.py');self.assertEqual(actual['path'],actual['resolved'])
if __name__=='__main__':unittest.main(verbosity=2)
