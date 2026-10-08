import importlib.util,pathlib,tempfile,unittest
D=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('real_checker',D/'check-hardware-final.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Mapping(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();m.A=pathlib.Path(self.tmp.name).resolve();self.local=m.A/'runs/test/identity-linux-fixture';self.local.mkdir(parents=True);self.remote=m.ArtifactPath(str(m.ROOT/'runs/test/identity-linux-fixture'))
 def tearDown(self):self.tmp.cleanup()
 def test_actual_parser_walk_contract_nested_cache(self):
  (self.local/'cache/nested').mkdir(parents=True);(self.local/'cache/nested/raw.json').write_text('raw');(self.local/'result.json').write_text('result');actual=list(self.remote.rglob('*'));self.assertEqual(sorted(str(p)for p in actual),sorted(str(m.ROOT/p.relative_to(m.A))for p in self.local.rglob('*')));self.assertEqual(sum(p.is_file()for p in actual),2);self.assertEqual(sum(p.is_dir()for p in actual),2);self.assertTrue(self.remote.exists());self.assertTrue(self.remote.is_dir());self.assertEqual((self.remote/'cache/nested/raw.json').read_text(),'raw')
 def test_missing_exists_false(self):self.assertFalse((self.remote/'missing').exists())
 def test_symlink_visible_and_canonical_reject(self):
  (self.local/'raw').write_text('raw');(self.local/'linked').symlink_to(self.local/'raw');items=list(self.remote.rglob('*'));link=next(p for p in items if p.name=='linked');self.assertTrue(link.is_symlink())
  with self.assertRaises(AssertionError):link.resolve()
 def test_symlink_directory_cannot_walk(self):
  (self.local/'linked-dir').symlink_to(self.local,target_is_directory=True)
  with self.assertRaises(AssertionError):list((self.remote/'linked-dir').rglob('*'))
if __name__=='__main__':unittest.main()
