import importlib.util,tempfile,shutil,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'deploy-head-existing-diagnostic-parent.py'
s=importlib.util.spec_from_file_location('head_candidate',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
DIAG=P.parents[1]/'fresh200-numeric-diagnostic-3310'
class Head(unittest.TestCase):
 def test_archive(self):self.assertEqual(len(m.validate_archive(P.parent.joinpath('tools-deployment.tar').read_bytes())),22)
 def test_existing_production_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();(root/m.CHILD).mkdir();(root/'logs').mkdir()
   with self.assertRaisesRegex(ValueError,'Only exact'):m.validate_parent(root)
 def test_symlink_parent(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();(root/'alias').symlink_to(root,target_is_directory=True)
   with self.assertRaisesRegex(ValueError,'Canonical'):m.validate_parent(root/'alias')
 def test_valid_diag_and_tamper(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();c=root/m.CHILD;(c/'logs').mkdir(parents=True);shutil.copytree(DIAG/'tools',c/'tools');shutil.copyfile(DIAG/'tools-manifest.json',c/'logs/tools-manifest.json');(c/'source-expected.txt').write_text(m.SOURCE+'\n');m.validate_parent(root)
   (c/'tools/stage.sh').write_text('tampered')
   with self.assertRaisesRegex(ValueError,'bytes'):m.validate_parent(root)
 def test_foreign_child(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();(root/m.CHILD).mkdir();(root/'unowned').mkdir()
   with self.assertRaisesRegex(ValueError,'Only exact'):m.validate_parent(root)
if __name__=='__main__':unittest.main()
