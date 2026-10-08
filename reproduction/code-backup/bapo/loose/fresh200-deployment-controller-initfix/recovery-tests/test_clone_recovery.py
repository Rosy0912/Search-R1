import importlib.util,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'recover-source-clone.py'
s=importlib.util.spec_from_file_location('recover',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Recovery(unittest.TestCase):
 def test_actual_bundle_exact_correct_branch(self):m.validate_bundle(P.parent/'source.bundle')
 def test_original_setup_branch_mismatch_detected(self):
  actual=m.setup_branch((P.parent/'setup-isolated.py').read_bytes())
  self.assertEqual(actual,m.FAILED_BRANCH)
  self.assertNotEqual(actual,m.BRANCH)
  self.assertEqual(m.bundle_heads(P.parent/'source.bundle'),[m.SOURCE+' refs/heads/'+m.BRANCH])
 def test_missing_branch_rejected(self):
  with patch.object(m,'bundle_heads',return_value=[m.SOURCE+' refs/heads/'+m.FAILED_BRANCH]):
   with self.assertRaisesRegex(ValueError,'real frozen branch'):m.validate_bundle(P.parent/'source.bundle')
 def test_noncanonical_parent_refused(self):
  with self.assertRaisesRegex(ValueError,'Canonical'):m.validate_existing(Path('relative'))
 def test_no_clone_until_existing_preconditions_pass(self):
  with patch.object(m,'validate_existing',side_effect=ValueError('no repo preconditions')),patch.object(m.subprocess,'run')as run:
   with self.assertRaisesRegex(ValueError,'preconditions'):m.finish(m.ROOT)
   run.assert_not_called()
if __name__=='__main__':unittest.main()
