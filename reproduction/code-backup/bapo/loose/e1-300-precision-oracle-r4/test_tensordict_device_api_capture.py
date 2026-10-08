"""Actual local API-source capture checks; not target Torch2.4/GPU proof."""
import ast,contextlib,io,json,os,unittest
from pathlib import Path
from unittest.mock import patch
class APISource(unittest.TestCase):
 def test_actual_tensordict_method_source_captured_without_cuda(self):
  text=(Path(__file__).parent/'tools/hardware.sh').read_text();script=text.split("python - <<'PYTD'",1)[1].split('\n',1)[1].split('\nPYTD',1)[0];ast.parse(script)
  repo=Path(__file__).parents[1]/'bapo-e1-300-devicefix'
  import torch
  with patch.dict(os.environ,{'SEARCH_R1_REPO':str(repo),'SOURCE':'733097feeae371b4efd8fd48940c46ff36da839b'}),contextlib.redirect_stdout(io.StringIO())as out:
   exec(compile(script,'local-api-capture','exec'),{})
  v=json.loads(out.getvalue());self.assertEqual(set(v['cuda_methods']),{'TensorDict.cuda','TensorDictBase.cuda'});self.assertFalse(torch.cuda.is_initialized())
  for record in v['cuda_methods'].values():self.assertTrue(record['signature']);self.assertTrue(record['source']);self.assertTrue(Path(record['file']).is_file())
 def test_api_original_requires_direct_hardware_and_capture_binding(self):
  base=Path(__file__).parent
  self.assertIn("'tensordict-device-api.json'",(base/'tools/verify_hardware_receipt.py').read_text())
  self.assertIn("'tensordict-device-api.json'",(base/'capture-hardware-devicefix.py').read_text())
if __name__=='__main__':unittest.main()
