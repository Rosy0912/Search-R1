import json,tempfile,unittest,sys
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).parents[1]/'control'))
from verify_checkpoint import verify
class AliasTest(unittest.TestCase):
 def setup(self,tied=True,extra=None):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name);a=root/'initial';b=root/'saved';a.mkdir();b.mkdir()
  for p in [a,b]:(p/'config.json').write_text(json.dumps({'tie_word_embeddings':tied}))
  initial={'model.embed_tokens.weight':torch.ones(3,2),'model.layers.0.self_attn.q_proj.weight':torch.ones(2,2)}
  saved={k:v.clone() for k,v in initial.items()};saved['model.layers.0.self_attn.q_proj.weight']+=.01
  if extra:saved[extra]=torch.ones(3,2)
  save_file(initial,str(a/'model.safetensors'));save_file(saved,str(b/'model.safetensors'));return a,b,saved
 def test_materialized_tied_head_requires_exact_values(self):
  a,b,saved=self.setup(extra='lm_head.weight');r=verify(a,b);self.assertTrue(r['accepted']);self.assertEqual(r['verified_aliases'],{'lm_head.weight':'model.embed_tokens.weight'})
  saved['lm_head.weight']+=.1;save_file(saved,str(b/'model.safetensors'))
  with self.assertRaisesRegex(ValueError,'differs'):verify(a,b)
 def test_untied_or_arbitrary_extra_rejected(self):
  a,b,_=self.setup(False,'lm_head.weight')
  with self.assertRaises(ValueError):verify(a,b)
  a,b,_=self.setup(True,'other.weight')
  with self.assertRaises(ValueError):verify(a,b)
 def test_missing_regular_tensor_rejected(self):
  a,b,saved=self.setup(extra='lm_head.weight');del saved['model.layers.0.self_attn.q_proj.weight'];save_file(saved,str(b/'model.safetensors'))
  with self.assertRaises(ValueError):verify(a,b)
 def test_unchanged_and_nonfinite_checkpoint_rejected(self):
  a,b,saved=self.setup();saved['model.layers.0.self_attn.q_proj.weight'].fill_(1);save_file(saved,str(b/'model.safetensors'));self.assertFalse(verify(a,b)['accepted'])
  saved['model.layers.0.self_attn.q_proj.weight'].fill_(float('nan'));save_file(saved,str(b/'model.safetensors'));self.assertFalse(verify(a,b)['accepted'])
if __name__=='__main__':unittest.main()
