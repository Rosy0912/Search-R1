#!/usr/bin/env python3
"""CPU-only counterexamples for both new tied-weight evidence tools."""
import ast, importlib.util, json, tempfile, unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parent
NAMES=['bapo-smoke2-evidence-gpu64-tied','bapo-accept5-evidence-gpu64-tied','bapo-formal500-evidence-gpu64-tied']
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class TiedTests(unittest.TestCase):
 def fixture(self,root):
  base,ck=root/'base',root/'ck';base.mkdir();ck.mkdir()
  config=dict(model_type='qwen2',architectures=['Qwen2ForCausalLM'],tie_word_embeddings=True,vocab_size=1030,hidden_size=2)
  for p in [base,ck]:(p/'config.json').write_text(json.dumps(config))
  a=torch.full((1030,2),.234567,dtype=torch.float32)
  before={'model.embed_tokens.weight':a,'z.weight':torch.ones(2)}
  after={'model.embed_tokens.weight':a.to(torch.bfloat16),'lm_head.weight':a.to(torch.bfloat16).clone(),'z.weight':torch.ones(2)}
  return base,ck,before,after,config
 def write(self,b,c,x,y):
  save_file(x,str(b/'model.safetensors'));save_file(y,str(c/'model.safetensors'))
 def test_exact_alias_rounding_is_not_learning_and_real_change_is(self):
  for name in NAMES:
   with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
    b,c,x,y,_=self.fixture(Path(d));self.write(b,c,x,y);m=load(name)
    r=m.weight_witness(b,c);self.assertFalse(r['changed']);self.assertEqual(r['tied_weight_resolution']['elements_compared'],2060)
    self.assertEqual(r['tied_weight_resolution']['base_storage_key'],'model.embed_tokens.weight')
    y['model.embed_tokens.weight'][1029,1]+=.03125;y['lm_head.weight']=y['model.embed_tokens.weight'].clone();self.write(b,c,x,y)
    r=m.weight_witness(b,c);self.assertTrue(r['changed']);self.assertEqual(r['comparison'],'base_cast_to_checkpoint_dtype_then_float64')
 def test_full_alias_validation_rejects_late_difference_nonfinite_dtype_and_shape(self):
  for name in NAMES:
   for kind in ['late-difference','nan','dtype','shape']:
    with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as d:
     b,c,x,y,_=self.fixture(Path(d));y['z.weight']+=1
     if kind=='late-difference':y['lm_head.weight'][1029,1]+=.03125
     if kind=='nan':y['lm_head.weight'][1029,1]=float('nan')
     if kind=='dtype':y['lm_head.weight']=y['lm_head.weight'].float()
     if kind=='shape':y['lm_head.weight']=torch.zeros(1031,2)
     self.write(b,c,x,y)
     with self.assertRaises(ValueError):load(name).weight_witness(b,c)
 def test_arbitrary_key_missing_extra_or_invalid_config_rejected(self):
  for name in NAMES:
   for kind in ['base-missing','extra','untied','wrong-model','wrong-architecture','dimensions']:
    with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as d:
     b,c,x,y,cfg=self.fixture(Path(d))
     if kind=='base-missing':del y['z.weight']
     elif kind=='extra':y['other.weight']=torch.zeros(2)
     else:
      if kind=='untied':cfg['tie_word_embeddings']=False
      if kind=='wrong-model':cfg['model_type']='qwen3'
      if kind=='wrong-architecture':cfg['architectures']=['Other']
      if kind=='dimensions':cfg['vocab_size']=1031
      (c/'config.json').write_text(json.dumps(cfg))
     self.write(b,c,x,y)
     with self.assertRaises(ValueError):load(name).weight_witness(b,c)
 def test_later_shape_error_rejected_before_changed_witness(self):
  for name in NAMES:
   with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
    b,c,x,y,_=self.fixture(Path(d));y['z.weight']=torch.zeros(3);y['model.embed_tokens.weight']+=.03125;y['lm_head.weight']=y['model.embed_tokens.weight'].clone();self.write(b,c,x,y)
    with self.assertRaisesRegex(ValueError,'shape'):load(name).weight_witness(b,c)
 def test_actual_hf_validation_ast_rejects_loading_info_and_untied_storage(self):
  # Execute the exact new HF guard AST; no model download or GPU allocation.
  class W:
   def __init__(self,ptr):self.weight=self;self.ptr=ptr
   def data_ptr(self):return self.ptr
  class Model:
   config=type('Config',(),dict(tie_word_embeddings=True))()
   def __init__(self,tied):self.tied=tied
   def get_input_embeddings(self):return W(1)
   def get_output_embeddings(self):return W(1 if self.tied else 2)
  for name in NAMES:
   tree=ast.parse((ROOT/(name+'.py')).read_text())
   main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
   block=next(n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.hf_load')
   nodes=[n for n in block.body if (isinstance(n,ast.If) and ('loading_info' in ast.unparse(n.test) or 'tied_storage' in ast.unparse(n.test))) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tied_storage' for t in n.targets))]
   code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/(name+'.py')),'exec')
   clean=dict(missing_keys=[],unexpected_keys=[],mismatched_keys=[],error_msgs=[])
   exec(code,dict(model=Model(True),loading_info=clean))
   for key in clean:
    bad=dict(clean);bad[key]=['bad']
    with self.assertRaises(ValueError):exec(code,dict(model=Model(True),loading_info=bad))
   with self.assertRaises(ValueError):exec(code,dict(model=Model(False),loading_info=clean))
if __name__=='__main__':unittest.main()
