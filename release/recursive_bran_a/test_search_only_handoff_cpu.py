"""Minimal immutable configuration allowance test; no CUDA/model loading."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
import ast, os, types
source=Path(__file__).resolve().parents[2]/'search_r1/baselines/recursive_a_schedule_resume.py'
node=ast.parse(source.read_text())
selected=[x for x in node.body if isinstance(x,ast.FunctionDef) and x.name in ('enabled','require','validate_entry')]
r=types.ModuleType('handoff_fixture');r.__dict__.update(os=os,Path=Path,json=json,hashlib=hashlib,HANDOFF='search_only_and_no_repeated_candidate_proof_v1',binding=lambda e:None)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),r.__dict__)
class Test(unittest.TestCase):
 def test_only_authorized_changes(self):
  old={'trainer':{'evaluation_steps':[5,50,100,150,200],'checkpoint_save_steps':[5,50,100,150,200]},
       'actor_rollout_ref':{'rollout':{},'actor':{'optim':{'lr':1e-6}}},'retriever':{'url':'http://10.120.17.101:18004/retrieve'}}
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'parent.yaml';OmegaConf.save(OmegaConf.create(old),f)
   b={'parent_config':str(f),'parent_config_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'handoff_contract':r.HANDOFF}
   new=OmegaConf.create(old);new.actor_rollout_ref.rollout.tree_action_tag_stop=True;new.actor_rollout_ref.rollout.tree_action_stop_mode='search_only'
   with patch.object(r,'binding',return_value=b),patch.object(r,'enabled',return_value=True):
    self.assertTrue(r.validate_entry(new))
    new.actor_rollout_ref.rollout.tree_action_stop_mode='both'
    with self.assertRaises(ValueError):r.validate_entry(new)
    new.actor_rollout_ref.rollout.tree_action_stop_mode='search_only';new.actor_rollout_ref.actor.optim.lr=5e-6
    with self.assertRaises(ValueError):r.validate_entry(new)
if __name__=='__main__':unittest.main()
