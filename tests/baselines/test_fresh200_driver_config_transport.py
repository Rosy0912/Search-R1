"""Real OmegaConf serialization/child-env regression; no Ray/GPU resources."""
import ast,base64,hashlib,json,os,pickle,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from search_r1.baselines.fresh200_contract import verify_fixed_policy_safety_artifact
MAIN=Path(__file__).resolve().parents[2]/'search_r1/baselines/main.py'
ENV=('BAPO_FRESH200_FIXED_THRESHOLD_ARTIFACT','BAPO_FRESH200_FIXED_THRESHOLD_SHA256')
class ConfigTransport(unittest.TestCase):
 def config(self):
  return OmegaConf.create({'actor_rollout_ref':{'actor':{'policy_safety':{'fixed_threshold_artifact':'${oc.env:BAPO_FRESH200_FIXED_THRESHOLD_ARTIFACT,}','fixed_threshold_artifact_sha256':'${oc.env:BAPO_FRESH200_FIXED_THRESHOLD_SHA256,}'}}}})
 def child(self,cfg):
  env=dict(os.environ)
  for n in ENV:env.pop(n,None)
  script='import pickle,base64,json;from omegaconf import OmegaConf;c=pickle.loads(base64.b64decode(__import__("sys").argv[1]));print(json.dumps(OmegaConf.to_container(c,resolve=True)))'
  out=subprocess.check_output([sys.executable,'-B','-c',script,base64.b64encode(pickle.dumps(cfg)).decode()],env=env,text=True)
  return json.loads(out)['actor_rollout_ref']['actor']['policy_safety']
 def test_original_unresolved_config_loses_post_ray_exports(self):
  with patch.dict(os.environ,{ENV[0]:'/own/threshold.json',ENV[1]:'a'*64}):
   cfg=self.config();self.assertEqual(OmegaConf.to_container(cfg,resolve=True)['actor_rollout_ref']['actor']['policy_safety']['fixed_threshold_artifact'],'/own/threshold.json')
   self.assertEqual(self.child(cfg),{'fixed_threshold_artifact':'','fixed_threshold_artifact_sha256':''})
 def test_actual_driver_resolve_statement_preserves_child_values(self):
  tree=ast.parse(MAIN.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main')
  resolve=next(x for x in fn.body if isinstance(x,ast.Expr)and isinstance(x.value,ast.Call)and ast.unparse(x.value.func)=='OmegaConf.resolve')
  remote=next(x for x in fn.body if isinstance(x,ast.Try))
  self.assertLess(fn.body.index(resolve),fn.body.index(remote))
  with patch.dict(os.environ,{ENV[0]:'/own/threshold.json',ENV[1]:'a'*64}):
   cfg=self.config();exec(compile(ast.Module(body=[resolve],type_ignores=[]),str(MAIN),'exec'),{'OmegaConf':OmegaConf,'config':cfg})
   self.assertEqual(self.child(cfg),{'fixed_threshold_artifact':'/own/threshold.json','fixed_threshold_artifact_sha256':'a'*64})
 def safety(self,path,sha):
  return dict(enabled=True,threshold_origin='user_fixed',dual_clip=3,target_old_policy_kl=.2,negative_tail_limit=.2,skip_on_policy_guard=True,kl_log_bound=30.,fixed_threshold_artifact=str(path),fixed_threshold_artifact_sha256=sha)
 def test_missing_and_wronghash_still_fail_strict_artifact_admission(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d).resolve()/'threshold.json'
   with self.assertRaisesRegex(ValueError,'Canonical immutable'):verify_fixed_policy_safety_artifact(self.safety(p,'0'*64))
   p.write_text('{}')
   with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_fixed_policy_safety_artifact(self.safety(p,'0'*64))
   link=p.with_name('alias');link.symlink_to(p)
   with self.assertRaisesRegex(ValueError,'Canonical immutable'):verify_fixed_policy_safety_artifact(self.safety(link,hashlib.sha256(p.read_bytes()).hexdigest()))
if __name__=='__main__':unittest.main()
