import importlib.util,json,pathlib,tempfile,unittest
spec=importlib.util.spec_from_file_location('recursive_rolling',pathlib.Path(__file__).parents[2]/'search_r1/baselines/recursive_rolling.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RollingTests(unittest.TestCase):
 def state(self,base,step,job='ours'):
  for k in ('training_state','actor','algorithm'):(base/k/f'global_step_{step}').mkdir(parents=True)
  (base/'training_state'/f'global_step_{step}'/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',identity=dict(completed_step=step,job_id=job,code_commit='c',config_sha256='h',data_identity='d',model_identity='m',actual_update_limit=200))))
 def test_keep_latest_two_protected_and_foreign(self):
  with tempfile.TemporaryDirectory() as t:
   b=pathlib.Path(t)
   for i in range(1,9):self.state(b,i,'foreign' if i==2 else 'ours')
   self.assertEqual(m.prune(b,8),[1,3,4,6])
   self.assertEqual({p.name for p in (b/'training_state').iterdir()},{'global_step_2','global_step_5','global_step_7','global_step_8'})
   self.assertEqual(len(list((b/'rolling-pruned-receipts').glob('*.json'))),4)
 def test_symlink_refused_before_any_delete(self):
  with tempfile.TemporaryDirectory() as t:
   b=pathlib.Path(t)
   for i in (1,2,3):self.state(b,i)
   (b/'actor/global_step_1/source-link').symlink_to(b/'actor/global_step_3')
   with self.assertRaises(RuntimeError):m.prune(b,3)
   self.assertTrue((b/'training_state/global_step_1').exists())
if __name__=='__main__':unittest.main()
