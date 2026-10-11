"""Execute the production method directly with CPU-only state writers."""
import ast,pathlib,types,unittest

source=pathlib.Path(__file__).resolve().parents[2]/'verl/trainer/ppo/ray_trainer.py'
node=ast.parse(source.read_text());cls=next(n for n in node.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_evaluate_with_checkpoint_protection')
namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)

class CheckpointBeforeEvalTests(unittest.TestCase):
    def trainer(self, fail=False):
        calls=[];state=types.SimpleNamespace(config=types.SimpleNamespace(trainer={'training_state_enabled':True}))
        state._save_checkpoint=lambda:calls.append('pre-fullstate')
        def evaluate():
            calls.append('eval')
            if fail:raise ValueError('real evaluator fault')
            return {'EM':.5}
        state._validate=evaluate
        state._save_complete_training_state=lambda boundary_phase:calls.append(boundary_phase)
        return state,calls

    def test_eval_failure_leaves_saved_training_state(self):
        trainer,calls=self.trainer(True)
        with self.assertRaisesRegex(ValueError,'real evaluator fault'):namespace['_evaluate_with_checkpoint_protection'](trainer,True)
        self.assertEqual(calls,['pre-fullstate','eval'])

    def test_success_uses_separate_post_eval_state_not_overwrite(self):
        trainer,calls=self.trainer()
        self.assertEqual(namespace['_evaluate_with_checkpoint_protection'](trainer,True),{'EM':.5})
        self.assertEqual(calls,['pre-fullstate','eval','post-evaluation'])

    def test_non_keypoint_does_not_store_extra_states(self):
        trainer,calls=self.trainer();namespace['_evaluate_with_checkpoint_protection'](trainer,False)
        self.assertEqual(calls,['eval'])

if __name__=='__main__':unittest.main()
