import tempfile
import unittest
from pathlib import Path
from .contract import recipe,patch_observation

class ContractTests(unittest.TestCase):
    def script(self):
        return '\n'.join(['algorithm.adv_estimator=tree \\', 'actor_rollout_ref.rollout.tree_search=true \\',
            'trainer.total_training_steps=180 \\', 'actor_rollout_ref.actor.optim.lr=$lr \\',
            'data.train_files=$DATA_DIR/train.parquet \\', 'actor_rollout_ref.model.path=$BASE_MODEL \\',
            'max_turns=3 \\'])
    def test_preserve_science_and_input_paths(self):
        d=dict(x.split('=',1) for x in recipe(self.script(),'/model','/data','/out','http://node1:42'))
        self.assertEqual(d['data.train_files'],'/data/train.parquet')
        self.assertEqual(d['actor_rollout_ref.model.path'],'/model')
        self.assertEqual(d['max_turns'],'3')
        self.assertEqual(d['actor_rollout_ref.actor.optim.lr'],'1e-6')
        self.assertEqual(d['trainer.total_training_steps'],'180')
    def test_reject_altered_tree_algorithm(self):
        with self.assertRaises(AssertionError):recipe(self.script().replace('=tree','=grpo'),'/m','/d','/o','http://n')
    def test_reject_altered_budget(self):
        with self.assertRaises(AssertionError):recipe(self.script().replace('=180','=181'),'/m','/d','/o','http://n')
    def test_reject_unbound_variable(self):
        with self.assertRaises(KeyError):recipe(self.script().replace('$lr','$unknown'),'/m','/d','/o','http://n')
    def test_patch_after_actual_step_only_and_no_replay(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p)
            for name,txt in [('verl/workers/actor/dp_actor.py','            self.actor_optimizer.step()\n'),('verl/trainer/ppo/ray_trainer_ts.py','                logger.log(data=metrics, step=self.global_steps)\n')]:
                q=root/name;q.parent.mkdir(parents=True);q.write_text(txt)
            observer=root/'observer.py';observer.write_text('pass\n')
            changes=patch_observation(root,observer);self.assertEqual(len(changes),2)
            self.assertIn('self.actor_optimizer.step()\n            from treegrpo_observe', (root/'verl/workers/actor/dp_actor.py').read_text())
            with self.assertRaises(AssertionError):patch_observation(root,observer)

if __name__=='__main__':unittest.main()
