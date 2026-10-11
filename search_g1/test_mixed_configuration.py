"""Compose the actual native shared-GPU recipe without creating GPU contexts."""
import json
import tempfile
import unittest
from pathlib import Path
from reproduction.common8 import configuration

class NativeSharedConfigurationTests(unittest.TestCase):
    def test_native_micro16_accum8_and_reduced_memory_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();(base/'logs').mkdir()
            model=str(base/'official-model')
            run=dict(profile='common8',run_id='cpu-config-only',completed_updates=0,
                contract_kind='fresh200_warmup57',target_updates=200,acceptance_update=5,
                policy_safety_mode='inline_first_five',actor_micro_batch_per_rank=2,
                initial_model=model,reference_model=model,
                model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
                budget=json.loads((configuration.REPO/'reproduction/common8/budget.json').read_text()),
                checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
                vllm_gpu_memory_utilization=.1,
                source_sha256={n:configuration.sha(configuration.REPO/n) for n in
                    ('reproduction/common8/recipe.json','reproduction/common8/budget.json',
                     'verl/trainer/config/ppo_trainer.yaml')})
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,config,_=configuration.resolved(base)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_micro_batch_size,16)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size,128)
            self.assertEqual((128//8)//(16//8),8)
            self.assertEqual(config.actor_rollout_ref.rollout.gpu_memory_utilization,.1)
            self.assertEqual(config.actor_rollout_ref.actor.compute_dtype,'float32')
            self.assertEqual(config.actor_rollout_ref.rollout.n_agent,5)
            self.assertEqual(config.trainer.n_gpus_per_node,8)

if __name__=='__main__':unittest.main()
