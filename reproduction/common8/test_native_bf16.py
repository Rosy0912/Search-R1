import copy,json,tempfile,unittest
from reproduction.common8.precision_contract import *
from reproduction.common8 import test_configuration
from reproduction.common8 import configuration
from search_g1.full_state_restore import execution_budget_compatible

class NativePrecisionTests(unittest.TestCase):
    def run_binding(self,run):
        run.update(precision_profile=NATIVE_BF16_PROFILE,training_world_size=8,
                   execution_backend='slurm',actor_micro_batch_per_rank=2,
                   vllm_gpu_memory_utilization=.1,behavior_probability_policy=SAMPLING_JOURNAL_POLICY)
        run['budget'].update(compute_dtype='bfloat16',rollout_dtype='bfloat16',allow_tf32=None)
        return run
    def test_actual_hydra_execution_and_unchanged_budget(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=test_configuration.ConfigurationTests().fixture(root)
            parent=copy.deepcopy(run);self.run_binding(run)
            execution_budget_compatible(run,parent)
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,cfg,_=configuration.resolved(base)
            self.assertTrue(cfg.actor_rollout_ref.model.use_remove_padding)
            self.assertEqual(cfg.actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype,'bf16')
            self.assertEqual(cfg.actor_rollout_ref.actor.fsdp_config.model_dtype,'fp32')
            self.assertEqual(cfg.actor_rollout_ref.actor.ppo_micro_batch_size,16)
            self.assertEqual(cfg.actor_rollout_ref.rollout.gpu_memory_utilization,.1)
            self.assertTrue(sampling_journal_only(run))
            run['budget']['learning_rate']=2e-6
            with self.assertRaises(AssertionError):execution_budget_compatible(run,parent)
    def test_wrong_topology_or_cache_rejected(self):
        run=self.run_binding(dict(budget={}))
        for key,value in [('training_world_size',4),('execution_backend','ssh_local'),('vllm_gpu_memory_utilization',.4)]:
            bad=copy.deepcopy(run);bad[key]=value
            with self.assertRaises(AssertionError):bf16_policy(bad)
    def test_legacy_profiles_unchanged(self):
        self.assertFalse(remove_padding({}))
        run=dict(precision_profile=BF16_PROFILE,execution_backend='ssh_local',training_world_size=4)
        self.assertTrue(bf16_policy(run));self.assertFalse(remove_padding(run))
