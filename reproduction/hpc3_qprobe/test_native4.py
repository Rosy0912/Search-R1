import copy
import json
import tempfile
import unittest

from reproduction.common8.native_allocation import allocated_indices, validate_native_four
from reproduction.common8.precision_contract import NATIVE4_BF16_PROFILE, SAMPLING_JOURNAL_POLICY, bf16_policy
from reproduction.common8.formal_contract import formal_mathematics
from reproduction.common8.topology import actor_local_micro, actor_local_mini, local_trajectories
from search_g1.qprobe_warmup import verify_frozen
from reproduction.common8 import test_configuration, configuration


class NativeFourTests(unittest.TestCase):
    def binding(self):
        return dict(profile='common8',execution_site='hpc3',execution_backend='slurm',
                    training_world_size=4,actor_micro_batch_per_rank=4,
                    precision_profile=NATIVE4_BF16_PROFILE,vllm_gpu_memory_utilization=.1,
                    behavior_probability_policy=SAMPLING_JOURNAL_POLICY)

    def test_full_batch_and_opportunities_with_doubled_rank_work(self):
        run=self.binding();math=formal_mathematics(run)
        self.assertTrue(bf16_policy(run))
        self.assertEqual((local_trajectories(run),actor_local_micro(run),actor_local_mini(run)),(640,4,32))
        self.assertEqual((math['world_size'],math['ppo_micro_batch_size'],math['optimizer_opportunities_per_outer']), (4,16,20))

    def test_actual_hydra_resolution_keeps_global_512_g5_and_mini128(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=test_configuration.ConfigurationTests().fixture(root)
            run.update(self.binding());run['budget']['world_size']=4
            run['budget'].update(formal_mathematics(run))
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,cfg,_=configuration.resolved(base)
            self.assertEqual((cfg.trainer.n_gpus_per_node,cfg.data.train_batch_size,
                              cfg.actor_rollout_ref.rollout.n_agent,cfg.actor_rollout_ref.actor.ppo_mini_batch_size), (4,512,5,128))
            self.assertEqual(cfg.actor_rollout_ref.actor.ppo_micro_batch_size,16)
            self.assertEqual(cfg.trainer.total_training_steps,201)

    def test_bound_four_frozen_readbacks_reject_missing_rank_and_changed_optimizer(self):
        before=[dict(rank=i,parameter_sha256='p',optimizer_sha256='o',scheduler_sha256='s',
                     actual_optimizer_writes=0,completed_outer=0) for i in range(4)]
        self.assertEqual(verify_frozen(before,copy.deepcopy(before),4)['actual_world_size'],4)
        for bad in (before[:-1],before+before,[] ):
            with self.assertRaises(ValueError):verify_frozen(before,bad,4)
        bad=copy.deepcopy(before);bad[3]['optimizer_sha256']='changed'
        with self.assertRaises(ValueError):verify_frozen(before,bad,4)
        with self.assertRaises(ValueError):verify_frozen(before,before,8)

    def test_four_slurm_gpu_mapping_preserves_foreign_unassigned_gpu(self):
        indices=allocated_indices(' Nodes=n CPU_IDs=0-47 GRES=gpu:4(IDX:4-7)\n','n',4)
        self.assertEqual(indices,[4,5,6,7])
        job=dict(JobId='123',JobState='RUNNING',NodeList='n',NumNodes='1',TresPerNode='gres/gpu:4')
        rows=[[str(i),'uuid-'+str(i),'80000'] for i in range(8)]
        args=(job,'n',['4','5','6','7'],['123','456'],rows,[['uuid-0','999']],indices,['uuid-'+str(i) for i in indices])
        self.assertTrue(validate_native_four(*args))
        bad=list(args);bad[5]=[['uuid-4','999']]
        with self.assertRaises(AssertionError):validate_native_four(*bad)
        bad=list(args);bad[7]=['uuid-0','uuid-5','uuid-6','uuid-7']
        with self.assertRaises(AssertionError):validate_native_four(*bad)


if __name__=='__main__':unittest.main()
