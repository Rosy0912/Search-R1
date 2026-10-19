import copy
import unittest
from .contract import overrides,validate_override_delta,patched_sources,validate_plan,PARENTS,ROOT,PARENT_SOURCE


class ContractTests(unittest.TestCase):
    def parent(self,size='3b'):
        return dict(size=size,adapter_commit=PARENT_SOURCE,root=ROOT+'/runs/'+PARENTS[size][1],
                    overrides=['trainer.total_steps=300','trainer.start_step=100',
                               'trainer.resume_mode=disable','actor_rollout_ref.actor.optim.lr=1e-6',
                               'actor_rollout_ref.rollout.n=4','data.train_batch_size=256'])
    def test_both_sizes_exact_150_and_ten_new(self):
        for size in PARENTS:
            p=self.parent(size);v=overrides(p,size)
            self.assertTrue(validate_override_delta(p['overrides'],v))
            self.assertIn('trainer.total_steps=250',v)
            self.assertIn('trainer.resume_from_path='+p['root']+'/checkpoints/global_step_240',v)
    def test_lr_change_rejected(self):
        p=self.parent();v=overrides(p,'3b');v=[x.replace('optim.lr=1e-6','optim.lr=2e-6') for x in v]
        with self.assertRaises(AssertionError):validate_override_delta(p['overrides'],v)
    def test_batch_change_rejected(self):
        p=self.parent();v=overrides(p,'3b');v=[x.replace('batch_size=256','batch_size=128') for x in v]
        with self.assertRaises(AssertionError):validate_override_delta(p['overrides'],v)
    def test_wrong_final_label_rejected(self):
        p=self.parent();v=[x.replace('total_steps=250','total_steps=150') for x in overrides(p,'3b')]
        with self.assertRaises(AssertionError):validate_override_delta(p['overrides'],v)
    def test_old_native_delete_rejected(self):
        p=self.parent();v=[x.replace('after_load=False','after_load=True') for x in overrides(p,'3b')]
        with self.assertRaises(AssertionError):validate_override_delta(p['overrides'],v)
    def test_patch_is_import_only_and_loop_unchanged(self):
        t=('from rllm.trainer.verl.agent_ppo_branch_trainer import AgentPPOTrainer\n'
           '            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker\n')
        b='original training loop\n';a,c=patched_sources(t,b)
        self.assertEqual(c,b)
        self.assertEqual(a.replace('from reproduction.branpo150.trainer import AgentPPOTrainer\n',
                                   'from rllm.trainer.verl.agent_ppo_branch_trainer import AgentPPOTrainer\n')
                         .replace('            from verl.workers.fsdp_workers import ActorRolloutRefWorker\n'
                                  '            from reproduction.branpo150.worker import AsyncActorRolloutRefWorker\n',
                                  '            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker\n'),t)


if __name__=='__main__':unittest.main()
