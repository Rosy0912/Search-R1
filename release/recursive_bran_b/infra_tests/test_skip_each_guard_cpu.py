"""Real tiny CPU Adam/native actor body; synthetic matched guard, no NCCL claim."""
import os,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from tests.test_recursive_mu_only_old_policy import MuOnlyTests
import test_tree_behavior_mu_actor_cpu as f
from search_r1.baselines.policy_safety_runtime import PolicySafetyRuntime,PolicySafetyRefusal
from search_r1.baselines.tree_guard_continue import PROTOCOL,validate_ledger
from search_r1.baselines.tree_fresh200 import validate_completed_outer

class EachGuard(unittest.TestCase):
 def execute(self,kind):
  fixture=MuOnlyTests();fixture.setUp()
  try:
   cfg=fixture.cfg;cfg.update(ppo_mini_batch_size=32,ppo_micro_batch_size=4,guard_continue_protocol=PROTOCOL)
   data=f.make_data(fixture.factory,cfg,rows=640);data.batch['ref_log_prob']=torch.full_like(data.batch['old_log_probs'],-.8)
   actor=f.Actor(cfg);actor._fresh_total_committed_optimizer_calls=0
   native=PolicySafetyRuntime.measure;refused=[False]
   def measure(runtime,phase,publish=True):
    if phase=='before_optimizer' and (kind=='all' or not refused[0]):
     refused[0]=True
     if kind=='programming':raise RuntimeError('synthetic programming error')
     raise PolicySafetyRefusal('synthetic matched finite numeric guard',matched_all_ranks=True,safety_guard=True)
    return native(runtime,phase,publish=publish)
   kl=f.ast_members(f.ROOT/'verl/trainer/ppo/core_algos.py',('kl_penalty',),dict(torch=torch))['kl_penalty']
   with patch.dict(os.environ,RECURSIVE_B_SKIP_ONLY_REFUSED_OPPORTUNITY='1'),patch.object(f.rank_audit,'enabled',lambda *a:False),patch.dict(f.actor_scope,core_algos=NS(kl_penalty=kl)),patch.object(PolicySafetyRuntime,'measure',measure):
    if kind=='programming':
     with self.assertRaisesRegex(RuntimeError,'programming'):actor.update_policy(data)
     self.assertEqual(actor._actual_committed_optimizer_calls,0);return
    m=actor.update_policy(data)
    want=0 if kind=='all' else 19
    self.assertEqual(m['actor/optimizer_steps'],want);self.assertEqual(actor._fresh_total_committed_optimizer_calls,want)
    ledger=actor._fresh_opportunity_ledger;validate_ledger(ledger,want,world_size=4)
    self.assertEqual(len(ledger),20);self.assertFalse(any(x['status']=='SKIPPED_BY_PRIOR_ROLLOUT_BRAKE' for x in ledger))
    if want:self.assertEqual(int(actor.actor_optimizer.state[actor.actor_module.weight]['step']),want)
    else:self.assertFalse(actor.actor_optimizer.state)
    metrics=dict(m,**{'actor/scheduler_step_before_outer':3,'actor/scheduler_step_after_outer':4})
    validate_completed_outer(metrics,4)
  finally:fixture.tearDown()
 def test_one_guard_skips_one_and_later19_actual_adam_writes(self):self.execute('one')
 def test_all20_guarded_zero_adam_is_a_sampled_outer_not_a_restart(self):self.execute('all')
 def test_programming_error_still_exits_instead_of_normal_skip(self):self.execute('programming')
if __name__=='__main__':unittest.main()
