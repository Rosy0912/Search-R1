"""Real CPU Adam state with constructed native identity; never GPU admission."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from reproduction.common8.scheduler import actor_scheduler
from search_g1 import policy_safety as safety


class MeasurementSafetyTests(unittest.TestCase):
    def make_optimizer(self):
        module=torch.nn.Linear(2,1,dtype=torch.float64)
        adam=torch.optim.AdamW(module.parameters(),lr=1e-6)
        schedule=actor_scheduler(adam,dict(lr_warmup_steps=57,lr_warmup_steps_ratio=0.,completed_updates=0),
                                 execution_role=safety.MEASUREMENT_ROLE)
        return module,adam,schedule

    def writes(self,module,adam,count):
        for _ in range(count):
            adam.zero_grad();module(torch.ones(1,2,dtype=torch.float64)).sum().backward();adam.step()

    def test_actual_warmup_two_outer_state_and_reject_inherited_or_third(self):
        module,adam,schedule=self.make_optimizer()
        initial=[p.detach().clone() for p in module.parameters()]
        safety.verify_measurement_phase_state(adam,1,0)
        self.writes(module,adam,20)
        self.assertTrue(all(torch.equal(before,after) for before,after in zip(initial,module.parameters())))
        schedule.step();safety.verify_measurement_phase_state(adam,2,20)
        self.writes(module,adam,20)
        self.assertTrue(any(not torch.equal(before,after) for before,after in zip(initial,module.parameters())))
        for step,writes in ((1,0),(2,20),(3,40),(True,0),(1,False)):
            with self.subTest(step=step,writes=writes),self.assertRaises(ValueError):
                safety.verify_measurement_phase_state(adam,step,writes)
        _,fresh,_=self.make_optimizer()
        with self.assertRaises(ValueError):safety.verify_fresh_phase_state(fresh,1,0)

    def test_second_outer_nonfinite_moments_and_counter_drift_reject(self):
        module,adam,schedule=self.make_optimizer();self.writes(module,adam,20);schedule.step()
        original=copy.deepcopy(adam.state_dict())
        for kind in ('step','moment','lr'):
            adam.load_state_dict(copy.deepcopy(original))
            if kind=='step':next(iter(adam.state.values()))['step'].fill_(19)
            if kind=='moment':next(iter(adam.state.values()))['exp_avg'].fill_(float('nan'))
            if kind=='lr':adam.param_groups[0]['lr']=1e-6
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                safety.verify_measurement_phase_state(adam,2,20)

    def test_measurement_observes_tail_but_retains_kl_and_numeric_rejection(self):
        args=dict(c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=None,
                  execution_role=safety.MEASUREMENT_ROLE)
        a=torch.tensor([-1.,1.]);m=torch.ones(2);d=torch.tensor([2.,0.])
        report=safety.monitor(d,a,m,torch.tensor([.001,.999]),**args)
        self.assertTrue(report['allowed']);self.assertEqual(report['negative_tail_weight_fraction'],1.)
        self.assertEqual(report['enforcement']['negative_tail'],'observe_only')
        report=safety.monitor(d,a,m,torch.tensor([.1,.9]),**args)
        self.assertIn('old_policy_kl',report['reasons'])
        for value in (float('nan'),31.):
            report=safety.monitor(torch.tensor([value,0.]),a,m,m,**args)
            self.assertFalse(report['allowed']);self.assertFalse(report['kl_estimate_complete'])
        with self.assertRaises(ValueError):safety.monitor(d,a,m,m,**dict(args,negative_tail_limit=.1))

    def test_real_initial_empty_and_final_forty_proofs_link_and_reject_replaced_initial(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'logs').mkdir()
            run=dict(run_id='a'*32,measurement_parent_run_id='b'*32)
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            module,adam,schedule=self.make_optimizer()
            identity=patch.object(safety,'_measurement_process_identity',return_value=(0,12345,67890,'constructed-cpu-uuid'))
            world=patch.object(safety.dist,'get_world_size',return_value=8)
            plan=patch('reproduction.common8.warmup_measurement_plan.load_plan',return_value=({},'c'*64))
            with patch.dict(os.environ,SEARCH_G1_BASE=str(base),SLURM_JOB_ID='123'),identity,world,plan:
                safety.publish_measurement_optimizer_proof(module,adam,plan_sha256='c'*64,
                    logical_update=0,optimizer_writes=0)
                path=base/'logs/warmup57-measurement-initial-optimizer-rank-0.json'
                original=path.read_bytes();before=json.loads(original)
                self.assertEqual(before['adam_steps'],[]);self.assertEqual(before['final_optimizer_writes'],0)
                self.writes(module,adam,20);schedule.step();self.writes(module,adam,20)
                replaced=dict(before,pid=54321);path.write_text(json.dumps(replaced))
                with self.assertRaisesRegex(RuntimeError,'measurement_actual_optimizer_state'):safety.publish_measurement_optimizer_proof(module,adam,
                    plan_sha256='c'*64,logical_update=2,optimizer_writes=40)
                target=base/'logs/warmup57-measurement-optimizer-rank-0.json'
                self.assertFalse(target.exists())
                path.write_bytes(original)
                safety.publish_measurement_optimizer_proof(module,adam,plan_sha256='c'*64,
                    logical_update=2,optimizer_writes=40)
                after=json.loads(target.read_bytes())
                self.assertEqual(after['final_optimizer_writes'],40)
                self.assertTrue(after['adam_steps'] and all(step==40 for step in after['adam_steps']))
                self.assertEqual(after['initial_optimizer_proof_sha256'],hashlib.sha256(original).hexdigest())
                self.assertTrue(after['all_parameters_and_adam_finite'])
                self.assertEqual(path.read_bytes(),original)
