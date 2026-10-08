"""Actual source observation boundary gates only; no FSDP/GPU acceptance."""
from pathlib import Path
from types import SimpleNamespace as N
import ast
import unittest
p=Path(__file__).parent/'search_r1/baselines/sampling_consistency.py'
t=ast.parse(p.read_text());fn=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='calibration_boundary_matches'];assert len(fn)==1
scope={};exec(compile(ast.Module(body=fn,type_ignores=[]),str(p),'exec'),scope)
check=scope['calibration_boundary_matches']
def fixture(corrected):
    consistency=N(enabled=not corrected,enforce=corrected)
    return N(global_steps=111,actual_outer_updates=110,config=N(actor_rollout_ref=N(
        actor=N(policy_safety=N(enabled=True)),rollout=N(sampling_consistency=consistency))))
class CalibrationBoundaryTests(unittest.TestCase):
    def test_corrected_source_boundary_preserves_original_gate_configured_enforce(self):
        self.assertTrue(check(fixture(True),110,True))
    def test_legacy_observation_boundary_remains_original_protocol(self):
        self.assertTrue(check(fixture(False),110,False))
        f=fixture(False);f.config.actor_rollout_ref.rollout.sampling_consistency.enabled=False
        self.assertFalse(check(f,110,False))
    def test_profiles_cannot_impersonate_one_another(self):
        self.assertFalse(check(fixture(True),110,False));self.assertFalse(check(fixture(False),110,True))
        f=fixture(True);f.config.actor_rollout_ref.rollout.sampling_consistency.enforce=False
        self.assertFalse(check(f,110,True))
    def test_no_completed_outer_or_scheduler_advance_can_be_hidden(self):
        for field,value in [('global_steps',112),('actual_outer_updates',111)]:
            f=fixture(True);setattr(f,field,value)
            self.assertFalse(check(f,110,True))
        f=fixture(True);f.config.actor_rollout_ref.actor.policy_safety.enabled=False
        self.assertFalse(check(f,110,True))
    def test_corrected_optins_are_actual_booleans(self):
        for field,value in [('enabled',0),('enforce',1)]:
            f=fixture(True);setattr(f.config.actor_rollout_ref.rollout.sampling_consistency,field,value)
            self.assertFalse(check(f,110,True))
        self.assertFalse(check(fixture(True),110,1))
if __name__=='__main__':unittest.main()
