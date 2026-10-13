"""CPU-only final-boundary checks; no GPU, Ray, model or training imports."""
import ast
from pathlib import Path
import unittest
from final_eval import validate_checkpoint
from boundary_supervisor import terminal_condition, matches


class Boundary(unittest.TestCase):
    def fixture(self):
        return dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',
            identity=dict(config_sha256='fixed', completed_step=200,
                guard_continue_accounting=dict(completed_outer_updates=200)),
            driver=dict(next_step=201),
            worker_states=[dict(identity=dict(rank=i)) for i in range(8)],
            hf_checkpoint='/remote/actor/global_step_200')

    def test_final_only(self):
        self.assertEqual(validate_checkpoint(self.fixture(), 'fixed'),
            '/remote/actor/global_step_200')

    def test_reject_wrong_boundary_and_partial(self):
        for key in ('completed_step', 'completed_outer_updates'):
            item = self.fixture()
            parent = item['identity'] if key == 'completed_step' else item['identity']['guard_continue_accounting']
            parent[key] = 199
            with self.assertRaises(AssertionError): validate_checkpoint(item, 'fixed')
        item = self.fixture(); item['worker_states'].pop()
        with self.assertRaises(AssertionError): validate_checkpoint(item, 'fixed')

    def test_reject_other_config(self):
        with self.assertRaises(AssertionError): validate_checkpoint(self.fixture(), 'other')

    def test_payload_syntax(self):
        for name in ('final_eval.py', 'final_worker.py', 'boundary_supervisor.py'):
            ast.parse(Path(__file__).with_name(name).read_text())

    def test_no_early_training_handoff(self):
        self.assertEqual(terminal_condition(None,'fixed',False,None),'WAIT_TRAINING')
        self.assertEqual(terminal_condition(self.fixture(),'fixed',False,None),'WAIT_TRAINING')
        with self.assertRaises(AssertionError): terminal_condition(self.fixture(),'fixed',True,None)
        manifest=self.fixture();manifest['boundary_phase']='post-evaluation'
        self.assertEqual(terminal_condition(manifest,'fixed',True,None),'HANDOFF_FINAL_ONLY')

    def test_do_not_duplicate_native_final(self):
        metrics={'val/qa_count/all':51713}
        self.assertEqual(terminal_condition(None,'fixed',False,metrics),'NATIVE_FINAL_COMPLETE')
        with self.assertRaises(AssertionError):
            terminal_condition(None,'fixed',False,{'val/qa_count/all':512})

    def test_signal_identity_checks_all_fields(self):
        owner=dict(pid=1,uid=0,tick=12)
        self.assertTrue(matches(owner,owner))
        self.assertFalse(matches(None,owner))
        for key in owner:
            self.assertFalse(matches(dict(owner,**{key:owner[key]+1}),owner))


if __name__ == '__main__': unittest.main()
