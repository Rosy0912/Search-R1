"""CPU-only final-boundary checks; no GPU, Ray, model or training imports."""
import ast
import copy
from pathlib import Path
import unittest
from final_eval import validate_checkpoint


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
        for name in ('final_eval.py', 'final_worker.py'):
            ast.parse(Path(__file__).with_name(name).read_text())


if __name__ == '__main__': unittest.main()
