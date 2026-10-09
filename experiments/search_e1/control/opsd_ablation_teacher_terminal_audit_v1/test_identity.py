"""One tiny negative actual-wait identity example; no archive or real data."""
import copy
import unittest
from audit import verify_external


class IdentityRejection(unittest.TestCase):
    def test_rejects_external_tick_different_from_embedded_child(self):
        child = dict(identity=dict(pid=123, uid=1005, starttick=456, run_id='r'), actual_wait_returncode=0)
        exits = dict(accepted=True, child_exit_evidence=[child], gpu_release=dict(all_owned_release=dict(accepted=True)))
        actual = dict(schema='opsd-ablation-actual-phase-terminal-v1', run_id='r', variant='C1',
                      config_sha256='s', phase='sampling-running', child_exit_evidence=[copy.deepcopy(child)])
        actual['child_exit_evidence'][0]['identity']['starttick'] = 457
        with self.assertRaisesRegex(ValueError, 'identity/evidence differs'):
            verify_external(actual, exits, 'r', 's', 'C1', 1005)


if __name__ == '__main__':
    unittest.main()
