import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

p = Path(__file__).resolve().parents[2] / 'reproduction/a800/stop_at_checkpoint.py'
spec = importlib.util.spec_from_file_location('stop_boundary', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class StopBoundaryTests(unittest.TestCase):
    def test_actual_completion_parser_does_not_count_started_rollout_or_validation(self):
        text = ('epoch0, step301\nstep:300 - val/test_score/nq:0.500\n'
                '\x1b[36m(main_task pid=1)\x1b[0m step:300 - actor/pg_loss:0.000 - critic/rewards/mean:0.400\n')
        self.assertEqual(m.logged_updates(text), [300])

    def test_duplicate_actual_update_cannot_satisfy_boundary(self):
        with self.assertRaises(AssertionError):
            m.logged_updates('step:300 - critic/rewards/mean:0.4\n' * 2)

    def test_pid_reuse_blocks_signal_after_pidfd_open(self):
        r = dict(pid=4, start_ticks=10, sid=3, uid=0)
        with patch.object(m.os,'pidfd_open',return_value=77,create=True), patch.object(m.os,'close') as close, patch.object(m,'identity',return_value=dict(r,start_ticks=11)), patch.object(m.signal,'pidfd_send_signal',create=True) as signal:
            with self.assertRaises(RuntimeError):
                m.send(r,m.signal.SIGSTOP)
            signal.assert_not_called()
            close.assert_called_once_with(77)

if __name__ == '__main__':
    unittest.main()
