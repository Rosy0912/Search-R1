import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


runner = load('replay_gpu'); core = load('replay_core')


class Retry(unittest.TestCase):
    def args(self):
        config = dict(gpu_index=0, gpu_uuid='exact', initial_min_free_mib=32768,
            running_min_free_mib=6144, max_replay_memory_mib=12032)
        own = dict(pid=42, start_ticks=90, sid=42, uid=0, command_sha256='owned')
        pf = types.SimpleNamespace(same_process=lambda a, b: all(a[k] == b[k] for k in ['pid', 'start_ticks', 'sid', 'uid']))
        return config, own, pf

    def good(self, **kwargs):
        return dict(candidate_gpu=dict(uuid='exact', actual_free_mib=60000), compute_processes=[], **kwargs)

    def test_vanished_proc_then_exact_read_can_continue(self):
        config, own, pf = self.args()
        pf.resource_snapshot = unittest.mock.Mock(side_effect=[FileNotFoundError(), self.good()])
        with patch.object(runner.time, 'sleep') as sleep:
            result = runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(pf.resource_snapshot.call_count, 2); self.assertEqual(sleep.call_args.args, (1.0,))
        self.assertEqual(result['bounded_read_retry_events'][0]['exception_type'], 'FileNotFoundError')

    def test_unbound_original_transition_never_accepted_until_exact(self):
        config, own, pf = self.args()
        bad = self.good(); bad['compute_processes'] = [dict(pid=99, start_ticks=1, sid=1, uid=0,
            command_sha256='unbound', memory_mib=1, matched_recorded_pid_tick_sid_uid=False)]
        pf.resource_snapshot = unittest.mock.Mock(side_effect=[bad, self.good()])
        with patch.object(runner.time, 'sleep'):
            result = runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(result['bounded_read_retry_events'][0]['observation']['processes'][0]['pid'], 99)

    def test_persistent_foreign_still_HOLD_four_times(self):
        config, own, pf = self.args(); bad = self.good()
        bad['compute_processes'] = [dict(pid=99, start_ticks=1, sid=1, uid=0,
            command_sha256='unbound', memory_mib=1, matched_recorded_pid_tick_sid_uid=False)]
        pf.resource_snapshot = unittest.mock.Mock(return_value=bad)
        with patch.object(runner.time, 'sleep'), self.assertRaises(core.Hold) as caught:
            runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(pf.resource_snapshot.call_count, 4)
        self.assertEqual(len(caught.exception.resource_retry_events), 4)

    def test_direct_free_or_memory_failure_never_retried(self):
        for kind in ['free', 'cap']:
            config, own, pf = self.args(); bad = self.good()
            if kind == 'free': bad['candidate_gpu']['actual_free_mib'] = 6143
            else: bad['compute_processes'] = [dict(own, memory_mib=12033, matched_recorded_pid_tick_sid_uid=False)]
            pf.resource_snapshot = unittest.mock.Mock(return_value=bad)
            with patch.object(runner.time, 'sleep') as sleep, self.assertRaises(core.Hold):
                runner.resource_gate(Path('/none'), {}, config, pf, own, core)
            self.assertEqual(pf.resource_snapshot.call_count, 1); sleep.assert_not_called()

    def test_foreign_plus_lowfree_HOLD_without_retry(self):
        config, own, pf = self.args(); bad = self.good(); bad['candidate_gpu']['actual_free_mib'] = 6143
        bad['compute_processes'] = [dict(pid=99, start_ticks=1, sid=1, uid=0,
            command_sha256='unbound', memory_mib=1, matched_recorded_pid_tick_sid_uid=False)]
        pf.resource_snapshot = unittest.mock.Mock(return_value=bad)
        with patch.object(runner.time, 'sleep') as sleep, self.assertRaises(core.Hold) as caught:
            runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(str(caught.exception), 'candidate_direct_free_below_floor')
        self.assertEqual(pf.resource_snapshot.call_count, 1); sleep.assert_not_called()

    def test_recorded_command_change_plus_owncap_HOLD_without_retry(self):
        config, own, pf = self.args(); bad = self.good()
        bad['compute_processes'] = [dict(pid=1, matched_recorded_pid_tick_sid_uid=True,
            recorded_command_sha256='old', command_sha256='new'),
            dict(own, memory_mib=12033, matched_recorded_pid_tick_sid_uid=False)]
        pf.resource_snapshot = unittest.mock.Mock(return_value=bad)
        with patch.object(runner.time, 'sleep') as sleep, self.assertRaises(core.Hold) as caught:
            runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(str(caught.exception), 'replay_GPU_memory_cap_exceeded')
        self.assertEqual(pf.resource_snapshot.call_count, 1); sleep.assert_not_called()

    def test_UUID_failure_not_retried(self):
        config, own, pf = self.args(); bad = self.good(); bad['candidate_gpu']['uuid'] = 'foreign'
        pf.resource_snapshot = unittest.mock.Mock(return_value=bad)
        with patch.object(runner.time, 'sleep'), self.assertRaises(core.Hold):
            runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(pf.resource_snapshot.call_count, 1)

    def test_retry_events_written_only_requested_remote_log_path(self):
        config, own, pf = self.args(); pf.resource_snapshot = unittest.mock.Mock(side_effect=[ProcessLookupError(), self.good()])
        with tempfile.TemporaryDirectory() as d, patch.object(runner.time, 'sleep'):
            p = Path(d) / 'events.jsonl'
            runner.resource_gate(Path('/none'), {}, config, pf, own, core, event_path=p)
            self.assertIn(b'ProcessLookupError', p.read_bytes())

    def test_exact_self_argv_changed_HOLD_without_retry(self):
        config, own, pf = self.args(); bad = self.good()
        bad['compute_processes'] = [dict(own, command_sha256='changed', memory_mib=1, matched_recorded_pid_tick_sid_uid=False)]
        pf.resource_snapshot = unittest.mock.Mock(return_value=bad)
        with patch.object(runner.time, 'sleep'), self.assertRaises(core.Hold):
            runner.resource_gate(Path('/none'), {}, config, pf, own, core)
        self.assertEqual(pf.resource_snapshot.call_count, 1)


if __name__ == '__main__': unittest.main()
