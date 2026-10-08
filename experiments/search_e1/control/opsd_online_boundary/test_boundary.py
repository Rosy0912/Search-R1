"""Independent CPU fixtures: all OS signals and node reads are mocked.

The target source is never altered. Temporary files are local-only. This is a
source review, not a deployed stop, live-gate verification, or training test.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).with_name('preserve_cpu_merge_owner.py')
spec = importlib.util.spec_from_file_location('boundary_review_subject', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def current(base=None):
    base = base or m.BASE
    return dict(m.OWNER, run_id=m.RUN, argv=('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python -B '
        + str(base / 'source/wait-recovery-2cbcc1b/opsd_online_wait_recovery/recovery.py')
        + ' --config ' + str(base / 'receipts/config.json') + ' --config-sha256 ' + m.CONFIG_SHA
        + ' --run-id ' + m.RUN + ' --resume'))


class SignalReached(Exception):
    pass


class BoundaryGuardTests(unittest.TestCase):
    def test_exact_owner(self):
        self.assertIsNone(m.match_owner(current()))

    def test_pid_uid_tick_run_source_config_resume_reject(self):
        for key, value in [('pid', 660174), ('uid', 1006), ('starttick', 924349237), ('run_id', 'another-run')]:
            c = current(); c[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                m.match_owner(c)
        for text in ['recovery.py', m.CONFIG_SHA, '--resume']:
            c = current(); c['argv'] = c['argv'].replace(text, 'wrong')
            with self.subTest(argv=text), self.assertRaises(RuntimeError):
                m.match_owner(c)

    def test_absent_owner_rejects(self):
        with self.assertRaises(RuntimeError): m.match_owner(None)

    def test_missing_pidfd_capability_fails_before_loading_or_signal(self):
        load = Mock(side_effect=AssertionError('Must not load source'))
        sent = Mock(side_effect=AssertionError('Must not signal'))
        with patch.object(m.os, 'getuid', return_value=1005), patch.dict(m.os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(m, 'hasattr', return_value=False, create=True), patch.object(m, 'load_wait', load), \
             patch.object(m.signal, 'pidfd_send_signal', sent, create=True):
            with self.assertRaisesRegex(RuntimeError, 'safe pidfd'): m.preserve(None)
        load.assert_not_called(); sent.assert_not_called()

    def test_source_sha_change_fails_before_import(self):
        with patch.object(m, 'sha', return_value='0' * 64), \
             patch.object(m.importlib.util, 'spec_from_file_location') as loader:
            with self.assertRaisesRegex(RuntimeError, 'Frozen wait helper changed'): m.load_wait()
        loader.assert_not_called()

    def test_existing_outer1_fails_before_signal(self):
        self._mock_preserve(mutator='outer_before', expect_signal=False)

    def test_wrong_owner_fails_before_signal(self):
        self._mock_preserve(mutator='owner_before', expect_signal=False)

    def test_cp_validation_failure_fails_before_signal(self):
        self._mock_preserve(mutator='cp_bad', expect_signal=False)

    def test_status_changed_during_cp_rejected_before_signal(self):
        self._mock_preserve(mutator='status_after_cp', expect_signal=False)

    def test_outer1_advanced_during_cp_rejected_before_signal(self):
        self._mock_preserve(mutator='outer_after_cp', expect_signal=False)

    def test_child_added_during_cp_rejected_before_signal(self):
        self._mock_preserve(mutator='child_after_cp', expect_signal=False)

    def test_owner_pid_reused_after_pidfd_open_rejected(self):
        self._mock_preserve(mutator='owner_after_cp', expect_signal=False)

    def test_fresh_full_audit_failure_rejected_before_signal(self):
        self._mock_preserve(mutator='audit_after_cp', expect_signal=False)

    def test_fresh_gpu_boundary_failure_rejected_before_signal(self):
        self._mock_preserve(mutator='gpu_after_cp', expect_signal=False)

    def test_all_mocked_fresh_guards_allow_only_one_pidfd_signal(self):
        self._mock_preserve(mutator='none', expect_signal=True)

    def test_argv_substring_or_duplicate_option_rejected(self):
        for transform in [lambda s: s.replace('recovery.py', 'recovery.py.not-script'),
                          lambda s: s.replace('--resume', '--resume-extra'),
                          lambda s: s + ' --config-sha256 ' + m.CONFIG_SHA]:
            with self.subTest(transform=transform), self.assertRaises(RuntimeError):
                c = current(); c['argv'] = transform(c['argv']); m.match_owner(c)

    def _mock_preserve(self, mutator, expect_signal):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for relative in ['runs/outers/outer-00000/training', 'runs/outers/outer-00001']:
                (base / relative).mkdir(parents=True)
            status = dict(run_id=m.RUN, phase='UNCERTAIN_RELEASE_LEASE_HELD', error='Owned descendant still holds GPU after parent exit')
            (base / 'runs/status.json').write_text(json.dumps(status))
            child = dict(pid=2147483000, uid=1005, run_id=m.RUN, starttick=1)
            (base / 'runs/child-history.jsonl').write_text(json.dumps(dict(identity=child)) + '\n')
            (base / 'runs/outers/outer-00000/outer-manifest.json').write_text('{"P":327}')
            if mutator == 'outer_before': (base / 'runs/outers/outer-00001/outer-manifest.json').write_text('{}')
            c = current(base)
            if mutator == 'owner_before': c['starttick'] += 1
            apps = dict(apps=[dict(pid=m.OWNER['pid'], uuid='GPU0', used_mib=500), dict(pid=m.ORACLE['pid'], uuid='GPU7', used_mib=574)])
            original = types.SimpleNamespace(process_identity=Mock(return_value=c), gpu_snapshot=Mock(return_value=apps))
            config = dict(owner=dict(gpu_uuids=['GPU0']), resources=dict(shared_gpu_lease_paths=[]))
            wait = types.SimpleNamespace(checked_json=Mock(return_value=(config, {})), load_original=Mock(return_value=(original, {})))
            def cp_check(*args):
                if mutator == 'cp_bad': raise ValueError('CP hash changed')
                if mutator == 'status_after_cp':
                    (base / 'runs/status.json').write_text(json.dumps(dict(run_id=m.RUN, phase='TRAINING_RUNNING', error='')))
                if mutator == 'outer_after_cp':
                    (base / 'runs/outers/outer-00001/outer-manifest.json').write_text('{}')
                if mutator == 'child_after_cp':
                    with (base / 'runs/child-history.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(identity=dict(child, pid=2147482999))) + '\n')
                if mutator == 'owner_after_cp': original.process_identity.return_value = dict(c, starttick=c['starttick']+1)
                if mutator == 'audit_after_cp': acceptance.require_owned_release.side_effect = RuntimeError('Full audit rejected')
                if mutator == 'gpu_after_cp': original.gpu_snapshot.return_value = dict(apps=apps['apps'] + [dict(pid=2147482888, uuid='GPU1', used_mib=100)])
                return dict(cumulative_pair_exposure=327), {}
            checkpoint = types.SimpleNamespace(check_checkpoint=Mock(side_effect=cp_check),
                check_outer_complete=Mock(return_value=(dict(checkpoint=dict(sha256=m.CP11_SHA)), {})),
                fileproof=Mock(return_value={}), exclusive_json=Mock(return_value={}))
            acceptance = types.SimpleNamespace(require_launch=Mock(return_value={}), require_owned_release=Mock(return_value={}))
            sent = Mock(side_effect=SignalReached())
            actual_read_bytes = Path.read_bytes
            def read_bytes(path):
                if str(path) == '/proc/' + str(m.OWNER['pid']) + '/cmdline':
                    return c['argv'].replace(' ', '\0').encode() + b'\0'
                return actual_read_bytes(path)
            with patch.object(m, 'BASE', base), patch.object(m.os, 'getuid', return_value=1005), \
                 patch.dict(m.os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(m, 'hasattr', return_value=True, create=True), \
                 patch.object(m, 'load_wait', return_value=wait), patch.dict('sys.modules', {'checkpoint': checkpoint, 'acceptance': acceptance}), \
                 patch.object(m.os, 'pidfd_open', return_value=77, create=True), patch.object(m.os, 'close'), \
                 patch.object(m.signal, 'pidfd_send_signal', sent, create=True), \
                 patch.object(Path, 'read_bytes', autospec=True, side_effect=read_bytes):
                if expect_signal:
                    with self.assertRaises(SignalReached): m.preserve(None)
                    sent.assert_called_once_with(77, m.signal.SIGTERM)
                else:
                    with self.assertRaises((RuntimeError, ValueError)): m.preserve(None)
                    sent.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
