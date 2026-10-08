"""New small publication boundaries only. No SSH or production call occurs."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
raw = (HERE / 'publish_post_terminal_integrity_once.py').read_bytes()
p = types.ModuleType('local_publisher_boundaries')
p.__file__ = str(HERE / 'publish_post_terminal_integrity_once.py')
exec(compile(raw, p.__file__, 'exec', dont_inherit=True), p.__dict__)


class PublicationBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.packet = p.prepare_packet()
        self.mraw, self.members = p.validate_packet(self.packet)

    def tearDown(self):
        for directory in self.root.rglob('*'):
            if directory.is_dir() and not directory.is_symlink():
                directory.chmod(0o700)
        self.tmp.cleanup()

    def test_fixed_actual_bundle_and_exact_manifest(self):
        self.assertEqual(hashlib.sha256(self.mraw).hexdigest(), p.PIN)
        self.assertEqual(set(self.members), set(p.MEMBERS))
        self.assertEqual(hashlib.sha256(self.members['post_terminal_integrity.py']).hexdigest(), p.MEMBERS['post_terminal_integrity.py'])

    def test_missing_extra_or_changed_members_rejected(self):
        for mode in ('missing', 'extra', 'changed'):
            packet = copy.deepcopy(self.packet)
            if mode == 'missing':
                del packet['members']['post_terminal_integrity.py']
            elif mode == 'extra':
                packet['members']['other.py'] = base64.b64encode(b'x=1\n').decode()
            else:
                packet['members']['post_terminal_integrity.py'] = base64.b64encode(b'raise RuntimeError("never")\n').decode()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                p.validate_packet(packet)

    def test_changed_manifest_or_foreign_target_protocol_rejected(self):
        packet = copy.deepcopy(self.packet)
        manifest = json.loads(self.mraw)
        manifest['source_commit'] = '0' * 40
        packet['manifest_b64'] = base64.b64encode(json.dumps(manifest).encode()).decode()
        with self.assertRaises(ValueError):
            p.validate_packet(packet)
        packet = dict(self.packet, target='other-cluster-or-dir')
        with self.assertRaises(ValueError):
            p.validate_packet(packet)

    def test_fixed_new_directory_manifest_last_and_full_readback(self):
        events = []
        original = p.os.open
        def traced_open(path, flags, *args, **kwargs):
            if flags & os.O_CREAT:
                events.append((str(path), flags))
            return original(path, flags, *args, **kwargs)
        with mock.patch.object(p.os, 'open', side_effect=traced_open):
            result = p.publish_files(self.root, self.mraw, self.members)
        target = self.root / p.TARGET_NAME
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
        self.assertEqual(events[-1][0], 'manifest.json')
        self.assertEqual(len(events), 6)
        self.assertTrue(all(flags & os.O_EXCL and flags & os.O_NOFOLLOW for _, flags in events))
        self.assertEqual(len({(r['device'], r['inode']) for r in result}), 6)
        for rec in result:
            self.assertEqual(p.read(target / rec['name'], rec['sha256']), self.mraw if rec['name']=='manifest.json' else self.members[rec['name']])
            self.assertEqual(stat.S_IMODE((target / rec['name']).stat().st_mode), 0o444)

    def test_existing_target_never_overwritten(self):
        target = self.root / p.TARGET_NAME
        target.mkdir()
        marker = target / 'old'
        marker.write_bytes(b'must remain')
        with self.assertRaises(FileExistsError):
            p.publish_files(self.root, self.mraw, self.members)
        self.assertEqual(marker.read_bytes(), b'must remain')
        self.assertEqual(list(target.iterdir()), [marker])

    def test_dangling_target_symlink_and_parent_symlink_rejected(self):
        (self.root / p.TARGET_NAME).symlink_to(self.root / 'missing-target')
        with self.assertRaises(FileExistsError):
            p.publish_files(self.root, self.mraw, self.members)
        (self.root / p.TARGET_NAME).unlink()
        parent = self.root / 'physical-parent'
        parent.mkdir()
        alias = self.root / 'parent-alias'
        alias.symlink_to(parent, target_is_directory=True)
        with self.assertRaises(OSError):
            p.publish_files(alias, self.mraw, self.members)
        self.assertFalse((parent / p.TARGET_NAME).exists())

    def test_prior_member_change_and_restore_detected_at_close(self):
        original = p.read
        changed = []
        def mutate_older(path, pin):
            value = original(path, pin)
            if Path(path).name == 'manifest.json' and not changed:
                older = Path(path).parent / 'post_terminal_integrity.py'
                older.chmod(0o600)
                older.write_bytes(self.members['post_terminal_integrity.py'])
                older.chmod(0o444)
                changed.append(True)
            return value
        with mock.patch.object(p, 'read', side_effect=mutate_older):
            with self.assertRaisesRegex(ValueError, 'Closed publication member changed'):
                p.publish_files(self.root, self.mraw, self.members)
        # Partial published target remains as failed-attempt evidence; no cleanup in publisher.
        self.assertTrue((self.root / p.TARGET_NAME / 'manifest.json').exists())

    def test_replaced_executable_refused_before_compile(self):
        path = self.root / 'transport.py'
        source = b'VALUE=1\n'
        path.write_bytes(source)
        pin = hashlib.sha256(source).hexdigest()
        marker = self.root / 'unreviewed-code-ran'
        path.write_text('from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        with self.assertRaises(ValueError):
            p.compiled_module('wrong_transport', path, pin)
        self.assertFalse(marker.exists())

    def test_same_fd_source_replacement_refused(self):
        path = self.root / 'source.py'
        replacement = self.root / 'replacement'
        content = b'VALUE=1\n'
        path.write_bytes(content)
        replacement.write_bytes(content)
        original = p.physical_fd
        calls = []
        def replace_on_reopen(value, directory=False):
            calls.append(value)
            if len(calls) == 2:
                replacement.replace(path)
            return original(value, directory)
        with mock.patch.object(p, 'physical_fd', side_effect=replace_on_reopen):
            with self.assertRaises(ValueError):
                p.read(path, hashlib.sha256(content).hexdigest())

    def test_local_original_output_symlink_parent_rejected(self):
        parent = self.root / 'real'
        parent.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(parent, target_is_directory=True)
        with self.assertRaises(OSError):
            p.exclusive(alias / 'original', b'known original')
        self.assertFalse((parent / 'original').exists())

    def test_real_approved_transport_compiled_without_any_remote_call(self):
        transport = p.compiled_module('fixed_transport_compile_only', p.W / 'transport.py', p.TRANSPORT)
        self.assertTrue(callable(transport.remote))
        self.assertEqual(transport.__file__, str(p.W / 'transport.py'))

    def test_remote_wrapper_refuses_changed_code_before_compile(self):
        pin = hashlib.sha256(raw).hexdigest()
        command = p.remote_command(raw, pin)
        self.assertEqual(shlex.split(command)[:2], ['python3', '-c'])
        wrapper = shlex.split(command)[2]
        bad = b'raise AssertionError("Unreviewed publisher executed")\n'
        wrapper = wrapper.replace(repr(base64.b64encode(raw).decode()), repr(base64.b64encode(bad).decode()))
        with self.assertRaisesRegex(ValueError, 'before compile'):
            exec(compile(wrapper, '<local changed-wrapper rejection fixture>', 'exec'), {})
        with self.assertRaises(ValueError):
            p.remote_command(raw, '0' * 64)

    def test_wrong_host_refused_before_helpers(self):
        with mock.patch.object(p.socket, 'gethostname', return_value='wrong-host'), mock.patch.object(p, 'compiled_module', side_effect=AssertionError('Must not load remote helper')):
            with self.assertRaises(ValueError):
                p.publish_packet(self.packet)

    def test_unknown_outcome_durable_and_second_attempt_never_calls_remote(self):
        attempts = []
        def timeout(command, data, timeout):
            attempts.append(command)
            raise subprocess.TimeoutExpired(command, timeout, output=b'partial remote output', stderr=b'actual timeout stderr')
        fake_transport = types.SimpleNamespace(remote=timeout)
        args = [p.__file__, '--deploy-reviewed', '--publisher-sha256', hashlib.sha256(raw).hexdigest()]
        with mock.patch.object(p, 'W', self.root), mock.patch.object(p, 'prepare_packet', return_value=self.packet), mock.patch.object(p, 'compiled_module', return_value=fake_transport), mock.patch.object(sys, 'argv', args):
            with self.assertRaises(subprocess.TimeoutExpired):
                p.main()
            with self.assertRaises(FileExistsError):
                p.main()
        self.assertEqual(len(attempts), 1)
        out = self.root / 'post-terminal-integrity-deployment-b8d8ff319cec-originals'
        failure = json.loads((out / 'publish-failure-or-unknown-original.json').read_bytes())
        self.assertTrue(failure['remote_outcome_unknown'])
        self.assertIsNone(failure['actual_returncode'])
        self.assertTrue(failure['must_not_retry'])
        self.assertEqual((out / 'publish-failed-stdout.original').read_bytes(), b'partial remote output')
        self.assertEqual((out / 'publish-failed-stderr.original').read_bytes(), b'actual timeout stderr')

    def test_known_ssh_exit_still_does_not_establish_remote_publication_outcome(self):
        def fail(command, data, timeout):
            raise subprocess.CalledProcessError(255, command, output=b'connection lost', stderr=b'known transport exit')
        fake_transport = types.SimpleNamespace(remote=fail)
        args = [p.__file__, '--deploy-reviewed', '--publisher-sha256', hashlib.sha256(raw).hexdigest()]
        with mock.patch.object(p, 'W', self.root), mock.patch.object(p, 'prepare_packet', return_value=self.packet), mock.patch.object(p, 'compiled_module', return_value=fake_transport), mock.patch.object(sys, 'argv', args):
            with self.assertRaises(subprocess.CalledProcessError):
                p.main()
        out = self.root / 'post-terminal-integrity-deployment-b8d8ff319cec-originals'
        failure = json.loads((out / 'publish-failure-or-unknown-original.json').read_bytes())
        self.assertEqual(failure['actual_returncode'], 255)
        self.assertTrue(failure['remote_outcome_unknown'])
        self.assertTrue(failure['must_not_retry'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
