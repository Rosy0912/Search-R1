"""Small CPU schema/archive fixtures; no SSH, production bytes or scientific tests."""
import gzip
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).absolute().parent


def load(name):
    path = HERE / name
    ns = dict(__name__='_fixture_' + name, __file__=str(path))
    exec(compile(path.read_bytes(), str(path), 'exec'), ns)
    return ns


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve() / 'study'; self.base.mkdir()
        self.m = load('source-operation.py'); self.m['BASE'] = self.base
        self.b = self.m['primitives'](HERE.parent / 'opsd_ablation_outer_backup_v2/backup.py')
        self.group = self.base / 'group-M'; self.runs = self.group / 'runs'
        self.run = '1' * 32
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True))
            return self.b['fileproof'](path)
        self.save = save
        cfg = save(self.group / 'receipts/authorized-launch-20261009/config.json', dict(
            schema='opsd-ablation-config-v1', variant='M', run_id=self.run,
            paths=dict(output_root=str(self.runs)), scientific=dict(outer_iterations=10)))
        rows = []
        for index in range(10):
            cp = save(self.runs / 'checkpoints' / ('update-%08d-outer-%05d' % (index + 1, index)) / 'manifest.json',
                dict(accepted=True, run_id=self.run, config_sha256=cfg['sha256'], optimizer_updates=index + 1,
                     cursor=dict(outer_index=index)))
            complete = save(self.runs / 'outers' / ('outer-%05d' % index) / 'training/outer-complete.json',
                dict(accepted=True, run_id=self.run, outer_index=index, optimizer_updates=index + 1, checkpoint=cp))
            rows.append(dict(outer_index=index, optimizer_updates=index + 1, completion=complete, checkpoint=cp))
        owner = dict(pid=123, uid=1005, starttick=456, run_id=self.run)
        status = save(self.runs / 'status.json', dict(phase='TRAIN_COMPLETE', outer_iterations=10,
            actual_optimizer_updates=10, run_id=self.run, variant='M', config_sha256=cfg['sha256'],
            latest_checkpoint=rows[-1]['checkpoint'], owner=owner))
        training = self.base / 'evaluation/runs/02-M/training'
        actual = save(training / 'actual-child-exit.json', dict(actual_wait_returncode=0,
            identity=owner, run_id=self.run, signal_termination_not_exit0=False))
        release = save(training / 'actual-child-release.json', dict(accepted=True,
            owned=dict(accepted=True), after=dict(accepted=True)))
        terminal = save(training / 'group-terminal.json', dict(schema='opsd-ablation-group-foreground-terminal-v1',
            accepted=True, run_id=self.run, config_sha256=cfg['sha256'], actual_exit_code=0,
            signal_termination_not_exit0=False, controller=owner, status=status,
            actual_child_exit=actual, actual_release=release))
        self.scope = dict(schema=self.m['SCHEMA'], accepted=True, cluster='HKUST', variant='M',
            run_id=self.run, config=cfg, output_root=str(self.runs), closed_outers=list(range(10)),
            outers=rows, status=status, group_terminal=terminal, global_files=[], include_outer_raw=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_ten_closures_regular_members_and_cache_exclusion(self):
        cache = self.runs / 'outers/outer-00000/merged-model-cache'; cache.mkdir(); (cache / 'cache.bin').write_bytes(b'cache')
        closed = self.m['closure'](self.b, self.scope)
        paths, excluded = self.m['enumerate_files'](self.b, self.scope, closed)
        self.assertEqual(len(paths), 10); self.assertEqual(excluded, [str(cache)])
        files = [dict(relative=p.relative_to(self.base).as_posix(), **self.b['fileproof'](p)) for p in paths]
        tar = io.BytesIO(); self.b['stream_archive'](dict(files=files), tar)
        archive = self.base / 'test.tar.gz'; archive.write_bytes(gzip.compress(tar.getvalue(), mtime=0))
        self.assertEqual(len(load('verify-archive.py')['verify_members'](self.b, archive, dict(files=files))), 10)

    def test_partial_or_changed_config_is_rejected(self):
        partial = dict(self.scope, closed_outers=list(range(9)))
        with self.assertRaises(ValueError): self.m['scope_shape'](partial)
        path = Path(self.scope['config']['path']); path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError): self.m['closure'](self.b, self.scope)

    def test_literal_scope_original_bytes_do_not_need_a_node_file(self):
        raw = json.dumps(self.scope, indent=2).encode() + b'\n'
        sha = hashlib.sha256(raw).hexdigest()
        value, proof = self.m['inline_scope'](base64.b64encode(raw).decode(), sha)
        self.assertEqual(value, self.scope)
        self.assertEqual(proof, dict(delivery='inline-base64', bytes=len(raw), sha256=sha))
        with self.assertRaises(ValueError):
            self.m['inline_scope'](base64.b64encode(raw + b' ').decode(), sha)

    def test_nonzero_wait_and_mutable_global_are_rejected(self):
        terminal, _ = self.m['read_ref'](self.b, self.scope['group_terminal'])
        actual, _ = self.m['read_ref'](self.b, terminal['actual_child_exit'])
        actual['actual_wait_returncode'] = 2
        terminal['actual_child_exit'] = self.save(Path(terminal['actual_child_exit']['path']), actual)
        self.scope['group_terminal'] = self.save(Path(self.scope['group_terminal']['path']), terminal)
        with self.assertRaises(ValueError): self.m['closure'](self.b, self.scope)
        ledger = self.save(self.base / 'runs/full-vocabulary/ledger.json', dict(bytes=0))
        self.scope['global_files'] = [ledger]
        with self.assertRaises(ValueError): self.m['enumerate_files'](self.b, self.scope, {})

    def test_corrupt_or_truncated_gzip_member_is_rejected(self):
        payload = self.base / 'payload'; payload.write_bytes(b'original')
        row = dict(relative='payload', **self.b['fileproof'](payload))
        tar = io.BytesIO(); self.b['stream_archive'](dict(files=[row]), tar)
        archive = self.base / 'test.tar.gz'; archive.write_bytes(gzip.compress(tar.getvalue(), mtime=0))
        verifier = load('verify-archive.py')['verify_members']
        wrong = dict(row, sha256='0' * 64)
        with self.assertRaises(ValueError): verifier(self.b, archive, dict(files=[wrong]))
        archive.write_bytes(archive.read_bytes()[:-8])
        with self.assertRaises((EOFError, OSError)): verifier(self.b, archive, dict(files=[row]))


if __name__ == '__main__':
    unittest.main()
