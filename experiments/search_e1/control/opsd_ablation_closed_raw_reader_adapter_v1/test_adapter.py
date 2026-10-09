"""Four small proof/interface fixtures, no Node/SSH/science or real backup scan."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).absolute().parent
GENERIC = HERE.parent / 'opsd_ablation_closed_raw_backup_v1'
TRANSPORT = HERE.parent / 'opsd_ablation_outer_backup_v2/backup.py'
READER = HERE.parent / 'opsd_ablation_recorded_clipmass_v1/aggregate_clipmass.py'


def load(path):
    namespace = dict(__name__='_adapter_fixture', __file__=str(path))
    exec(compile(path.read_bytes(), str(path), 'exec'), namespace)
    return namespace


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.a = load(HERE / 'adapt.py')
        self.b = load(TRANSPORT)
        self.refs = {}; self.values = {}
        def save(key, path, value):
            path.write_text(json.dumps(value, sort_keys=True) + '\n')
            self.refs[key] = self.b['fileproof'](path); self.values[key] = value
            return self.refs[key]
        self.save = save
        self.refs['source_manifest'] = self.b['fileproof'](GENERIC / 'source-manifest.json')
        self.refs['transport_dependency'] = self.b['fileproof'](TRANSPORT)
        operation = load(GENERIC / 'source-operation.py'); base = operation['BASE']; group = base / 'group-M'
        def ref(path): return dict(path=str(path), sha256='a' * 64)
        config = ref(group / 'receipts/authorized-launch-20261009/config.json')
        closures = [dict(outer_index=i, optimizer_updates=i + 1,
            completion=ref(group / 'runs/outers' / ('outer-%05d' % i) / 'training/outer-complete.json'),
            checkpoint=ref(group / 'runs/checkpoints' / ('update-%08d-outer-%05d' % (i + 1, i)) / 'manifest.json')) for i in range(10)]
        scope = dict(schema=operation['SCHEMA'], accepted=True, cluster='HKUST', variant='M', run_id='1' * 32,
            config=config, output_root=str(group / 'runs'), closed_outers=list(range(10)), outers=closures,
            status=ref(group / 'runs/status.json'), group_terminal=ref(base / 'study/runs/02-M/training/group-terminal.json'),
            include_outer_raw=True, global_files=[])
        sp = save('scope', self.root / 'scope.json', scope)
        payload = b'closed toy payload, no questions or weights\n'
        name = 'group-M/runs/outers/outer-00000/training/toy.jsonl'
        row = dict(path=str(base / name), relative=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        closed = dict(config=config, outers=[dict(outer_index=x['outer_index'], optimizer_updates=x['optimizer_updates'],
            complete=x['completion'], checkpoint=x['checkpoint']) for x in closures], status=scope['status'], group_terminal=scope['group_terminal'])
        inv = dict(schema=operation['INVENTORY'], source=str(base), run_id=scope['run_id'], variant='M', config=config,
            scope=dict(delivery='inline-base64', bytes=sp['bytes'], sha256=sp['sha256']), closed_outers=list(range(10)),
            closures=closed, files=[row], expected_file_count=1, bytes=len(payload), excluded_subtrees=[],
            source_helper_sha256=self.a['DEPENDENCY_SHA'])
        ip = save('inventory', self.root / 'inventory.json', inv)
        after = dict(schema='opsd-ablation-closed-raw-source-after-v1', accepted=True, scope_sha256=sp['sha256'],
            inventory_sha256=ip['sha256'], closures=closed, source=str(base), run_id=scope['run_id'], files=1,
            all_source_metadata_stable=True, all_noncache_file_set_unchanged=True)
        save('after_metadata', self.root / 'after.json', after)
        self.destination = self.root / 'backup'; self.destination.mkdir(mode=0o700)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            member = tarfile.TarInfo(name); member.size = len(payload); tar.addfile(member, io.BytesIO(payload))
        archive = self.destination / 'closed-raw.tar.gz.partial'; archive.write_bytes(gzip.compress(stream.getvalue(), mtime=0))
        op = self.b['fileproof'](GENERIC / 'source-operation.py')
        save('transfer_launch', self.destination / 'transfer-launch.json', dict(inventory_sha256=ip['sha256'],
            scope=sp, operation_sha256=op['sha256'], scope_transport='inline-base64'))
        save('transfer_terminal', self.destination / 'transfer-terminal.json', dict(actual_returncode=0, local_error=None,
            partial_bytes=archive.stat().st_size))
        # Real published local verifier runs once on toy archive/proofs. This is
        # not a Node stream, live completion/ownership proof, or scientific test.
        program = self.b['fileproof'](GENERIC / 'verify-archive.py')
        argv = [sys.executable, '-B', program['path'], '--scope', sp['path'], '--scope-sha256', sp['sha256'],
            '--inventory', ip['path'], '--after-metadata', self.refs['after_metadata']['path'], '--destination', str(self.destination),
            '--source-helper', str(TRANSPORT), '--source-operation', op['path'], '--operation-sha256', op['sha256']]
        actual = subprocess.run(argv, capture_output=True, check=False)
        self.assertEqual(actual.returncode, 0, actual.stderr.decode())
        cp = self.b['fileproof'](self.destination / 'complete-sha-verified.json')
        self.refs['complete'] = cp; self.values['complete'] = json.loads(Path(cp['path']).read_bytes())
        save('verifier_execution', self.root / 'actual-verifier.json', dict(schema='opsd-ablation-closed-raw-verifier-execution-v1',
            actual_returncode=actual.returncode, program=program, argv=argv, complete=cp))

    def tearDown(self): self.temp.cleanup()

    def test_accepted_adapter_matches_actual_legacy_reader_interface(self):
        adapted, _ = self.a['adapt'](self.refs)
        self.assertEqual(adapted['schema'], self.a['SCHEMA'])
        self.assertEqual(adapted['original_complete']['sha256'], self.refs['complete']['sha256'])
        self.assertFalse(adapted['fresh_archive_SHA_or_member_scan_performed'])
        path = self.root / 'adapter.json'; path.write_text(json.dumps(adapted))
        raw = READER.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a')
        reader = load(READER)
        backups = reader['FixedBackups']([dict(kind='gzip', inventory=self.refs['inventory'], complete=self.b['fileproof'](path))], '1' * 32)
        self.assertEqual(len(backups.files), 1)

    def test_verifier_nonzero_wrong_invocation_or_missing_evidence_is_hold(self):
        original = self.values['verifier_execution']
        for changed in (dict(original, actual_returncode=1), dict(original, argv=original['argv'][:-2]), {}):
            self.save('verifier_execution', self.root / 'actual-verifier.json', changed)
            with self.assertRaises((ValueError, KeyError)): self.a['adapt'](self.refs)

    def test_after_exact_set_and_stream_identity_are_required(self):
        self.save('after_metadata', self.root / 'after.json', dict(self.values['after_metadata'], all_noncache_file_set_unchanged=False))
        with self.assertRaises(ValueError): self.a['adapt'](self.refs)
        self.save('after_metadata', self.root / 'after.json', dict(self.values['after_metadata'], all_noncache_file_set_unchanged=True))
        self.save('transfer_launch', self.destination / 'transfer-launch.json', dict(self.values['transfer_launch'], operation_sha256='0' * 64))
        with self.assertRaises(ValueError): self.a['adapt'](self.refs)

    def test_wrong_member_or_changed_current_archive_is_hold(self):
        original = dict(self.values['complete'])
        complete = dict(original); complete['members'] = [dict(complete['members'][0], sha256='0' * 64)]
        cp = self.save('complete', Path(self.refs['complete']['path']), complete)
        self.save('verifier_execution', self.root / 'actual-verifier.json', dict(self.values['verifier_execution'], complete=cp))
        with self.assertRaises(ValueError): self.a['adapt'](self.refs)
        cp = self.save('complete', Path(self.refs['complete']['path']), original)
        self.save('verifier_execution', self.root / 'actual-verifier.json', dict(self.values['verifier_execution'], complete=cp))
        archive = self.destination / 'closed-raw.tar.gz'; archive.write_bytes(archive.read_bytes() + b'changed')
        with self.assertRaises(ValueError): self.a['adapt'](self.refs)


if __name__ == '__main__': unittest.main()
