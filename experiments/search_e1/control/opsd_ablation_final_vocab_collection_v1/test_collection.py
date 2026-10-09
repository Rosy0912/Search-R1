"""Tiny real filesystem/subprocess fixtures only; no production/SSH/NumPy."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect
import capture


def raw(value):
    return json.dumps(value, sort_keys=True).encode()


class CollectionIO(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.root = Path(self.temp.name) / 'study'; self.root.mkdir()
        self.root_patch = patch.object(collect, 'STUDY_ROOT', self.root); self.root_patch.start()
        self.uid_patch = patch.object(collect.os, 'getuid', return_value=1005); self.uid_patch.start()
        self.request = dict(schema='opsd-ablation-final-vocab-collection-inputs-v1',
                            study_root=str(self.root), study_id='tiny-study', groups=[], accepted_source_inventories=[])
        for n, variant in enumerate(collect.VARIANTS):
            run = 'run-' + variant; runs = self.root / ('group-' + variant) / 'runs'
            (runs / 'outers').mkdir(parents=True)
            owner = dict(pid=100+n, uid=1005, starttick=1000+n, run_id=run, argv=['original-controller', run])
            audit = dict(schema='same-uid-owned-process-audit-v2', run_id=run, uid=1005, controller_starttick=owner['starttick'],
                         accepted_no_owned_or_opaque=True, blockers=[], no_exit_code_inferred_from_absence=True,
                         known_owned=[owner], ended_unix=1)
            owned = dict(accepted=True, no_signals_sent=True, audit=audit)
            config = dict(schema='opsd-ablation-config-v1', variant=variant, run_id=run, study_id='tiny-study',
                          scientific=dict(outer_iterations=10), paths=dict(output_root=str(runs)), precision=dict(mode='bf16'),
                          logging=dict(full_vocab_study_root=str(self.root / 'full-vocabulary')))
            for field in ('source_model_manifest','runtime_manifest','common_initial_adapter_manifest','common_cohort_manifest','helper_source_manifest'):
                config[field] = dict(sha256='1'*64)
            cfg = self.save(self.root / ('group-' + variant) / 'receipts/config.json', config)
            contract = dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
                            variant=variant, precision='bf16', mixture_alpha=.1 if variant in ('M','J') else 0.)
            outers = []; previous = None
            for index in range(10):
                cp = dict(path=str(runs / 'checkpoints' / ('update-%02d/manifest.json' % index)), bytes=1, sha256='2'*64)
                value = dict(schema='opsd-online-outer-complete-v1', accepted=True, run_id=run, outer_index=index,
                             optimizer_updates=index+1, checkpoint=cp, contract=contract)
                if previous is not None:
                    value['outer_old_adapter'] = previous
                outers.append(self.save(runs / 'outers' / ('outer-%05d' % index) / 'training/outer-complete.json', value)); previous = cp
            status = self.save(runs / 'status.json', dict(schema='opsd-ablation-controller-status-v1', phase='TRAIN_COMPLETE',
                       outer_iterations=10, variant=variant, run_id=run, config_sha256=cfg['sha256'], contract=contract,
                       actual_optimizer_updates=10, latest_checkpoint=previous, owner=owner,
                       resource_release=dict(all_owned_release=owned, resource_admission=dict(owned_gpu_processes=[]))))
            td = self.root / 'original-terminals' / variant
            wait = self.save(td / 'actual-child-exit.json', dict(actual_wait_returncode=0, run_id=run, no_signals=True,
                             signal_termination_not_exit0=False, identity=owner))
            release = self.save(td / 'actual-release.json', dict(accepted=True, owned=owned,
                                after=dict(accepted=True, phase='cpu-merge', run_id=run, owned_gpu_processes=[], checked_unix=1)))
            terminal = self.save(td / 'group-terminal.json', dict(schema='opsd-ablation-group-foreground-terminal-v1',
                                 accepted=True, actual_exit_code=0, no_signals=True, signal_termination_not_exit0=False,
                                 run_id=run, config_sha256=cfg['sha256'], status=status, controller=owner,
                                 actual_child_exit=wait, actual_release=release))
            self.request['groups'].append(dict(variant=variant, run_id=run, config=cfg, status=status, terminal=terminal, outers=outers))
        self.ledger = self.root / 'full-vocabulary/ledger.json'; self.save(self.ledger, {'bytes':0,'reservations':{}})
        self.npy = self.root / 'group-C0/runs/outers/outer-00000/full-vocabulary/tiny.npy'
        self.npy.parent.mkdir(); self.npy.write_bytes(b'tiny-NPY-fixture-not-a-finite-proof')

    def tearDown(self):
        self.uid_patch.stop(); self.root_patch.stop(); self.temp.cleanup()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw(value))
        return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def accepted(self, file=None):
        f = file or dict(collect.current(self.npy), path=str(self.npy), sha256=hashlib.sha256(self.npy.read_bytes()).hexdigest())
        content = raw(dict(accepted=True, files=[f])); ref = dict(path='/private/accepted/source-inventory.json', bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        g = self.request['groups'][0]
        self.request['accepted_source_inventories'] = [dict(reference=ref, content_base64=base64.b64encode(content).decode(),
                    variant=g['variant'], run_id=g['run_id'], config_sha256=g['config']['sha256'])]

    def test_closed_scope_reuses_sha_and_includes_unknown_failed_retry(self):
        self.accepted()
        extra = self.npy.parent / 'failed' / 'retry-2.npy'; extra.parent.mkdir(); extra.write_bytes(b'unknown-extra')
        original = collect.hash_npy
        with patch.object(collect, 'hash_npy', wraps=original) as hashed:
            result = collect.collect(self.request)
        self.assertEqual(hashed.call_count, 1)
        self.assertEqual({f['path'] for f in result['files']}, {str(self.npy), str(extra)})
        self.assertEqual(result['sha_reuse_count'], 1); self.assertEqual(result['fresh_sha_count'], 1)
        self.assertFalse(result['finite_checked']); self.assertEqual(len(result['closures']), 7)
        self.assertEqual(base64.b64decode(result['ledger_content_base64']), self.ledger.read_bytes())

    def test_missing_closure_refuses_before_npy_scan(self):
        Path(self.request['groups'][-1]['terminal']['path']).unlink()
        with patch.object(collect, 'enumerate_npy') as scan:
            with self.assertRaises(OSError): collect.collect(self.request)
        scan.assert_not_called()

    def test_published_generic_inventory_needs_separate_exact_acceptance(self):
        self.accepted(); item = self.request['accepted_source_inventories'][0]
        inv = json.loads(base64.b64decode(item['content_base64'])); del inv['accepted']
        inv.update(schema='opsd-ablation-closed-raw-inventory-v1', variant=item['variant'], run_id=item['run_id'],
                   config=dict(sha256=item['config_sha256']))
        content = raw(inv); item['content_base64'] = base64.b64encode(content).decode()
        item['reference'].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        with self.assertRaises(KeyError): collect.collect(self.request)
        acceptance = raw(dict(schema='opsd-ablation-closed-raw-compressed-local-backup-v1', accepted=True,
                    inventory_sha256=item['reference']['sha256'], actual_ssh_returncode=0,
                    all_local_decompressed_member_SHA_match=True, original_NPY_bytes_not_reserialized=True,
                    run_id=item['run_id'], variant=item['variant'], config=dict(sha256=item['config_sha256'])))
        item['acceptance'] = dict(reference=dict(path='/private/accepted/complete.json', bytes=len(acceptance),
                                 sha256=hashlib.sha256(acceptance).hexdigest()), content_base64=base64.b64encode(acceptance).decode())
        result = collect.collect(self.request)
        self.assertEqual(result['sha_reuse_count'], 1)
        self.assertEqual(result['files'][0]['sha256_origin']['acceptance'], item['acceptance']['reference'])

    def test_nonzero_actual_wait_refuses(self):
        g = self.request['groups'][0]; terminal = json.loads(Path(g['terminal']['path']).read_bytes())
        wait = json.loads(Path(terminal['actual_child_exit']['path']).read_bytes()); wait['actual_wait_returncode'] = 9
        terminal['actual_child_exit'] = self.save(Path(terminal['actual_child_exit']['path']), wait)
        g['terminal'] = self.save(Path(g['terminal']['path']), terminal)
        with self.assertRaisesRegex(ValueError, 'parent wait0'): collect.collect(self.request)

    def test_symlink_is_refused(self):
        (self.npy.parent / 'escape.npy').symlink_to(self.npy)
        with self.assertRaisesRegex(ValueError, 'Symlink'): collect.collect(self.request)

    def test_changed_npy_between_enumerations_is_refused(self):
        original = collect.enumerate_npy; calls = []
        def enumerate_changed(roots):
            if calls: self.npy.write_bytes(b'changed')
            calls.append(1); return original(roots)
        with patch.object(collect, 'enumerate_npy', side_effect=enumerate_changed):
            with self.assertRaisesRegex(ValueError, 'between enumerations'): collect.collect(self.request)

    def test_changed_ledger_is_refused(self):
        original = collect.enumerate_npy
        def enumerate_changed(roots):
            value = original(roots); self.ledger.write_bytes(b'{"changed":true}'); return value
        with patch.object(collect, 'enumerate_npy', side_effect=enumerate_changed):
            with self.assertRaisesRegex(ValueError, 'ledger bytes/stat changed'): collect.collect(self.request)

    def test_unknown_accepted_metadata_and_changed_reuse_are_refused(self):
        f = dict(collect.current(self.npy), path=str(self.npy), sha256=hashlib.sha256(self.npy.read_bytes()).hexdigest())
        bad = dict(f); del bad['inode']; self.accepted(bad)
        with self.assertRaisesRegex(ValueError, 'Unknown accepted source metadata'): collect.collect(self.request)
        self.accepted(f); self.npy.write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError, 'metadata changed'): collect.collect(self.request)

    def test_wrapper_preserves_actual_nonzero_wait_without_acceptance_receipts(self):
        out = Path(self.temp.name) / 'capture-failure'
        argv = [sys.executable, '-B', '-c', 'import sys; print("synthetic failure"); sys.exit(7)']
        with self.assertRaisesRegex(ValueError, 'nonzero'): capture.capture(argv, b'fixture', out, '1'*64)
        execution = json.loads((out / 'ssh-execution.private.json').read_text())
        self.assertEqual(execution['actual_returncode'], 7); self.assertTrue(execution['actual_wait'])
        self.assertFalse((out / 'physical_inventory_execution.private.json').exists())
        self.assertFalse((out / 'ledger-copy-execution.private.json').exists())

    def test_wrapper_wait0_keeps_exact_stdout_and_ledger_copy(self):
        result = collect.collect(self.request); request_raw = raw(self.request)
        result.update(request_sha256=hashlib.sha256(request_raw).hexdigest(), collector_source=dict(sha256='1'*64))
        response = json.dumps(result)
        argv = [sys.executable, '-B', '-c', 'import sys; sys.stdout.write(sys.argv[1])', response]
        out = Path(self.temp.name) / 'capture-success'; capture.capture(argv, request_raw, out, '1'*64)
        refs = json.loads((out / 'consumer-copy-refs.private.json').read_text())
        self.assertEqual((out / 'ledger.original.json').read_bytes(), self.ledger.read_bytes())
        self.assertEqual(refs['physical_inventory']['sha256'], hashlib.sha256(response.encode()).hexdigest())
        receipt = json.loads((out / 'ledger-copy-execution.private.json').read_text())
        self.assertEqual(receipt['actual_returncode'], 0); self.assertEqual(receipt['source_before'], receipt['source_after'])


if __name__ == '__main__':
    unittest.main()
