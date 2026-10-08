"""Synthetic CPU metadata fixtures; no actual assets, model, GPU, or SSH calls."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'search_r1/baselines/probability_probe_contracts.py'
spec = importlib.util.spec_from_file_location('probability_probe_contracts', MODULE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def temporary_directory():
    # macOS /var is a symlink; fixtures use the actual canonical local root.
    return tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))


def config():
    result = {}
    for key, value in P.FIXED_CONFIG.items():
        current = result
        parts = key.split('.')
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def selection():
    inventory, selections = {}, []
    for rank in range(8):
        raw_key, monitor_key = 'raw-%d' % rank, 'monitor-%d' % rank
        inventory[raw_key] = dict(path=str(P.ASSET_ROOT / ('synthetic/rank-%d-raw.pt' % rank)),
            sha256='a' * 64, rank=rank, kind='raw', row_count=64,
            original_micro4_groups=[list(range(start, start + 4)) for start in range(0, 64, 4)])
        inventory[monitor_key] = dict(path=str(P.ASSET_ROOT / ('synthetic/rank-%d-monitor.pt' % rank)),
            sha256='b' * 64, rank=rank, kind='monitor', row_count=32)
        selections.append(dict(rank=rank, raw_groups=[dict(asset=raw_key, source_group=0,
            row_indices=[0, 1, 2, 3])], monitor=dict(asset=monitor_key, row_indices=list(range(32)))))
    identity = {field: 'c' * 64 for field in P.IDENTITY_SHA_FIELDS}
    identity.update(core_commit='3d10326fe8cde3bf7a62bc9c7567b308cc6afc3a',
        source_files=[dict(path=str(P.ASSET_ROOT / 'synthetic/source.py'), sha256='d' * 64)])
    identity['assets_sha256'] = P.files_aggregate({asset['path']: asset['sha256'] for asset in inventory.values()})
    manifest = dict(schema_version=1, asset_root=str(P.ASSET_ROOT), identity=copy.deepcopy(identity),
                    selections=selections, status=P.UNFROZEN, frozen=False)
    return manifest, identity, inventory


def parameters():
    serialized = ['model.embed_tokens.weight', 'lm_head.weight'] + ['synthetic.%03d.weight' % i for i in range(433)]
    unique = [name for name in serialized if name != 'lm_head.weight']
    mapping = {name: name for name in serialized}
    mapping['lm_head.weight'] = 'model.embed_tokens.weight'
    return serialized, unique, mapping


class Tests(unittest.TestCase):
    def test_finite_strict_json_round_trip(self):
        value = {'null': None, 'list': [True, 2, .125, '\u4e2d\u6587']}
        with temporary_directory() as directory:
            path = Path(directory) / 'receipt.json'
            path.write_bytes(P.strict_json_bytes(value))
            self.assertEqual(P.strict_json_load(path), value)

    def test_nested_nonfinite_json_is_refused_before_artifact_write(self):
        with temporary_directory() as directory:
            for value in (float('nan'), float('inf'), float('-inf')):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    P.atomic_write_json_no_clobber(Path(directory) / 'receipt.json', {'nested': [value]}, directory)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_json_nonfinite_literals_overflow_and_duplicate_keys_refused(self):
        with temporary_directory() as directory:
            path = Path(directory) / 'input.json'
            for payload in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}', b'{"a":1,"a":2}'):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    P.strict_json_load(path)

    def test_non_json_key_and_value_types_refused(self):
        for value in ({1: 'integer key'}, {'tuple': (1, 2)}, {'bytes': b'raw'}, {'set': {1}}):
            with self.assertRaises(ValueError):
                P.strict_json_bytes(value)

    def test_host_requires_exact_h100_9(self):
        self.assertEqual(P.require_local_host('h100-9.cluster.local'), 'h100-9.cluster.local')
        for host in ('a800', 'h100-8', 'h100-9-other', 'MacBook', 'h1009'):
            with self.assertRaises(ValueError):
                P.require_local_host(host)

    def test_path_relative_traversal_outside_and_symlink_escape_refused(self):
        with temporary_directory() as root, temporary_directory() as outside:
            path = Path(root) / 'same.txt'; path.write_text('same node')
            self.assertEqual(P.local_asset_path(path, root), path)
            escaped = Path(outside) / 'foreign.txt'; escaped.write_text('outside')
            link = Path(root) / 'escape'; link.symlink_to(outside, target_is_directory=True)
            for candidate in ('relative.pt', Path(root) / '..' / Path(outside).name / 'foreign.txt', escaped, link / 'foreign.txt'):
                with self.assertRaises(ValueError):
                    P.local_asset_path(candidate, root)

    def test_symlink_root_is_refused(self):
        with temporary_directory() as directory:
            root = Path(directory) / 'real'; root.mkdir()
            link = Path(directory) / 'root'; link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'root must not contain a symlink'):
                P.local_asset_path(link / 'new.json', link, must_exist=False)

    def test_atomic_publication_is_complete_and_preserves_existing_target(self):
        with temporary_directory() as directory:
            target = Path(directory) / 'receipt.json'
            receipt = P.atomic_write_json_no_clobber(target, {'actual_optimizer_writes': 0}, directory)
            original = target.read_bytes()
            self.assertEqual(receipt['sha256'], hashlib.sha256(original).hexdigest())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                P.atomic_write_json_no_clobber(target, {'actual_optimizer_writes': 1}, directory)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['receipt.json'])

    def test_atomic_target_dangling_symlink_refused(self):
        with temporary_directory() as directory:
            target = Path(directory) / 'dangling.json'; target.symlink_to('missing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                P.atomic_write_bytes_no_clobber(target, b'never publish', directory)
            self.assertTrue(target.is_symlink())

    def test_competing_publishers_have_exactly_one_winner_and_no_temp_files(self):
        with temporary_directory() as directory:
            target = Path(directory) / 'one.bin'
            def publish(number):
                try:
                    receipt = P.atomic_write_bytes_no_clobber(target, ('payload%d' % number).encode(), directory)
                    return number, receipt
                except (ValueError, FileExistsError):
                    return number, None
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(publish, range(8)))
            winners = [(number, receipt) for number, receipt in results if receipt is not None]
            self.assertEqual(len(winners), 1)
            number, receipt = winners[0]
            self.assertEqual(target.read_bytes(), ('payload%d' % number).encode())
            self.assertEqual(receipt['sha256'], P.file_sha(target))
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['one.bin'])

    def test_target_symlink_created_after_first_exists_check_cannot_redirect_publication(self):
        with temporary_directory() as directory:
            target = Path(directory) / 'evidence.json'
            redirected = Path(directory) / 'redirected.json'
            original = P.local_asset_path
            def concurrent_link(path, root, must_exist):
                target.symlink_to(redirected)
                return original(path, root, must_exist)
            with mock.patch.object(P, 'local_asset_path', side_effect=concurrent_link):
                with self.assertRaises(FileExistsError):
                    P.atomic_write_bytes_no_clobber(target, b'forbidden redirection', directory)
            self.assertTrue(target.is_symlink())
            self.assertFalse(redirected.exists())
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['evidence.json'])

    def test_exact_complete_files_use_original_default_json_aggregate(self):
        with temporary_directory() as directory:
            bundle = Path(directory) / 'hf'; bundle.mkdir()
            (bundle / 'config.json').write_bytes(b'config')
            (bundle / 'weights.safetensors').write_bytes(b'synthetic bytes')
            hashes = {name: P.file_sha(bundle / name) for name in ('config.json', 'weights.safetensors')}
            expected = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            compact = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.assertNotEqual(expected, compact)
            result = P.bind_exact_files(bundle, hashes, expected, directory)
            self.assertTrue(result['full_files_read'])
            self.assertFalse(result['acceptance'])

    def test_exact_file_set_rejects_extra_missing_changed_and_bad_aggregate(self):
        with temporary_directory() as directory:
            bundle = Path(directory) / 'hf'; bundle.mkdir()
            path = bundle / 'model.bin'; path.write_bytes(b'frozen')
            hashes = {'model.bin': P.file_sha(path)}; aggregate = P.files_aggregate(hashes)
            (bundle / 'unbound.bin').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'file set mismatch'):
                P.bind_exact_files(bundle, hashes, aggregate, directory)
            (bundle / 'unbound.bin').unlink(); path.unlink()
            with self.assertRaisesRegex(ValueError, 'file set mismatch'):
                P.bind_exact_files(bundle, hashes, aggregate, directory)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes SHA mismatch'):
                P.bind_exact_files(bundle, hashes, aggregate, directory)
            with self.assertRaisesRegex(ValueError, 'aggregate disagrees'):
                P.bind_exact_files(bundle, hashes, '0' * 64, directory)

    def test_exact_file_bundle_refuses_symlink_even_inside_root(self):
        with temporary_directory() as directory:
            bundle = Path(directory) / 'hf'; bundle.mkdir()
            path = Path(directory) / 'weights'; path.write_bytes(b'weights')
            (bundle / 'model.bin').symlink_to(path)
            hashes = {'model.bin': P.file_sha(path)}
            with self.assertRaisesRegex(ValueError, 'symlink'):
                P.bind_exact_files(bundle, hashes, P.files_aggregate(hashes), directory)

    def test_source_binding_reads_full_bytes_and_rejects_changed_duplicate_escape(self):
        with temporary_directory() as directory:
            path = Path(directory) / 'source.py'; path.write_bytes(b'original source')
            entry = dict(path=str(path), sha256=P.file_sha(path))
            self.assertEqual(P.bind_sources([entry], directory), [entry])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                P.bind_sources([entry, entry], directory)
            path.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                P.bind_sources([entry], directory)
            with self.assertRaises(ValueError):
                P.bind_sources([dict(path='/a800/source.py', sha256='a' * 64)], directory)

    def test_original_fixed_protocol_and_local_batch_numbers(self):
        receipt = P.validate_fixed_config(config())
        self.assertEqual((receipt['original_local_old_micro'], receipt['original_local_actor_micro'],
                          receipt['original_local_mini']), (4, 2, 48))
        self.assertFalse(receipt['acceptance'])

    def test_config_batch_temperature_offload_dynamic_world_and_boolean_changes_refused(self):
        mutations = [('data', 'train_batch_size', 641),
                     ('actor', 'ppo_micro_batch_size', 64), ('actor', 'use_dynamic_bsz', True),
                     ('rollout', 'temperature', .95), ('rollout', 'top_p', .95),
                     ('rollout', 'log_prob_micro_batch_size', 16), ('rollout', 'n', True)]
        for section, key, value in mutations:
            cfg = config(); target = cfg['data'] if section == 'data' else cfg['actor_rollout_ref'][section]
            target[key] = value
            with self.assertRaisesRegex(ValueError, 'Changed fixed config'):
                P.validate_fixed_config(cfg)
        cfg = config(); cfg['actor_rollout_ref']['actor']['fsdp_config']['param_offload'] = False
        with self.assertRaises(ValueError):
            P.validate_fixed_config(cfg)
        with self.assertRaises(ValueError):
            P.validate_fixed_config(config(), world_size=4)
        with self.assertRaises(ValueError):
            P.validate_fixed_config(config(), additional_expected={'data.train_batch_size': 512})

    def test_missing_config_and_nonfinite_unknown_fields_refused(self):
        cfg = config(); del cfg['actor_rollout_ref']['rollout']['temperature']
        with self.assertRaisesRegex(ValueError, 'Missing fixed config'):
            P.validate_fixed_config(cfg)
        cfg = config(); cfg['unexpected'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            P.validate_fixed_config(cfg)

    def test_serialized435_unique434_are_separate_without_alias_or_byte_proof(self):
        receipt = P.validate_parameter_inventory(*parameters())
        self.assertEqual(receipt['serialized_tensor_keys'], 435)
        self.assertEqual(receipt['unique_parameters'], 434)
        self.assertTrue(receipt['all_serialized_keys_bound'])
        self.assertFalse(receipt['same_gather_window_alias_proven'])
        self.assertFalse(receipt['full_tensor_bytes_read'])
        self.assertFalse(receipt['acceptance'])

    def test_omitted_head_duplicate_unique_nonexhaustive_map_and_wrong_alias_refused(self):
        serialized, unique, mapping = parameters()
        with self.assertRaisesRegex(ValueError, 'Serialized key count'):
            P.validate_parameter_inventory(serialized[1:], unique, mapping)
        with self.assertRaises(ValueError):
            P.validate_parameter_inventory(serialized, unique + ['lm_head.weight'], mapping)
        broken = dict(mapping); broken.pop('lm_head.weight')
        with self.assertRaisesRegex(ValueError, 'Exhaustive serialized'):
            P.validate_parameter_inventory(serialized, unique, broken)
        broken = dict(mapping); broken['lm_head.weight'] = unique[-1]
        with self.assertRaisesRegex(ValueError, 'structural alias differs'):
            P.validate_parameter_inventory(serialized, unique, broken)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            P.validate_parameter_inventory(serialized, unique, mapping, 434, 434)

    def test_unfrozen_selection_default_cannot_run(self):
        manifest, identity, inventory = selection()
        del manifest['status']; del manifest['frozen']
        receipt = P.validate_selection(manifest, identity, inventory)
        self.assertEqual(receipt['status'], P.UNFROZEN)
        self.assertFalse(receipt['runtime_selection_precondition'])
        self.assertFalse(receipt['real_cpu_evidence_bytes_verified'])
        with self.assertRaisesRegex(ValueError, 'execution forbidden'):
            P.validate_selection(manifest, identity, inventory, allow_frozen=True)

    def test_actual_snapshot64_rows_not_confused_with_selected20(self):
        manifest, identity, inventory = selection()
        for chosen in manifest['selections']:
            chosen['raw_groups'] = [dict(asset='raw-%d' % chosen['rank'], source_group=group,
                row_indices=inventory['raw-%d' % chosen['rank']]['original_micro4_groups'][group])
                for group in (0, 4, 8, 12, 15)]
        receipt = P.validate_selection(manifest, identity, inventory)
        self.assertEqual(receipt['raw_selected_rows_per_rank'], [20] * 8)
        self.assertEqual(receipt['monitor_selected_rows_per_rank'], [32] * 8)

    def test_original_partial_last_group_is_complete_not_cropped(self):
        manifest, identity, inventory = selection()
        inventory['raw-0']['row_count'] = 66
        inventory['raw-0']['original_micro4_groups'].append([64, 65])
        manifest['selections'][0]['raw_groups'] = [dict(asset='raw-0', source_group=16, row_indices=[64, 65])]
        self.assertEqual(P.validate_selection(manifest, identity, inventory)['raw_selected_rows_per_rank'][0], 2)

    def test_raw_groups_crop_reorder_duplicate_out_of_range_excess_and_bool_refused(self):
        for rows in ([0, 1], [3, 2, 1, 0], [0, 1, 1, 3], [64, 65, 66, 67], [0, True, 2, 3]):
            manifest, identity, inventory = selection()
            manifest['selections'][0]['raw_groups'][0]['row_indices'] = rows
            with self.assertRaises(ValueError):
                P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection()
        group = manifest['selections'][0]['raw_groups'][0]
        manifest['selections'][0]['raw_groups'].append(copy.deepcopy(group))
        with self.assertRaises(ValueError):
            P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection()
        manifest['selections'][0]['raw_groups'] = [dict(asset='raw-0', source_group=group,
            row_indices=inventory['raw-0']['original_micro4_groups'][group]) for group in range(6)]
        with self.assertRaisesRegex(ValueError, 'exceeds 20'):
            P.validate_selection(manifest, identity, inventory)

    def test_selection_schema_boolean_and_original_group_order_refused(self):
        manifest, identity, inventory = selection(); manifest['schema_version'] = True
        with self.assertRaisesRegex(ValueError, 'selection schema'):
            P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection()
        with self.assertRaisesRegex(ValueError, 'must be a boolean'):
            P.validate_selection(manifest, identity, inventory, allow_frozen=1)
        manifest['selections'][0]['raw_groups'] = [dict(asset='raw-0', source_group=number,
            row_indices=inventory['raw-0']['original_micro4_groups'][number]) for number in (3, 1)]
        with self.assertRaisesRegex(ValueError, 'groups reordered'):
            P.validate_selection(manifest, identity, inventory)

    def test_monitor_crop_reorder_wrong_rank_missing_rank_and_asset_mismatch_refused(self):
        for rows in (list(range(31)), list(reversed(range(32)))):
            manifest, identity, inventory = selection()
            manifest['selections'][0]['monitor']['row_indices'] = rows
            with self.assertRaises(ValueError):
                P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection(); manifest['selections'][0]['monitor']['asset'] = 'monitor-1'
        with self.assertRaisesRegex(ValueError, 'rank mismatch'):
            P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection(); manifest['selections'][0]['rank'] = 1
        with self.assertRaises(ValueError):
            P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection(); manifest['selections'].pop()
        with self.assertRaisesRegex(ValueError, 'eight rank selections'):
            P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection(); inventory['raw-0']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'aggregate differs'):
            P.validate_selection(manifest, identity, inventory)

    def test_selection_ready_config_parent_source_bytes_bound_to_expected_identity(self):
        for field in P.IDENTITY_SHA_FIELDS:
            manifest, identity, inventory = selection(); manifest['identity'][field] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'identities changed'):
                P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection(); manifest['identity']['source_files'][0]['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'identities changed'):
            P.validate_selection(manifest, identity, inventory)

    def test_frozen_metadata_requires_both_cpu_evidence_bindings_and_never_gpu_acceptance(self):
        manifest, identity, inventory = selection(); manifest.update(status=P.FROZEN, frozen=True)
        with self.assertRaisesRegex(ValueError, 'actual position/action'):
            P.validate_selection(manifest, identity, inventory, allow_frozen=True)
        manifest['cpu_results'] = {name: dict(path=str(P.ASSET_ROOT / ('synthetic/' + name + '.json')),
            sha256='e' * 64) for name in ('position', 'action')}
        receipt = P.validate_selection(manifest, identity, inventory, allow_frozen=True)
        self.assertTrue(receipt['runtime_selection_precondition'])
        self.assertFalse(receipt['real_cpu_evidence_bytes_verified'])
        self.assertFalse(receipt['real_gpu_execution_verified'])
        self.assertFalse(receipt['acceptance'])

    def test_selection_cannot_hide_foreign_assets_in_unused_inventory(self):
        manifest, identity, inventory = selection(); inventory['raw-7']['path'] = '/a800/foreign.pt'
        with self.assertRaisesRegex(ValueError, 'escaped same-node root'):
            P.validate_selection(manifest, identity, inventory)
        manifest, identity, inventory = selection(); manifest['asset_root'] = '/a800'
        with self.assertRaisesRegex(ValueError, 'asset root differs'):
            P.validate_selection(manifest, identity, inventory)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(status='SOURCE_CONTRACT_CPU_FIXTURES_PASS' if result.wasSuccessful() else 'FAILED',
        tests_run=result.testsRun, module_sha256=P.file_sha(MODULE), test_sha256=P.file_sha(__file__),
        real_asset_reads=False, actual_selection_frozen=False, gpu_calls=False, remote_calls=False,
        acceptance=False), sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
