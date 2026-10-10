"""Full bind_inputs fixtures with the original native publication admission.

All files below are small synthetic metadata. Only verify_runtime/verify_common
are fixture stubs; real publication byte checks, scope checks, source imports and
native config/science validation run unchanged. No tensors/models are loaded.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

try:
    from . import reader
    from .common import bind_helper, exclusive, read
except ImportError:
    import reader
    from common import bind_helper, exclusive, read


HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent


def ref(path):
    raw = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


NATIVE = bind_helper(CONTROL / 'opsd_signal_weight_v1',
                     ref(CONTROL / 'opsd_signal_weight_v1' / 'source-manifest.json'),
                     ('prepare', 'evaluation'))


class Topology:
    """Five separate publications, sharing only the exact published00fec source."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.study = self.root / 'study'
        self.study.mkdir(); self.study_id = 'synthetic-four-distinct-publications'
        self.configs = {}; self.config_refs = {}; self.group_publications = {}
        self.counter = 0
        self.sources = {}
        entries = [('reader', HERE, 'a' * 40),
                   ('original', CONTROL / 'opsd_signal_weight_v1', reader.COMMITS['original']),
                   ('completion', CONTROL / 'opsd_signal_weight_completion_v1', reader.COMMITS['completion']),
                   ('analysis', CONTROL / 'opsd_signal_weight_analysis_v1', reader.COMMITS['analysis'])]
        for name, directory, commit in entries:
            manifest = ref(directory / 'source-manifest.json')
            publication = self.store(name + '-publication', dict(
                accepted=True, branch='search-e1', source_commit=commit,
                remote_readback_sha=commit, source_manifest_sha256=manifest['sha256']))
            self.sources[name + '_source'] = dict(helper_root=str(directory), source_manifest=manifest,
                                                 publication=publication)
        common = self.store('synthetic-common', dict(synthetic=True, not_model_data=True))
        for index, variant in enumerate(reader.VARIANTS):
            cfg = NATIVE['prepare'].template(str(self.study / ('group-' + variant)), variant,
                                            self.study_id, str(self.study))
            cfg['run_id'] = ('%032x' % (index + 1))
            cfg['source_commit'] = reader.COMMITS['original']
            cfg['paths']['helper_root'] = self.sources['original_source']['helper_root']
            for name in ('source_model_manifest', 'runtime_manifest', 'train_identity', 'validation_identity',
                         'common_cohort_manifest', 'common_initial_adapter_manifest'):
                cfg[name] = copy.deepcopy(common)
            cfg['common_initial_adapter_tensor_sha256'] = 'f' * 64
            cfg['helper_source_manifest'] = copy.deepcopy(self.sources['original_source']['source_manifest'])
            cfg['authorization'] = dict(four_signal_groups_25_outer=True, four_gpu_execution=True)
            for gate in ('storage', 'resource_transition', 'retriever_admission'):
                cfg['gates'][gate] = copy.deepcopy(common)
            publication = self.store(variant + '-publication', dict(
                accepted=True, branch='search-e1', source_commit=reader.COMMITS['original'],
                remote_readback_sha=reader.COMMITS['original'],
                source_manifest_sha256=cfg['helper_source_manifest']['sha256'],
                scope=dict(study_id=self.study_id, variant=variant, run_id=cfg['run_id'],
                           output_root=cfg['paths']['output_root'])))
            cfg['gates']['publication'] = publication
            self.configs[variant] = cfg; self.group_publications[variant] = publication
        self.request = dict(schema=reader.SCHEMA, analysis_run_id='synthetic-new-independent-reader',
                            **copy.deepcopy(self.sources))
        # No target/common_zero is provided: bind-only must not read endpoint evidence.
        self.pin_configs_and_study()

    def store(self, name, value):
        self.counter += 1
        return exclusive(self.root / ('%03d-%s.json' % (self.counter, name)), value)

    def pin_configs_and_study(self):
        for variant, cfg in self.configs.items():
            self.config_refs[variant] = self.store(variant + '-config', cfg)
        self.request['configs'] = copy.deepcopy(self.config_refs)
        original = self.sources['original_source']
        self.request['study_manifest'] = self.store('study-manifest', dict(
            study_id=self.study_id, study_base=str(self.study),
            helper_root=original['helper_root'], helper_source_manifest=original['source_manifest'],
            publication=original['publication'],
            groups=[dict(variant=v, config=self.config_refs[v]) for v in reader.VARIANTS]))

    def change_publication(self, variant, function):
        old, _ = read(self.configs[variant]['gates']['publication'])
        function(old)
        self.configs[variant]['gates']['publication'] = self.store(variant + '-changed-publication', old)
        self.pin_configs_and_study()

    def bind(self):
        with mock.patch.object(reader, 'STUDY_ROOT', str(self.study)), \
             mock.patch.dict(os.environ, {'E1_RUN_ID': self.request['analysis_run_id']}), \
             mock.patch.object(NATIVE['prepare'], 'verify_runtime', return_value=dict(
                 fixture_stub=True, all305_equal=True)) as runtime, \
             mock.patch.object(NATIVE['prepare'], 'verify_common', return_value=dict(
                 fixture_stub=True, cohort=dict(accepted=True))) as common:
            result = reader.bind_inputs(self.request)
            runtime.assert_called_once(); common.assert_called_once()
            return result

    def preflight(self):
        request_ref = self.store('bindings-only-request', self.request)
        with mock.patch.object(reader, 'STUDY_ROOT', str(self.study)), \
             mock.patch.dict(os.environ, {'E1_RUN_ID': self.request['analysis_run_id'], 'CUDA_VISIBLE_DEVICES': ''}), \
             mock.patch.object(reader.os, 'getuid', return_value=1001), \
             mock.patch.object(NATIVE['prepare'], 'verify_runtime', return_value=dict(
                 fixture_stub=True, all305_equal=True)), \
             mock.patch.object(NATIVE['prepare'], 'verify_common', return_value=dict(
                 fixture_stub=True, cohort=dict(accepted=True))), \
             mock.patch.object(reader, 'verify_target', side_effect=AssertionError('Endpoint evidence is forbidden')), \
             mock.patch.object(reader, 'verify_zero', side_effect=AssertionError('Zero evidence is forbidden')):
            return reader.bind_only(request_ref)


class BindInputsTopologyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.topology = Topology(self.temp.name)

    def reject(self, mutate, message):
        mutate(self.topology)
        with self.assertRaisesRegex((ValueError, RuntimeError), message):
            self.topology.bind()

    def test_real_topology_accepts_studywide_and_four_distinct_scoped_publications(self):
        sources, study, configs, refs, *_ = self.topology.bind()
        native_pub = self.topology.sources['original_source']['publication']
        group_refs = [configs[v]['gates']['publication'] for v in reader.VARIANTS]
        self.assertEqual(len({r['path'] for r in [native_pub] + group_refs}), 5)
        self.assertEqual(sources['original']['publication'], native_pub)
        self.assertEqual(set(sources['config_publications']), set(reader.VARIANTS))
        for variant in reader.VARIANTS:
            evidence = sources['config_publications'][variant]
            self.assertEqual(evidence['reference'], configs[variant]['gates']['publication'])
            self.assertEqual(evidence['admission'], dict(source_commit=reader.COMMITS['original'],
                source_manifest_sha256=reader.ORIGINAL_SHA))
        self.assertEqual(study, self.topology.request['study_manifest'])
        self.assertEqual(refs, self.topology.request['configs'])
        self.assertNotIn('torch', sys.modules)

    def test_bind_only_with_no_endpoint_fields_has_explicit_nonendpoint_scope(self):
        before = set(Path(self.temp.name).rglob('*'))
        result = self.topology.preflight()
        after = set(Path(self.temp.name).rglob('*'))
        # Only the fixture explicitly created the request; the reader wrote nothing.
        self.assertEqual(len(after - before), 1)
        self.assertEqual(result['schema'], 'opsd-signal-endpoint-bindings-v2')
        self.assertTrue(result['accepted'] and result['no_sampling'] and result['no_output_files_written'])
        for key in ('endpoint_accepted', 'full_study_accepted', 'source100_weight_payload_fresh_SHA',
                    'fresh_numeric_Adam_or_tensor_deserialization', 'torch_imported'):
            self.assertFalse(result[key])
        self.assertNotIn('target', self.topology.request)
        self.assertNotIn('postraw', result); self.assertNotIn('optimizer_updates', result)
        self.assertNotIn('formal_endpoint_slots', result)

    def test_wrong_run_rejected_by_real_native_scope(self):
        self.reject(lambda t: t.change_publication('G_POS', lambda p: p['scope'].update(run_id='f' * 32)),
                    'Receipt does not bind this explicit variant/run/output')

    def test_wrong_variant_rejected_by_real_native_scope(self):
        self.reject(lambda t: t.change_publication('G_NEG', lambda p: p['scope'].update(variant='G_POS')),
                    'Receipt does not bind this explicit variant/run/output')

    def test_wrong_output_root_rejected_by_real_native_scope(self):
        self.reject(lambda t: t.change_publication('H_POS', lambda p: p['scope'].update(output_root=str(t.root / 'foreign'))),
                    'Receipt does not bind this explicit variant/run/output')

    def test_wrong_study_rejected_by_real_native_scope(self):
        self.reject(lambda t: t.change_publication('H_NEG', lambda p: p['scope'].update(study_id='another-study')),
                    'Receipt is outside this new ablation study')

    def test_wrong_group_publication_manifest_rejected(self):
        self.reject(lambda t: t.change_publication('G_POS', lambda p: p.update(source_manifest_sha256='b' * 64)),
                    'Published manifest is not this deployed helper tree')

    def test_wrong_group_publication_readback_rejected(self):
        self.reject(lambda t: t.change_publication('G_NEG', lambda p: p.update(remote_readback_sha='b' * 40)),
                    'Missing actual push and remote SHA readback')

    def test_wrong_group_publication_commit_rejected(self):
        self.reject(lambda t: t.change_publication('H_POS', lambda p: p.update(
            source_commit='b' * 40, remote_readback_sha='b' * 40)),
                    'Configured source commit differs from publication')

    def test_group_publication_not_accepted_rejected(self):
        self.reject(lambda t: t.change_publication('H_NEG', lambda p: p.update(accepted=False)),
                    'Latest implementation was not published')

    def test_group_publication_wrong_branch_rejected(self):
        self.reject(lambda t: t.change_publication('G_POS', lambda p: p.update(branch='main')),
                    'Latest implementation was not published')

    def test_changed_group_publication_payload_with_original_pin_rejected(self):
        def mutate(t):
            path = Path(t.configs['G_POS']['gates']['publication']['path'])
            path.write_bytes(path.read_bytes() + b'\n')
        self.reject(mutate, 'File changed or SHA differs')

    def test_wrong_group_publication_sha_rejected(self):
        def mutate(t):
            t.configs['G_POS']['gates']['publication']['sha256'] = 'b' * 64
            t.pin_configs_and_study()
        self.reject(mutate, 'File changed or SHA differs')

    def test_wrong_group_publication_bytes_rejected(self):
        def mutate(t):
            t.configs['G_POS']['gates']['publication']['bytes'] += 1
            t.pin_configs_and_study()
        self.reject(mutate, 'File changed or SHA differs')

    def test_study_publication_still_must_match_original_source(self):
        study, _ = read(self.topology.request['study_manifest'])
        study['publication'] = self.topology.configs['G_POS']['gates']['publication']
        self.topology.request['study_manifest'] = self.topology.store('different-study-publication', study)
        with self.assertRaisesRegex(ValueError, 'Original ordered study/helper/publication differs'):
            self.topology.bind()

    def test_config_wrong_helper_root_remains_rejected(self):
        def mutate(t):
            t.configs['G_POS']['paths']['helper_root'] = str(t.root / 'wrong-helper-root')
            t.pin_configs_and_study()
        self.reject(mutate, 'Config quartet differs from immutable original study/source')

    def test_config_wrong_helper_manifest_remains_rejected(self):
        def mutate(t):
            t.configs['G_POS']['helper_source_manifest'] = dict(
                t.configs['G_POS']['helper_source_manifest'], sha256='b' * 64)
            t.pin_configs_and_study()
        self.reject(mutate, 'Config quartet differs from immutable original study/source')

    def test_config_wrong_source_commit_remains_rejected(self):
        def mutate(t):
            t.configs['G_POS']['source_commit'] = 'b' * 40
            t.pin_configs_and_study()
        self.reject(mutate, 'Config quartet differs from immutable original study/source')

    def test_four_rank_accumulation8_science_remains_enforced(self):
        def mutate(t):
            t.configs['G_POS']['scientific']['gradient_accumulation_full_batch_per_rank'] = 4
            t.pin_configs_and_study()
        self.reject(mutate, 'Scientific setting changed: gradient_accumulation_full_batch_per_rank')

    def test_foreign_reader_run_marker_rejected(self):
        self.topology.request['analysis_run_id'] = self.topology.configs['G_POS']['run_id']
        with self.assertRaisesRegex(ValueError, 'Reader needs its independent explicit run marker'):
            self.topology.bind()

    def test_original_manifest_expected_pin_remains_enforced(self):
        manifest, _ = read(self.topology.sources['original_source']['source_manifest'])
        changed = self.topology.store('wrong-original-manifest', dict(manifest, synthetic=True))
        pub = self.topology.store('wrong-original-publication', dict(
            accepted=True, branch='search-e1', source_commit=reader.COMMITS['original'],
            remote_readback_sha=reader.COMMITS['original'], source_manifest_sha256=changed['sha256']))
        self.topology.request['original_source'].update(source_manifest=changed, publication=pub)
        with self.assertRaisesRegex(ValueError, 'Wrong frozen helper manifest'):
            self.topology.bind()

    def test_fresh_original_source_payload_change_rejected(self):
        copied = self.topology.root / 'changed-native-source'
        shutil.copytree(CONTROL / 'opsd_signal_weight_v1', copied)
        (copied / 'gates.py').write_bytes((copied / 'gates.py').read_bytes() + b'\n# fixture mutation\n')
        self.topology.request['original_source']['helper_root'] = str(copied)
        with self.assertRaisesRegex(ValueError, 'File changed or SHA differs'):
            self.topology.bind()

    def test_binding_only_rejects_nonempty_cuda(self):
        request_ref = self.topology.store('request-cuda', self.topology.request)
        with mock.patch.object(reader.os, 'getuid', return_value=1001), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4'}):
            with self.assertRaisesRegex(ValueError, 'empty CUDA'):
                reader.bind_only(request_ref)

    def test_binding_only_rejects_wrong_uid(self):
        request_ref = self.topology.store('request-uid', self.topology.request)
        with mock.patch.object(reader.os, 'getuid', return_value=1005), \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(ValueError, 'UID1001'):
                reader.bind_only(request_ref)


if __name__ == '__main__':
    unittest.main()
