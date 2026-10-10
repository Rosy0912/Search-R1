"""Pure local tiny controller fixtures; no node calls, actors, GPU or raw data."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('statistics_control', HERE / 'statistics_control_v2.py')
ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)


class ControllerFixtures(unittest.TestCase):
    def context(self):
        return ctl.context(HERE / 'historical_replay/parallel_selected/cohort-config.json',
                           HERE / 'historical_replay/parallel_recovery_v1/cohort-config.json')

    def observed(self, context):
        rows = copy.deepcopy(context['producers'])
        for i, item in enumerate(rows):
            item.pop('input_path'); item['inputs_sha256'] = format(i, '064x')
        return dict(phase='PREPARE_COMPACT_BINDINGS_ONLY', C_inputs_sha256='a' * 64, R_sources=rows)

    def test_actual_small_blueprints_exact_scope_and_no_dense_rows(self):
        context = self.context()
        self.assertEqual(sorted(s for item in context['producers'] for s in item['selected_steps']), ctl.SELECTED)
        self.assertEqual(context['producers'][0]['selected_steps'], [1])
        self.assertEqual(context['producers'][0]['config_sha256'], ctl.OLD_CONFIG_SHA)
        self.assertTrue(context['paths']['source'].endswith('statistics-source-ff98085b6918-v2'))
        self.assertEqual(context['paths']['C_helper_path'].split('/')[-2], 'source')

    def test_config_schema_frozen_consumer_matches(self):
        context = self.context(); cfg = ctl.prepare_config(context, self.observed(context))
        required = ['selected_steps', 'native_run_id', 'original_base', 'diagnostic_base',
            'run_manifest_sha256', 'C_directory', 'C_inputs_sha256', 'C_helper_sha256',
            'pairwise_source_path', 'pairwise_source_sha256', 'R_sources', 'output_new', 'expected_node']
        self.assertTrue(all(k in cfg for k in required)); self.assertEqual(cfg['selected_steps'], ctl.SELECTED)
        self.assertEqual(len(cfg['R_sources']), 7); self.assertTrue(cfg['per_row_outputs_remote_only'])

    def test_mixed_sources_preserve_only_accepted_original_points(self):
        c=self.context(); old=[x for x in c['producers'] if 'R-parallel-source-' in x['source_directory']]
        new=[x for x in c['producers'] if 'R-recovery-source-' in x['source_directory']]
        self.assertEqual(sorted(s for x in old for s in x['selected_steps']),[20,40,100])
        self.assertEqual(sorted(s for x in new for s in x['selected_steps']),[60,80,120,140])
        self.assertEqual(len(c['blueprint_paths']),10)
        self.assertTrue(all('gpu2.json' in x['config_path'] or 'gpu5.json' in x['config_path'] for x in old))
        self.assertTrue(c['paths']['output'].endswith('-v2'))

    def test_original_no_recovery_scope_supported(self):
        c=ctl.context(HERE/'historical_replay/parallel_selected/cohort-config.json')
        self.assertEqual(sorted(s for x in c['producers'] for s in x['selected_steps']),ctl.SELECTED)
        self.assertFalse(c['paths']['output'].endswith('-v2'))

    def test_missing_selected_or_duplicated_scope_rejected(self):
        context = self.context(); observed = self.observed(context)
        observed['R_sources'].pop()
        with self.assertRaises(AssertionError): ctl.prepare_config(context, observed)
        observed = self.observed(context); observed['R_sources'].append(observed['R_sources'][0])
        with self.assertRaises(AssertionError): ctl.prepare_config(context, observed)

    def test_context_covers_only_seven_new_points_and_first_old(self):
        c = self.context()
        self.assertEqual(sum(len(x['selected_steps']) for x in c['producers']), 8)
        self.assertTrue(all(Path(x['source_directory']).is_relative_to(c['paths']['diagnostic_base']) for x in c['producers']))
        self.assertEqual(c['payload_sha256']['selected_pair_statistics.py'], '88baa58b218a8c23e107ca9283cb674761cc300280917af555f8280f7bc344b1')

    def test_remote_snippets_compile_without_execution(self):
        for code in [ctl.REMOTE_DEPLOY, ctl.REMOTE_PREPARE, ctl.REMOTE_EXECUTE]:
            ast.parse(code)
        self.assertIn("'Ran 77 tests", ctl.REMOTE_DEPLOY)
        self.assertIn("CUDA_VISIBLE_DEVICES=''", ctl.REMOTE_COMMON)
        self.assertNotIn('pidfd_send_signal', ctl.REMOTE_COMMON)
        self.assertNotIn('nvidia-smi', ctl.REMOTE_COMMON)

    def test_remote_execute_rejects_incomplete_before_consumer(self):
        code = ctl.REMOTE_EXECUTE
        self.assertLess(code.index("assert sorted(selected)==cfg['selected_steps']"), code.index('r=subprocess.run'))
        self.assertLess(code.index("assert not pathlib.Path(cfg['output_new']).exists()"), code.index("target=S/'config-selected8.json'"))
        self.assertIn('historical_R_reconstructed_not_online_bitwise=True', code)
        self.assertIn('len(encoded)<=30*1024', code)

    def test_publication_mismatch_rejected_without_node(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve(); (base / 'code.py').write_text('synthetic local fixture')
            p = base / 'publication.json'; data = dict(source_commit='fixture', files=[dict(path=ctl.PREFIX + 'code.py', sha256='0' * 64)])
            p.write_text(json.dumps(data))
            with patch.object(ctl, 'HERE', base):
                with self.assertRaises(AssertionError): ctl.published(p, ['code.py'])
                data['files'][0]['sha256'] = hashlib.sha256((base / 'code.py').read_bytes()).hexdigest()
                p.write_text(json.dumps(data)); self.assertEqual(ctl.published(p, ['code.py'])['source_commit'], 'fixture')


if __name__ == '__main__': unittest.main()
