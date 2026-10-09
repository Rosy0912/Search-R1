"""Synthetic CPU fixtures; no Node, production data, GPU, HTTP, or generation."""
import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
try:
    from . import bank, common, replay_math as replay, summary, replay as launcher
except ImportError:
    import bank, common, replay_math as replay, summary
    import replay as launcher

TRAINING = Path(os.environ.get('OPSD_ANALYSIS_FROZEN_HELPER_ROOT',
    str(Path(__file__).resolve().parent.parent / 'opsd_signal_weight_v1')))


def source_module(name):
    path = TRAINING / (name + '.py'); spec = importlib.util.spec_from_file_location('_analysis_fixture_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1] * (2 if messages == 'ordinary' else int(messages))
    def __call__(self, response, **kwargs):
        return dict(input_ids=list(range(len(response))), offset_mapping=[[i, i+1] for i in range(len(response))])


class Core:
    @staticmethod
    def prompt(question, ref): return 'ordinary' if ref is None else ref
    @staticmethod
    def response_encoding(tokenizer, response, spans):
        return list(range(len(response))), [any(a <= i < b for a, b in spans) for i in range(len(response))]


class Regions:
    @staticmethod
    def character_regions(response, spans): return [(0, len(response), 'think_text')]
    @staticmethod
    def category(offset, segments): return 'think_text'


def encoded(**overrides):
    value = dict(response=[0, 1], mask=[True, True], offsets=[[0, 1], [1, 2]],
        regions=['think_text', 'final_answer'], answer_positions=[1],
        ordinary_missing_reason=None, teacher_missing_reason=None,
        ordinary_context_tokens=4, teacher_context_tokens=6, sha256='a'*64)
    value.update(overrides); return value


def token_rows(teacher=True):
    return [dict(token_index=i, token_id=i, token_offsets=[i, i+1], region='think_text' if i == 0 else 'final_answer',
        student_entropy=1. + i, source100_ordinary_reverse_kl=.2+i*.2,
        raw_teacher_reverse_kl=(.4+i*.2) if teacher else None,
        post_clip_teacher_loss=(.3+i*.2) if teacher else None,
        student_observed_logp=-2.+i, teacher_observed_logp=(-1. + .5*i) if teacher else None,
        source_observed_logp=-1.5, clipped_vocabulary_count=1 if teacher else None,
        clipped_student_probability_mass=.1 if teacher else None, clipped_teacher_probability_mass=.05 if teacher else None)
        for i in range(2)]


def fixture_bank():
    data = [dict(question_id=str(i), question='q'+str(i), answers=['gold'], data_source='nq' if i < 256 else 'hotpotqa') for i in range(512)]
    rows = []; traces = {}
    for i, q in enumerate(data):
        result = dict(response='right' if i % 2 == 0 else 'wrong', policy_spans=[[0, 5]], termination='answer', search_calls=0)
        protocol = dict(protocol_em=int(i % 2 == 0), protocol_valid_final_action=True,
            failure_class=None if i % 2 == 0 else 'valid_gold_mismatch', extracted_final_answer=result['response'])
        traces[i] = dict(validation_index=i, question_id=q['question_id'], question=q['question'], answers=q['answers'], native_result=result, protocol=protocol)
        rows.append(dict(validation_index=i, question_id=q['question_id'], primary=True, version='source100', completed_outer=0,
            protocol_em=protocol['protocol_em'], row_semantic_sha256=common.semantic_sha(q),
            native_result_sha256=common.semantic_sha(result), trace={'fixture': i}))
    def reader(ref): return traces[ref['fixture']]
    def scorer(result, answers):
        hit = int(result['response'] == 'right')
        return dict(protocol_em=hit, protocol_valid_final_action=True, failure_class=None if hit else 'valid_gold_mismatch',
            extracted_final_answer=result['response'])
    return data, rows, reader, scorer


class AnalysisTests(unittest.TestCase):
    def test_cpu_imports_do_not_load_torch(self):
        code = "import common, bank, summary, replay, replay_worker, replay_math, sys; assert 'torch' not in sys.modules"
        p = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_common_bank_all512_and_only_strict_success(self):
        data, rows, read, score = fixture_bank(); result = bank.build_bank(data, rows, read, score)
        self.assertEqual((result['questions'], result['available']), (512, 256))
        self.assertFalse(result['entries'][1]['available']); self.assertIsNone(result['entries'][1]['reference_response'])
        self.assertEqual(result['entries'][1]['source_zero_em'], 0)

    def test_common_bank_no_reference_is_valid_population(self):
        data, rows, read, score = fixture_bank()
        for row in rows:
            trace = read(row['trace']); trace['native_result']['response'] = 'wrong'
            trace['protocol'] = score(trace['native_result'], ['gold']); row['protocol_em'] = 0
            row['native_result_sha256'] = common.semantic_sha(trace['native_result'])
        result = bank.build_bank(data, rows, read, score)
        self.assertEqual(result['available'], 0); self.assertEqual(len(result['entries']), 512)

    def test_bank_rejects_wrong_checkpoint_and_duplicate_slots(self):
        data, rows, read, score = fixture_bank()
        changed = copy.deepcopy(rows); changed[0]['completed_outer'] = 15
        with self.assertRaises(ValueError): bank.build_bank(data, changed, read, score)
        changed = copy.deepcopy(rows); changed[1] = changed[0]
        with self.assertRaises(ValueError): bank.build_bank(data, changed, read, score)

    def test_bank_rejects_gold_label_lie(self):
        data, rows, read, score = fixture_bank(); rows[1]['protocol_em'] = 1
        with self.assertRaises(ValueError): bank.build_bank(data, rows, read, score)

    def test_encoder_no_B_keeps_ordinary_coverage(self):
        x = replay.encode_response(Tokenizer(), 'q', dict(response='xy', policy_spans=[[0, 2]]), None, Core, Regions)
        self.assertIsNone(x['ordinary_missing_reason']); self.assertEqual(x['teacher_missing_reason'], 'no_common_zero_successful_reference')
        self.assertEqual(x['mask'], [True, True]); self.assertIsNone(x['teacher_prefix'])

    def test_teacher_8192_limit_does_not_delete_ordinary(self):
        x = replay.encode_response(Tokenizer(), 'q', dict(response='xy', policy_spans=[[0, 2]]), '8191', Core, Regions)
        self.assertIsNone(x['ordinary_missing_reason']); self.assertEqual(x['teacher_missing_reason'], 'teacher_joint_context_limit')
        self.assertEqual(x['ordinary_context_tokens'], 4); self.assertEqual(x['teacher_context_tokens'], 8193)

    def test_ordinary_limit_and_empty_mask_report_NA(self):
        x = replay.encode_response(Tokenizer(), 'q', dict(response='xy', policy_spans=[[0, 2]]), '2', Core, Regions, max_context=3)
        self.assertEqual(x['ordinary_missing_reason'], 'ordinary_context_limit')
        item = encoded(ordinary_missing_reason='ordinary_context_limit', teacher_missing_reason='ordinary_context_limit')
        result = replay.summarize_tokens(item, [])
        self.assertEqual(result['valid_policy_tokens'], 0); self.assertTrue(all(v is None for v in result['metrics'].values()))
        x = replay.encode_response(Tokenizer(), 'q', dict(response='xy', policy_spans=[]), None, Core, Regions)
        self.assertEqual(x['ordinary_missing_reason'], 'empty_policy_mask')

    def test_encoder_rejects_native_alignment_or_span_change(self):
        with self.assertRaises(ValueError):
            replay.encode_response(Tokenizer(), 'q', dict(response='xy', policy_spans=[[1, 0]]), None, Core, Regions)
        class Bad(Core):
            @staticmethod
            def response_encoding(tokenizer, response, spans): return [1, 0], [True, True]
        with self.assertRaises(ValueError): replay.encode_response(Tokenizer(), 'q', dict(response='xy', policy_spans=[[0, 2]]), None, Bad, Regions)

    def test_gap_is_signed_answer_token_mean_not_KL(self):
        result = replay.summarize_tokens(encoded(), token_rows())
        self.assertAlmostEqual(result['metrics']['answer_gap'], -.5)
        self.assertAlmostEqual(result['metrics']['raw_teacher_reverse_kl'], .5)
        self.assertAlmostEqual(result['metrics']['student_entropy'], 1.5)

    def test_no_answer_payload_only_gap_NA(self):
        result = replay.summarize_tokens(encoded(answer_positions=[]), token_rows())
        self.assertIsNone(result['metrics']['answer_gap']); self.assertAlmostEqual(result['metrics']['raw_teacher_reverse_kl'], .5)
        self.assertEqual(result['missing_reasons']['answer_gap'], 'no_effective_final_answer_payload_tokens')

    def test_no_B_and_teacher_overflow_still_entropy_and_source_drift(self):
        for reason in ('no_common_zero_successful_reference', 'teacher_joint_context_limit'):
            result = replay.summarize_tokens(encoded(teacher_missing_reason=reason), token_rows(False))
            self.assertIsNone(result['metrics']['answer_gap']); self.assertIsNone(result['metrics']['raw_teacher_reverse_kl'])
            self.assertAlmostEqual(result['metrics']['student_entropy'], 1.5)
            self.assertAlmostEqual(result['metrics']['source100_ordinary_reverse_kl'], .3)
            self.assertEqual(result['missing_reasons']['raw_teacher_reverse_kl'], reason)

    def test_every_effective_token_required(self):
        with self.assertRaises(ValueError): replay.summarize_tokens(encoded(), token_rows()[:1])
        rows = token_rows(); rows[1]['token_offsets'] = [0, 1]
        with self.assertRaises(ValueError): replay.summarize_tokens(encoded(), rows)

    def test_nonfinite_is_failure_not_NA(self):
        rows = token_rows(); rows[0]['student_entropy'] = float('nan')
        with self.assertRaises(ValueError): replay.summarize_tokens(encoded(), rows)

    def test_regions_reconstruct_trajectory_mean_and_not_gradient(self):
        value = replay.summarize_tokens(encoded(), token_rows())
        row = dict(replay=value); got = summary.regional_summary([row])
        self.assertAlmostEqual(sum(x['equal_question_mean_contribution'] for x in got['raw_teacher_reverse_kl']['regions'].values()), .5)
        self.assertTrue(value['region_contributions_are_loss_not_parameter_gradients'])

    def relation_rows(self, values, labels):
        return [dict(protocol_em=y, protocol_valid_final_action=True, failure_class=None if y else 'valid_gold_mismatch',
            dataset='nq' if i % 2 == 0 else 'hotpotqa', search_calls=i % 3,
            replay=dict(valid_policy_tokens=2+i, metrics={f: x for f in replay.FEATURES},
                missing_reasons={f: None for f in replay.FEATURES})) for i, (x, y) in enumerate(zip(values, labels))]

    def test_binary_and_zero_variance_all_associations_NA(self):
        for values, labels, reason in (([1., 2.], [1, 1], 'only_one_EM_class_in_metric_subset'),
                ([1., 1.], [0, 1], 'zero_feature_variance')):
            got = summary.association(self.relation_rows(values, labels), 'answer_gap')
            self.assertEqual(got['association_NA_reason'], reason)
            self.assertTrue(all(got[k] is None for k in ('pearson', 'spearman', 'original_increasing_direction_EM_AUC')))

    def test_auc_direction_and_tied_ranks(self):
        got = summary.association(self.relation_rows([1., 1., 2., 3.], [0, 0, 1, 1]), 'answer_gap')
        self.assertEqual(got['original_increasing_direction_EM_AUC'], 1.)
        got = summary.association(self.relation_rows([3., 2., 1., 1.], [0, 0, 1, 1]), 'answer_gap')
        self.assertEqual(got['original_increasing_direction_EM_AUC'], 0.)

    def test_metric_coverage_does_not_filter_EM(self):
        rows = self.relation_rows([1., 2., 3.], [1, 0, 0]); rows[2]['replay']['metrics']['answer_gap'] = None
        rows[2]['replay']['missing_reasons']['answer_gap'] = 'no_common_zero_successful_reference'
        got = summary.association(rows, 'answer_gap')
        self.assertEqual((got['questions'], got['metric_available'], got['metric_unavailable']), (3, 2, 1))
        self.assertEqual(summary.counts(rows)['questions'], 3); self.assertAlmostEqual(summary.counts(rows)['em'], 1/3)

    def test_linear_controls_exactly_explained_variable_not_spurious(self):
        rx, dropped = summary.residualize([0., 1., 0., 1.], dict(dataset=[0., 1., 0., 1.], constant=[1.]*4))
        self.assertLess(sum(x*x for x in rx), 1e-20); self.assertIn('constant', dropped)

    def test_paired_case_metrics_are_question_not_token_alignment(self):
        left = self.relation_rows([1., 2.], [1, 0]); right = self.relation_rows([3., 4.], [0, 1])
        for i, row in enumerate(left): row['validation_index'] = i
        for i, row in enumerate(right): row['validation_index'] = i
        value = summary.paired_cases(left, right)
        self.assertEqual((value['lost'], value['gained']), (1, 1)); self.assertTrue(value['different_response_tokens_not_position_paired'])

    def test_strict_remote_output_blocks_Mac_and_symlink_escape(self):
        with self.assertRaises(ValueError): common.remote_output('/tmp/analysis-output')

    def test_publication_rejects_unpublished_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'one.py').write_text('x=1\n')
            mp = common.exclusive_json(root / 'manifest.json', dict(files={'one.py': common.file_ref(root / 'one.py')}))
            pp = common.exclusive_json(root / 'published.json', dict(accepted=True, branch='search-e1', source_commit='a'*40,
                remote_readback_sha='a'*40, source_manifest_sha256=mp['sha256']))
            launcher.verify_analysis_publication(root, mp, pp)
            (root / 'one.py').write_text('x=2\n')
            with self.assertRaises(ValueError): launcher.verify_analysis_publication(root, mp, pp)

    def final_rows(self):
        data, rows, read, score = fixture_bank(); b = bank.build_bank(data, rows, read, score); result = []
        for version, t in sorted(common.ENDPOINTS):
            for i, q in enumerate(data):
                eligible = b['entries'][i]['available']; hit = int(eligible) if t == 0 else int(i % 3 == 0)
                tok = token_rows(eligible)
                tok[0]['student_entropy'] += (i % 7) * .1
                enc = encoded(teacher_missing_reason=None if eligible else 'no_common_zero_successful_reference')
                replayed = replay.summarize_tokens(enc, tok)
                result.append(dict(version=version, completed_outer=t, validation_index=i, question_id=q['question_id'],
                    dataset=q['data_source'], row_semantic_sha256=common.semantic_sha(q), bank_available=eligible,
                    protocol_em=hit, protocol_valid_final_action=True, failure_class=None if hit else 'valid_gold_mismatch',
                    search_calls=i % 4, replay=replayed))
        return b, result

    def test_final_all4608_endpoint_EM_and_bank_NA_boundaries(self):
        b, rows = self.final_rows(); result = summary.summarize(rows, b)
        self.assertEqual(result['formal_EM_rows'], 4608)
        zero = result['models']['source100/0']
        self.assertEqual(zero['all512']['questions'], 512); self.assertEqual(zero['all512']['hits'], 256)
        self.assertEqual(zero['bank_subset']['em'], 1.)
        self.assertEqual(zero['features']['answer_gap']['association_NA_reason'], 'only_one_EM_class_in_metric_subset')
        self.assertIsNone(zero['features']['answer_gap']['original_increasing_direction_EM_AUC'])
        self.assertEqual(zero['features']['student_entropy']['metric_available'], 512)
        for v in common.VARIANTS:
            self.assertEqual(result['models'][v+'/25']['all512']['questions'], 512)
            self.assertEqual(result['models'][v+'/25']['features']['answer_gap']['metric_available'], 256)

    def test_final_missing_endpoint_or_duplicate_is_rejected(self):
        b, rows = self.final_rows()
        with self.assertRaises(ValueError): summary.summarize(rows[:-1], b)
        rows[-1] = rows[0]
        with self.assertRaises(ValueError): summary.summarize(rows, b)

    def endpoint_identities(self, rows):
        import hashlib
        result={}
        for row in rows:
            key=row['version']+'/'+str(row['completed_outer']);digest=hashlib.sha256(key.encode()).hexdigest()
            row['current_adapter_tensor_sha256']=digest
            result[key]=dict(version=row['version'],completed_outer=row['completed_outer'],adapter_tensor_sha256=digest,
                raw_checkpoint=dict(kind='common_zero_adapter' if row['completed_outer']==0 else 'raw_end_of_outer',
                    manifest=dict(path='/synthetic/'+key+'/checkpoint.json',sha256=digest)))
        return result

    def test_cross_signal_all512_pairs_keep_NA_cases_and_exact_checkpoint_ids(self):
        b,rows=self.final_rows();identities=self.endpoint_identities(rows)
        changed=next(row for row in rows if row['version']=='H_POS' and row['completed_outer']==25 and row['validation_index']==1)
        self.assertIsNone(changed['replay']['metrics']['answer_gap'])
        changed['protocol_em']=1;changed['failure_class']=None
        result=summary.summarize(rows,b,identities);cross=result['descriptive_cross_signal_pairs']
        self.assertEqual(len(cross),8)
        value=cross['G_POS/25->H_POS/25']
        self.assertEqual((value['questions'],value['gained'],value['lost']),(512,1,0))
        self.assertEqual({k:v['questions'] for k,v in value['datasets'].items()},{'nq':256,'hotpotqa':256})
        self.assertEqual(value['left_checkpoint_identity'],identities['G_POS/25'])
        self.assertEqual(value['right_checkpoint_identity'],identities['H_POS/25'])
        self.assertTrue(value['no_significance_test_or_new_family'])

    def test_cross_signal_identity_requires_all_saved_endpoint_digests(self):
        b,rows=self.final_rows();identities=self.endpoint_identities(rows)
        identities.pop('G_POS/15')
        with self.assertRaises(ValueError):summary.summarize(rows,b,identities)
        identities=self.endpoint_identities(rows);rows[-1]['current_adapter_tensor_sha256']='f'*64
        with self.assertRaises(ValueError):summary.summarize(rows,b,identities)

    def test_resource_boundary_blocks_foreign_gpu_and_cpu_cuda(self):
        cfg = dict(owner=dict(uid=1001, physical_gpus=[4, 5, 6, 7], gpu_uuids=['u4', 'u5', 'u6', 'u7']),
            gates={'retriever_admission': {}}, resources=dict(startup_minimum_free_mib=20480, training_minimum_free_mib=6144))
        owner = dict(pid=10, uid=1001, run_id='new'); devices = [dict(index=i, uuid='u'+str(i), actual_free_mib=24000) for i in range(4, 8)]
        g = types.SimpleNamespace(process=lambda pid: owner, same_process=lambda a, b: a == b,
            bound=lambda x: x, retriever_admission=lambda c, r: dict(owner={'pid': 77}), inventory=lambda: (devices, []))
        self.assertTrue(launcher.resource_boundary(cfg, {'gates': g}, owner, 'new', released=True)['accepted'])
        def foreign(pid, run, parent, children): raise ValueError('Foreign process')
        g.owned_gpu_process = foreign; g.inventory = lambda: (devices, [dict(gpu_uuid='u4', pid=99, used_mib=1)])
        with self.assertRaises(ValueError): launcher.resource_boundary(cfg, {'gates': g}, owner, 'new')
        g.inventory = lambda: (devices, [dict(gpu_uuid='u4', pid=10, used_mib=1)])
        with self.assertRaises(ValueError): launcher.resource_boundary(cfg, {'gates': g}, owner, 'new')
        g.inventory = lambda: (devices, [dict(gpu_uuid='u7', pid=77, used_mib=1025)])
        with self.assertRaises(ValueError): launcher.resource_boundary(cfg, {'gates': g}, owner, 'new')

    def test_resource_boundary_actualfree_and_UUID(self):
        cfg = dict(owner=dict(uid=1001, physical_gpus=[4, 5, 6, 7], gpu_uuids=['u4', 'u5', 'u6', 'u7']),
            gates={'retriever_admission': {}}, resources=dict(startup_minimum_free_mib=20480, training_minimum_free_mib=6144))
        owner = dict(pid=10, uid=1001, run_id='new'); devices = [dict(index=i, uuid='u'+str(i), actual_free_mib=24000) for i in range(4, 8)]
        g = types.SimpleNamespace(process=lambda pid: owner, same_process=lambda a, b: a == b,
            bound=lambda x: x, retriever_admission=lambda c, r: dict(owner={'pid': 77}), inventory=lambda: (devices, []))
        devices[0]['actual_free_mib'] = 20479
        with self.assertRaises(ValueError): launcher.resource_boundary(cfg, {'gates': g}, owner, 'new', released=True)
        devices[0]['actual_free_mib'] = 24000; devices[0]['uuid'] = 'wrong'
        with self.assertRaises(ValueError): launcher.resource_boundary(cfg, {'gates': g}, owner, 'new', released=True)


if __name__ == '__main__': unittest.main()
