import ast
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import summarize_regions as s


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(s.canonical(value))


def make_probe(directory, case_id, version, name, response, origin, value, teacher=True, spans=None):
    spans = spans if spans is not None else [[0, len(response)]]
    info = [m.span() for m in re.finditer(r'<information>.*?</information>', response, re.S)]
    rows = []
    valid = [any(a <= i and i + 1 <= b for a, b in spans) and not any(a <= i < b for a, b in info) for i in range(len(response))]
    m = sum(valid)
    for i, char in enumerate(response):
        rows.append(dict(token_index=i, token_id=ord(char), character_offset=[i, i + 1], text=char,
                         policy_mask=valid[i], information_mask=any(a <= i < b for a, b in info),
                         teacher_student_clipped_kl=value if teacher else None,
                         teacher_student_unclipped_kl=value + .1 if teacher else None,
                         base_ordinary_to_student_kl=value + .2,
                         clipped_vocabulary_contributions=1 if teacher else None,
                         clipped_teacher_probability_mass=.125 if teacher else None,
                         distributions={'teacher_privileged': dict(target_logprob=-2., target_probability=.1353352832366127, entropy=1.5,
                           topk=[dict(token_id=9, logprob=-1., probability=.36787944117144233), dict(token_id=10, logprob=-2., probability=.1353352832366127)])} if teacher else {},
                         actual_masked_trajectory_loss_contribution=value / m if teacher and valid[i] else 0. if teacher else None))
    path = directory / ('tokens-' + version + '-' + name) / 'tokens.jsonl'
    path.parent.mkdir(parents=True)
    data = b''.join(s.canonical(x) + b'\n' for x in rows); path.write_bytes(data)
    summary = dict(status='complete', case_id=case_id, version=version, response_origin=origin,
                   original_training_objective=origin == 'preserved_student', response_tokens=len(rows),
                   valid_policy_tokens=m, reference_status='successful_original_reference' if teacher else 'not_applicable_no_successful_reference',
                   tokens_file=dict(path=str(path), bytes=len(data), rows=len(rows), sha256=s.digest(data)),
                   full_vocabulary_dump=dict(status='skipped_cap'), trajectory_mean=value if teacher else None,
                   frozen_pair_loss=value if teacher else None, per_token_contribution_sum_float64=value if teacher else None,
                   clipped_policy_vocabulary_contributions=m if teacher else None,
                   frozen_pair_loss_check=dict(passed=True, per_token_contribution_sum_matches=True) if teacher else dict(status='not_applicable_no_successful_reference'))
    if origin.startswith('preserved_'):
        summary['preserved_trajectory_name'] = name
    write_json(directory / ('token-summary-' + version + '-' + name + '.json'), summary)
    return summary, path, spans


def fixture(root, hard=False):
    cases_dir = root / 'selection'; run_dir = root / 'run'; directory = run_dir / 'worker-0' / 'case-a'
    response = '<think>a</think><search>q</search><information>d</information><answer>x</answer>'
    spans = [[0, response.index('<information>')], [response.index('<answer>'), len(response)]]
    siblings = [dict(row=dict(sibling=i, response=response, policy_spans=spans)) for i in range(5)]
    case = dict(case_id='case-a', question_id='fixture:1', dataset='nq', group='all5_wrong' if hard else 'correct_wrong',
                original_pair=None if hard else dict(row=dict(student=response, student_policy_spans=spans)),
                original_siblings=siblings)
    case['case_semantic_sha256'] = s.digest(s.canonical(case))
    cases_dir.mkdir(); data = s.canonical(case) + b'\n'; (cases_dir / 'cases.jsonl').write_bytes(data)
    write_json(cases_dir / 'manifest.json', dict(cases=dict(sha256=s.digest(data))))
    write_json(run_dir / 'config.json', dict(run_id='fixture-private-run'))
    result = dict(case_id='case-a', dataset=case['dataset'], group=case['group'], status='complete',
                  source_case_sha256=case['case_semantic_sha256'], versions={})
    for index, version in enumerate(s.VERSIONS):
        originals = []
        for i in range(5 if hard else 1):
            name = 'original-failed-sibling-' + str(i) if hard else 'original-mined-student'
            summary, _, _ = make_probe(directory, 'case-a', version, name, response,
                        'preserved_failed_sibling' if hard else 'preserved_student', i + 1 + index / 10., not hard, spans)
            originals.append(summary)
        generation = dict(active_adapter=version, privileged_reference=False, kind='actual_autoregressive_generation',
                          teacher_forced=False, response='<answer>x</answer>', policy_spans=[[0, 18]], protocol_em=int(index == 0))
        # Compute length rather than relying on the fixture spelling.
        generation['policy_spans'] = [[0, len(generation['response'])]]
        write_json(directory / ('student-generation-' + version + '.json'), generation)
        generated, _, _ = make_probe(directory, 'case-a', version, 'actual-student-generation', generation['response'],
                                     'new_student_generation', index + .5, not hard)
        result['versions'][version] = dict(generation_em=generation['protocol_em'], token_probes=originals,
                                           actual_student_response_teacher_forced_probe=generated)
    if hard:
        teacher = dict(status='not_applicable_no_successful_reference')
    else:
        teacher = dict(active_adapter='disabled', privileged_reference=True, response='<answer>x</answer>', policy_spans=[[0, 18]])
        teacher['policy_spans'] = [[0, len(teacher['response'])]]
        summary, _, _ = make_probe(directory, 'case-a', 'source100', 'actual-teacher-generation', teacher['response'], 'new_teacher_generation', .5)
        result['actual_teacher_response_teacher_forced_probe'] = summary
    write_json(directory / 'teacher-generation.json', teacher)
    results = run_dir / 'worker-0' / 'case-results.jsonl'; results.write_bytes(s.canonical(result) + b'\n')
    return cases_dir, run_dir, directory, results


class AnalysisTests(unittest.TestCase):
    def test_region_precedence_boundary_and_last_authored_answer(self):
        text = '<think>a</think><search>q</search><information><answer>bad</answer></information><answer>x</answer>'
        regions = s.character_regions(text, [[0, text.index('<information>')], [text.rindex('<answer>'), len(text)]])
        self.assertEqual(s.category((text.index('q'), text.index('q') + 1), regions), 'search_query')
        self.assertEqual(s.category((text.rindex('x'), text.rindex('x') + 1), regions), 'final_answer')
        self.assertEqual(s.category((text.index('bad'), text.index('bad') + 3), regions), 'other')
        self.assertEqual(s.category((6, 9), regions), 'boundary')
        self.assertEqual(s.category((0, 7), regions), 'structural_tags')

    def test_region_sum_reconstructs_native_and_information_never_supervised(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, path, spans = make_probe(Path(tmp), 'a', 'source100', 'x', '<think>a</think><information>z</information><answer>x</answer>', 'preserved_student', .4)
            result = s.probe(summary, path, '<think>a</think><information>z</information><answer>x</answer>', spans)
            self.assertAlmostEqual(sum(x['trajectory_contributions']['teacher_student_clipped_kl'] for x in result['regions'].values()), .4)
            self.assertAlmostEqual(result['means']['clipped_teacher_probability_mass'], .125)
            summary['frozen_pair_loss'] = .8
            with self.assertRaisesRegex(ValueError, 'native trajectory'):
                s.probe(summary, path, '<think>a</think><information>z</information><answer>x</answer>', spans)

    def test_hard_five_siblings_are_question_mean_not_five_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, run, _, _ = fixture(root, hard=True)
            summary = s.run(cases, run, root / 'out')
            first = summary['original_common_history']['overall']['source100']
            self.assertEqual(first['questions'], 1)
            self.assertAlmostEqual(first['means']['base_ordinary_to_student_kl'], 3.2)
            self.assertIsNone(first['means']['teacher_student_clipped_kl'])
            self.assertEqual(summary['skip_counts']['probe_skip:new_teacher_generation:no_successful_reference'], 1)

    def test_full_fixture_pairs_only_original_not_new_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, run, _, results = fixture(root)
            with results.open('ab') as f: f.write(b'{"partial":')
            summary = s.run(cases, run, root / 'out')
            self.assertEqual(summary['completed_cases'], 1)
            self.assertEqual(summary['skip_counts']['trailing_partial_result_line'], 1)
            item = json.loads((root / 'out' / 'cases.jsonl').read_text())
            self.assertAlmostEqual(item['paired_changes']['epoch1']['original_history_mean_metric_change']['teacher_student_clipped_kl'], .3)
            self.assertEqual(item['paired_changes']['epoch1']['strict_flip'], '1->0')
            self.assertFalse(item['generated']['epoch1']['paired_checkpoint_loss_comparison_allowed'])
            self.assertEqual(item['teacher_invariance']['epoch1:original-mined-student']['status'], 'verified')
            with self.assertRaises(FileExistsError): s.run(cases, run, root / 'out')

    def test_duplicate_completed_case_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, run, _, results = fixture(root)
            results.write_bytes(results.read_bytes() * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                s.run(cases, run, root / 'out')
            self.assertEqual(json.loads((root / 'out' / 'failed.json').read_text())['status'], 'failed_partial_not_accepted')

    def test_nonfinite_token_and_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = '<answer>x</answer>'; summary, path, spans = make_probe(Path(tmp), 'a', 'source100', 'x', text, 'preserved_student', .4)
            data = path.read_bytes(); path.write_bytes(data.replace(b'"base_ordinary_to_student_kl":0.6000000000000001', b'"base_ordinary_to_student_kl":NaN', 1))
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                s.probe(summary, path, text, spans)
            path.write_bytes(data + b'{}\n')
            with self.assertRaises((ValueError, KeyError)):
                s.probe(summary, path, text, spans)

    def test_native_objective_flag_not_relabelled_for_new_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, run, directory, _ = fixture(root)
            path = directory / 'token-summary-epoch1-actual-student-generation.json'
            value = json.loads(path.read_text()); value['original_training_objective'] = True; write_json(path, value)
            with self.assertRaisesRegex(ValueError, 'Generated probe identity'):
                s.run(cases, run, root / 'out')

    def test_four_checkpoint_token_alignment_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases, run, directory, result_path = fixture(root)
            path = directory / 'tokens-epoch1-original-mined-student' / 'tokens.jsonl'
            rows = [json.loads(x) for x in path.read_bytes().splitlines()]
            rows[0]['token_id'] += 1
            data = b''.join(s.canonical(x) + b'\n' for x in rows); path.write_bytes(data)
            summary_path = directory / 'token-summary-epoch1-original-mined-student.json'
            value = json.loads(summary_path.read_text()); value['tokens_file'].update(sha256=s.digest(data), bytes=len(data)); write_json(summary_path, value)
            result = json.loads(result_path.read_text()); result['versions']['epoch1']['token_probes'][0] = value
            result_path.write_bytes(s.canonical(result) + b'\n')
            with self.assertRaisesRegex(ValueError, 'history/token/mask mismatch'):
                s.run(cases, run, root / 'out')

    def test_empty_region_does_not_nullify_hard_question_contribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for i, text in enumerate(('<think>a</think><answer>x</answer>', '<answer>x</answer>')):
                summary, path, spans = make_probe(Path(tmp), 'a', 'source100', str(i), text, 'preserved_failed_sibling', .4, teacher=False)
                results.append(s.probe(summary, path, text, spans))
            average = s.mean_records(results)
            self.assertEqual(average['regions']['think_text']['token_mean_nonempty_trajectories'], 1)
            self.assertIsNotNone(average['regions']['think_text']['mean_trajectory_token_means']['base_ordinary_to_student_kl'])
            self.assertAlmostEqual(average['regions']['think_text']['mean_trajectory_contributions']['base_ordinary_to_student_kl'], results[0]['regions']['think_text']['trajectory_contributions']['base_ordinary_to_student_kl'] / 2)

    def test_teacher_q_changed_and_missing_fields_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); text = '<answer>x</answer>'
            a, ap, spans = make_probe(directory, 'a', 'source100', 'x', text, 'preserved_student', .4)
            b, bp, _ = make_probe(directory, 'a', 'epoch1', 'x', text, 'preserved_student', .3)
            ar = s.probe(a, ap, text, spans); br = s.probe(b, bp, text, spans)
            self.assertEqual(s.teacher_invariance(ap, bp, ar['token_file'], br['token_file'])['status'], 'verified')
            rows = [json.loads(x) for x in bp.read_bytes().splitlines()]; rows[0]['distributions']['teacher_privileged']['entropy'] = 2.
            data = b''.join(s.canonical(x) + b'\n' for x in rows); bp.write_bytes(data)
            proof = {}; list(s.lines(bp, proof=proof))
            with self.assertRaisesRegex(ValueError, 'teacher q differs'):
                s.teacher_invariance(ap, bp, ar['token_file'], proof)
            for row in rows: row['distributions'] = {}
            data = b''.join(s.canonical(x) + b'\n' for x in rows); bp.write_bytes(data)
            proof = {}; list(s.lines(bp, proof=proof))
            result = s.teacher_invariance(ap, bp, ar['token_file'], proof)
            self.assertEqual(result['status'], 'available_fields_invariant_with_missing_fields')
            self.assertEqual(result['missing_fields']['entropy:one'], len(text))

    def test_cpu_only_imports(self):
        tree = ast.parse(Path(s.__file__).read_text())
        names = {n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
        self.assertFalse(names & {'torch', 'transformers', 'numpy', 'requests', 'subprocess'})


if __name__ == '__main__':
    unittest.main()
