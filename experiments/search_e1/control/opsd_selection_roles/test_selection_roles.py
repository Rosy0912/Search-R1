import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import selection_roles as roles


def literal_structure(text):
    # Deliberately distinct from strict final validity: an earlier open think
    # can coexist with a later independently valid authored answer span.
    balanced = text.count('<think>') == text.count('</think>')
    return dict(balanced=balanced, unexpected_closing=0,
                unclosed_opening=int(not balanced), nested_reserved_opening=0)


def save_jsonl(path, rows):
    path.write_bytes(b''.join(roles.canonical(row) + b'\n' for row in rows))


def fixture(directory):
    directory.mkdir()
    db = sqlite3.connect(directory / 'source.private.sqlite')
    db.executescript('''CREATE TABLE questions(qid TEXT PRIMARY KEY,first_line INTEGER,runs INTEGER,
      valid INTEGER,eligible INTEGER,dataset TEXT,cell TEXT,expected_pair TEXT,pair_seen INTEGER);
      CREATE TABLE rows(qid TEXT,sibling INTEGER,source_line INTEGER,payload TEXT,
      provenance TEXT,metrics TEXT,PRIMARY KEY(qid,sibling));''')
    cases, pairs, populations = [], [], []
    for dataset in ('nq', 'hotpotqa'):
        for group in ('correct_correct', 'correct_wrong', 'all5_wrong'):
            qid = dataset + '-' + group; eligible = group != 'all5_wrong'
            entries = []
            for sibling in range(5):
                reward = int(group == 'correct_correct' or group == 'correct_wrong' and sibling == 0)
                first = '<think>open<search>query</search>'
                last = '<answer>x</answer>' if reward else '<answer>y</answer>'
                failure = None if reward else 'valid_gold_mismatch'
                if group == 'correct_wrong' and sibling == 1 and dataset == 'nq':
                    last = '<think><answer>x</answer>'; failure = 'final_invalid'
                text = first + last
                spans = [[0, len(first)], [len(first), len(text)]]
                row = dict(question_id=qid, question='PRIVATE QUESTION ' + qid, answers=['x'],
                    data_source='hotpot' if dataset == 'hotpotqa' else dataset,
                    sibling=sibling, response=text, policy_spans=spans, reward=reward,
                    search_calls=0 if sibling < 2 else 2, termination='answer')
                if sibling != 0:
                    row['token_count'] = 37
                metric = dict(strict_reward=reward, reward_matches=True, failure=failure,
                    transcript_structure=literal_structure(text),
                    policy_structure=[literal_structure(first), literal_structure(last)])
                provenance = dict(line=len(cases)*5+sibling+1, offset=sibling*100,
                    bytes=len(roles.canonical(row))+1, row_sha256=hashlib.sha256(roles.canonical(row)).hexdigest())
                entries.append((row, provenance, metric))
                db.execute('INSERT INTO rows VALUES(?,?,?,?,?,?)',
                    (qid, sibling, provenance['line'], roles.canonical(row), roles.canonical(provenance), roles.canonical(metric)))
            ref, student = entries[0][0], entries[1][0]
            expected = dict(question_id=qid, question=ref['question'], answers=ref['answers'],
                reference=ref['response'], student=student['response'], reference_reward=ref['reward'],
                student_reward=student['reward'], student_policy_spans=student['policy_spans']) if eligible else None
            case = dict(question_id=qid, dataset=dataset, group=group, successes=sum(x[0]['reward'] for x in entries),
                sibling_order=list(range(5)), provenance=[x[1] for x in entries],
                sibling_diagnostics=[dict(sibling=x[0]['sibling'], **x[2]) for x in entries],
                eligible=eligible, reference=dict(sibling=0, search_calls=0,
                    token_count_or_length=len(ref['response']), tied_siblings=[0]) if eligible else None,
                selection=dict(full_chosen=1, candidate_siblings=list(range(1, 5))) if eligible else None,
                native_shadow_exact=True)
            cases.append(case)
            db.execute('INSERT INTO questions VALUES(?,?,?,?,?,?,?,?,?)',
                (qid, len(cases)*5, 1, 1, int(eligible), dataset, group,
                 roles.canonical(expected) if expected else None, int(eligible)))
            if eligible:
                digest = hashlib.sha256(roles.canonical(expected)).hexdigest()
                ref_matches = [dict(sibling=r['sibling'], provenance=p) for r, p, m in entries
                               if r['response'] == ref['response'] and r['reward'] == ref['reward']]
                student_matches = [dict(sibling=r['sibling'], provenance=p) for r, p, m in entries
                                   if r['response'] == student['response'] and r['reward'] == student['reward']
                                   and r['policy_spans'] == student['policy_spans']]
                pairs.append(dict(question_id=qid, previous_pair_rows=0, native_pair_exact=True,
                    mismatch_fields=[], actual_semantic_sha256=digest, expected_semantic_sha256=digest,
                    reference_matching_source_rows=ref_matches, student_matching_source_rows=student_matches))
            populations.append(dict(dataset=dataset, group=group, questions=1, eligible_pairs=int(eligible)))
    db.commit(); db.close()
    save_jsonl(directory/'cases.private.jsonl', cases); save_jsonl(directory/'pairs.private.jsonl', pairs)
    (directory/'issues.private.jsonl').write_bytes(b'')
    summary = dict(schema='opsd-selection-audit-v1', accepted=True, status='ACCEPTED', errors={}, fatal=None,
        counters=dict(audited_questions=6, unique_rollout_questions=6, native_expected_pairs=4, pair_native_exact=4),
        cells=populations, private_artifacts=[str(directory/name) for name in
            ('source.private.sqlite', 'cases.private.jsonl', 'pairs.private.jsonl', 'issues.private.jsonl')])
    (directory/'summary.json').write_bytes(roles.canonical(summary))
    return cases, pairs, summary


class RolesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.directory = self.root/'accepted'; self.cases, self.pairs, self.summary = fixture(self.directory)

    def tearDown(self):
        self.tmp.cleanup()

    def run_roles(self):
        return roles.run(self.directory, self.root/'result', production=False)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern): self.run_roles()
        self.assertFalse((self.root/'result'/'summary.json').exists())

    def save_cases(self):
        save_jsonl(self.directory/'cases.private.jsonl', self.cases)

    def save_pairs(self):
        save_jsonl(self.directory/'pairs.private.jsonl', self.pairs)

    def test_exact_role_denominators_not_all_siblings(self):
        before = roles.sha(self.directory/'source.private.sqlite'); value = self.run_roles()
        self.assertEqual(value['counters']['questions'], 6)
        self.assertEqual(value['counters']['hard_questions_without_roles'], 2)
        self.assertEqual(value['global_roles']['reference']['denominator'], 4)
        self.assertEqual(value['global_roles']['student']['denominator'], 4)
        self.assertEqual(value['global_roles']['reference']['strict_success_count'], 4)
        self.assertEqual(value['global_roles']['student']['strict_success_count'], 2)
        self.assertEqual(value['global_roles']['student']['final_invalid'], 1)
        self.assertEqual(value['global_roles']['student']['valid_final_gold_mismatch'], 1)
        self.assertEqual(value['global_roles']['reference']['token_count_missing_rows'], 4)
        self.assertEqual(value['global_roles']['reference']['token_count_mean_denominator'], 0)
        self.assertIsNone(value['global_roles']['reference']['full_response_token_count_mean'])
        self.assertEqual(value['global_roles']['student']['token_count_mean_denominator'], 4)
        self.assertEqual(value['global_roles']['student']['full_response_token_count_mean'], 37)
        self.assertEqual(value['global_roles']['reference']['strict_success_with_unbalanced_whole_transcript'], 4)
        self.assertEqual(value['global_roles']['reference']['role_with_any_nonfinal_think_defect'], 4)
        self.assertEqual(before, roles.sha(self.directory/'source.private.sqlite'))
        self.assertFalse(value['original_raw_files_reread'])
        compact = roles.canonical(value)
        self.assertNotIn(b'PRIVATE QUESTION', compact); self.assertNotIn(b'<answer>', compact)
        self.assertTrue(all(not item['roles'] for item in value['cells'] if item['cell'].endswith('all5_wrong')))

    def test_ambiguous_origin_keeps_native_selected_id(self):
        value = self.run_roles()
        cells = {item['cell']: item for item in value['cells']}
        self.assertEqual(cells['nq/correct_correct']['population']['reference_ambiguous_stored_origin_questions'], 1)
        private = [json.loads(x) for x in (self.root/'result'/'roles.private.jsonl').read_text().splitlines()]
        row = next(x for x in private if x['group'] == 'correct_correct')
        self.assertEqual(row['roles']['reference']['native_predicted_sibling'], 0)
        self.assertFalse(row['roles']['reference']['stored_pair_unique_origin_proven'])
        self.assertEqual(row['roles']['reference']['stored_matching_siblings'], [0, 1, 2, 3, 4])

    def test_unaccepted_producer_rejected(self):
        self.summary['accepted'] = False
        (self.directory/'summary.json').write_bytes(roles.canonical(self.summary))
        self.assert_rejected('producer_audit_not_accepted')

    def test_production_needs_bound_summary_sha(self):
        with self.assertRaisesRegex(ValueError, 'expected_accepted_summary_sha'):
            roles.run(self.directory, self.root/'result')
        self.assertFalse((self.root/'result').exists())

    def test_private_artifact_escape_rejected(self):
        self.summary['private_artifacts'][0] = '/elsewhere/source.private.sqlite'
        (self.directory/'summary.json').write_bytes(roles.canonical(self.summary))
        self.assert_rejected('producer_private_artifact_paths')

    def test_cell_population_rejected(self):
        self.summary['cells'][0]['eligible_pairs'] = 2
        (self.directory/'summary.json').write_bytes(roles.canonical(self.summary))
        self.assert_rejected('accepted_cell_denominator')

    def test_group_or_success_count_rejected(self):
        self.cases[0]['successes'] = 4; self.save_cases()
        self.assert_rejected('case_success_group')

    def test_duplicate_case_rejected(self):
        self.cases.append(copy.deepcopy(self.cases[0])); self.save_cases()
        self.assert_rejected('private_case_duplicate')

    def test_hard_has_no_fabricated_role(self):
        self.cases[2]['reference'] = {'sibling': 0}; self.save_cases()
        self.assert_rejected('hard_case_fabricated')

    def test_origin_wrong_provenance_rejected(self):
        self.pairs[0]['reference_matching_source_rows'][0]['provenance']['line'] = 999
        self.save_pairs(); self.assert_rejected('stored_pair_origin_provenance')

    def test_duplicate_origin_rejected(self):
        self.pairs[0]['reference_matching_source_rows'].append(copy.deepcopy(self.pairs[0]['reference_matching_source_rows'][0]))
        self.save_pairs(); self.assert_rejected('private_pair_origin_missing_or_duplicate')

    def test_spool_selected_response_mismatch_rejected(self):
        db = sqlite3.connect(self.directory/'source.private.sqlite')
        text = db.execute('SELECT payload FROM rows WHERE qid=? AND sibling=0', ('nq-correct_correct',)).fetchone()[0]
        row = json.loads(text); row['response'] += 'changed'
        db.execute('UPDATE rows SET payload=? WHERE qid=? AND sibling=0', (roles.canonical(row), 'nq-correct_correct'))
        db.commit(); db.close(); self.assert_rejected('selected_role_pair_response')

    def test_spool_pending_wal_rejected(self):
        Path(str(self.directory/'source.private.sqlite')+'-wal').write_bytes(b'pending')
        self.assert_rejected('producer_spool_still_writable')

    def test_jsonl_partial_rejected(self):
        path = self.directory/'pairs.private.jsonl'; path.write_bytes(path.read_bytes()[:-1])
        self.assert_rejected('partial_or_oversize_private_jsonl')


if __name__ == '__main__':
    unittest.main()
