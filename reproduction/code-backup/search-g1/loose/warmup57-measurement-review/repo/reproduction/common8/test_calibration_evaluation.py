"""CPU tensor/protocol tests; fixtures cannot establish native or healthy admission."""
import copy
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from search_r1.qa_protocol import decode_model_spans, extract_final_answer

# Import the real primary-source reward functions without requiring the unrelated
# DataProto/Ray dependency stack for these typed CPU batch fixtures.
_qa_path = Path(__file__).resolve().parents[2] / 'verl/utils/reward_score/qa_em.py'
_spec = importlib.util.spec_from_file_location('_calibration_test_actual_qa_em', _qa_path)
qa_em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa_em)
_reward_package = types.ModuleType('verl.utils.reward_score')
_reward_package.qa_em = qa_em
with patch.dict(sys.modules, {'verl.utils.reward_score': _reward_package}):
    from reproduction.common8.calibration_evaluation import aggregate, rows_from_batch


class CharacterTokenizer:
    def decode(self, ids):
        return ''.join(chr(int(value)) for value in ids)


@dataclass
class BatchItem:
    batch: dict
    non_tensor_batch: dict


@dataclass
class TensorBatch:
    batch: dict
    non_tensor_batch: dict
    meta_info: dict

    def __len__(self):
        return self.batch['responses'].shape[0]

    def __getitem__(self, index):
        return BatchItem({key: value[index] for key, value in self.batch.items()},
                         {key: value[index] for key, value in self.non_tensor_batch.items()})


def tensor_batch(specifications):
    prompts, responses, masks = [], [], []
    for row in specifications:
        prompts.append([ord(value) for value in row.get('prompt', 'Question: name the place.\n')])
        responses.append([ord(value) for text, _ in row['segments'] for value in text])
        masks.append([int(authored) for text, authored in row['segments'] for _ in text])
    plen = max(map(len, prompts))
    rlen = max(1, max(map(len, responses)))
    count = len(specifications)
    prompt_tensor = torch.zeros(count, plen, dtype=torch.long)
    response_tensor = torch.zeros(count, rlen, dtype=torch.long)
    attention = torch.zeros(count, plen + rlen, dtype=torch.long)
    info = torch.zeros_like(attention)
    for index, (prompt, response, mask) in enumerate(zip(prompts, responses, masks)):
        prompt_tensor[index, plen - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        response_tensor[index, :len(response)] = torch.tensor(response, dtype=torch.long)
        attention[index, plen - len(prompt):plen + len(response)] = 1
        info[index, plen - len(prompt):plen] = 1
        info[index, plen:plen + len(mask)] = torch.tensor(mask, dtype=torch.long)
    return TensorBatch(dict(prompts=prompt_tensor, responses=response_tensor,
                            attention_mask=attention, info_mask=info,
                            token_level_scores=torch.randn(count, rlen)),
                       dict(__g1_question_id=[row.get('id', 'question-' + str(index))
                                             for index, row in enumerate(specifications)],
                            data_source=[row.get('source', 'nq') for row in specifications],
                            reward_model=[dict(ground_truth=dict(target=row.get('gold', ['Paris'])))
                                          for row in specifications]),
                       dict(actual_retrieval_stats=[row.get('actual', 0) for row in specifications],
                            valid_action_stats=[row.get('valid', 1) for row in specifications],
                            turns_stats=[row.get('turns', 1) for row in specifications],
                            valid_search_stats=[row.get('tag_search', 0) for row in specifications]))


def canonical_rows():
    ids = ['question-' + str(index) for index in range(512)]
    sources = ['nq'] * 256 + ['hotpotqa'] * 256
    specifications = [dict(id=qid, source=source, segments=[('<answer>Paris</answer>', True)])
                      for qid, source in zip(ids, sources)]
    return rows_from_batch(tensor_batch(specifications), CharacterTokenizer()), ids, sources


class CalibrationEvaluationTests(unittest.TestCase):
    def test_strict_em_matches_existing_generated_span_rule_and_excludes_spoofing(self):
        specifications = [
            dict(segments=[('<think>reason</think><answer>Paris</answer>', True)]),
            dict(segments=[('<search>place</search>', True),
                           ('<information><answer>Paris</answer></information>', False)],
                 actual=1, valid=1, turns=2),
            dict(segments=[('<answer>Pa', True), ('ri', False), ('s</answer>', True)], valid=0),
            dict(segments=[('<answer>Paris</answer>', True), ('evidence', False),
                           ('invalid later action', True)], valid=0),
            dict(segments=[('<answer>Paris</answer><answer>Paris</answer>', True)], valid=0),
            dict(segments=[('<answer><search>Paris</search></answer>', True)], valid=0),
            dict(prompt='Example: <answer>Paris</answer>\nQuestion: place?\n',
                 segments=[('No valid answer.', True)], valid=0),
            dict(segments=[], valid=0),
        ]
        batch = tensor_batch(specifications)
        rows = rows_from_batch(batch, CharacterTokenizer())
        original_em = []
        for item in (batch[index] for index in range(len(batch))):
            plen = item.batch['prompts'].numel()
            length = int(item.batch['attention_mask'][plen:].sum())
            spans = decode_model_spans(CharacterTokenizer(), item.batch['responses'][:length],
                                       item.batch['info_mask'][plen:plen + length])
            answer = extract_final_answer(spans)
            original_em.append(int(answer is not None and bool(qa_em.em_check(
                answer, item.non_tensor_batch['reward_model']['ground_truth']['target']))))
        self.assertEqual([row['em'] for row in rows], original_em)
        self.assertEqual(original_em, [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual([row['f1'] for row in rows], [1., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(rows[0]['answer_sha256'], hashlib.sha256(b'Paris').hexdigest())
        self.assertTrue(all(row['answer_sha256'] is None for row in rows[1:]))

    def test_f1_aliases_normalization_and_multiset_overlap(self):
        batch = tensor_batch([
            dict(segments=[('<answer>New York</answer>', True)], gold=['London', 'New York City']),
            dict(segments=[('<answer>red red blue</answer>', True)], gold=['red blue blue']),
            dict(segments=[('<answer>The New, York!</answer>', True)], gold=['new york']),
            dict(segments=[('<answer>Paris</answer>', True)], gold=['London', 'Paris']),
            dict(segments=[('<answer>unknown</answer>', True)], gold=['Paris']),
        ])
        rows = rows_from_batch(batch, CharacterTokenizer())
        self.assertEqual([row['em'] for row in rows], [0, 0, 1, 1, 0])
        for row, expected in zip(rows, [.8, 2. / 3., 1., 1., 0.]):
            self.assertAlmostEqual(row['f1'], expected)

    def test_executed_search_uses_executor_metadata_excluding_final_unexecuted_tag(self):
        batch = tensor_batch([dict(segments=[('<search>real</search>', True),
                                            ('<information>doc</information>', False),
                                            ('<search>final unexecuted</search>', True)],
                                   actual=1, valid=2, turns=2, tag_search=2)])
        row = rows_from_batch(batch, CharacterTokenizer())[0]
        self.assertEqual((row['actual_calls'], row['valid_action'], row['turns']), (1, 2, 2))
        self.assertEqual(row['em'], 0)
        batch.meta_info['actual_retrieval_stats'] = [0]
        self.assertEqual(rows_from_batch(batch, CharacterTokenizer())[0]['actual_calls'], 0)

    def test_real_tensor_lengths_padding_and_no_input_or_reward_mutation(self):
        batch = tensor_batch([dict(prompt='Short?', segments=[('<answer>Paris</answer>', True)]),
                              dict(prompt='A considerably longer prompt?', segments=[], valid=0)])
        before = {key: value.clone() for key, value in batch.batch.items()}
        metadata = copy.deepcopy(batch.meta_info)
        rows = rows_from_batch(batch, CharacterTokenizer())
        self.assertEqual(rows[0]['response_token_count'], len('<answer>Paris</answer>'))
        self.assertEqual(rows[0]['model_token_count'], rows[0]['response_token_count'])
        self.assertEqual(rows[1]['response_token_count'], 0)
        self.assertEqual(batch.meta_info, metadata)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in batch.batch.items()))
        encoded = json.dumps(rows)
        self.assertNotIn('Paris', encoded)
        self.assertNotIn('Short?', encoded)
        self.assertNotIn('<answer>', encoded)

    def test_bad_executor_metadata_collectively_unused_but_locally_rejected(self):
        original = tensor_batch([dict(segments=[('<answer>Paris</answer>', True)])])
        for key in ('actual_retrieval_stats', 'valid_action_stats', 'turns_stats'):
            for value in (None, [], [0, 0], ['1'], [True], [1.0], [-1]):
                with self.subTest(key=key, value=value):
                    batch = copy.deepcopy(original)
                    batch.meta_info[key] = value
                    with self.assertRaises(ValueError):
                        rows_from_batch(batch, CharacterTokenizer())
        for actual, valid, turns in ((2, 1, 2), (0, 2, 1), (5, 5, 5), (0, 0, 0), (0, 0, 6)):
            batch = copy.deepcopy(original)
            batch.meta_info.update(actual_retrieval_stats=[actual], valid_action_stats=[valid], turns_stats=[turns])
            with self.subTest(counts=(actual, valid, turns)), self.assertRaises(ValueError):
                rows_from_batch(batch, CharacterTokenizer())

    def test_bad_tensor_masks_and_row_identity_rejected(self):
        original = tensor_batch([dict(segments=[('<answer>Paris</answer>', True)])])
        for variant in ('attention_nonbinary', 'attention_hole', 'authorship_nonbinary',
                        'response_wrong_dtype', 'mask_wrong_width', 'missing_id', 'bad_source', 'bad_gold'):
            batch = copy.deepcopy(original)
            plen = batch.batch['prompts'].shape[1]
            if variant == 'attention_nonbinary': batch.batch['attention_mask'][0, plen] = 2
            elif variant == 'attention_hole': batch.batch['attention_mask'][0, plen + 1] = 0
            elif variant == 'authorship_nonbinary': batch.batch['info_mask'][0, plen] = 2
            elif variant == 'response_wrong_dtype': batch.batch['responses'] = batch.batch['responses'].float()
            elif variant == 'mask_wrong_width': batch.batch['info_mask'] = batch.batch['info_mask'][:, :-1]
            elif variant == 'missing_id': del batch.non_tensor_batch['__g1_question_id']
            elif variant == 'bad_source': batch.non_tensor_batch['data_source'] = ['triviaqa']
            elif variant == 'bad_gold': batch.non_tensor_batch['reward_model'][0]['ground_truth']['target'] = []
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                rows_from_batch(batch, CharacterTokenizer())

    def test_fixed512_micro_action_denominator_and_weighted_metrics(self):
        rows, ids, sources = canonical_rows()
        for index, row in enumerate(rows):
            if index >= 256:
                row.update(em=0, f1=.5, actual_calls=2, valid_action=3, turns=5)
        result = aggregate(rows, ids, sources)
        self.assertEqual(result['weighted_em'], .5)
        self.assertEqual(result['weighted_f1'], .75)
        self.assertEqual(result['executed_search_question_rate'], .5)
        self.assertEqual(result['executed_search_calls_per_question'], 1.)
        self.assertEqual(result['valid_action_count'], 1024)
        self.assertEqual(result['action_opportunity_count'], 1536)
        self.assertAlmostEqual(result['valid_action_rate'], 2. / 3.)
        self.assertNotEqual(result['valid_action_rate'], (1. + 3. / 5.) / 2.)
        self.assertEqual(result['question_ids'], ids)
        self.assertEqual(result['data_sources'], sources)
        self.assertIs(result['production_admission'], False)
        self.assertNotIn('accepted', result)
        self.assertNotIn('healthy', result)

    def test_missing_extra_reordered_unpaired_or_wrong_source_counts_rejected(self):
        rows, ids, sources = canonical_rows()
        for variant in ('missing', 'extra', 'reordered', 'wrong_id', 'wrong_source', 'wrong_canonical_count'):
            altered = copy.deepcopy(rows)
            expected_ids, expected_sources = list(ids), list(sources)
            if variant == 'missing': altered.pop()
            elif variant == 'extra': altered.append(copy.deepcopy(altered[-1]))
            elif variant == 'reordered': altered[0], altered[1] = altered[1], altered[0]
            elif variant == 'wrong_id': altered[0]['question_id'] = altered[1]['question_id']
            elif variant == 'wrong_source': altered[0]['data_source'] = 'hotpotqa'
            else: expected_sources[0] = 'hotpotqa'
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                aggregate(altered, expected_ids, expected_sources)
        before = aggregate(rows, ids, sources)
        after = aggregate(rows, ids, sources)
        self.assertEqual(before['question_ids'], after['question_ids'])

    def test_canonical_duplicate_occurrences_are_retained_without_deduplication(self):
        rows, ids, sources = canonical_rows()
        ids[1] = ids[0]
        rows[1]['question_id'] = ids[0]
        result = aggregate(rows, ids, sources)
        self.assertEqual(result['question_count'], 512)
        self.assertEqual(result['question_ids'][:2], [ids[0], ids[0]])

    def test_malformed_scalar_or_raw_qa_rows_cannot_enter_aggregate(self):
        rows, ids, sources = canonical_rows()
        for key, value in (('em', True), ('em', 2), ('f1', float('nan')), ('f1', -.1),
                           ('actual_calls', True), ('answer_sha256', 'bad'), ('answer_present', 1),
                           ('model_token_count', -1), ('turns', 0), ('answer', 'Paris')):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                aggregate(bad, ids, sources)
        bad = copy.deepcopy(rows)
        bad[0].update(answer_present=False, answer_sha256=None)
        with self.assertRaises(ValueError):
            aggregate(bad, ids, sources)


if __name__ == '__main__':
    unittest.main()
