"""Bounded CPU fixtures; synthetic tokenizer is not official/GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import diagnose_actions_v1 as diagnostic
from diagnose_actions_v1 import (GENERATION_SHA, checked_file, extract_frozen_postprocess,
    first_difference, mapping_check, retained_replay, row_diagnostic,
    selected_scoring_action, text_boundary_evidence, trim_terminal_pad, byte_decoder)

SOURCE = Path(__file__).parents[2] / 'gigpo-e1-quality-resume/search_r1/llm_agent/generation.py'


class FixtureTokenizer:
    pad_token_id = 99
    eos_token_id = 98
    all_special_ids = [98, 99]
    def __init__(self, pieces, recode):
        self.pieces = pieces
        self.recode = recode
    def convert_ids_to_tokens(self, ids):
        return [self.pieces[value] if value not in self.all_special_ids else '<special>' for value in ids]
    def decode(self, ids, skip_special_tokens=True):
        inverse = byte_decoder()
        raw = b''.join(bytes(inverse[char] for char in self.pieces[value])
                       for value in ids if not skip_special_tokens or value not in self.all_special_ids)
        return raw.decode('utf-8', errors='replace')
    def batch_decode(self, values, skip_special_tokens=True):
        return [self.decode(row.tolist(), skip_special_tokens) for row in values]
    def encode(self, text, add_special_tokens=False):
        return self.recode[text]
    def __call__(self, texts, **kwargs):
        values = [self.encode(text, add_special_tokens=False) for text in texts]
        width = max(map(len, values))
        return {'input_ids': torch.tensor([row + [self.pad_token_id] * (width - len(row)) for row in values])}


class DiagnosticFixtures(unittest.TestCase):
    def setUp(self):
        self.processor, self.ast_sha = extract_frozen_postprocess(SOURCE, torch)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), GENERATION_SHA)

    def replay(self, tokenizer, raw):
        self.processor.tokenizer = tokenizer
        tokens, texts = self.processor._postprocess_responses(torch.tensor([raw]))
        return tokens[0][tokens[0].ne(tokenizer.pad_token_id)].tolist(), texts[0]

    def test_exact_frozen_postprocess_keeps_identical_token_path(self):
        tok = FixtureTokenizer({1: 'abc'}, {'abc': [1]})
        self.assertEqual(self.replay(tok, [1, 99])[0], [1])

    def test_incomplete_utf8_tail_expands_after_real_postprocess(self):
        # GPT2 ByteLevel alphabet maps F0 to 'ð'; incomplete UTF8 decodes U+FFFD.
        tok = FixtureTokenizer({1: 'a', 2: 'ð', 3: 'ï', 4: '¿', 5: '½'}, {'a\ufffd': [1, 3, 4, 5]})
        recoded, text = self.replay(tok, [1, 2])
        self.assertEqual(recoded, [1, 3, 4, 5])
        proof = text_boundary_evidence(tok, [1, 2], 'a\ufffd', text)
        self.assertFalse(proof['strict_utf8_valid'])
        self.assertTrue(proof['strict_utf8_error']['reaches_end'])

    def test_closing_inside_token_is_not_sampled_prefix(self):
        tok = FixtureTokenizer({1: 'a</answer>x', 2: 'a', 3: '</answer>'},
                               {'a</answer>x': [1], 'a</answer>': [2, 3]})
        recoded, text = self.replay(tok, [1])
        proof = text_boundary_evidence(tok, [1], 'a</answer>x', text)
        self.assertTrue(proof['closing_inside_sampled_token'])
        self.assertEqual(recoded, [2, 3])

    def test_search_closing_precedence_is_preserved(self):
        value = '<answer>a</answer>x<search>b</search>tail'
        truncated = '<answer>a</answer>x<search>b</search>'
        tok = FixtureTokenizer({1: value, 2: truncated}, {value: [1], truncated: [2]})
        self.assertEqual(self.replay(tok, [1]), ([2], truncated))

    def test_absent_closing_not_synthesized(self):
        tok = FixtureTokenizer({1: '<answer>a'}, {'<answer>a': [1]})
        self.assertEqual(self.replay(tok, [1]), ([1], '<answer>a'))

    def test_terminal_pad_only_middle_pad_retained(self):
        self.assertEqual(trim_terminal_pad([1, 99, 2, 99, 99], 99), [1, 99, 2])

    def test_eos_is_explicit_removed_special_not_extra_token(self):
        tok = FixtureTokenizer({1: 'abc'}, {'abc': [1]})
        row = dict(source=0, step_id=0, generation_prefix=[6], scoring_prefix=[6], retained_action_tokens=[1])
        diag = row_diagnostic(tok, [1, 98, 99], [1, 98, 99], [1], 'abc', row, 1,
            dict(worker_prompt_matches_observer=True, engine_prompt_matches_observer=True,
                 worker_engine_prefix_match=True), [1], 3, 99, 98)
        self.assertEqual(diag['eos_positions'], [1])
        self.assertTrue(diag['full_decode_reencode_equals_sampled_after_explicit_special_removal'])

    def test_wrong_worker_row_prefix_is_detected(self):
        raw = dict(tensors={'prompts': torch.tensor([[99, 4], [99, 5]])},
                   actual_engine_prompt_token_ids=[[4], [5]])
        row = dict(generation_prefix=[5])
        self.assertFalse(mapping_check(raw, row, 0, 99)['worker_prompt_matches_observer'])
        self.assertTrue(mapping_check(raw, row, 1, 99)['worker_prompt_matches_observer'])

    def test_scoring_permutation_and_observation_mask(self):
        batch = dict(tensors=dict(step_ids=torch.tensor([[0, -1, 1, -1]]),
                     loss_mask=torch.tensor([[1, 0, 1, 0]]), responses=torch.tensor([[3, 70, 4, 80]])))
        self.assertEqual(selected_scoring_action(batch, 0, 1), [4])

    def test_response_cap_shortens_only(self):
        self.assertEqual(retained_replay([1, 2, 3], {'scoring_prefix': list(range(4097))}, 2), [1])

    def test_file_sha_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = (Path(temp).resolve() / 'original.json')
            path.write_bytes(b'abc')
            record = dict(path=str(path), bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
            before = checked_file(record)
            self.assertEqual(before, checked_file(record))
            path.write_bytes(b'abd')
            with self.assertRaisesRegex(RuntimeError, 'bytes/SHA differ'):
                checked_file(record)

    def test_first_difference_preserves_length_failure(self):
        self.assertEqual(first_difference([1], [1, 2]), {'index': 1, 'left': None, 'right': 2})

    def actual_metadata_fixture(self):
        base = Path(__file__).parents[3] / 'outputs/gigpo-e1-policy-safety'
        failure_path = base / 'CALIBRATION3289-FAILURE-PAYLOAD-v1.json'
        self.assertEqual(hashlib.sha256(failure_path.read_bytes()).hexdigest(), diagnostic.INVENTORY)
        failure = json.loads(failure_path.read_text())
        spec = json.loads((base / 'CALIBRATION3289-ACTION-IDENTITY-CPU-INPUT-v1.json').read_text())
        return spec, failure

    def test_actual_failure_metadata_scope_accepts_string_job_id(self):
        spec, failure = self.actual_metadata_fixture()
        self.assertIs(type(failure['job_id']), str)
        diagnostic.validate_spec(spec, failure)

    def test_actual_failure_metadata_wrong_job_type_rejects(self):
        spec, failure = self.actual_metadata_fixture()
        failure['job_id'] = 3289
        with self.assertRaisesRegex(RuntimeError, 'Wrong failure original'):
            diagnostic.validate_spec(spec, failure)

    def test_actual_mmap_and_all_serial_rank_orchestration_fixture(self):
        # The target-only entry is intentionally not invoked on Mac. Core uses
        # actual Torch mmap/PT/JSON and the exact frozen postprocess AST.
        tok = FixtureTokenizer({1: 'abc'}, {'abc': [1]})
        self.processor.tokenizer = tok
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp).resolve()
            directory = run / 'probabilities/train'
            directory.mkdir(parents=True)
            inventory, turns = [], []
            for serial in range(4):
                prefix = [6] + [1] * serial
                turns.append([dict(source=i, step_id=serial, generation_prefix=prefix,
                    scoring_prefix=prefix, retained_action_tokens=[1]) for i in range(2560)])
                for rank in range(8):
                    path = directory / f'call-{serial}-rank-{rank}.pt'
                    raw = dict(phase='train', serial=serial, rank=rank, sampled_ids=torch.ones(320, 1, dtype=torch.long),
                        tensors=dict(responses=torch.ones(320, 1, dtype=torch.long),
                                     prompts=torch.tensor([prefix] * 320)),
                        actual_engine_prompt_token_ids=[prefix] * 320)
                    torch.save(raw, path)
                    record = dict(path=str(path), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    inventory.append(record)
                    path.with_suffix('.json').write_text(json.dumps(dict(phase='train', serial=serial,
                        rank=rank, rows=320, source_commit=diagnostic.SOURCE, raw=record)))
            (run / 'training-prefix-turns.json').write_text(json.dumps(turns))
            scored = dict(tensors=dict(responses=torch.ones(2560, 4, dtype=torch.long),
                step_ids=torch.tensor([[0, 1, 2, 3]] * 2560), loss_mask=torch.ones(2560, 4, dtype=torch.long)),
                non_tensors=dict(audit_trajectory_row=list(range(2560))),
                metadata=dict(baseline_outer_step=26), scope='GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION')
            torch.save(scored, run / 'actual-rollout-2560.pt')
            spec = dict(raw_material_inventory=inventory, pad_token_id=99, eos_token_id=98)
            with patch.object(diagnostic, 'RUN', run):
                result = diagnostic.analyze(spec, torch, tok, self.processor)
            self.assertEqual(result['unique_source_steps'], 10240)
            self.assertEqual(result['counters']['sampled_prefix_mismatch'], 0)
            self.assertEqual(result['bounded_bad_examples'], [])



if __name__ == '__main__':
    unittest.main()
