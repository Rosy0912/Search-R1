"""Synthetic CPU-only sentinel and complete diagnostic orchestration fixtures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
import sys
BASE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPO = BASE / 'work/gigpo-e1-raw-action-identity-20261006'
sys.path.insert(0, str(REPO))
from tests.baselines.test_action_identity import ByteTokenizer
from search_r1.baselines import action_identity
spec = importlib.util.spec_from_file_location('raw_support_diagnostic', Path(__file__).with_name('diagnose_raw_support_v1.py'))
diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)


class Sentinels(unittest.TestCase):
    def test_true_pad_has_negative_logprob_and_is_kept(self):
        ids = torch.tensor([[1, 151643, 2, 151643]])
        logp = torch.tensor([[-1., -7., -2., 151643.]])
        lengths, masks = diag.derive_real_lengths(ids, logp, 151643, torch)
        self.assertEqual(lengths, [3]); self.assertEqual(masks, [[1, 1, 1, 0]])
    def test_full_empty_and_terminal_eos_lengths(self):
        ids = torch.tensor([[1, 151645], [151643, 151643], [1, 2]])
        logp = torch.tensor([[-1., -.2], [151643., 151643.], [-1., -2.]])
        self.assertEqual(diag.derive_real_lengths(ids, logp, 151643, torch)[0], [2, 0, 2])
    def test_nonterminal_sentinel_and_other_positive_reject(self):
        for logp in [[-1., 151643., -2.], [-1., .001, 151643.], [-1., float('nan'), 151643.]]:
            with self.subTest(logp=logp), self.assertRaises(RuntimeError):
                diag.derive_real_lengths(torch.tensor([[1, 151643, 151643]]), torch.tensor([logp]), 151643, torch)
    def test_tail_token_and_dtype_mismatch_reject(self):
        with self.assertRaises(RuntimeError):
            diag.derive_real_lengths(torch.tensor([[1, 2]]), torch.tensor([[-1., 151643.]]), 151643, torch)
        with self.assertRaises(RuntimeError):
            diag.derive_real_lengths(torch.tensor([[1., 151643.]]), torch.tensor([[-1., 151643.]]), 151643, torch)


class CompleteFixture(unittest.TestCase):
    def build(self, root):
        tok = ByteTokenizer()
        # The fixture has an explicit positive padding sentinel, with EOS99.
        tok.pad_token_id = 151643; tok.all_special_ids = [151643, 98, 99]
        prompt = tok.encode('Question?'); width = len(prompt)
        inventory, turns = [], []
        for serial, active in enumerate((2560, 2553, 761, 271)):
            rows = [dict(source=i, step_id=serial, generation_prefix=prompt,
                         retained_action_tokens=tok.encode('<search>q</search>')) for i in range(active)]
            turns.append(rows); per_rank = (active + 7) // 8
            for rank in range(8):
                path = root / f'probabilities/train/call-{serial}-rank-{rank}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
                ids = torch.tensor([[2, 3, 99, 151643]]).repeat(per_rank, 1)
                attention = torch.ones((per_rank, width + 4), dtype=torch.long); attention[:, -1] = 0
                old = torch.tensor([[-.4, -.8, -.3, 0.]]).repeat(per_rank, 1)
                raw = dict(rank=rank, serial=serial, sampled_ids=ids,
                    sampled_log_probs=torch.tensor([[-.5, -1., -.3, 151643.]]).repeat(per_rank, 1),
                    actor_repeated_log_probs=old.clone(), actual_engine_prompt_token_ids=[prompt] * per_rank,
                    tensors=dict(prompts=torch.tensor([prompt]).repeat(per_rank, 1), responses=ids,
                        attention_mask=attention, position_ids=torch.arange(width + 4).repeat(per_rank, 1), old_log_probs=old))
                torch.save(raw, path)
                receipt = dict(path=str(path), bytes=path.stat().st_size, sha256=diag.sha256(path)); inventory.append(receipt)
                record = dict(rank=rank, serial=serial, phase='train', source_commit=diag.SOURCE, rows=per_rank,
                              raw=receipt, repeated_probability_probe_rng_unchanged=True,
                              repeated_probability_probe_complete_state_unchanged=True,
                              repeated_probe_identity_before={'state':'same'}, repeated_probe_identity_after={'state':'same'})
                path.with_suffix('.json').write_text(json.dumps(record))
        (root / 'training-prefix-turns.json').write_text(json.dumps(turns))
        class Processor:
            def _postprocess_responses(self, ids):
                texts = [action_identity.canonical_environment_text(tok.decode(row, skip_special_tokens=True)) for row in ids]
                encoded = [tok.encode(text) for text in texts]
                return torch.tensor(encoded), texts
        return tok, Processor(), dict(raw_material_inventory=inventory)
    def test_all_four_turns_eight_ranks_real_analyze_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); tok, processor, parent = self.build(root)
            previous = diag.RUN; diag.RUN = root
            try:
                output = diag.analyze({}, parent, torch, tok, processor, action_identity)
            finally:
                diag.RUN = previous
            self.assertEqual(output['counters']['actions'], 6145)
            self.assertEqual(output['covered_sources'], 2560)
            for key in ('support_rejected','hf_response_token_mismatch','hf_position_or_mask_mismatch','canonical_replay_mismatch'):
                self.assertEqual(output['counters'][key], 0)
            self.assertEqual(output['counters']['raw_tokens'], 6145 * 2)
            self.assertAlmostEqual(output['token_residuals']['hf_vllm_max_abs'], .2, places=6)
            self.assertAlmostEqual(output['complete_action_joint_residuals']['hf_vllm_max_abs'], .3, places=6)
            self.assertFalse(output['new_full_action_prefix_scorer_executed'])
            self.assertFalse(output['probability_accepted'])
    def test_original_row_prefix_tampering_rejects_real_analyze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); tok, processor, parent = self.build(root)
            p = root / 'training-prefix-turns.json'; turns = json.loads(p.read_text()); turns[0][0]['generation_prefix'] = [1]; p.write_text(json.dumps(turns))
            previous = diag.RUN; diag.RUN = root
            try:
                with self.assertRaisesRegex(RuntimeError, 'canonical prefix differs'):
                    diag.analyze({}, parent, torch, tok, processor, action_identity)
            finally:
                diag.RUN = previous


if __name__ == '__main__': unittest.main()
