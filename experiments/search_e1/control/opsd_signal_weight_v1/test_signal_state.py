"""Independent four-rank reduction, original stream, and durable cursor checks."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cohort_v1
import miner_v1
from trainer import batches, rank_batch, validate_science, verify_resume_cursor


class PortContracts(unittest.TestCase):
    def test_four_rank_sum_matches_global_mean_with_tails(self):
        import torch
        for count in (1, 7, 31, 32):
            parameter = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
            rows = torch.arange(count * 2, dtype=torch.float64).reshape(count, 2) / 19
            # Separate rank-local autograd graphs, each with actual global denominator.
            gradients = []
            for rank in range(4):
                local = parameter.detach().clone().requires_grad_(True)
                indices = rank_batch(list(range(count)), rank, 4)
                loss = sum(((local * rows[i]).sum() - 1).square() / count for i in indices)
                gradients.append(torch.autograd.grad(loss, local)[0] if indices else torch.zeros_like(local))
            full = sum(((parameter * row).sum() - 1).square() / count for row in rows)
            expected = torch.autograd.grad(full, parameter)[0]
            self.assertTrue(torch.allclose(sum(gradients), expected, atol=1e-12, rtol=1e-12))
            if count > 1:
                self.assertFalse(torch.allclose(sum(gradients) / 4, expected))

    def test_uniform_selection_original_first_ten_stream_and_extension(self):
        for outer in range(25):
            before = random.getstate()
            selected, proof = miner_v1.uniform_sibling('same-private-fixture', outer)
            payload = ['search-e1-opsd-uniform-five-sha256-u64-rejection-v1', 42, outer, 'same-private-fixture', 0]
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).digest()
            self.assertEqual(selected, int.from_bytes(digest[:8], 'big') % 5)
            self.assertEqual(proof['accepted_sha256'], digest.hex())
            self.assertEqual(before, random.getstate())
        for outer in (-1, 25):
            with self.assertRaises(ValueError):
                miner_v1.uniform_sibling('fixture', outer)

    def test_cohort_prefix_original_shuffle_no_new_rng_or_replacements(self):
        train = [dict(question_id='train-' + str(i), question='q-' + str(i), answers=['x']) for i in range(169103)]
        validation = [dict(question_id='held-' + str(i), question='held-q-' + str(i), answers=['x']) for i in range(512)]
        refs = [dict(path='/remote/' + name, sha256='a' * 64) for name in ('train', 'validation')]
        frozen = cohort_v1.freeze_cohorts(train, validation, *refs)
        original = list(range(169103)); random.Random(42).shuffle(original)
        self.assertEqual(frozen['outer_count'], 25)
        actual = [i for cohort in frozen['cohorts'] for i in cohort['train_indices']]
        self.assertEqual(actual, original[:12800])
        self.assertEqual(actual[:5120], original[:5120])
        self.assertEqual(len(set(actual)), 12800)

    def test_committed_cursor_does_not_repeat_prior_updates(self):
        state = dict(cursor=dict(outer_index=14, next_pair_cursor=64, order_sha256='order', encoded_sha256='encoding', usable_pair_count=71))
        cursor = verify_resume_cursor(state, 14, 'order', 'encoding', 71)
        self.assertEqual(batches(list(range(71)), cursor), [list(range(64, 71))])
        for field, invalid in [('outer_index', 13), ('order_sha256', 'new'), ('encoded_sha256', 'new'), ('next_pair_cursor', 63)]:
            changed = copy.deepcopy(state); changed['cursor'][field] = invalid
            with self.assertRaises(ValueError):
                verify_resume_cursor(changed, 14, 'order', 'encoding', 71)


if __name__ == '__main__':
    unittest.main()
