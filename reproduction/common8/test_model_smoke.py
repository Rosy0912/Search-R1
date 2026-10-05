"""CPU rejection tests only; these never instantiate the real GPU worker."""
import copy
import unittest
import torch
from reproduction.common8.model_smoke import compare, aggregate, LOGPROB_ATOL


class ModelSmokeTests(unittest.TestCase):
    def samples(self):
        old = torch.tensor([[-0.1, -0.2, float('nan')], [-0.3, float('nan'), float('nan')]])
        reference = old.clone()
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        tokens = torch.tensor([[5, 6], [7, 0]])
        raw = torch.tensor([[-0.1, -0.2], [-0.3, float('nan')]])
        responses = torch.tensor([[5, 6, 0], [7, 0, 0]])
        return old, reference, mask, tokens, raw, responses

    def proofs(self):
        ids = [str(i) for i in range(64)]; uuids = [str(i) for i in range(8)]
        rows = [dict(accepted=True, rank=r, run_id='cpu-only', slurm_job_id='cpu-only', phase='initial',
                     cuda_uuid=uuids[r], actual_optimizer_writes=0, sync_calls=1, sync_load_format='dtensor',
                     question_ids=ids[r*8:(r+1)*8], resolved_config_sha256='same',
                     parity=dict(tolerance=LOGPROB_ATOL, actor_reference_max_abs=0., actor_vllm_max_abs=0.,
                                 valid_tokens_per_question=[1]*8)) for r in range(8)]
        return rows, ids, uuids

    def test_valid_tokens_only_without_replacing_bad_padding(self):
        proof = compare(*self.samples())
        self.assertEqual(proof['valid_tokens_per_question'], [2, 1])
        self.assertEqual(proof['actor_vllm_max_abs'], 0)

    def test_valid_token_nonfinite_or_backend_mismatch_rejected(self):
        for field, value in ((0, float('nan')), (1, float('inf')), (4, -4.0)):
            with self.subTest(field=field):
                inputs = self.samples(); inputs[field][0, 0] = value
                with self.assertRaises(AssertionError): compare(*inputs)

    def test_wrong_token_empty_row_and_mask_hole_rejected(self):
        for case in ('token', 'empty', 'hole'):
            with self.subTest(case=case):
                inputs = self.samples()
                if case == 'token': inputs[3][0, 0] = 99
                elif case == 'empty': inputs[2][0].zero_()
                else: inputs[2][0] = torch.tensor([1, 0, 1])
                with self.assertRaises(AssertionError): compare(*inputs)

    def test_all_eight_proofs_cover_exact_ordered_sixty_four(self):
        rows, ids, uuids = self.proofs()
        self.assertEqual(aggregate(rows, 'cpu-only', 'cpu-only', 'initial', ids, uuids), 'same')
        for change in ('missing', 'identity', 'different_config', 'wrong_sync', 'optimizer_write'):
            with self.subTest(change=change):
                bad = copy.deepcopy(rows)
                if change == 'missing': bad.pop()
                elif change == 'identity': bad[0]['question_ids'].reverse()
                elif change == 'different_config': bad[0]['resolved_config_sha256'] = 'other'
                elif change == 'wrong_sync': bad[0]['sync_load_format'] = 'hf'
                else: bad[0]['actual_optimizer_writes'] = 1
                with self.assertRaises(AssertionError): aggregate(bad, 'cpu-only', 'cpu-only', 'initial', ids, uuids)

    def test_wrong_job_gpu_and_nonfinite_proof_rejected(self):
        rows, ids, uuids = self.proofs()
        for field, value in (('slurm_job_id', 'other'), ('cuda_uuid', 'other')):
            bad = copy.deepcopy(rows); bad[0][field] = value
            with self.assertRaises(AssertionError): aggregate(bad, 'cpu-only', 'cpu-only', 'initial', ids, uuids)
        rows[0]['parity']['actor_vllm_max_abs'] = float('nan')
        with self.assertRaises(AssertionError): aggregate(rows, 'cpu-only', 'cpu-only', 'initial', ids, uuids)


if __name__ == '__main__': unittest.main()
