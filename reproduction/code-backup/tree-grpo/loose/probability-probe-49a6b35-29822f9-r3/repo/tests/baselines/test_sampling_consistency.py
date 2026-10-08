import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from search_r1.baselines.sampling_consistency import reduce_pair_statistics, record_raw_turn, finalize_calibration
from search_r1.baselines.training_state import file_sha


NEUTRAL = dict(temperature=1., top_p=1., top_k=-1, min_p=0., presence_penalty=0.,
               frequency_penalty=0., repetition_penalty=1., n=1, logits_processors=[])


def gloo_pair(rank, rendezvous, directory):
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        old = torch.zeros(rank+1, dtype=torch.float64)
        current = torch.full_like(old, .1 + .1*rank)
        good = reduce_pair_statistics(old, current, torch.ones_like(old), NEUTRAL, 1., 1.)
        if rank == 1: current[0] = float('nan')
        bad = reduce_pair_statistics(old, current, torch.ones_like(old), NEUTRAL, 1., 1.)
        try:
            reduce_pair_statistics(torch.zeros(2,2) if rank == 1 else old, current,
                                   torch.ones_like(old), NEUTRAL, 1., 1.)
            raise AssertionError('One-rank shape mismatch must synchronously refuse')
        except RuntimeError as exc:
            self_refusal = 'Global sampling diagnostic preparation refusal' in str(exc)
        (Path(directory)/f'{rank}.json').write_text(json.dumps(dict(good=good, bad=bad,global_shape_refusal=self_refusal)))
    finally:
        dist.destroy_process_group()


class Output:
    def __init__(self, batch, meta): self.batch, self.meta_info = batch, meta
    def __len__(self): return len(self.batch['responses'])


class SamplingConsistencyTests(unittest.TestCase):
    def test_masked_nonfinite_and_numeric_bound(self):
        old = torch.tensor([0., float('nan')], dtype=torch.float64)
        current = torch.tensor([.1, float('inf')], dtype=torch.float64)
        decision = reduce_pair_statistics(old, current, torch.tensor([1,0]), NEUTRAL, 1., .2)
        self.assertTrue(decision['allowed'])
        self.assertAlmostEqual(decision['sampled_k3_mean'], math.expm1(.1)-.1)
        self.assertFalse(reduce_pair_statistics(old, current, torch.ones(2), NEUTRAL, 1., .2)['allowed'])
        large = reduce_pair_statistics(torch.zeros(1), torch.tensor([800.]), torch.ones(1), NEUTRAL, 1., 1.)
        self.assertIsNone(large['sampled_k3_mean'])
        self.assertEqual(large['numeric_bound_exceeded'], 1)

    def test_broadcast_refused_and_large_finite_diagnostic_serializable(self):
        with self.assertRaisesRegex(RuntimeError, 'preparation refusal'):
            reduce_pair_statistics(torch.zeros(1), torch.zeros(2,3), torch.ones(2,3), NEUTRAL, 1., .01)
        result = reduce_pair_statistics(torch.zeros(1,dtype=torch.float64),
                    torch.tensor([-1e200],dtype=torch.float64), torch.ones(1), NEUTRAL, 1., .01)
        self.assertTrue(math.isfinite(result['rms_delta']))
        json.dumps(result,allow_nan=False)

    def test_distribution_not_silently_corrected(self):
        old = torch.zeros(1)
        for key, value in [('top_p', .9), ('repetition_penalty', 1.1), ('temperature', .7)]:
            filtered = dict(NEUTRAL, **{key:value})
            decision = reduce_pair_statistics(old, old, torch.ones(1), filtered, 1., .01)
            self.assertFalse(decision['allowed'])
            self.assertIn('sampling_distribution_not_plain_actor_softmax_or_invalid_limit', decision['reasons'])

    def test_real_gloo_global_count_weight_and_invalid_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(gloo_pair, args=(directory+'/rendezvous', directory), nprocs=2, join=True)
            results = [json.loads((Path(directory)/f'{rank}.json').read_text()) for rank in range(2)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]['good']['valid_tokens'], 3)
        expected = ((math.expm1(.1)-.1)+2*(math.expm1(.2)-.2))/3
        self.assertAlmostEqual(results[0]['good']['sampled_k3_mean'], expected)
        self.assertEqual(results[0]['bad']['invalid_tokens'], 1)
        self.assertFalse(results[0]['bad']['allowed'])
        self.assertTrue(results[0]['global_shape_refusal'])

    def test_private_probabilities_removed_without_old_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = dict(os.environ)
            try:
                os.environ.update(SEARCH_R1_BASELINE_OUTPUT=directory, SEARCH_R1_BASELINE_COMMIT='test-source',
                                  SEARCH_R1_BASELINE_CONFIG_SHA256='test-config',SEARCH_R1_BASELINE_DATA_IDENTITY='test-data',
                                  SEARCH_R1_BASELINE_RUN_ID='test-run',SLURM_JOB_ID='test',SEARCH_R1_MODEL_IDENTITY='test-model')
                worker = SimpleNamespace(config=SimpleNamespace(rollout=SimpleNamespace(temperature=1.,
                    sampling_consistency=SimpleNamespace(max_abs_delta=.01,enforce=True))),
                    actor_lr_scheduler=SimpleNamespace(last_epoch=110))
                old = torch.zeros((2,3))
                output = Output(dict(old_log_probs=old, __sampling_log_probs=old.clone(), responses=torch.ones(2,3),
                                     input_ids=torch.ones(2,5), position_ids=torch.ones(2,5),attention_mask=torch.ones(2,5)),
                                dict(__sampling_distribution=NEUTRAL))
                before = old.clone()
                record_raw_turn(worker, output)
                self.assertNotIn('__sampling_log_probs', output.batch)
                self.assertNotIn('__sampling_distribution', output.meta_info)
                torch.testing.assert_close(output.batch['old_log_probs'], before)
                record = json.loads(Path(worker._sampling_consistency_records[0]).read_text())
                self.assertTrue(record['decision']['allowed'])
                self.assertTrue(Path(record['snapshot_path']).exists())
            finally:
                os.environ.clear(); os.environ.update(previous)

    def test_driver_terminal_observation_does_not_credit_outer(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = dict(os.environ)
            try:
                os.environ.update(SEARCH_R1_BASELINE_OUTPUT=directory,SEARCH_R1_BASELINE_COMMIT='source',
                  SEARCH_R1_BASELINE_CONFIG_SHA256='config',SEARCH_R1_BASELINE_DATA_IDENTITY='data',
                  SEARCH_R1_MODEL_IDENTITY='model',SLURM_JOB_ID='test',SEARCH_R1_BASELINE_RUN_ID='test-run')
                root = Path(directory)/'probability-calibration'; root.mkdir()
                identity = dict(code_commit='source',config_sha256='config',data_identity='data',
                                model_identity='model',job_id='test',run_id='test-run')
                for rank in range(8):
                    snapshot = root/f'{rank}-raw.pt';torch.save(torch.ones(1),snapshot)
                    monitor = root/f'{rank}-monitor.pt';torch.save(torch.ones(1),monitor)
                    raw_path = root/f'{rank}-raw.json'
                    raw_path.write_text(json.dumps(dict(identity,rank=rank,scheduler_step=110,
                        snapshot_path=str(snapshot),snapshot_sha256=file_sha(snapshot))))
                    (root/f'rank-{rank:02d}.json').write_text(json.dumps(dict(rank=rank,world_size=8,
                        optimizer_committed=0,scheduler_advanced=False,scheduler_step=110,
                        parent_completed_outer=110,observed_outer=111,**identity,
                        monitor_path=str(monitor),monitor_sha256=file_sha(monitor),
                        diagnostic_records=[str(raw_path)],diagnostic_entries=[dict(path=str(raw_path),sha256=file_sha(raw_path),
                             snapshot_path=str(snapshot),snapshot_sha256=file_sha(snapshot))])))
                # No update_actor attribute exists: invoking it would fail this test.
                group = SimpleNamespace(calibrate_policy_safety=lambda batch:None)
                config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(policy_safety=SimpleNamespace(enabled=True)),
                    rollout=SimpleNamespace(temperature=1.,sampling_consistency=SimpleNamespace(enabled=True,enforce=False))))
                trainer = SimpleNamespace(config=config,actor_rollout_wg=group,global_steps=111,actual_outer_updates=110,
                  checkpoint_sampler=SimpleNamespace(cursor=58240),recovery_provenance=dict(parent_completed_step=110,parent_cursor=57600))
                batch = Output(dict(token_level_rewards=torch.ones(8,3)),{})
                ready = finalize_calibration(trainer,batch)
                self.assertEqual(ready['optimizer_committed'],0)
                self.assertEqual(ready['new_completed_outer'],0)
                self.assertEqual(trainer.actual_outer_updates,110)
                self.assertFalse(ready['scheduler_advanced'])
            finally:
                os.environ.clear(); os.environ.update(previous)


if __name__ == '__main__': unittest.main()
