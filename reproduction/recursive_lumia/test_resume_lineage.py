import copy
import unittest

from .resume_lineage import validate_link, merge_rounds


class ResumeLineage(unittest.TestCase):
    def boundary(self):
        parent = dict(run_id='run', group='C1-root-only', source_commit='source',
            config_sha256='config', host='gpu-4090-3', allocation='1', output='/run/output/a')
        child = dict(parent, allocation='2', output='/run/output/b', resume_manifest='/native/manifest.json')
        identity = dict(job_id='1', code_commit='source', config_sha256='config', data_identity='data',
                        completed_step=2, actual_update_limit=150)
        workers = [dict(identity=dict(identity, rank=i, world_size=8), actual_total_optimizer_calls=33,
            scheduler_last_epoch=2, full_worker_state_finite=True, observed_adam_steps=[33], sha256=str(i)) for i in range(8)]
        manifest = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=identity,
                        worker_states=workers, driver=dict(next_step=3))
        restored = dict(status='ABLATION_CURRENT_CONDITION_NATIVE_STATE_RESTORED',
            completed_outer=2, next_outer=3, parent_manifest='/native/manifest.json', parent_manifest_sha256='hash',
            workers=[dict(rank=i, actual_state_equal=True, scheduler_step=2,
                          actual_total_optimizer_calls=33, parent_state_sha256=str(i)) for i in range(8)])
        return parent, child, manifest, restored, '/native/manifest.json', 'hash'

    def test_real_restoration_shape_preserves_partial_actual_adam_total(self):
        result = validate_link(*self.boundary())
        self.assertEqual(result['completed_outer'], 2)
        self.assertEqual(result['actual_adam_per_rank'], {str(i): 33 for i in range(8)})

    def test_missing_rank_wrong_parent_state_or_cold_reset_is_rejected(self):
        for mode in ('missing_rank', 'wrong_file', 'reset_adam', 'reset_scheduler', 'wrong_manifest', 'different_host'):
            args = copy.deepcopy(self.boundary())
            if mode == 'missing_rank': args[3]['workers'].pop()
            if mode == 'wrong_file': args[3]['workers'][0]['parent_state_sha256'] = 'different'
            if mode == 'reset_adam': args[3]['workers'][0]['actual_total_optimizer_calls'] = 0
            if mode == 'reset_scheduler': args[3]['workers'][0]['scheduler_step'] = 0
            if mode == 'wrong_manifest': args[3]['parent_manifest_sha256'] = 'different'
            if mode == 'different_host': args[1]['host'] = 'gpu-l40s-1'
            with self.assertRaises(ValueError): validate_link(*args)

    def row(self, outer, writes, total):
        guard = (dict(status='NORMAL_SAFETY_GUARD_ROUND_CONSUMED', optimizer_writes=writes,
            opportunities_processed=20, opportunities_skipped=20-writes) if writes < 20 else None)
        return dict(outer=outer, actual_adam_writes_per_rank=writes, normal_guard=guard,
                    cumulative_adam_writes_per_rank={str(i): total for i in range(8)})

    def chain(self):
        link = validate_link(*self.boundary())
        # Parent wrote outer3, but crashed before saving it; child restores2.
        parent = dict(job='1', attempts=[self.row(1,20,20), self.row(2,13,33), self.row(3,20,53)])
        child = dict(job='2', attempts=[self.row(3,0,33), self.row(4,20,53)])
        return [parent, child], [link]

    def test_rollback_discards_unsaved_updates_and_counts_zero_guard_as_one_round(self):
        result = merge_rounds(*self.chain())
        self.assertEqual(result['completed_rounds'], 4)
        self.assertEqual(result['actual_adam_per_rank']['0'], 53)
        self.assertEqual(result['discarded_after_restored_boundary'][0]['actual_adam_per_rank'], 20)
        self.assertEqual([r['outer'] for r in result['attempts']], [1,2,3,4])

    def test_duplicate_round_gap_double_count_reset_and_unguarded_skip_rejected(self):
        for mode in ('duplicate', 'gap', 'double_count', 'reset', 'missing_guard', 'false_parent'):
            segments, links = self.chain()
            if mode == 'duplicate': segments[1]['attempts'][0]['outer'] = 2
            if mode == 'gap': segments[1]['attempts'][0]['outer'] = 4
            if mode == 'double_count': segments[1]['attempts'][0]['cumulative_adam_writes_per_rank']['0'] = 53
            if mode == 'reset': segments[1]['attempts'][0]['cumulative_adam_writes_per_rank']['0'] = 0
            if mode == 'missing_guard': segments[1]['attempts'][0]['normal_guard'] = None
            if mode == 'false_parent': links[0]['actual_adam_per_rank']['0'] = 40
            with self.assertRaises(ValueError): merge_rounds(segments, links)


if __name__ == '__main__':
    unittest.main()
