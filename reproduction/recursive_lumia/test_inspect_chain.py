"""Physical-file continuation audits using small synthetic rank ledgers."""
import json
from pathlib import Path
import tempfile
import unittest

from .archive_entry import archive
from .inspect_chain import capture_boundary, inspect_chain
from .inspect_run import sha
from .prepare_ablation import condition_for


class ChainEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name).resolve()
        (self.run / 'receipts').mkdir()
        (self.run / 'output').mkdir()
        config = self.run / 'receipts/resolved.yaml'
        config.write_text('frozen: true\n')
        self.config = dict(base=str(self.run), config=str(config), config_sha256=sha(config),
            execution_backend='slurm_lumia', protocol='recursive_mechanism_ablation150_v1',
            group='C1-root-only', condition=condition_for('C1-root-only'), run_id='run',
            source_commit='a' * 40, data_identity='data')
        (self.run / 'receipts/configuration.json').write_text(json.dumps(self.config))

    def attempt(self, job, rows, resume=None):
        out = self.run / 'output' / ('attempt-' + str(job) * 32)
        out.mkdir()
        for name in ('fresh-state', 'attempt-results'):
            (out / name).mkdir()
        entry = dict(run_id='run', group='C1-root-only', source_commit=self.config['source_commit'],
            config_sha256=self.config['config_sha256'], host='gpu-4090-3', uid=1066,
            pid=100 + job, tick=1000 + job, allocation=str(job), output=str(out), resume_manifest=resume)
        (self.run / 'receipts/current-entry.json').write_text(json.dumps(entry))
        identity = dict(job_id=str(job), run_id='run', code_commit=self.config['source_commit'],
            config_sha256=self.config['config_sha256'], data_identity='data')
        (out / 'fresh-initialization.json').write_text(json.dumps(identity))
        log = self.run / ('job-%d.log' % job)
        lines = ['LUMIA_ABLATION_ENTRY=' + json.dumps(entry)]
        for outer, writes, total in rows:
            ranks = []
            for rank in range(8):
                ledger = [dict(index=i, rows=16, local_row_indices=list(range((i-1)*16, i*16)),
                    actual_writes=int(i <= writes), status='SUCCESS_ADAM_WRITE' if i <= writes else 'GUARD_SKIPPED',
                    gradient_preclip=.25 if i <= writes else None) for i in range(1,21)]
                direct = dict(identity, status='DIRECT_RANK_COMPLETE_OUTER_OBSERVED',
                    rank=rank, world_size=8, actual_optimizer_calls=writes, scheduler_step_before=outer-1,
                    scheduler_step_after=outer, optimizer_opportunity_ledger=ledger, attempt_id=outer)
                file = out / 'fresh-state' / ('%d-%d.json' % (outer, rank))
                file.write_text(json.dumps(direct))
                ranks.append(dict(rank=rank, optimizer_calls=writes, total_optimizer_calls=total,
                    scheduler_before=outer-1, scheduler_after=outer,
                    direct_rank_record=dict(path=str(file), sha256=sha(file))))
            guard = (dict(status='NORMAL_SAFETY_GUARD_ROUND_CONSUMED', optimizer_writes=writes,
                opportunities_processed=20, opportunities_skipped=20-writes) if writes < 20 else None)
            record = dict(accepted_outer=outer, attempt_id=outer,
                          all_rank_result=dict(ranks=ranks), normal_guard=guard)
            (out / 'attempt-results' / ('attempt-%06d.json' % outer)).write_text(json.dumps(record))
            lines.append('step:%d - actor/pg_loss:0.1 training/actual_outer_updates:%d' % (outer, outer))
        log.write_text('\n'.join(lines) + '\n')
        archived = archive(self.run, log, str(job), lambda _: '%d|jmzhang|RUNNING|0:0|\n' % job)
        return entry, out, dict(path=archived['archive'], sha256=archived['archive_sha256'])

    def fixture(self):
        parent, old, parent_ref = self.attempt(1, [(1,20,20), (2,20,40)])
        directory = self.run / 'checkpoints/native-rolling/global_step_1'
        directory.mkdir(parents=True)
        path = directory / 'manifest.json'
        identity = dict(job_id='1', code_commit=self.config['source_commit'],
            config_sha256=self.config['config_sha256'], data_identity='data', completed_step=1,
            actual_update_limit=150)
        workers = [dict(identity=dict(identity, rank=i, world_size=8), actual_total_optimizer_calls=20,
            scheduler_last_epoch=1, full_worker_state_finite=True, observed_adam_steps=[20], sha256=str(i)) for i in range(8)]
        manifest = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=identity,
                        worker_states=workers, driver=dict(next_step=2))
        path.write_text(json.dumps(manifest))
        progress = dict(completed_outer=1, run_id='run', source_commit=self.config['source_commit'],
            config_sha256=self.config['config_sha256'], checkpoint=dict(path=str(path), sha256=sha(path)),
            optimizer_calls_per_rank={str(i):20 for i in range(8)})
        (old / 'completed-outers.jsonl').write_text(json.dumps(progress) + '\n')
        boundary_ref = capture_boundary(self.run, path, sha(path))
        child, new, child_ref = self.attempt(2, [(2,13,33), (3,20,53)], resume=str(path))
        restored = dict(status='ABLATION_CURRENT_CONDITION_NATIVE_STATE_RESTORED',
            completed_outer=1, next_outer=2, parent_manifest=str(path), parent_manifest_sha256=sha(path),
            workers=[dict(rank=i, actual_state_equal=True, scheduler_step=1,
                          actual_total_optimizer_calls=20, parent_state_sha256=str(i)) for i in range(8)])
        (new / 'native-resumed.json').write_text(json.dumps(restored))
        chain = dict(version=1, run_id='run', attempts=[dict(entry=parent_ref),
                     dict(entry=child_ref, parent_boundary=boundary_ref)])
        chain_path = self.run / 'receipts/attempt-chain.json'
        chain_path.write_text(json.dumps(chain))
        return old, new, chain_path, path

    def query(self, _):
        return '1|jmzhang|TIMEOUT|0:0|\n2|jmzhang|RUNNING|0:0|\n'

    def test_reads_real_rank_files_and_discards_unsaved_parent_round(self):
        old, new, chain, path = self.fixture()
        path.unlink()  # Rolling cleanup is allowed after actual restore.
        result = inspect_chain(self.run, chain, query=self.query)
        self.assertEqual(result['completed_rounds'], 3)
        self.assertEqual(result['actual_adam_per_rank']['0'], 53)
        self.assertEqual([r['allocation'] for r in result['attempts']], ['1','2','2'])
        self.assertEqual(result['discarded_after_restored_boundary'][0]['outer'], 2)
        self.assertTrue(all(r['reported_pg_loss_finite'] for r in result['attempts']))
        self.assertFalse(result['initial_acceptance_verified'])

    def test_live_parent_or_missing_real_restoration_cannot_form_chain(self):
        old, new, chain, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'live'):
            inspect_chain(self.run, chain, query=lambda _: self.query(None).replace('TIMEOUT', 'RUNNING'))
        (new / 'native-resumed.json').unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_chain(self.run, chain, query=self.query)

    def test_changed_physical_rank_bytes_or_current_entry_is_rejected(self):
        old, new, chain, _ = self.fixture()
        file = old / 'fresh-state/1-0.json'
        original = file.read_bytes()
        file.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'modified'):
            inspect_chain(self.run, chain, query=self.query)
        file.write_bytes(original)
        current = self.run / 'receipts/current-entry.json'
        value = json.loads(current.read_text());value['pid'] += 1
        current.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'current'):
            inspect_chain(self.run, chain, query=self.query)

    def test_missing_parent_prefix_or_tampered_boundary_archive_is_rejected(self):
        old, new, chain, _ = self.fixture()
        value = json.loads(chain.read_text())
        original = chain.read_text()
        value['attempts'] = value['attempts'][1:]
        chain.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'first'):
            inspect_chain(self.run, chain, query=self.query)
        chain.write_text(original)
        value = json.loads(original)
        archive_path = Path(value['attempts'][1]['parent_boundary']['path'])
        archive_path.write_text(archive_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            inspect_chain(self.run, chain, query=self.query)


if __name__ == '__main__':
    unittest.main()
