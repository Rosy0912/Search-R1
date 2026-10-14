import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .archive_entry import archive
from .prepare_ablation import condition_for


class EntryArchive(unittest.TestCase):
    def fixture(self, root):
        run = Path(root).resolve()
        receipts = run / 'receipts'
        receipts.mkdir()
        config = receipts / 'resolved.yaml'
        config.write_text('frozen: true\n')
        record = dict(base=str(run), config=str(config),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            execution_backend='slurm_lumia', run_id='run', group='C1-root-only',
            condition=condition_for('C1-root-only'), source_commit='a' * 40)
        entry = dict(record, allocation='123', uid=1066, pid=42, tick=99,
                     output=str(run / 'output' / ('attempt-' + 'a' * 32)))
        (receipts / 'configuration.json').write_text(json.dumps(record))
        (receipts / 'current-entry.json').write_text(json.dumps(entry))
        log = run / 'job.log'
        log.write_text('LUMIA_ABLATION_ENTRY=' + json.dumps(entry) + '\n')
        return run, log, entry

    def test_archive_survives_current_entry_replacement_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            run, log, entry = self.fixture(root)
            query = lambda _: '123|jmzhang|RUNNING|0:0|\n'
            first = archive(run, log, '123', query)
            second = archive(run, log, '123', query)
            self.assertEqual(first['archive_sha256'], second['archive_sha256'])
            self.assertFalse(first['restart_authorized_by_this_receipt'])
            (run / 'receipts/current-entry.json').write_text(json.dumps(dict(entry, allocation='124')))
            self.assertEqual(json.loads(Path(first['archive']).read_text())['entry'], entry)

    def test_wrong_scheduler_owner_and_launch_log_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run, log, entry = self.fixture(root)
            for scheduler in ('123|other|RUNNING|0:0|\n', '124|jmzhang|RUNNING|0:0|\n'):
                with self.assertRaises(ValueError):
                    archive(run, log, '123', lambda _: scheduler)
            log.write_text('LUMIA_ABLATION_ENTRY=' + json.dumps(dict(entry, pid=99)) + '\n')
            with self.assertRaises(ValueError):
                archive(run, log, '123', lambda _: '123|jmzhang|RUNNING|0:0|\n')

    def test_config_mutation_or_capture_race_cannot_produce_valid_archive(self):
        with tempfile.TemporaryDirectory() as root:
            run, log, entry = self.fixture(root)
            def changed_during_query(_):
                (run / 'receipts/current-entry.json').write_text(json.dumps(dict(entry, allocation='124')))
                return '123|jmzhang|RUNNING|0:0|\n'
            with self.assertRaises(ValueError):
                archive(run, log, '123', changed_during_query)
            (run / 'receipts/current-entry.json').write_text(json.dumps(entry))
            (run / 'receipts/resolved.yaml').write_text('frozen: false\n')
            with self.assertRaises(ValueError):
                archive(run, log, '123', lambda _: '123|jmzhang|RUNNING|0:0|\n')


if __name__ == '__main__':
    unittest.main()
