"""CPU controls for memory sampling after long state inspection; no GPU calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from .checkpoint import exclusive_json, fileproof
    from .verify_five_updates import fresh_memory_snapshot
except ImportError:
    from checkpoint import exclusive_json, fileproof
    from verify_five_updates import fresh_memory_snapshot


class FreshMemoryTests(unittest.TestCase):
    def fixture(self, root):
        identity = dict(pid=10, uid=1005, starttick=99, run_id='observer-independent', source_sha256='source')
        manifest = exclusive_json(root/'helper.json', dict(files={'memory_observer.py':'source'}))
        current = root/'current.json'
        binding = exclusive_json(root/'binding.json', dict(schema='opsd-online-memory-observer-binding-v1',
            accepted=True, experiment_run_id='training', config_sha256='config',
            live_report_path=str(current), observer_identity=identity))
        config = dict(run_id='training', _config_sha256='config', paths={'receipts':str(root)},
            helper_source_manifest=manifest, owner={'uid':1005, 'gpu_uuids':[str(i) for i in range(8)]},
            resources={'training_minimum_free_mib':1024})
        report = dict(schema='opsd-online-direct-memory-v1', run_id='training', config_sha256='config',
            observer_identity=identity, samples=5, samples_below_limit=0, updated_unix=100.,
            minimum_actual_free_mib={str(i):2000 for i in range(8)},
            current={'devices':[{'uuid':str(i)} for i in range(8)]})
        current.write_text(json.dumps(report))
        return config,binding,current,report,identity

    def test_long_cpu_inspection_binds_latest_report_not_old_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,binding,current,report,identity=self.fixture(root)
            old=fileproof(current)
            # CPU state inspection took two minutes; the independent observer kept sampling.
            report.update(updated_unix=219.,samples=65)
            temporary=root/'replacement.json';temporary.write_text(json.dumps(report));temporary.replace(current)
            with patch('time.time',return_value=220.):
                snapshot,proof=fresh_memory_snapshot(binding,config,root/'accepted.json',reader=lambda _:identity)
            self.assertNotEqual(old['sha256'],proof['observed_current_report']['sha256'])
            self.assertEqual(json.loads(Path(snapshot['path']).read_text())['samples'],65)
            self.assertEqual(json.loads(Path(snapshot['path']).read_text()),json.loads(current.read_text()))

    def test_stopped_observer_old_sample_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,binding,current,report,identity=self.fixture(root)
            with patch('time.time',return_value=220.),self.assertRaisesRegex(ValueError,'stale'):
                fresh_memory_snapshot(binding,config,root/'accepted.json',reader=lambda _:identity)
            self.assertFalse((root/'accepted.json.direct-memory.json').exists())

    def test_reused_observer_pid_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,binding,current,report,identity=self.fixture(root)
            with patch('time.time',return_value=101.),self.assertRaisesRegex(ValueError,'no longer live'):
                fresh_memory_snapshot(binding,config,root/'accepted.json',reader=lambda _:dict(identity,starttick=100))

    def test_observed_memory_failure_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,binding,current,report,identity=self.fixture(root)
            report['samples_below_limit']=1;current.write_text(json.dumps(report))
            with patch('time.time',return_value=101.),self.assertRaisesRegex(ValueError,'limit failed'):
                fresh_memory_snapshot(binding,config,root/'accepted.json',reader=lambda _:identity)


if __name__=='__main__': unittest.main()
