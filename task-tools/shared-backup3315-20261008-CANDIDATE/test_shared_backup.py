import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('backup', Path(__file__).with_name('shared_backup.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


class Tests(unittest.TestCase):
    def test_no_idle_io_credit(self):
        clock = [10.0]
        with patch.object(b.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(b.time, 'sleep', side_effect=lambda n: clock.__setitem__(0, clock[0] + n)), patch.object(b.Budget, 'pressure_gate'):
            budget = b.Budget(); budget.consume(b.RATE)
            self.assertEqual(clock[0], 11.0)
            clock[0] = 1000.0
            budget.consume(b.RATE)
            self.assertEqual(clock[0], 1001.0)

    def test_mount_mismatch_rejected(self):
        data = {'filesystems': [{'target': '/mnt/shared', 'source': '/dev/local', 'fstype': 'ext4'}]}
        with patch.object(b.subprocess, 'check_output', return_value=json.dumps(data)):
            with self.assertRaises(RuntimeError): b.mount()

    def test_future_reserve_in_capacity(self):
        st = type('S', (), {'f_bavail': b.RESERVE + 99, 'f_frsize': 1})()
        with patch.object(b.os, 'statvfs', return_value=st):
            with self.assertRaises(RuntimeError): b.reserve(0)

    def test_symlink_component_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t).resolve(); (base/'real').mkdir(); (base/'link').symlink_to(base/'real')
            with self.assertRaises(RuntimeError): b.safe(base/'link'/'file')

    def test_atomic_no_replace_preserves_destination(self):
        if not hasattr(b.ctypes.CDLL(None), 'renameat2'):
            self.skipTest('Linux renameat2 unavailable; actual Linux required')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve(); a=p/'a'; c=p/'c'; a.mkdir(); c.mkdir(); (c/'original').write_text('keep')
            with self.assertRaises(OSError): b.no_replace(a,c)
            self.assertEqual((c/'original').read_text(),'keep'); self.assertTrue(a.is_dir())

    def test_copy_and_hash_exact(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t).resolve(); src=base/'src'; stage=base/'stage'; stage.mkdir(); src.write_bytes(b'abcd'*10000)
            entry={'path':str(src),'relative':'src','bytes':src.stat().st_size,'stat':b.signature(src.stat()),'scope':'immutable-checkpoint'}
            budget=unittest.mock.Mock()
            with patch.object(b,'SHARED',base), patch.object(b,'ROOT',base), patch.object(b,'mount'):
                item=b.copy_file(entry,stage,budget)
            self.assertEqual((stage/'src').read_bytes(),src.read_bytes())
            self.assertEqual(item['sha256'],b.hashlib.sha256(src.read_bytes()).hexdigest())
            with patch.object(b,'SHARED',base),patch.object(b,'ROOT',base),patch.object(b,'mount'):
                with self.assertRaises(FileExistsError): b.copy_file(entry,stage,budget)

    def test_immutable_change_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t).resolve(); src=base/'src'; stage=base/'stage'; stage.mkdir();src.write_bytes(b'old')
            e={'path':str(src),'relative':'file','bytes':3,'stat':b.signature(src.stat()),'scope':'immutable-checkpoint'}
            src.write_bytes(b'changed')
            with patch.object(b,'ROOT',base):
                with self.assertRaises(RuntimeError): b.copy_file(e,stage,unittest.mock.Mock())

    def test_wrong_probe_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t).resolve(); (base/'prepared.json').write_text(json.dumps({'probe_sha256':'a','probe':{}}))
            (base/'retained18-probe.actual.json').write_text(json.dumps({'hostname':'foreign','target':'h100-10','probe_sha256':'a','probe':{}}))
            with patch.object(b,'BASE',base):
                with self.assertRaises(RuntimeError): b.crosshost_gate()

    def test_absolute_destination_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve(); src=p/'src';src.write_bytes(b'keep');stage=p/'stage';stage.mkdir()
            e={'path':str(src),'relative':str(p/'escape'),'bytes':4,'stat':b.signature(src.stat()),'scope':'immutable-checkpoint'}
            with self.assertRaises(RuntimeError): b.copy_file(e,stage,unittest.mock.Mock())
            self.assertFalse((p/'escape').exists())

    def test_latest_complete_pause_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve()
            safe={'slurm_job_id':'3315','run':str(root),'pause':False,'reason':'within_threshold','time':b.time.time()}
            paused=dict(safe,pause=True,reason='pressure')
            (root/'resource-watch.jsonl').write_text(json.dumps(safe)+'\n'+json.dumps(paused)+'\n')
            class WaitWitness(Exception): pass
            with patch.object(b,'RUN',root),patch.object(b.time,'sleep',side_effect=WaitWitness):
                with self.assertRaises(WaitWitness): b.Budget().pressure_gate()

    def test_checkpoint_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve(); native=root/'checkpoints/fresh200_state/outer_100_COMPLETE';hf=root/'checkpoints/actor/global_step_100';algorithm=root/'checkpoints/algorithm/global_step_100'
            for p in [native,hf,algorithm]:p.mkdir(parents=True)
            (root/'outside').write_text('data');(native/'driver.pt').symlink_to(root/'outside')
            with patch.object(b,'RUN',root):
                with self.assertRaisesRegex(RuntimeError,'nonregular'):b.checkpoint_gate(100)


if __name__ == '__main__': unittest.main(verbosity=2)
