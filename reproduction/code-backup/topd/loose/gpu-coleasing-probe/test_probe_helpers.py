import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import probe_data,probe_owned
class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'logs').mkdir()
        self.folder=self.root/'outputs/common-data/actual-arbitrary-name';self.folder.mkdir(parents=True)
        self.file=self.folder/'train.parquet';self.file.write_bytes(b'fixture payload')
        self.sha=hashlib.sha256(self.file.read_bytes()).hexdigest();self.patch=patch.object(probe_data,'TRAIN_SHA',self.sha);self.patch.start()
        self.manifest={'content_identity':{'sha256':probe_data.CONTENT},'sha256':{'train.parquet':self.sha}}
        for file in (self.root/'logs/common-data-manifest.json',self.folder/'manifest.json'):file.write_text(json.dumps(self.manifest))
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def test_actual_manifest_path_no_specification_no_short_hash(self):self.assertEqual(probe_data.resolve_train(self.root)[0],self.file.resolve())
    def test_changed_train_bytes(self):
        self.file.write_bytes(b'changed')
        with self.assertRaises(ValueError):probe_data.resolve_train(self.root)
    def test_duplicate_verified_paths_rejected(self):
        other=self.root/'assets-root/common-data/second';other.mkdir(parents=True)
        (other/'manifest.json').write_text(json.dumps(self.manifest));(other/'train.parquet').write_bytes(self.file.read_bytes())
        with self.assertRaises(ValueError):probe_data.resolve_train(self.root)
    def test_wrong_content_identity(self):
        self.manifest['content_identity']['sha256']='wrong';(self.root/'logs/common-data-manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):probe_data.resolve_train(self.root)
    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            target=Path(outside)/'train.parquet';target.write_bytes(self.file.read_bytes());self.file.unlink();self.file.symlink_to(target)
            with self.assertRaises(ValueError):probe_data.resolve_train(self.root)
class CleanupTests(unittest.TestCase):
    def test_baseline_guardian_and_spawned_descendants_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=Path(tmp)/'cgroup';scope.mkdir();procs=scope/'cgroup.procs';procs.write_text('10\n')
            owner=probe_owned.OwnedProcesses(scope,'/slurm/job_42/step_batch')
            procs.write_text('10\n20\n21\n');called=[]
            original=Path
            def paths(value):
                value=str(value)
                if value.startswith('/proc/'):
                    p=Path(tmp)/value.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_text('0::/slurm/job_42/step_batch\n');return p
                return original(value)
            def send(fd,sig):
                called.append(fd);pending=set(map(int,procs.read_text().split()));pending.discard(fd)
                if fd==20:pending.add(22) # child spawned during shutdown must be re-scanned
                procs.write_text(''.join(str(pid)+'\n' for pid in sorted(pending)))
            with patch.object(probe_owned,'Path',side_effect=paths),patch.object(probe_owned.os,'pidfd_open',side_effect=lambda pid:pid,create=True),patch.object(probe_owned.os,'close'),patch.object(probe_owned.signal,'pidfd_send_signal',side_effect=send,create=True),patch.object(probe_owned.time,'sleep'):
                owner.cleanup()
            self.assertEqual(procs.read_text(),'10\n');self.assertEqual(set(called),{20,21,22});self.assertNotIn(10,called)
    def test_other_step_membership_never_signaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=Path(tmp)/'cgroup';scope.mkdir();procs=scope/'cgroup.procs';procs.write_text('10\n')
            owner=probe_owned.OwnedProcesses(scope,'/slurm/job_42/step_batch');procs.write_text('10\n99\n')
            p=Path(tmp)/'membership';p.write_text('0::/slurm/job_999/step_batch\n')
            def paths(value):return p if str(value).startswith('/proc/') else Path(value)
            with patch.object(probe_owned,'Path',side_effect=paths),patch.object(probe_owned.os,'pidfd_open',side_effect=lambda pid:pid,create=True),patch.object(probe_owned.os,'close'),patch.object(probe_owned.signal,'pidfd_send_signal',create=True) as send,patch.object(probe_owned.time,'sleep'):
                with self.assertRaises(RuntimeError):owner.cleanup()
                send.assert_not_called()
class MappingTests(unittest.TestCase):
    def setUp(self):
        from probe_gpu_mapping import numeric_worker_mapping
        self.map=numeric_worker_mapping;self.rows=[dict(index=3,uuid='GPU-A'),dict(index=7,uuid='GPU-B')]
    def test_numeric_scheduler_order_uuid_preserved(self):
        self.assertEqual(self.map(self.rows,{'GPU-A','GPU-B'},['7','3']),[dict(index=7,uuid='GPU-B'),dict(index=3,uuid='GPU-A')])
    def test_uuid_input_becomes_numeric_only(self):
        self.assertEqual([r['index'] for r in self.map(self.rows,{'GPU-A','GPU-B'},['GPU-A','GPU-B'])],[3,7])
    def test_foreign_numeric_device_rejected(self):
        with self.assertRaises(ValueError):self.map(self.rows,{'GPU-A','GPU-B'},['3','0'])
    def test_foreign_inventory_uuid_rejected(self):
        with self.assertRaises(ValueError):self.map(self.rows,{'GPU-A','GPU-C'},['3','7'])
    def test_duplicate_selectors_rejected(self):
        with self.assertRaises(ValueError):self.map(self.rows,{'GPU-A','GPU-B'},['3','GPU-A'])
    def test_duplicate_numeric_inventory_rejected(self):
        with self.assertRaises(ValueError):self.map([dict(index=3,uuid='GPU-A'),dict(index=3,uuid='GPU-B')],{'GPU-A','GPU-B'},['3','7'])
if __name__=='__main__':unittest.main()
