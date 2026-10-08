"""Meaningful CPU gate controls: input is not approval; reused/foreign PIDs are refused."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

try:
    from .acceptance import Hold, owned_gpu_process, storage_admission, resource_limits, sha
except ImportError:
    from acceptance import Hold, owned_gpu_process, storage_admission, resource_limits, sha


class LaunchAdmissionTests(unittest.TestCase):
    def test_foreign_gpu_process_and_reused_root_refused(self):
        owner = dict(pid=10, uid=1005, starttick=90)
        child = dict(pid=20, uid=1005, starttick=100)
        nodes = {20: dict(child, ppid=10, run_id='ours'),
                 21: dict(pid=21, uid=1005, starttick=101, ppid=20, run_id='ours')}
        self.assertEqual(owned_gpu_process(21, 'ours', owner, [child], nodes.__getitem__)['pid'], 21)
        nodes[21]['run_id'] = 'other-experiment'
        with self.assertRaises(Hold):
            owned_gpu_process(21, 'ours', owner, [child], nodes.__getitem__)
        nodes[21]['run_id'] = 'ours'
        nodes[20]['starttick'] += 1
        with self.assertRaises(Hold):
            owned_gpu_process(21, 'ours', owner, [child], nodes.__getitem__)

    def test_restart_request_does_not_supply_storage_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(paths=dict(output_root=directory), resources=dict(minimum_disk_bytes=100))
            receipt = dict(accepted=True, cluster='HKUST', output_root=directory,
                           mode='human_hkust_local_exception', human_authorization={}, backup_policy={})
            with self.assertRaises(Hold):
                storage_admission(config, receipt, lambda _: SimpleNamespace(free=1000))
            receipt.update(human_authorization=dict(explicit_exception=True, text='本次允许HKUST本地盘并持久备份',
                                                   observed_unix=time.time(), thread_id='explicit-human-thread'),
                           backup_policy=dict(durable_target='controller-durable-backup',
                                              synchronize_completed_checkpoints=True))
            self.assertEqual(storage_admission(config, receipt, lambda _: SimpleNamespace(free=1000))['mode'],
                             'human_hkust_local_exception')
            with self.assertRaises(Hold):
                storage_admission(config, receipt, lambda _: SimpleNamespace(free=99))

    def test_local_ext4_and_unverified_peer_are_not_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(paths=dict(output_root=directory), resources=dict(minimum_disk_bytes=100))
            mount = dict(source='/dev/sdb1', fstype='ext4', target='/data')
            probe = Path(directory) / 'shared-probe.txt'
            probe.write_text('actual same-object probe')
            digest = sha(probe)
            evidence = Path(directory) / 'peer-evidence.json'
            evidence.write_text(json.dumps(dict(accepted=True, hostname='HKUST4', sha256=digest)))
            receipt = dict(accepted=True, cluster='HKUST', output_root=directory, mode='same_cluster_shared',
                           mount=mount, local_hostname='HKUST1',
                           cross_node_readback=dict(accepted=True, cluster='HKUST', hostname='HKUST4', sha256=digest,
                                                   evidence=dict(path=str(evidence),sha256=sha(evidence))),
                           probe_sha256=digest, probe_path=str(probe), retention_verified_until_unix=time.time()+1000)
            with self.assertRaises(Hold):
                storage_admission(config, receipt, lambda _: SimpleNamespace(free=1000), lambda _: mount)
            mount.update(source='hkust-storage:/shared', fstype='nfs4')
            receipt['cross_node_readback']['accepted'] = False
            with self.assertRaises(Hold):
                storage_admission(config, receipt, lambda _: SimpleNamespace(free=1000), lambda _: mount)
            receipt['cross_node_readback']['accepted'] = True
            self.assertEqual(storage_admission(config, receipt, lambda _: SimpleNamespace(free=1000), lambda _: mount)['mode'],
                             'same_cluster_shared')
            probe.write_text('changed actual probe')
            with self.assertRaises(Hold):
                storage_admission(config, receipt, lambda _: SimpleNamespace(free=1000), lambda _: mount)

    def test_running_process_does_not_weaken_next_stage_or_unused_card(self):
        config = dict(resources=dict(startup_minimum_free_mib=20480, training_minimum_free_mib=1024))
        cards = [dict(uuid='gpu0', actual_free_mib=3000), dict(uuid='gpu1', actual_free_mib=24000)]
        resource_limits(config, 'training-running', cards, [dict(pid=20)], {'gpu0'})
        with self.assertRaises(Hold):
            resource_limits(config, 'sampling-start', cards, [dict(pid=20)], {'gpu0'})
        cards[1]['actual_free_mib'] = 10000
        with self.assertRaises(Hold):
            resource_limits(config, 'sampling-running', cards, [dict(pid=20)], {'gpu0'})
        with self.assertRaises(Hold):
            resource_limits(config, 'UNKNOWN_STAGE', cards, [], set())

    def test_future_output_root_can_be_admitted_without_creating_training_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created-yet' / 'runs'
            config = dict(paths=dict(output_root=str(output)), resources=dict(minimum_disk_bytes=100))
            receipt = dict(accepted=True, cluster='HKUST', output_root=str(output), mode='human_hkust_local_exception',
                           human_authorization=dict(explicit_exception=True, text='明确允许本次HKUST例外',
                                                    observed_unix=time.time(), thread_id='human'),
                           backup_policy=dict(durable_target='durable-controller',synchronize_completed_checkpoints=True))
            checked = []
            result = storage_admission(config, receipt, lambda path: checked.append(str(path)) or SimpleNamespace(free=1000))
            self.assertEqual(result['realpath'], str(output.resolve()))
            self.assertEqual(checked, [str(Path(directory).resolve())])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
