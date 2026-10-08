"""Small CPU ownership fixtures. No real /proc, GPU, network or process launch."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import owned_process_audit as audit

RUN = 'owned-run'
CONTROLLER = dict(pid=100, uid=1005, starttick=1000)
WORKER = dict(pid=101, uid=1005, starttick=1001)
ROOTS = ['/owned/source/worker.py', '/owned/source/launch_gpu_debug.py', '/owned/runs/debug-run']


def process(pid, uid=1005, tick=900, ppid=1, state='S', comm='python', argv=None):
    argv = list(argv or []); raw = ('\0'.join(argv) + ('\0' if argv else '')).encode()
    return dict(pid=pid, uid=uid, starttick=tick, ppid=ppid, pgid=pid, state=state,
                comm=comm, argv=argv, kernel_uids=[uid]*4, argv_sha256=hashlib.sha256(raw).hexdigest())


class ScannerTests(unittest.TestCase):
    def scan(self, processes, envs=None, **kwargs):
        table = {1: process(1, uid=0, tick=1, ppid=0, comm='systemd')}
        table.update({p['pid']: p for p in processes})
        def reader(pid):
            if pid not in table: raise FileNotFoundError()
            return copy.deepcopy(table[pid])
        def env_reader(pid):
            result = (envs or {}).get(pid, {})
            if isinstance(result, Exception): raise result
            return result
        return audit.audit_same_uid(RUN, 1005, 1000, [CONTROLLER, WORKER], ROOTS,
            snapshot_reader=reader, environment_reader=env_reader,
            pid_reader=lambda: list(table), **kwargs)

    def test_readable_foreign_and_changed_pid_tick(self):
        result = self.scan([process(101, tick=1500), process(200, tick=1600)], {101: {}, 200: {'E1_RUN_ID': 'other'}})
        self.assertTrue(result['accepted_no_owned_or_opaque'])
        self.assertTrue(next(x for x in result['exclusions'] if x['pid']==101)['reused_known_pid'])

    def test_known_owned_live_and_zombie_cannot_be_excluded(self):
        for state in ('S', 'Z'):
            value = process(101, tick=1001, state=state)
            result = self.scan([value], {101: PermissionError()}, exclusions=[dict(identity=value)])
            self.assertFalse(result['accepted_no_owned_or_opaque'])
            self.assertEqual(result['blockers'][0]['reason'], 'known_owned_pid_same_starttick_present_even_if_uid_changed')

    def test_known_pid_same_tick_changed_uid_or_opaque_kernel_never_reused(self):
        changed = process(101, uid=0, tick=1001)
        result = self.scan([changed])
        self.assertFalse(result['accepted_no_owned_or_opaque'])
        self.assertEqual(result['blockers'][0]['expected_known_identity'], WORKER)
        def unreadable(pid):
            raise PermissionError('known PID kernel status unreadable')
        result = audit.audit_same_uid(RUN, 1005, 1000, [WORKER], ROOTS,
            snapshot_reader=unreadable, pid_reader=lambda: [101])
        self.assertFalse(result['accepted_no_owned_or_opaque'])
        self.assertEqual(result['blockers'][0]['reason'], 'kernel_identity_unreadable:PermissionError')

    def test_kernel_effective_uid_not_proc_directory_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root/'101'; directory.mkdir()
            fields = ['S', '1', '101'] + ['0']*16 + ['1001']
            (directory/'stat').write_text('101 (python) ' + ' '.join(fields))
            (directory/'comm').write_text('python\n')
            (directory/'status').write_text('Uid:\t1005\t1005\t1005\t1005\nPPid:\t1\n')
            real_path = Path
            with mock.patch.object(audit, 'Path', side_effect=lambda x: root if x == '/proc' else real_path(x)):
                result = audit.proc_kernel_identity(101)
            self.assertEqual(result['uid'], 1005)
            self.assertEqual(result['proc_directory_owner_uid'], directory.stat().st_uid)
            self.assertEqual(result['starttick'], 1001)

    def test_snapshot_identity_change_after_environment_fails_closed(self):
        values = {1: process(1,uid=0,tick=1,ppid=0,comm='systemd'), 209: process(209,tick=1600)}
        calls = {209: 0}
        def reader(pid):
            value=copy.deepcopy(values[pid])
            if pid==209:
                calls[pid]+=1
                if calls[pid]>1: value['starttick']+=1
            return value
        result=audit.audit_same_uid(RUN,1005,1000,[CONTROLLER],ROOTS,
            snapshot_reader=reader,environment_reader=lambda _: {},pid_reader=lambda: [209])
        self.assertFalse(result['accepted_no_owned_or_opaque'])
        self.assertEqual(result['blockers'][0]['reason'],'kernel_identity_or_args_changed_after_environment')

    def test_marked_run_always_blocks_even_old_systemd_or_bound_zombie(self):
        for p in (process(200, tick=10, comm='systemd'), process(200, tick=1600, state='Z')):
            result = self.scan([p], {200: {'E1_RUN_ID': RUN}}, exclusions=[dict(identity=p)])
            self.assertFalse(result['accepted_no_owned_or_opaque'])

    def test_true_owned_parent_chain_and_envless_orphan_block(self):
        controller = process(100, tick=1000)
        orphan = process(205, tick=1600, ppid=100)
        result = self.scan([controller, orphan], {100: {}, 205: PermissionError()})
        self.assertFalse(result['accepted_no_owned_or_opaque'])
        self.assertTrue(next(x for x in result['blockers'] if x['pid']==205)['owned_parent_chain_matches'])
        result = self.scan([process(205, tick=1600, argv=['python','/owned/source/worker.py'])], {205: {}})
        self.assertFalse(result['accepted_no_owned_or_opaque'])

    def test_original_output_owned_but_cpu_run_input_not_owned(self):
        cpu = process(218104, tick=1700, argv=['python', '-B', '/other/source/summarize_regions.py',
                    '--run-dir', '/owned/runs/debug-run', '--output', '/other/full192'])
        result = self.scan([cpu], {218104: {}})
        self.assertTrue(result['accepted_no_owned_or_opaque'])
        self.assertEqual(result['exclusions'][0]['non_execution_input_argv_roots'], ['/owned/runs/debug-run'])
        for argv in (['python','other.py','--output','/owned/runs/debug-run/worker-0'],
                     ['python','other.py','--output=/owned/runs/debug-run/worker-0']):
            self.assertFalse(self.scan([process(207,tick=1600,argv=argv)])['accepted_no_owned_or_opaque'])

    def test_predating_systemd_pam_verified_kernel_parent_chain_only(self):
        systemd = process(2372931, tick=884, comm='systemd')
        pam = process(2372933, tick=885, ppid=2372931, comm='sd-pam')
        for actual_comm in ('sd-pam', '(sd-pam)'):
            pam['comm'] = actual_comm
            result = self.scan([systemd,pam], {systemd['pid']: PermissionError(),pam['pid']: PermissionError()})
            self.assertTrue(result['accepted_no_owned_or_opaque'])
            self.assertTrue(all(x['origin_environ_unknown'] for x in result['exclusions']))
        pam['comm']='fake-sd-pam'
        self.assertFalse(self.scan([systemd,pam],{systemd['pid']:PermissionError(),pam['pid']:PermissionError()})['accepted_no_owned_or_opaque'])
        late = process(200,tick=1500,comm='systemd')
        self.assertFalse(self.scan([late],{200: PermissionError()})['accepted_no_owned_or_opaque'])
        fake = process(200,tick=500,comm='python',argv=['systemd'])
        self.assertFalse(self.scan([fake],{200: PermissionError()})['accepted_no_owned_or_opaque'])

    def test_actual_root_parent_sshd_not_mutable_argv_claim(self):
        root = process(300,uid=0,tick=800,comm='sshd')
        child = process(301,tick=1500,ppid=300,comm='sshd',argv=['sshd: user@notty'])
        result = self.scan([root,child], {301: PermissionError()})
        self.assertTrue(result['accepted_no_owned_or_opaque'])
        self.assertEqual(result['exclusions'][0]['reason'],'actual_root_parent_sshd_service')
        root['uid']=1005;root['kernel_uids']=[1005]*4
        self.assertFalse(self.scan([root,child],{301: PermissionError()})['accepted_no_owned_or_opaque'])
        root['uid']=0;root['comm']='python'
        self.assertFalse(self.scan([root,child],{301: PermissionError()})['accepted_no_owned_or_opaque'])

    def test_unknown_opaque_live_and_unknown_z_even_old_tick_fail_closed(self):
        for state,tick in (('S',1600),('Z',1600),('Z',500)):
            result = self.scan([process(208,tick=tick,state=state)],{208: PermissionError()})
            self.assertFalse(result['accepted_no_owned_or_opaque'])
            self.assertNotIn('E1_RUN_ID', result['blockers'][0])

    def test_explicit_cpu_zombie_proof_only_cannot_override_parent_or_run(self):
        value = process(206996,tick=1600,state='Z')
        proof = dict(identity=value,receipt={'path':'fixture','sha256':'x'},source_sha256='cpu-source',evidence=[])
        result = self.scan([value],{206996: PermissionError()},exclusions=[proof])
        self.assertTrue(result['accepted_no_owned_or_opaque'])
        self.assertTrue(result['exclusions'][0]['origin_environ_unknown'])
        value['ppid']=100;controller=process(100,tick=1000)
        self.assertFalse(self.scan([value,controller],{206996: PermissionError()},exclusions=[proof])['accepted_no_owned_or_opaque'])

    def test_cpu_provenance_references_hash_and_required_source_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); evidence=root/'actual-rejected.json';evidence.write_text('{"accepted":false}')
            receipt=root/'cpu-provenance.json';value=dict(schema='independent-cpu-process-provenance-v1',
                verified_non_m1_owned_cpu_task=True,no_gpu_or_training=True,identity=process(206996,tick=1600,state='Z'),
                source_sha256='a'*64,evidence=[dict(path=str(evidence),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())])
            receipt.write_text(json.dumps(value));ref=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            self.assertEqual(len(audit.load_exclusions([ref])),1)
            evidence.write_text('changed')
            with self.assertRaises(ValueError):audit.load_exclusions([ref])


if __name__=='__main__':unittest.main()
