"""Local identity fixtures only: never query GPUs, SSH, or signal processes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('fresh_watch', TOOLS / 'resource_watch.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeAnchor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.original_source = w.FRESH200_SOURCE
        w.FRESH200_SOURCE = '8e01320bd4ba06f5b64cb6a3164373e6f25f35f7'
        self.original = w.FRESH200_ROOT
        w.FRESH200_ROOT = self.root
        (self.root / 'logs').mkdir()
        (self.root / 'runs').mkdir()
        (self.root / 'env/bin').mkdir(parents=True)
        self.exe = self.root / 'native-python'
        self.exe.write_bytes(b'actual executable bytes fixture')
        (self.root / 'env/bin/python').symlink_to(self.exe)
        (self.root / 'source-expected.txt').write_text(w.FRESH200_SOURCE)
        self.record = dict(source_commit=w.FRESH200_SOURCE, root=str(self.root),
                           sys_prefix=str(self.root / 'env'), host='h100-5',
                           scope='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE',
                           executable=w.executable_identity(self.exe), executable_sha256=digest(self.exe))
        self.manifest = dict(source_commit=w.FRESH200_SOURCE,
                             files={'resource_watch.py': digest(TOOLS / 'resource_watch.py')})
        self.approval = dict(protocol=w.FRESH200_PROTOCOL, root=str(self.root),
                             source_commit=w.FRESH200_SOURCE)
        self.write()

    def tearDown(self):
        w.FRESH200_SOURCE = self.original_source
        w.FRESH200_ROOT = self.original
        self.tmp.cleanup()

    def write(self):
        anchor = self.root / 'logs/runtime-executable.json'
        anchor.write_text(json.dumps(self.record))
        manifest = self.root / 'logs/tools-manifest.json'
        manifest.write_text(json.dumps(self.manifest))
        self.approval.update(runtime_executable=dict(path=str(anchor), bytes=anchor.stat().st_size,
                                                    sha256=digest(anchor)), tools_manifest_sha256=digest(manifest))
        self.write_approval()

    def write_approval(self):
        (self.root / 'logs/launch-approval.json').write_text(json.dumps(self.approval))

    def load(self, current=False):
        return w.load_runtime_anchor(str(self.root / 'runs/diagnostic-discovery128-test-3302'), current)

    def test_real_r2_missing_status_fault_reproduced_then_fresh_pass(self):
        # Exact former admission expression crashes on the real Fresh200 schema.
        with self.assertRaisesRegex(KeyError, 'status'):
            self.approval['status']
        result = self.load()
        self.assertEqual(result['source_commit'], w.FRESH200_SOURCE)
        self.assertEqual(result['executable'], self.record['executable'])

    def test_old_format_missing_status_fails_closed(self):
        self.approval.pop('protocol')
        self.write_approval()
        with self.assertRaisesRegex(ValueError, 'own launch protocol'):
            self.load()

    def test_legacy_e1_pass_cannot_authorize_fresh(self):
        self.approval['status'] = 'PASS_INDEPENDENT_E1_LAUNCH_REVIEW'
        self.write_approval()
        with self.assertRaisesRegex(ValueError, 'Legacy E1'):
            self.load()

    def test_genuine_legacy_branch_preserves_original_contract(self):
        self.approval.pop('protocol')
        self.approval['status'] = 'PASS_INDEPENDENT_E1_LAUNCH_REVIEW'
        self.write_approval()
        with patch.object(w, 'FRESH200_ROOT', self.root / 'different-fresh-root'), patch.object(w,'LEGACY_E1_ROOTS',frozenset((self.root,))):
            self.load()
            self.approval.pop('status')
            self.write_approval()
            with self.assertRaises(KeyError):
                self.load()

    def test_wrong_source_root_protocol_and_binding_rejected(self):
        original = json.loads(json.dumps(self.approval))
        for field, value in [('source_commit', 'f' * 40), ('root', '/old/root'),
                             ('protocol', 'E1'), ('runtime_executable', None),
                             ('tools_manifest_sha256', '0' * 64)]:
            with self.subTest(field=field):
                self.approval = dict(original, **{field: value})
                self.write_approval()
                with self.assertRaises(ValueError):
                    self.load()
        self.approval = original
        del self.approval['runtime_executable']
        self.write_approval()
        with self.assertRaises(ValueError):
            self.load()

    def test_fixed_source_marker_cannot_be_self_declared(self):
        (self.root / 'source-expected.txt').write_text('f' * 40)
        self.approval['source_commit'] = 'f' * 40
        self.write_approval()
        with self.assertRaises(ValueError):
            self.load()

    def test_unfrozen_new_training_source_blocks_launch(self):
        with patch.object(w, 'FRESH200_SOURCE', None):
            with self.assertRaisesRegex(ValueError, 'not frozen/bound'):
                self.load()

    def test_foreign_root_cannot_fabricate_legacy_e1_approval(self):
        self.approval.pop('protocol')
        self.approval['status']='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'
        self.write_approval()
        with patch.object(w,'FRESH200_ROOT',self.root/'different-new-root'):
            with self.assertRaisesRegex(ValueError, 'genuinely old E1 ROOT'):
                self.load()

    def test_binding_hash_bytes_path_types_and_missing_capture_rejected(self):
        original = json.loads(json.dumps(self.approval))
        for field, value in [('path', str(self.root / 'logs/other.json')), ('bytes', True),
                             ('bytes', original['runtime_executable']['bytes'] + 1),
                             ('sha256', '0' * 64)]:
            with self.subTest(field=field, value=value):
                self.approval = json.loads(json.dumps(original))
                self.approval['runtime_executable'][field] = value
                self.write_approval()
                with self.assertRaises(ValueError):
                    self.load()
        self.approval = original
        self.write_approval()
        (self.root / 'logs/runtime-executable.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Ordinary canonical'):
            self.load()

    def test_capture_source_prefix_host_scope_and_executable_metadata_rejected(self):
        original = dict(self.record)
        for field, value in [('source_commit', 'f' * 40), ('root', '/old/root'),
                             ('sys_prefix', '/old/env'), ('host', 'h100-16'),
                             ('scope', 'E1_PASS'), ('executable_sha256', '0' * 64)]:
            with self.subTest(field=field):
                self.record = dict(original, **{field: value})
                self.write()
                with self.assertRaises(ValueError):
                    self.load()
        for field in original:
            with self.subTest(missing=field):
                self.record = dict(original)
                del self.record[field]
                self.write()
                with self.assertRaises(ValueError):
                    self.load()

    def test_actual_executable_replacement_and_newenv_redirect_rejected(self):
        self.exe.write_bytes(b'changed executable')
        with self.assertRaisesRegex(ValueError, 'bytes/identity'):
            self.load()
        self.record['executable_sha256'] = digest(self.exe)
        self.write()
        other = self.root / 'other-python'
        other.write_bytes(self.exe.read_bytes())
        (self.root / 'env/bin/python').unlink()
        (self.root / 'env/bin/python').symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'environment executable'):
            self.load()

    def test_manifest_wrong_source_missing_watcher_and_hash_rejected(self):
        original = dict(self.manifest)
        for manifest in [dict(original, source_commit='f' * 40),
                         dict(original, files={}),
                         dict(original, files={'resource_watch.py': '0' * 64})]:
            self.manifest = manifest
            self.write()
            with self.assertRaises(ValueError):
                self.load()

    def test_current_interpreter_identity_prefix_are_checked(self):
        original = w.executable_identity
        def current(path):
            if str(path) in (w.sys.executable, '/proc/self/exe'):
                return self.record['executable']
            return original(path)
        with patch.object(w, 'executable_identity', side_effect=current), patch.object(w.sys, 'prefix', str(self.root / 'env')):
            self.load(True)
        with patch.object(w, 'executable_identity', side_effect=current), patch.object(w.sys, 'prefix', '/old/env'):
            with self.assertRaisesRegex(ValueError, 'Actual interpreter'):
                self.load(True)

    def test_actual_main_literal_and_phase_ancestry(self):
        proc = self.root / 'proc'
        proc.mkdir()
        run = str(self.root / 'runs/diagnostic-discovery128-test-3302')
        commands = {100: ['bash', 'stage-gpu-inner.sh'],
                    101: ['python3', '-m', 'search_r1.baselines.main'],
                    102: ['python3', 'helper.py'],
                    103: ['ray::WorkerDict.actor_rollout_capture_fresh200_diagnostic', '']}
        for pid, parent in [(100, 1), (101, 100), (102, 101), (103, 101)]:
            directory = proc / str(pid)
            directory.mkdir()
            fields = ['S', str(parent)] + ['0'] * 17 + [str(pid * 100)]
            (directory / 'stat').write_text(str(pid) + ' (python) ' + ' '.join(fields))
            (directory / 'cmdline').write_bytes(('\0'.join(commands[pid]) + '\0').encode())
            leases = json.dumps([{'uuid':'GPU-'+str(i)} for i in range(8)])
            (directory / 'environ').write_bytes(('BAPO_RUN_ID=' + run + '\0SLURM_JOB_ID=3302\0SEARCH_R1_GPU_LEASES='+leases+'\0').encode())
            (directory / 'exe').symlink_to(self.exe)
        original = w.executable_identity
        def current(path):
            if str(path) in (w.sys.executable, '/proc/self/exe'):
                return self.record['executable']
            return original(path)
        with patch.object(w, 'executable_identity', side_effect=current), patch.object(w.sys, 'prefix', str(self.root / 'env')):
            rows = w.freeze_inventory(run, '3302', {'GPU-' + str(i) for i in range(8)}, 100, proc)
        self.assertEqual(rows[101]['role'], 'trainer')
        self.assertEqual(rows[102]['role'], 'phase_descendant')
        self.assertEqual(rows[101]['parent_start_ticks'], rows[100]['start_ticks'])
        self.assertEqual(rows[102]['parent_start_ticks'], rows[101]['start_ticks'])
        self.assertEqual(rows[103]['role'], 'ray_worker')
        self.assertEqual(rows[103]['runtime_anchor_sha256'], self.load()['sha256'])
        for command in [['python3', 'search_r1.baselines.main'],
                        ['python3', '-m', 'search_r1.baselines.main.fake'],
                        ['sh', '-m', 'search_r1.baselines.main'],
                        ['pythonfake', '-m', 'search_r1.baselines.main'],
                        ['python3', '-c', 'pass', '-m', 'search_r1.baselines.main'],
                        ['python3', '-m', 'verl.trainer.main_ppo']]:
            self.assertFalse(w.is_trainer_command(command, run))

    def test_exact_diagnostic_rpc_anchor_and_title_tails(self):
        run = str(self.root / 'runs/diagnostic-discovery128-test-3302')
        anchor = self.load()
        proof = dict(pid=102, ppid=101, start_ticks=22, parent_start_ticks=21,
                     run=run, slurm_job_id='3302', uid=os.getuid(), role='ray_worker',
                     executable=self.record['executable'], allocated_uuid_set=['GPU-'+str(i) for i in range(8)],
                     gpu_leases_inherited=True, inherited_gpu_lease_uuids=['GPU-'+str(i) for i in range(8)],
                     runtime_anchor_sha256=anchor['sha256'],
                     command=['ray::WorkerDict.actor_rollout_capture_fresh200_diagnostic', ''])
        w.certificate_identity(proof, anchor)
        for command in [['ray::WorkerDict.ref_capture_fresh200_diagnostic'],
                        ['ray::WorkerDict.actor_rollout_capture_fresh200_diagnostic_fake'],
                        ['ray::WorkerDict.actor_rollout_capture_fresh200_diagnostic', 'foreign']]:
            with self.assertRaises(ValueError):
                w.certificate_identity(dict(proof, command=command), anchor)
        wrong = dict(anchor, sha256='0'*64)
        with self.assertRaises(ValueError):
            w.certificate_identity(proof, wrong)

    def test_integrated_anchor_main_ray_freeze_stable_recovery_replay(self):
        import copy
        spec = importlib.util.spec_from_file_location('fresh_journal', TOOLS/'guard_journal.py')
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        tools = self.root/'tools'
        tools.mkdir()
        watcher = tools/'resource_watch.py'
        watcher.write_bytes((TOOLS/'resource_watch.py').read_bytes())
        run = str(self.root/'runs/diagnostic-discovery128-test-3302')
        Path(run).mkdir()
        journal = Path(run)/'resource-watch.jsonl'
        uuids = ['GPU-'+str(i) for i in range(8)]
        anchor = self.load()
        def sample(unsafe=False, recovery=False):
            start, end = 1000000, 1200000
            sm = 16 if unsafe else 0
            inventory = '\n'.join('%s, %s, 1000, %s'%(i,uuids[i],sm if i==0 else 0) for i in range(8))+'\n'
            compute = '\n'.join('%s, %s, 1'%(uuids[i],500+i) for i in range(8))+'\n'
            pmon = '# gpu pid type sm mem enc dec jpg ofa command\n'+'\n'.join('%s %s C %s 0 - - - - python'%(i,500+i,sm if i==0 else 0) for i in range(8))+'\n'
            a = w.inventory(inventory,[str(i) for i in range(8)])
            nv = {}
            for i in range(8):
                nv[uuids[i]] = dict(status='ok',whole_gpu_sm_percent=sm if i==0 else 0,
                    whole_gpu_sm_query_us=end-5,query_start_us=end-20,query_end_us=end,
                    api='nvmlDeviceGetProcessUtilization',complete_query=True,last_seen_timestamp_us=0,
                    mig_mode=0,returned_sample_count=1,samples=[dict(pid=102,timestamp_us=end-10,
                    sm_percent=0,mem_percent=0,enc_percent=0,dec_percent=0)])
                a[str(i)].update(whole_gpu_sm_source='nvml_query',whole_gpu_sm_query_us=end-5)
            own = {str(p):dict(start_ticks=10+p,uid=os.getuid(),run=run,slurm_job_id='3302',lease_uuids=uuids) for p in (101,102,103)}
            ticks = {str(500+i):dict(before=900+i,after=900+i) for i in range(8)}
            ticks['102'] = dict(before=112,after=112)
            row = dict(run=run,slurm_job_id='3302',phase_pid=101,time=1.,raw_inventory=inventory,
                raw_compute=compute,raw_pmon=pmon,raw_nvml=nv,allocated_gpus=a,window_start_us=start,
                window_end_us=end,owned_before_query=own,pid_start_ticks=ticks)
            row.update(w.assess_with_bounds(a,w.parse_pmon(pmon,a),w.parse_memory(compute,a),
                       lambda p:p in (101,102,103),nv,start))
            row['reason']='external_occupancy_exceeds_15_percent' if row['pause'] else 'within_threshold'
            if recovery:row.update(event='recovery_sample',elapsed_seconds=1)
            return row
        def event(name,**values):
            return dict(run=run,slurm_job_id='3302',phase_pid=101,time=1.,event=name,**values)
        def certificate(pid):
            role={101:'phase',102:'trainer',103:'ray_worker'}[pid]
            command={101:['bash','stage-gpu-inner.sh'],102:['python','-m','search_r1.baselines.main'],
                103:['ray::WorkerDict.actor_rollout_capture_fresh200_diagnostic','']}[pid]
            return dict(pid=pid,ppid=1 if pid==101 else 101,parent_start_ticks=11 if pid==101 else 111,
                start_ticks=10+pid,state='S',uid=os.getuid(),run=run,slurm_job_id='3302',role=role,
                command=command,executable=self.record['executable'],allocated_uuid_set=uuids,
                gpu_leases_inherited=True,inherited_gpu_lease_uuids=uuids,runtime_anchor_sha256=anchor['sha256'])
        rows = [sample(True),event('recovery_started',grace_seconds=60)]
        for pid in (101,102,103):
            rows.append(event('owned_freeze_signal',pid=pid,start_ticks=10+pid,uid=os.getuid(),
                              role=certificate(pid)['role'],lease_uuids=uuids,certificate=certificate(pid)))
        stable = event('owned_freeze_stable',pids=[101,102,103],epochs={str(p):10+p for p in (101,102,103)},
            before_certificates={str(p):dict(certificate(p),state='T') for p in (101,102,103)},
            after_certificates={str(p):dict(certificate(p),state='T') for p in (101,102,103)})
        rows.append(stable)
        for _ in range(3):rows.append(sample(False,True))
        rows += [event('recovery_validated',consecutive_safe_samples=3),event('recovery_resume',pause=False)]
        rows += [event('owned_resume_signal',pid=p,start_ticks=10+p,signal=18) for p in (101,102,103)]
        rows += [event('recovery_finished',elapsed_seconds=6,reserved_gpu_hours=6*8/3600),sample()]
        def replay(data,source=None):
            source = w.FRESH200_SOURCE if source is None else source
            journal.write_text(''.join(json.dumps(row)+'\n' for row in data))
            with patch.object(g,'NEW_ROOT',self.root),patch.object(g,'FRESH200_SOURCE',w.FRESH200_SOURCE),patch.object(g,'load_bound_watcher',return_value=w):
                return g.replay(journal,3302,run,source,watcher)
        result = replay(rows)
        self.assertEqual(result['freeze_events'],3)
        self.assertEqual(result['safe_resume_events'],1)
        self.assertEqual(result['watch_sha256'],digest(watcher))
        bad = copy.deepcopy(rows)
        bad[3]['certificate']['command']=['python','-c','pass','-m','search_r1.baselines.main']
        with self.assertRaises(g.JournalError):replay(bad)
        bad = copy.deepcopy(rows)
        bad[4]['certificate']['command']=['ray::WorkerDict.ref_capture_fresh200_diagnostic']
        with self.assertRaises(g.JournalError):replay(bad)
        bad = copy.deepcopy(rows)
        bad[3]['certificate']['parent_start_ticks']=999
        with self.assertRaises(g.JournalError):replay(bad)
        bad = copy.deepcopy(rows)
        bad[3]['certificate']['allocated_uuid_set'][0]='GPU-foreign'
        with self.assertRaises(g.JournalError):replay(bad)
        with self.assertRaises(g.JournalError):replay(rows[:-3])
        with self.assertRaises(g.JournalError):replay(rows,'f'*40)
        with patch.object(g,'WATCH_SHA','0'*64):
            with self.assertRaises(g.JournalError):replay(rows)


if __name__ == '__main__':
    unittest.main()
