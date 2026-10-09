"""Synthetic CPU-only tests; no process, tmux, GPU or model is accessed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    from . import capture as c
except ImportError:
    import capture as c


def raw(value): return json.dumps(value, sort_keys=True).encode()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve(); self.n = 0

    def save(self, value, name=None):
        self.n += 1; path = self.root / (name or ('f%d.json' % self.n)); path.write_bytes(raw(value))
        return dict(path=str(path), sha256=c.digest(path.read_bytes()), bytes=path.stat().st_size)

    def native(self):
        owner = dict(c.OWNER, euid=1001, ppid=c.SERVER_PID, pgid=c.OWNER['pid'], argv=['python', 'entry'], run_id=c.MARKER)
        studyref = self.save({'id': 'study'}); configs = []; groups = []
        for index, variant in enumerate(c.VARIANTS):
            config = dict(variant=variant, run_id='run-' + variant, owner={'uid': 1001}); cp = self.save(config)
            configs.append((config, cp))
            controller = dict(pid=200 + index, uid=1001, starttick=300 + index, run_id=config['run_id'])
            actual = self.save(dict(actual_wait_returncode=0, controller=controller))
            ledger = self.save(dict(accepted=True, variant=variant, run_id=config['run_id'], outer_iterations=25,
                rows=[dict(outer_index=i, run_id=config['run_id'], config_sha256=cp['sha256']) for i in range(25)]))
            groups.append(dict(variant=variant, actual_wait_returncode=0, summary=ledger, actual_terminal=actual,
                release=dict(accepted=True, audit=dict(accepted_no_owned_or_opaque=True, blockers=[]))))
        value = dict(accepted=True, completed_groups=4, completed_outers=100, full_training_em_rows=100,
            study_manifest=studyref, actual_cpu_supervisor=owner, groups=groups)
        return value, studyref, configs, owner

    def pane(self, dead='0', status=''):
        return dict(session_name=c.SESSION, session_id='$45', pane_id='%45', pane_pid=str(c.OWNER['pid']),
            pane_dead=dead, pane_dead_status=status, pane_start_command='original-start', pid=str(c.SERVER_PID),
            socket_path='/original-socket', remain_on_exit=True)

    def binding(self):
        request = dict(owner=c.OWNER, supervisor_run_id=c.MARKER, log_path='/log')
        originals = dict(expected_argv=['python', 'entry'], study_manifest={'path': '/study', 'sha256': '1'*64})
        owner = dict(c.OWNER, euid=1001, status='PRESENT', state='S', ppid=c.SERVER_PID,
            pgid=c.OWNER['pid'], argv=originals['expected_argv'], run_id=c.MARKER,
            cuda_visible_devices='', dont_write_bytecode='1')
        server = dict(pid=c.SERVER_PID, uid=1001, euid=1001, starttick=123, status='PRESENT', state='S')
        socket = dict(path='/original-socket', uid=1001, inode=2, device=3, mode=4, ctime_ns=5)
        binding = dict(schema='opsd-signal-weight-driver-live-binding-v1', accepted=True, mode='live', completion=False,
            supervisor_run_id=c.MARKER, originals=originals, owner=owner, server=server, socket=socket,
            tmux=self.pane(), exec_same_pid=True)
        return request, originals, owner, server, socket, binding


class FilesAndLaunchTests(Fixture):
    def test_bound_sha_and_single_json_bytes(self):
        ref = self.save({'correct': 1}); value, proof = c.bound(ref)
        self.assertEqual(value, {'correct': 1}); self.assertEqual(ref, proof)
        ref['sha256'] = '0' * 64
        with self.assertRaises(c.Hold): c.bound(ref)

    def test_symlink_is_rejected(self):
        path = self.root / 'real'; path.write_text('{}'); link = self.root / 'link'; link.symlink_to(path)
        with self.assertRaises(c.Hold): c.read(link)

    def test_script_is_single_exec_and_no_trailing_zero(self):
        script = b"#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=''\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONUNBUFFERED=1\nexport E1_RUN_ID=marker\nexec python entry --resume\n"
        self.assertTrue(c.launch_contract(script, ['python', 'entry', '--resume'], 'marker')['no_following_command'])
        for bad in (script + b'exit 0\n', script.replace(b'--resume', b'--resume; true'), script.replace(b'PYTHONUNBUFFERED=1', b'PYTHONUNBUFFERED=0')):
            with self.assertRaises(c.Hold): c.launch_contract(bad, ['python', 'entry', '--resume'], 'marker')

    def test_tree_checks_actual_file_and_remote_bytes(self):
        helper = self.root / 'helper'; helper.mkdir(); (helper / 'capture.py').write_text('x=1\n')
        payload = c.read(helper / 'capture.py')[1]
        manifest = dict(files={'capture.py': {'sha256': payload['sha256'], 'bytes': payload['bytes']}})
        (helper / 'source-manifest.json').write_bytes(raw(manifest)); mr = c.read(helper / 'source-manifest.json')[1]
        publication = dict(accepted=True, branch='search-e1', source_commit='a'*40, remote_readback_sha='a'*40,
            source_manifest_sha256=mr['sha256'], files=[dict(path='experiments/search_e1/control/helper/'+name, sha256=fp['sha256'], bytes=fp['bytes'])
                for name, fp in [('capture.py', payload), ('source-manifest.json', mr)]])
        pr = self.save(publication)
        self.assertEqual(c.tree(helper, mr, pr)['source_commit'], 'a'*40)
        bare = copy.deepcopy(publication)
        for row in bare['files']: row['name'] = row.pop('path').rsplit('/', 1)[1]
        self.assertEqual(c.tree(helper, mr, self.save(bare))['source_commit'], 'a'*40)
        bad=copy.deepcopy(publication);bad['files'][0]['name']='another.py'
        with self.assertRaises(c.Hold):c.tree(helper,mr,self.save(bad))
        (helper / 'capture.py').write_text('x=2\n')
        with self.assertRaises(c.Hold): c.tree(helper, mr, pr)

    def test_tree_refuses_extra_python(self):
        helper = self.root / 'helper'; helper.mkdir(); (helper / 'capture.py').write_text('x=1\n')
        fp=c.read(helper / 'capture.py')[1]; (helper / 'source-manifest.json').write_bytes(raw({'files':{'capture.py':{'sha256':fp['sha256'],'bytes':fp['bytes']}}}))
        mr=c.read(helper / 'source-manifest.json')[1]; (helper / 'evil.py').write_text('x=2\n')
        with self.assertRaises(c.Hold): c.tree(helper, mr, self.save({}))


class NativeCompletionTests(Fixture):
    def check(self, value, sr, configs, owner):
        with patch.object(c, 'process', lambda pid: dict(pid=pid, status='ABSENT')):
            return c.verify_native_completion(self.save(value), {}, sr, configs, owner)

    def test_full_native_wait0_and_rows_accept(self):
        value, sr, configs, owner = self.native(); captureowner=dict(owner, state='S', cuda_visible_devices='', status='PRESENT')
        result=self.check(value, sr, configs, captureowner)
        self.assertEqual(result['completed_outers'],100); self.assertEqual(len(result['groups']),4)

    def test_nonzero_or_unknown_wait_rejected(self):
        value, sr, configs, owner = self.native()
        for rc in [1, -15, 'UNKNOWN']:
            bad=copy.deepcopy(value); bad['groups'][0]['actual_wait_returncode']=rc
            with self.assertRaises(c.Hold): self.check(bad,sr,configs,owner)

    def test_bound_actual_wait_differs_rejected(self):
        value,sr,configs,owner=self.native(); actual, _=c.bound(value['groups'][0]['actual_terminal'])
        actual['actual_wait_returncode']=1; value['groups'][0]['actual_terminal']=self.save(actual)
        with self.assertRaises(c.Hold): self.check(value,sr,configs,owner)

    def test_native_driver_argv_or_tick_reuse_rejected(self):
        value,sr,configs,owner=self.native()
        for key,badvalue in [('starttick', owner['starttick']+1),('argv',['wrong'])]:
            bad=copy.deepcopy(value);bad['actual_cpu_supervisor'][key]=badvalue
            with self.assertRaises(c.Hold):self.check(bad,sr,configs,owner)

    def test_missing_duplicate_ledger_outer_rejected(self):
        value,sr,configs,owner=self.native(); ledger,_=c.bound(value['groups'][0]['summary'])
        for rows in [ledger['rows'][:-1], ledger['rows'][:-1]+[ledger['rows'][0]]]:
            bad=copy.deepcopy(value);new=dict(ledger,rows=rows);bad['groups'][0]['summary']=self.save(new)
            with self.assertRaises(c.Hold):self.check(bad,sr,configs,owner)

    def test_ledger_config_or_run_changed_rejected(self):
        value,sr,configs,owner=self.native();ledger,_=c.bound(value['groups'][0]['summary'])
        for key in ['run_id','config_sha256']:
            bad=copy.deepcopy(value);new=copy.deepcopy(ledger);new['rows'][2][key]='wrong';bad['groups'][0]['summary']=self.save(new)
            with self.assertRaises(c.Hold):self.check(bad,sr,configs,owner)

    def test_owned_opaque_or_live_child_rejected(self):
        value,sr,configs,owner=self.native();value['groups'][0]['release']['audit']['blockers']=[{'pid':1}]
        with self.assertRaises(c.Hold):self.check(value,sr,configs,owner)
        with patch.object(c,'process',return_value=dict(pid=100,uid=1001,starttick=5,status='PRESENT')):
            with self.assertRaises(c.Hold):c.absent_original(dict(pid=100,uid=1001,starttick=5))


class LiveAndTerminalTests(Fixture):
    def run_terminal(self, binding=None, pane=None, socket=None, server=None, native=None, release=None, raw_sha=None):
        request,orig,owner,sv,so,default=self.binding(); binding=default if binding is None else binding
        encoded=raw(binding); native= native or dict(accepted=True,native_completion={'path':'/native','sha256':'2'*64},groups=[])
        with patch.object(c,'tmux_snapshot',return_value=pane or self.pane('1','0')), \
             patch.object(c,'process',return_value=server or sv), \
             patch.object(c,'socket_identity',return_value=socket or so), \
             patch.object(c,'absent_original',return_value={'original_identity_released':True}), \
             patch.object(c,'verify_native_completion',return_value=native), \
             patch.object(c,'owned_release',return_value=release or {'accepted':True}):
            return c.terminal(request,orig,{},[],encoded,raw_sha or c.digest(encoded),{'path':'/native','sha256':'2'*64})

    def test_same_retained_exit0_accepts_with_explicit_source(self):
        result=self.run_terminal()
        self.assertEqual(result['actual_supervisor_wait_returncode'],0)
        self.assertEqual(result['driver_exit_evidence']['source'],'same-retained-tmux-pane')
        self.assertTrue(result['driver_exit_evidence']['no_exit_code_inferred_from_absence'])
        self.assertTrue(result['analysis_still_required'])

    def test_absence_does_not_allow_live_or_nonzero_pane(self):
        for pane in [self.pane('0',''),self.pane('1','1'),self.pane('1',''),self.pane('1','143')]:
            with self.assertRaises(c.Hold):self.run_terminal(pane=pane)

    def test_original_binding_bytes_not_reformatted(self):
        with self.assertRaises(c.Hold):self.run_terminal(raw_sha='0'*64)

    def test_another_binding_source_rejected(self):
        *_,binding=self.binding();binding['originals']['expected_argv']=['wrong']
        with self.assertRaises(c.Hold):self.run_terminal(binding=binding)

    def test_completion_cannot_pose_as_live_binding(self):
        *_,binding=self.binding();binding['completion']=True
        with self.assertRaises(c.Hold):self.run_terminal(binding=binding)

    def test_server_tick_or_socket_reuse_rejected(self):
        request,orig,owner,server,socket,binding=self.binding()
        with self.assertRaises(c.Hold):self.run_terminal(server=dict(server,starttick=server['starttick']+1))
        with self.assertRaises(c.Hold):self.run_terminal(socket=dict(socket,inode=socket['inode']+1))

    def test_pane_birth_command_changed_rejected(self):
        pane=self.pane('1','0');pane['pane_start_command']='exit 0'
        with self.assertRaises(c.Hold):self.run_terminal(pane=pane)

    def test_live_same_pid_exec_and_actual_log(self):
        request,orig,owner,server,socket,binding=self.binding()
        original_readlink=c.os.readlink
        with patch.object(c,'tmux_snapshot',return_value=self.pane()), \
             patch.object(c,'process',side_effect=lambda pid: owner if pid==owner['pid'] else server), \
             patch.object(c,'socket_identity',return_value=socket), \
             patch.object(c.os,'readlink',side_effect=lambda path,*a,**k:'/log' if str(path).startswith('/proc/') else original_readlink(path,*a,**k)):
            result=c.live(request,orig)
        self.assertFalse(result['completion']);self.assertTrue(result['no_exit_code_claimed'])

    def test_live_wrong_foreground_or_parent_rejected(self):
        request,orig,owner,server,socket,binding=self.binding()
        for wrong in [dict(owner,argv=['wrong']),dict(owner,ppid=1),dict(owner,cuda_visible_devices='0')]:
            with patch.object(c,'tmux_snapshot',return_value=self.pane()), \
                 patch.object(c,'process',side_effect=lambda pid: wrong if pid==owner['pid'] else server), \
                 patch.object(c,'socket_identity',return_value=socket):
                with self.assertRaises(c.Hold):c.live(request,orig)


class GPUReleaseTests(Fixture):
    def release(self, apps):
        source=self.root/'retriever_stream.py';source.write_text('only-source\n')
        oracle=dict(pid=1749337,uid=1001,euid=1001,starttick=641133475,status='PRESENT',argv=[str(source)])
        admission=dict(accepted=True,pid=oracle['pid'],uid=1001,starttick=oracle['starttick'],
            source_path=str(source),source_identity_sha256=c.digest(source.read_bytes()))
        refs=[self.save(dict(admission,scope={'variant':v})) for v in c.VARIANTS]
        configs=[(dict(owner=dict(gpu_uuids=['g4','g5','g6','g7']),run_id='r'+v,
            paths={'output_root':'/out/'+v},gates={'retriever_admission':ref}),{}) for v,ref in zip(c.VARIANTS,refs)]
        devices=[dict(index=i,uuid='g'+str(i),actual_free_mib=40000) for i in range(8)]
        originals=dict(original_source={'manifest':{'path':'/old/source-manifest.json'}},
                       recovery_source={'manifest':{'path':'/new/source-manifest.json'}})
        with patch.object(c,'gpu_inventory',return_value=(devices,apps)),patch.object(c,'process',return_value=oracle),patch.object(Path,'iterdir',return_value=[]):
            return c.owned_release(configs,originals)

    def test_only_approved_gpu7_oracle_and_foreign0_3_ignored(self):
        result=self.release([dict(gpu_uuid='g7',pid=1749337,used_mib=738),dict(gpu_uuid='g0',pid=999,used_mib=40000)])
        self.assertEqual(len(result['owned_gpu_applications']),1);self.assertTrue(result['accepted'])

    def test_owned_foreign_wrong_gpu_or_oracle_memory_rejected(self):
        for app in [dict(gpu_uuid='g4',pid=999,used_mib=1),dict(gpu_uuid='g6',pid=1749337,used_mib=738),dict(gpu_uuid='g7',pid=1749337,used_mib=1025)]:
            with self.assertRaises(c.Hold):self.release([app])


if __name__=='__main__':unittest.main()
