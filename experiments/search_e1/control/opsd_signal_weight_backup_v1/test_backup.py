"""Tiny synthetic CPU/USTAR/transport fixtures, no SSH or actual state tensors."""
import base64
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;PACKAGE='_signal_backup_test_package'
package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package
peer=importlib.import_module(PACKAGE+'.peer');transfer=importlib.import_module(PACKAGE+'.transfer')


class BackupContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'source-study';self.target=self.root/'remote-target'
        self.variant='G_POS';self.run_id='a'*32;self.cp_dir=self.source/'group-G_POS/runs/checkpoints/update-00000005-outer-00000'
        self.cp_dir.mkdir(parents=True)
        runtime=self.write(self.source/'receipts/runtime-identity.json',{'tiny':'runtime identity'})
        model=self.write(self.source/'receipts/source100-identity.json',{'tiny':'model identity'})
        initial=self.write(self.source/'receipts/commonzero-identity.json',{'tiny':'initial identity'})
        cfg=dict(schema='opsd-signal-weight-config-v1',variant='G_POS',run_id=self.run_id,study_id='tiny-study',owner={'uid':1001},
            precision={'mode':'bf16'},baseline_variant='R',runtime_manifest=runtime,source_model_manifest=model,common_initial_adapter_manifest=initial,
            scientific=dict(outer_iterations=25,world_size=4,micro_batch=1,global_batch=32,gradient_accumulation_full_batch_per_rank=8,
                original_eight_rank_accumulation=4,learning_rate=1e-6,component_clip=10,loss_reduction='trajectory_mean',teacher_frozen=True))
        self.config=self.write(self.source/'group-G_POS/receipts/config.json',cfg)
        ranks=[dict(rank=r,file=self.raw(self.cp_dir/('rank-%d-state.pt'%r),('synthetic rank'+str(r)).encode())) for r in range(4)]
        shared=self.raw(self.cp_dir/'optimizer-shared.pt',b'synthetic metadata-only optimizer identity')
        files=[self.raw(self.cp_dir/'adapter/adapter_config.json',b'{}'),self.raw(self.cp_dir/'adapter/adapter_model.safetensors',b'tiny synthetic bytes; no tensor data')]
        body=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=self.run_id,config_sha256=self.config['sha256'],optimizer_updates=5,
            cursor={'outer_index':0,'next_pair_cursor':160,'usable_pair_count':327},complete_rank_count=4,equal_optimizer_steps=True,
            contract=peer.science_contract(cfg),runtime_identity=runtime['sha256'],source_model_identity=model['sha256'],common_initial_adapter=initial,
            all_rank_optimizer_tensor_sha256=['c'*64]*4,all_rank_adapter_sha256=['d'*64]*4,all_rank_scaler_state=[{'enabled':False}]*4,
            all_rng=['python','numpy','torch_cpu','cuda_all'],scheduler={'kind':'none','warmup_updates':0},cumulative_pair_exposure=160,
            optimizer_storage='shared_exact_four_rank_optimizer_v1',ranks=ranks,shared_optimizer=shared,
            adapter={'path':str(self.cp_dir/'adapter'),'files':files},protection_reason=['update5'])
        self.checkpoint=self.write(self.cp_dir/'manifest.json',body);self.peer_sha=peer.proof(peer.__file__)['sha256']
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(peer,'SOURCE',str(self.source)))
        self.stack.enter_context(patch.object(peer,'TARGET',str(self.target)))
        self.stack.enter_context(patch.dict(peer.os.environ,{'CUDA_VISIBLE_DEVICES':''}))
        self.stack.enter_context(patch.object(peer.shutil,'disk_usage',return_value=types.SimpleNamespace(free=200<<30)))

    def raw(self,path,raw):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return peer.proof(path)

    def write(self,path,value):return peer.write_once(path,value)

    def inventory(self):
        with patch.object(peer.os,'getuid',return_value=1001):
            return peer.inventory(self.config,self.checkpoint,'G_POS',5,0,self.peer_sha)

    def tar_bytes(self,value,mutate=None):
        output=io.BytesIO()
        with tarfile.open(fileobj=output,mode='w',format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(value['members']):
                raw=Path(value['members'][name]['path']).read_bytes();info=tarfile.TarInfo(name);info.size=len(raw)
                if mutate:info,raw=mutate(info,raw)
                archive.addfile(info,io.BytesIO(raw))
        return output.getvalue()

    def terminal(self,value):
        return dict(schema='opsd-signal-source-stream-terminal-v1',accepted=True,inventory_sha256=peer.object_sha(value),
            source_tar_actual_wait_returncode=0,before_after_inventory_equal=True,source_peer_sha256=self.peer_sha)

    def transport_evidence(self):
        return dict(source_ssh_actual_wait_returncode=0,target_ssh_actual_wait_returncode=0,source_alias='hkust4',target_alias='hkust1',
            direct_ssh_options=list(peer.DIRECT_SSH_OPTIONS),connection_mode='direct_no_proxy_no_multiplex',local_checkpoint_payload_files_created=0)

    def test_actual_inventory_requires_expected_update_and_protects_partial_u5(self):
        value=self.inventory();self.assertEqual(value['member_count'],8);self.assertEqual(value['optimizer_updates'],5)
        self.assertEqual(value['cursor']['next_pair_cursor'],160);self.assertIn('no optimizer/tensor',value['numeric_scope'])
        with patch.object(peer.os,'getuid',return_value=1001):
            with self.assertRaisesRegex(ValueError,'checkpoint/run/count'):peer.inventory(self.config,self.checkpoint,'G_POS',4,0,self.peer_sha)

    def test_pure_json_scientific_identity_corruption_does_not_accept_byte_backup(self):
        original=json.loads(Path(self.checkpoint['path']).read_text())
        mutations=(lambda x:x.__setitem__('runtime_identity','0'*64),lambda x:x.__setitem__('source_model_identity','0'*64),
            lambda x:x['contract'].__setitem__('target_version','wrong-target'),lambda x:x.__setitem__('common_initial_adapter',{'sha256':'0'*64}),
            lambda x:x['all_rank_optimizer_tensor_sha256'].__setitem__(3,'e'*64),lambda x:x['all_rank_adapter_sha256'].__setitem__(2,'e'*64),
            lambda x:x['all_rank_scaler_state'].__setitem__(1,{'enabled':True}),lambda x:x.__setitem__('all_rng',['torch_cpu']),
            lambda x:x.__setitem__('scheduler',{'kind':'linear','warmup_updates':57}),lambda x:x.__setitem__('cumulative_pair_exposure',-1))
        for mutate in mutations:
            changed=copy.deepcopy(original);mutate(changed);Path(self.checkpoint['path']).write_text(json.dumps(changed));self.checkpoint=peer.proof(self.checkpoint['path'])
            with self.subTest(mutation=repr(mutate)),self.assertRaises(ValueError):self.inventory()

    def test_rank_alias_inode_and_adapter_inventory_declarations_are_checked(self):
        original=json.loads(Path(self.checkpoint['path']).read_text())
        for key,mutate in [('rankalias',lambda x:x['ranks'][1].__setitem__('file',x['ranks'][0]['file'])),
            ('sharedpath',lambda x:x.__setitem__('shared_optimizer',x['ranks'][0]['file'])),
            ('adapterdirectory',lambda x:x['adapter'].__setitem__('path',str(self.cp_dir))),
            ('adaptermissing',lambda x:x['adapter'].__setitem__('files',x['adapter']['files'][:1]))]:
            changed=copy.deepcopy(original);mutate(changed);Path(self.checkpoint['path']).write_text(json.dumps(changed));self.checkpoint=peer.proof(self.checkpoint['path'])
            with self.subTest(key=key),self.assertRaises(ValueError):self.inventory()
        path=self.cp_dir/'rank-1-state.pt';path.unlink();os.link(self.cp_dir/'rank-0-state.pt',path)
        original['ranks'][1]['file']=peer.proof(path);Path(self.checkpoint['path']).write_text(json.dumps(original));self.checkpoint=peer.proof(self.checkpoint['path'])
        with self.assertRaisesRegex(ValueError,'inode'):self.inventory()

    def test_unprotected_partial_checkpoint_is_rejected(self):
        body=json.loads(Path(self.checkpoint['path']).read_text());body['optimizer_updates']=6
        new=self.cp_dir.with_name('update-00000006-outer-00000');self.cp_dir.rename(new)
        for rank in body['ranks']:rank['file']['path']=str(new/Path(rank['file']['path']).name)
        body['shared_optimizer']['path']=str(new/'optimizer-shared.pt')
        for file in body['adapter']['files']:file['path']=str(new/'adapter'/Path(file['path']).name)
        body['adapter']['path']=str(new/'adapter')
        path=new/'manifest.json';path.write_text(json.dumps(body));checkpoint=peer.proof(path)
        with patch.object(peer.os,'getuid',return_value=1001):
            with self.assertRaisesRegex(ValueError,'Only protected'):peer.inventory(self.config,checkpoint,'G_POS',6,0,self.peer_sha)

    def test_source_file_change_or_undocumented_extra_is_rejected(self):
        Path(self.config['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.inventory()

    def test_unexplained_extra_checkpoint_file_is_not_silently_copied(self):
        (self.cp_dir/'uncommitted-junk.pt').write_bytes(b'tiny unexplained fixture')
        with self.assertRaisesRegex(ValueError,'unexplained/symlink'):self.inventory()

    def test_receive_waits_for_true_source_and_both_ssh_codes_before_complete(self):
        value=self.inventory();data=self.tar_bytes(value)
        with patch.object(peer.os,'getuid',return_value=1005):
            pending=peer.receive(value,self.peer_sha,io.BytesIO(data));destination=Path(pending['destination'])
            self.assertFalse(pending['complete']);self.assertFalse((destination/'complete-sha-verified.json').exists())
            bad=self.transport_evidence();bad['source_ssh_actual_wait_returncode']=1
            with self.assertRaisesRegex(ValueError,'dual SSH wait0'):peer.commit(value,self.peer_sha,pending['pending_receipt'],self.terminal(value),bad)
            self.assertFalse((destination/'complete-sha-verified.json').exists())
            accepted=peer.commit(value,self.peer_sha,pending['pending_receipt'],self.terminal(value),self.transport_evidence())
            self.assertTrue(accepted['accepted']);self.assertEqual(accepted['member_count'],8)
            for name,entry in value['members'].items():self.assertEqual(peer.proof(destination/'payload'/name)['sha256'],entry['sha256'])
            again=peer.verify_existing(value,destination);self.assertTrue(again['accepted'])
            self.assertEqual(peer.proof(destination/'source-config.json')['sha256'],self.config['sha256'])
            (destination/'payload/optimizer-shared.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'SHA differs'):peer.verify_existing(value,destination)

    def test_same_pinned_commit_retries_marker_failure_after_move_without_overwrite(self):
        value=self.inventory();actual_write=peer.write_once
        with patch.object(peer.os,'getuid',return_value=1005):
            pending=peer.receive(value,self.peer_sha,io.BytesIO(self.tar_bytes(value)));destination=Path(pending['destination'])
            def fail_marker(path,body):
                if Path(path).name=='complete-sha-verified.json':raise OSError('Synthetic durable marker I/O failure')
                return actual_write(path,body)
            with patch.object(peer,'write_once',side_effect=fail_marker):
                with self.assertRaisesRegex(OSError,'marker I/O'):peer.commit(value,self.peer_sha,pending['pending_receipt'],self.terminal(value),self.transport_evidence())
            self.assertTrue((destination/'payload').exists());self.assertFalse((destination/'complete-sha-verified.json').exists())
            before={name:peer.proof(destination/'payload'/name) for name in value['members']}
            result=peer.commit(value,self.peer_sha,pending['pending_receipt'],self.terminal(value),self.transport_evidence())
            self.assertTrue(result['accepted'])
            self.assertEqual(before,{name:peer.proof(destination/'payload'/name) for name in value['members']})
            (destination/'payload/unexpected.pt').write_bytes(b'junk')
            with self.assertRaisesRegex(ValueError,'unexpected/symlink'):peer.verify_existing(value,destination)

    def test_accepted_probe_rechecks_actual_direct_wait_evidence(self):
        value=self.inventory()
        with patch.object(peer.os,'getuid',return_value=1005):
            pending=peer.receive(value,self.peer_sha,io.BytesIO(self.tar_bytes(value)))
            result=peer.commit(value,self.peer_sha,pending['pending_receipt'],self.terminal(value),self.transport_evidence())
            body=json.loads(Path(result['receipt']['path']).read_text());body['actual_transport']['direct_ssh_options']=['-o','ProxyJump=proxy']
            Path(result['receipt']['path']).write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError,'direct dual SSH'):peer.verify_existing(value,Path(result['destination']))

    def test_actual_local_ustar_command_is_accepted_for_tiny_identity_fixture(self):
        value=self.inventory();path=self.root/'tiny-source-command.tar'
        with patch.object(peer.os,'getuid',return_value=1001),patch.dict(peer.os.environ,{'LC_ALL':'C'}),path.open('wb') as stream:
            terminal=peer.send(value,self.peer_sha,stream)
        self.assertEqual(terminal['source_tar_actual_wait_returncode'],0);self.assertTrue(terminal['before_after_inventory_equal'])
        extracted=peer.extract_ustar(path,self.root/'tiny-command-extracted',value['members'])
        self.assertEqual(set(extracted),set(value['members']))

    def test_truncated_stream_keeps_remote_partial_and_does_not_claim_complete(self):
        value=self.inventory();data=self.tar_bytes(value)[:1000]
        with patch.object(peer.os,'getuid',return_value=1005):
            with self.assertRaises(ValueError):peer.receive(value,self.peer_sha,io.BytesIO(data))
        partial=list(self.target.rglob('checkpoint.tar.partial'));self.assertEqual(len(partial),1);self.assertEqual(partial[0].read_bytes(),data)
        self.assertFalse(list(self.target.rglob('complete-sha-verified.json')))

    def test_remote_backup_disk_reserve_failure_starts_no_payload_or_gpu_work(self):
        value=self.inventory()
        with patch.object(peer.os,'getuid',return_value=1005),patch.object(peer.shutil,'disk_usage',return_value=types.SimpleNamespace(free=50<<30)):
            with self.assertRaisesRegex(ValueError,'100GiB reserve'):peer.receive(value,self.peer_sha,io.BytesIO(self.tar_bytes(value)))
        self.assertFalse(list(self.target.rglob('checkpoint.tar.partial')))
        self.assertFalse(list(self.target.rglob('complete-sha-verified.json')))

    def test_ustar_rejects_symlink_duplicate_unknown_and_missing_terminator(self):
        value=self.inventory()
        def mutate(info,raw):info.type=tarfile.SYMTYPE;info.linkname='/outside';info.size=0;return info,b''
        tests=[self.tar_bytes(value,mutate),self.tar_bytes(value)[:-1024]]
        # A byte-level trailer mutation cannot be hidden after the first valid terminator.
        tests.append(self.tar_bytes(value)+b'X'+bytes(511))
        for index,data in enumerate(tests):
            path=self.root/('bad-%d.tar'%index);path.write_bytes(data)
            if index==1:
                # Standard USTAR includes record padding; cut inside the payload instead.
                path.write_bytes(data[:1000])
            with self.assertRaises(ValueError):peer.extract_ustar(path,self.root/('extract-'+str(index)),value['members'])

    def test_duplicate_and_unexpected_ustar_members_fail_closed(self):
        value=self.inventory()
        for kind in ('duplicate','unexpected'):
            output=io.BytesIO()
            with tarfile.open(fileobj=output,mode='w',format=tarfile.USTAR_FORMAT) as archive:
                first=next(iter(value['members']));data=Path(value['members'][first]['path']).read_bytes()
                info=tarfile.TarInfo(first);info.size=len(data);archive.addfile(info,io.BytesIO(data))
                info=tarfile.TarInfo(first if kind=='duplicate' else 'unexpected.pt');info.size=len(data);archive.addfile(info,io.BytesIO(data))
            path=self.root/(kind+'.tar');path.write_bytes(output.getvalue())
            with self.assertRaisesRegex(ValueError,'Unexpected/duplicate'):peer.extract_ustar(path,self.root/('bad-'+kind),value['members'])

    def test_readonly_inputs_do_not_match_production_execution_roots(self):
        old=Path(os.environ.get('OPSD_SIGNAL_TEST_PRODUCTION_HELPER',HERE.parent/'opsd_signal_weight_v1'))/'owned_process_audit.py'
        module=types.ModuleType('_backup_ownership_fixture');module.__file__=str(old)
        exec(compile(old.read_bytes(),str(old),'exec'),module.__dict__)
        root=str(self.source/'group-G_POS/runs');cp_path=str(self.cp_dir/'manifest.json')
        arguments=['/env/python','/independent/peer.py','--checkpoint',cp_path]
        execution,inputs=module.matched_roots(arguments,[root,str(self.source/'source/frozen/opsd_signal_weight_v1/trainer.py')])
        self.assertEqual(execution,[]);self.assertEqual(inputs,[root])
        execution,inputs=module.matched_roots(['tar','--format=ustar','-C',str(self.cp_dir),'manifest.json'],[root])
        self.assertEqual(execution,[]);self.assertEqual(inputs,[root])

    def test_target_rejects_wrong_uid_cuda_peer_source_or_escaping_member(self):
        value=self.inventory()
        with patch.object(peer.os,'getuid',return_value=1001):
            with self.assertRaisesRegex(ValueError,'remote UID'):peer.target_root(value,self.peer_sha)
        with patch.object(peer.os,'getuid',return_value=1005),patch.dict(peer.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'CUDA visibility'):peer.target_root(value,self.peer_sha)
        with patch.object(peer.os,'getuid',return_value=1005):
            with self.assertRaisesRegex(ValueError,'publication differs'):peer.target_root(value,'0'*64)
            bad=copy.deepcopy(value);first=next(iter(bad['members']));bad['members']['../outside']=bad['members'].pop(first)
            with self.assertRaisesRegex(ValueError,'Unsafe checkpoint'):peer.target_root(bad,self.peer_sha)

    def test_remote_shell_arguments_are_quoted_and_only_hkust_aliases_allowed(self):
        command=transfer.ssh_command('hkust4','/env/python',"/path/peer.py",self.peer_sha,'inventory',['--config',"/path/$(private);`secret`.json"])
        self.assertIn("'/path/$(private);`secret`.json'",command[-1]);self.assertIn('CUDA_VISIBLE_DEVICES=',command[-1])
        self.assertEqual(command[:-2],['ssh']+peer.DIRECT_SSH_OPTIONS);self.assertEqual(command[-2],'hkust4')
        with self.assertRaisesRegex(ValueError,'approved HKUST'):transfer.ssh_command('HKUST4','/env/python','/peer.py',self.peer_sha,'probe',[])
        with self.assertRaisesRegex(ValueError,'approved HKUST'):transfer.ssh_command('other-cluster','/env/python','/peer.py',self.peer_sha,'probe',[])

    def test_diagnostics_are_drained_beyond_capture_cap_with_full_sha(self):
        raw=b'SSH diagnostics\n'*12000;drain=transfer.DiagnosticDrain(io.BytesIO(raw),cap=128<<10)
        captured,evidence=drain.finish();self.assertEqual(len(captured),128<<10)
        self.assertEqual(evidence['bytes'],len(raw));self.assertFalse(evidence['capture_complete'])
        self.assertIsNone(evidence['error_type']);self.assertEqual(evidence['sha256'],hashlib.sha256(raw).hexdigest())

    def test_existing_target_short_circuits_without_launching_any_payload_pipe(self):
        value=self.inventory();calls=[]
        def run(command,**kwargs):
            calls.append(command);result=value if command[-2]=='hkust4' else dict(accepted=True,state='COMPLETE_FRESH_SHA_READBACK',receipt={'sha256':'1'*64})
            return types.SimpleNamespace(returncode=0,stdout=json.dumps(result).encode(),stderr=b'')
        with patch.object(transfer,'sys',types.SimpleNamespace(modules={})):
            result=transfer.backup(self.config['path'],self.config['sha256'],self.checkpoint['path'],self.checkpoint['sha256'],'G_POS',5,0,
                '/python','/peer.py','/python','/peer.py',self.peer_sha,runner=run,popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No transfer should launch')))
        self.assertTrue(result['already_complete']);self.assertEqual(result['local_checkpoint_payload_files_created'],0);self.assertEqual(len(calls),2)

    def test_remote_nonzero_control_wait_is_not_accepted_and_no_restart(self):
        result=types.SimpleNamespace(returncode=7,stdout=b'{}',stderr=b'synthetic failure')
        with self.assertRaisesRegex(transfer.TransferError,'SSH returned nonzero') as context:transfer.read_call(['ssh'],runner=lambda *a,**k:result)
        self.assertEqual(context.exception.evidence['actual_ssh_wait_returncode'],7)

    def test_stream_transport_never_reads_or_creates_local_checkpoint_payload(self):
        value=self.inventory();calls=[];popen_calls=[]
        receipt={'path':str(self.target/'pending.json'),'sha256':'1'*64}
        class NeverReadPayload:
            closed=False
            def read(self,*args):raise AssertionError('Local controller must never capture payload bytes')
            def close(self):self.closed=True
        source_stdout=NeverReadPayload()
        sender=types.SimpleNamespace(stdout=source_stdout,stderr=io.BytesIO(json.dumps({'source_stream_terminal':self.terminal(value)}).encode()+b'\n'),wait=lambda:0)
        receiver=types.SimpleNamespace(stdout=io.BytesIO(json.dumps({'accepted':True,'complete':False,'pending_receipt':receipt}).encode()),
            stderr=io.BytesIO(b''),returncode=0,wait=lambda:0)
        def popen(command,**kwargs):
            popen_calls.append((command,kwargs))
            if len(popen_calls)==1:return sender
            self.assertIs(kwargs['stdin'],source_stdout);return receiver
        def run(command,**kwargs):
            calls.append(command)
            if command[-2]=='hkust4':body=value
            elif ' commit ' in command[-1]:body={'accepted':True,'state':'COMPLETE_FRESH_SHA_READBACK','receipt':receipt}
            else:body={'accepted':False,'state':'NOT_COMPLETE'}
            return types.SimpleNamespace(returncode=0,stdout=json.dumps(body).encode(),stderr=b'')
        with patch.object(transfer,'sys',types.SimpleNamespace(modules={})):
            result=transfer.backup(self.config['path'],self.config['sha256'],self.checkpoint['path'],self.checkpoint['sha256'],'G_POS',5,0,
                '/python','/peer.py','/python','/peer.py',self.peer_sha,runner=run,popen=popen)
        self.assertTrue(result['accepted']);self.assertEqual(result['local_checkpoint_payload_files_created'],0)
        self.assertTrue(source_stdout.closed);self.assertEqual(len(popen_calls),2)
        self.assertEqual(result['actual_transport']['source_ssh_actual_wait_returncode'],0)
        self.assertEqual(len(calls),3,'Only inventory/probe/commit control calls surround the streamed payload')


if __name__=='__main__':unittest.main()
