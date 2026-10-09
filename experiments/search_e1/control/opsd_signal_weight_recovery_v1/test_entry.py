"""Synthetic CPU entry contracts. No SSH/GPU/process launch/real artifacts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import entry
import recovery_common


class Tests(unittest.TestCase):
    def setup_commands(self):
        reference=dict(path='/original/config.json',sha256='a'*64)
        config=dict(paths=dict(python='/original/python'),run_id='b'*32)
        return reference,config,{reference['path']:(config,reference)},Path('/original/helper'),dict(path='/new/repair.json',sha256='c'*64)

    def test_exact_controller_dispatch_keeps_original_config_run_and_resume(self):
        ref,cfg,configs,root,repair=self.setup_commands()
        command=[cfg['paths']['python'],str(root/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],
            '--run-id',cfg['run_id'],'--resume']
        result,role=entry.rewrite_command(command,configs,root,Path('/new/entry.py'),repair)
        self.assertEqual(role,'controller');self.assertEqual(result[-7:],command[2:]);self.assertIn('--resume',result)
        self.assertEqual(result[:4],['/original/python','/new/entry.py','--mode','controller'])

    def test_exact_torchrun_keeps_world_and_resume_retry_checkpoint_arguments(self):
        ref,cfg,configs,root,repair=self.setup_commands()
        args=['--config',ref['path'],'--config-sha256',ref['sha256'],'--outer-manifest','/outer/manifest.json','--outer-manifest-sha256','d'*64,
            '--resume-checkpoint','/cp/manifest.json','--resume-checkpoint-sha256','e'*64,
            '--retry-capsule-manifest','/retry/manifest.json','--retry-capsule-sha256','f'*64]
        command=[cfg['paths']['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'trainer.py')]+args
        result,role=entry.rewrite_command(command,configs,root,Path('/new/entry.py'),repair)
        self.assertEqual(role,'trainer');self.assertEqual(result[:5],command[:5]);self.assertEqual(result[-len(args):],args)
        self.assertEqual(result[5:8],['/new/entry.py','--mode','trainer'])

    def test_sampling_evaluation_teacher_and_foreign_jobs_not_rewritten(self):
        _,_,configs,root,repair=self.setup_commands()
        for command in (['/original/python',str(root/name),'--anything'] for name in ('sampling.py','evaluation.py','teacher_diag.py')):
            result,role=entry.rewrite_command(command,configs,root,Path('/new/entry.py'),repair)
            self.assertIsNone(role);self.assertEqual(result,command)
        for command in ('echo private', ['/other/python','/foreign/controller.py','--config','x']):
            result,role=entry.rewrite_command(command,configs,root,Path('/new/entry.py'),repair)
            self.assertIsNone(role);self.assertEqual(result,command)

    def test_controller_changed_sha_run_python_and_duplicate_flags_rejected(self):
        ref,cfg,configs,root,repair=self.setup_commands()
        command=[cfg['paths']['python'],str(root/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',cfg['run_id']]
        for i,replacement in ((0,'/wrong/python'),(5,'f'*64),(7,'wrongrun')):
            bad=list(command);bad[i]=replacement
            with self.assertRaises(ValueError):entry.rewrite_command(bad,configs,root,Path('/new/entry.py'),repair)
        with self.assertRaises(ValueError):entry.rewrite_command(command+['--config',ref['path']],configs,root,Path('/new/entry.py'),repair)

    def test_dispatch_verifies_before_popen_keeps_env_handle_and_restores(self):
        ref,cfg,configs,root,repair=self.setup_commands();events=[];env={'E1_RUN_ID':cfg['run_id'],'CUDA_VISIBLE_DEVICES':''}
        command=[cfg['paths']['python'],str(root/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',cfg['run_id']]
        handle=object()
        def popen(argv,*args,**kwargs):events.append(('spawn',argv,kwargs));return handle
        with patch.object(entry.subprocess,'Popen',popen):
            with entry.exact_dispatch(configs,root,repair,lambda:events.append(('verify',)),lambda x:events.append(('record',x))):
                self.assertIs(entry.subprocess.Popen(command,env=env,start_new_session=True),handle)
                self.assertEqual([x[0] for x in events],['verify','record','spawn'])
                self.assertIs(events[-1][2]['env'],env);self.assertTrue(events[-1][2]['start_new_session'])
            self.assertIs(entry.subprocess.Popen,popen)

    def test_failed_preflight_never_spawns_and_dispatch_context_restores(self):
        ref,cfg,configs,root,repair=self.setup_commands()
        command=[cfg['paths']['python'],str(root/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',cfg['run_id']]
        def fail():raise ValueError('Changed source')
        with patch.object(entry.subprocess,'Popen') as native:
            with self.assertRaises(ValueError),entry.exact_dispatch(configs,root,repair,fail,lambda x:None):entry.subprocess.Popen(command)
            native.assert_not_called();self.assertIs(entry.subprocess.Popen,native)

    def test_runpy_original_filename_and_function_globals_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'study_runner.py';path.write_text('def role():return "old"\ndef run():return __file__,role()\n')
            loaded=runpy.run_path(str(path),run_name='_recovery_original')
            loaded['run'].__globals__['role']=lambda:'overlay'
            self.assertEqual(loaded['run'](),(str(path),'overlay'))

    def test_pinned_JSON_uses_same_SHA_verified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.json';path.write_text('{"a":1}\n');reference=entry.file_ref(path)
            with patch.object(Path,'read_text',side_effect=AssertionError('Second unchecked JSON read')):
                value,proof=entry.checked_json(reference)
                self.assertEqual(value,{'a':1});self.assertEqual(proof,reference)
            path.write_text('{"a":2}\n')
            with self.assertRaises(ValueError):entry.checked_json(reference)

    def test_study_requires_explicit_resume_and_unknown_option_rejected(self):
        args=['--study-manifest','/original/study.json','--study-manifest-sha256','a'*64]
        with self.assertRaises(ValueError):entry.parse_original('study',args)
        self.assertTrue(entry.parse_original('study',args+['--resume']).resume)
        with self.assertRaises(SystemExit):entry.parse_original('study',args+['--resume','--fresh'])

    def scoped_fixture(self,directory):
        root=Path(directory).resolve();helper=root/'old';helper.mkdir();receipts=root/'receipts';receipts.mkdir()
        for name in ('study_runner.py','memory_observer.py'):(helper/name).write_text('original='+repr(name)+'\n')
        study_ref=dict(path=str(root/'study.json'),sha256='a'*64);repair_ref=dict(path=str(root/'repair.json'),sha256='b'*64)
        old_manifest=dict(path='/old/manifest.json',sha256='c'*64);old_pub=dict(path='/old/pub.json',sha256='d'*64)
        new_manifest=dict(path='/new/manifest.json',sha256='e'*64);new_pub=dict(path='/new/pub.json',sha256='f'*64)
        config=dict(run_id='group',study_id='study',source_commit='1'*40,owner=dict(uid=1001),
            paths=dict(python='/python',helper_root=str(helper),receipts=str(receipts)),helper_source_manifest=old_manifest,
            gates=dict(scoped_cpu_processes_path=str(receipts/'scoped-cpu-processes.json'),publication=old_pub),
            _config_ref_path=str(root/'config.json'),_config_sha256='2'*64)
        config_ref=dict(path=config['_config_ref_path'],sha256=config['_config_sha256'])
        repair=dict(publication=new_pub,source_manifest=new_manifest,supervisor_run_id='new-driver')
        argv=entry.overlay_command('/python',Path(entry.__file__).resolve(),'study',repair_ref,
            ['--study-manifest',study_ref['path'],'--study-manifest-sha256',study_ref['sha256'],'--resume'])
        driver=dict(pid=10,uid=1001,starttick=100,ppid=1,pgid=10,argv=argv,run_id='new-driver')
        observer=dict(pid=11,uid=1001,starttick=110,ppid=1,pgid=11,
            argv=['/python',str(helper/'memory_observer.py'),'--config',config_ref['path']],run_id='old-observer')
        def role(name,identity,input_ref,source):
            result=dict(role=name,identity=identity,no_gpu_or_training=True,input_config=input_ref,
                argv_sha256=hashlib.sha256(json.dumps(identity['argv'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                source=entry.file_ref(source),helper_source_manifest=old_manifest,publication=old_pub)
            if name=='study_driver':result['execution_overlay']=dict(repair_manifest=repair_ref,source=entry.file_ref(Path(entry.__file__)),
                publication=new_pub,helper_source_manifest=new_manifest,actual_entry_argv=argv)
            return result
        roles=[role('study_driver',driver,study_ref,helper/'study_runner.py'),role('memory_observer',observer,config_ref,helper/'memory_observer.py')]
        registry=dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id='group',config_sha256='2'*64,roles=roles)
        mapping={study_ref['path']:dict(study_id='study',helper_root=str(helper),helper_source_manifest=old_manifest,publication=old_pub,source_commit='1'*40),
            config_ref['path']:{k:v for k,v in config.items() if not k.startswith('_')},old_manifest['path']:dict(files={
                name:entry.file_ref(helper/name) for name in ('study_runner.py','memory_observer.py')}),
            old_pub['path']:dict(accepted=True,branch='search-e1',source_commit='1'*40,remote_readback_sha='1'*40,source_manifest_sha256='c'*64),
            config['gates']['scoped_cpu_processes_path']:registry}
        identities={10:driver,11:observer}
        gates=types.SimpleNamespace(bound=lambda ref:mapping[ref['path']],sha=lambda path:entry.file_ref(path)['sha256'] if Path(path).exists() else '3'*64,
            inventory=lambda:([dict(uuid='u4',actual_free_mib=24000)],[]),process=lambda pid:identities[pid])
        return config,gates,repair_ref,repair,study_ref,argv,registry

    def test_scoped_overlay_preserves_external_release_API_and_original_observer(self):
        with tempfile.TemporaryDirectory() as d,patch.object(recovery_common,'bind',return_value=None):
            cfg,gates,ref,repair,study,argv,registry=self.scoped_fixture(d)
            value=entry.scoped_roles(cfg,gates,ref,repair,study,argv)
            self.assertTrue(value['accepted']);self.assertFalse(value['observer_optional'])
            self.assertEqual(next(r['identity'] for r in value['roles'] if r['role']=='study_driver')['argv'],argv)
            self.assertEqual(value['identities'][1]['run_id'],'old-observer')
            self.assertEqual(value['receipt']['path'],cfg['gates']['scoped_cpu_processes_path'])
            self.assertIn('direct_gpu_inventory',value);self.assertIn('checked_unix',value)

    def test_scoped_overlay_requires_new_source_and_real_argv_not_old_claim(self):
        for kind in ('old_argv','missing_overlay','foreign_gpu','wrong_publication'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d,patch.object(recovery_common,'bind',return_value=None):
                cfg,gates,ref,repair,study,argv,registry=self.scoped_fixture(d);role=registry['roles'][0]
                if kind=='old_argv':
                    role['identity']['argv']=['/python',str(Path(d)/'old/study_runner.py'),'--study-manifest',study['path']]
                    role['argv_sha256']=hashlib.sha256(json.dumps(role['identity']['argv'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                elif kind=='missing_overlay':role.pop('execution_overlay')
                elif kind=='foreign_gpu':gates.inventory=lambda:([],[dict(pid=10)])
                else:role['execution_overlay']['publication']={'path':'/foreign/pub','sha256':'0'*64}
                with self.assertRaises(ValueError):entry.scoped_roles(cfg,gates,ref,repair,study,argv)

    def test_scoped_driver_only_allowed_during_observer_bootstrap(self):
        with tempfile.TemporaryDirectory() as d,patch.object(recovery_common,'bind',return_value=None):
            cfg,gates,ref,repair,study,argv,registry=self.scoped_fixture(d);registry['roles']=registry['roles'][:1]
            self.assertTrue(entry.scoped_roles(cfg,gates,ref,repair,study,argv,observer_optional=True)['accepted'])
            with self.assertRaises(ValueError):entry.scoped_roles(cfg,gates,ref,repair,study,argv)

    def failure_fixture(self,directory):
        root=Path(directory).resolve();output=root/'runs';output.mkdir();copies=root/'preserved';copies.mkdir()
        def put(path,value):path.write_text(json.dumps(value)+'\n');return entry.file_ref(path)
        config_ref=dict(path=str(root/'config.json'),sha256='a'*64)
        config=dict(variant='G_POS',run_id='group',paths=dict(output_root=str(output)),common_initial_adapter_manifest={'sha256':'init'},
            evaluation=dict(common_zero_receipt_path=str(root/'zero.json')))
        configs={config_ref['path']:(config,config_ref)}
        for variant in ('G_NEG','H_POS','H_NEG'):
            configs[str(root/variant)]=(dict(variant=variant,paths=dict(output_root=str(root/variant/'runs'))),{})
        owner=dict(pid=123,uid=1001,starttick=456,run_id='group');preserved=[]
        for index in range(11):
            name='status.json' if index==0 else 'study-status.json' if index==1 else 'evidence%d.json'%index
            value=dict(phase='FAILED_OWNED_RELEASE_CONFIRMED',owner=owner,known_children=[]) if index==0 else {'evidence':index}
            original=put(output/name,value);copy_path=copies/name;copy_path.write_bytes(Path(original['path']).read_bytes())
            preserved.append(dict(original=original,preserved=entry.file_ref(copy_path)))
        failure=dict(accepted=True,run_id='group',config_sha256='a'*64,olddriver_actual_pane_wait_returncode=1,
            oldcontroller_actual_parent_wait_returncode=1,oldknown_owner_children_absent=12,checkpoints=0,actual_adam_updates=0,
            reuse_accepted_zero_and_sampling=True,preserved=preserved)
        artifacts={name:put(root/(name+'.jsonl'),[]) for name in ('rollouts','pairs','questions')}
        outer=dict(run_id='group',config_sha256='a'*64,outer_index=0,cumulative_optimizer_updates_before=0,
            previous_complete_checkpoint=None,explicit_fresh=True,outer_old_adapter=config['common_initial_adapter_manifest'],P=327,**artifacts)
        sampling=dict(accepted=True,run_id='group',stats=dict(P=327),rollouts=artifacts['rollouts'],pairs=artifacts['pairs'],
            train_em=dict(accepted=True,unique_slots=2560,questions=512,post_training_em=False,overall=dict(denominator=2560,strict_correct=1280)))
        repair=dict(first_failure=put(root/'failure.json',failure),first_outer=put(root/'outer.json',outer),
            sampling_complete=put(root/'sampling.json',sampling),common_zero=put(root/'zero.json',{'accepted':True}))
        def absent(pid):raise FileNotFoundError(pid)
        gates=types.SimpleNamespace(process=absent,same_process=lambda a,b:a==b)
        return config,configs,repair,gates,failure

    def test_first_failure_reuses_original_2560_P327_and_preserves_nonzero_exits(self):
        with tempfile.TemporaryDirectory() as d:
            _,configs,repair,gates,_=self.failure_fixture(d)
            result=entry.first_recovery_gate(repair,configs,gates)
            self.assertTrue(result['no_new_sampling_or_optimizer_reset']);self.assertEqual(result['old_wait_returncodes_preserved'],[1,1])

    def test_first_failure_rejects_new_Adam_checkpoint_or_live_previous_owner(self):
        for change in ('adam','checkpoint','live_owner','changed_evidence'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,configs,repair,gates,failure=self.failure_fixture(d)
                if change=='adam':
                    failure['actual_adam_updates']=1;Path(repair['first_failure']['path']).write_text(json.dumps(failure)+'\n');repair['first_failure']=entry.file_ref(repair['first_failure']['path'])
                elif change=='checkpoint':
                    path=Path(cfg['paths']['output_root'])/'checkpoints'/'new';path.mkdir(parents=True);(path/'manifest.json').write_text('{}\n')
                elif change=='live_owner':gates.process=lambda pid:dict(pid=123,uid=1001,starttick=456,run_id='group')
                else:Path(failure['preserved'][0]['preserved']['path']).write_text('{"changed":true}\n')
                with self.assertRaises(ValueError):entry.first_recovery_gate(repair,configs,gates)


if __name__=='__main__':unittest.main()
