"""CPU metadata/control fixtures only. No Torch, models, network or real processes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

import entry
from evaluation_retry import retry_hold, validate_actual_evaluation_entry
from failure_gate import validate_observed_failure, prove_old_absent, untouched_future_arms, CP150_SHA, FAILURE_SCHEMA
from preserve import archive_tree, file_ref
from recovery_common import CONFIG_SHAS


class Hold(RuntimeError):
    pass


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return file_ref(path)


def failed_metadata():
    identity=dict(pid=44,starttick=900,uid=1001,run_id='same-run')
    return dict(schema=FAILURE_SCHEMA,variant='G_POS',run_id='same-run',completed_outers=15,
        actual_optimizer_updates=150,cumulative_pair_exposure=4568,raw_checkpoint=dict(path='/cp150',sha256=CP150_SHA),
        native_endpoint_accepted=False,missing_original_wave1_actual_wait_evidence=True,
        original_evaluator_actual_parent_wait_returncode=1,original_controller_actual_supervisor_wait_returncode=1,
        original_driver_actual_same_pane_exit_returncode=1,
        same_server_socket_and_pane=dict(pane=dict(pane_dead='1',pane_dead_status='1',pane_id='%45',session_id='$45')),
        original_native_evaluator_parent_terminal=[dict(value=dict(config_sha256=CONFIG_SHAS['G_POS'],run_id='same-run',
            phase='evaluation-running',child_exit_evidence=[dict(actual_wait_returncode=1,identity=identity)]))],
        original_group_controller_terminal=[dict(actual_wait_returncode=1)],
        old_controller_and_supervisor_processes=[dict(expected=dict(pid=90,starttick=300,uid=1001)),dict(expected=dict(pid=91,starttick=301,uid=1001))],
        eight_old_worker_processes_actual_absent=[dict(pid=100+i,starttick=1000+i,uid=1001,run_id='same-run',state='ABSENT',wait_returncode='UNKNOWN') for i in range(8)],
        filtered_failure_stack=['memory_observer_admission','gates.Hold: Receipt hash changed: /owned/memory-live.json'])


class RetryTests(unittest.TestCase):
    def test_hold_retries_exact_args_until_original_success(self):
        calls=[];pauses=[];reports=[];marker=object()
        def original(*args,**kw):
            calls.append((args,kw))
            if len(calls)<3:raise Hold('unchanged original gate')
            return marker
        wrapped=retry_hold(original,Hold,pause=pauses.append,report=reports.append)
        self.assertIs(wrapped('cfg','owner',released=True),marker)
        self.assertEqual(calls,[(('cfg','owner'),dict(released=True))]*3)
        self.assertEqual(pauses,[1,1]);self.assertEqual(len(reports),2)

    def test_valueerror_propagates_unchanged(self):
        error=ValueError('original failure');pauses=[]
        def original():raise error
        with self.assertRaises(ValueError) as caught:retry_hold(original,Hold,pause=pauses.append)()
        self.assertIs(caught.exception,error);self.assertEqual(pauses,[])

    def test_other_runtimeerror_is_not_hold(self):
        error=RuntimeError('not gates.Hold')
        with self.assertRaises(RuntimeError) as caught:retry_hold(lambda:(_ for _ in ()).throw(error),Hold)()
        self.assertIs(caught.exception,error)

    def test_no_retry_or_model_work_after_success(self):
        events=[];marker={'accepted':True}
        self.assertIs(retry_hold(lambda:marker,Hold,pause=events.append)(),marker);self.assertEqual(events,[])


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'original';self.root.mkdir()
        self.configs={};self.arguments={};self.repair=dict(path='/repair',sha256='a'*64)
        for variant,sha in CONFIG_SHAS.items():
            output=Path(self.tmp.name)/variant;cf=dict(path=str(output/'config.json'),sha256=sha)
            cfg=dict(variant=variant,run_id='run-'+variant,paths=dict(python='/python',helper_root=str(self.root),output_root=str(output)))
            self.configs[cf['path']]=(cfg,cf)
            for number in (15,25):
                cp=save(output/('cp-%d.json'%number),dict(accepted=True,run_id=cfg['run_id'],config_sha256=sha,
                    optimizer_updates=150 if number==15 else 250,cursor=dict(outer_index=number-1,next_pair_cursor=308,usable_pair_count=308)))
                save(output/'outers'/('outer-%05d'%(number-1))/'training/outer-complete.json',dict(accepted=True,run_id=cfg['run_id'],outer_index=number-1,
                    optimizer_updates=150 if number==15 else 250,checkpoint=cp))
                directory=output/'evaluations'/('outer-%02d'%number)
                args=['--config',cf['path'],'--config-sha256',sha,'--completed-outer',str(number),'--checkpoint',cp['path'],
                      '--checkpoint-sha256',cp['sha256'],'--output-dir',str(directory/'execution'),'--receipt',str(directory/'child-complete.json')]
                self.arguments[variant,number]=args

    def tearDown(self):self.tmp.cleanup()

    def dispatch(self,args):return entry.rewrite_command(['/python',str(self.root/'evaluation.py')]+args,self.configs,self.root,'/new/entry.py',self.repair)

    def test_actual_four_variants_fifteen_twentyfive_dispatch(self):
        for variant in CONFIG_SHAS:
            for number in (15,25):
                changed,role=self.dispatch(self.arguments[variant,number]);self.assertEqual(role,'evaluation')
                self.assertIn(str(self.root/'evaluation.py'),changed)
                self.assertEqual(changed[changed.index('--original-entry')+1],str(self.root/'evaluation.py'))
                self.assertEqual(changed[changed.index('--')+1:],self.arguments[variant,number])

    def mutation(self,index,value):
        args=list(self.arguments['G_POS',15]);args[index]=value
        with self.assertRaises((ValueError,SystemExit)):self.dispatch(args)

    def test_wrong_config_rejected(self):self.mutation(1,'/foreign-config')
    def test_wrong_config_sha_rejected(self):self.mutation(3,'b'*64)
    def test_zero_boundary_rejected(self):self.mutation(5,'0')
    def test_wrong_canonical_output_rejected(self):self.mutation(11,'/foreign-output')
    def test_wrong_receipt_rejected(self):self.mutation(13,'/foreign-receipt')
    def test_duplicate_config_option_rejected(self):
        args=self.arguments['G_POS',15]+['--config','/foreign']
        with self.assertRaises(ValueError):self.dispatch(args)
    def test_wrong_checkpoint_hash_rejected(self):self.mutation(9,'b'*64)
    def test_wrong_checkpoint_outer_rejected(self):
        args=self.arguments['G_POS',15];path=Path(args[7]);value=json.loads(path.read_bytes());value['cursor']['outer_index']=13
        cp=save(path,value);args=list(args);args[9]=cp['sha256']
        with self.assertRaises(ValueError):self.dispatch(args)
    def test_incomplete_checkpoint_cursor_rejected(self):
        args=self.arguments['G_POS',15];path=Path(args[7]);value=json.loads(path.read_bytes());value['cursor']['next_pair_cursor']=307
        cp=save(path,value);args=list(args);args[9]=cp['sha256']
        with self.assertRaises(ValueError):self.dispatch(args)
    def test_checkpoint_update_count_must_match_actual_end(self):
        args=self.arguments['G_POS',15];path=Path(args[7]);value=json.loads(path.read_bytes());value['optimizer_updates']=149
        cp=save(path,value);args=list(args);args[9]=cp['sha256']
        with self.assertRaises(ValueError):self.dispatch(args)
    def test_foreign_or_worker_command_not_redirected(self):
        command=['/python',str(self.root/'evaluation_worker.py'),'--request','/exact']
        self.assertEqual(entry.rewrite_command(command,self.configs,self.root,'/new/entry.py',self.repair),(command,None))
    def test_original_eval_entry_is_real_argv_and_actual_source(self):
        path=self.root/'evaluation.py';path.write_text('# original evaluator\n');proof=file_ref(path)
        manifest=dict(files={'evaluation.py':proof});argv=['/python','/new/entry.py','--mode','evaluation','--original-entry',str(path)]
        self.assertEqual(validate_actual_evaluation_entry(str(path),self.root,argv,manifest,file_ref),proof)
    def test_sysargv_alone_or_missing_real_token_rejected(self):
        path=self.root/'evaluation.py'
        with self.assertRaises(ValueError):validate_actual_evaluation_entry(str(path),self.root,['/new/entry.py'],{},file_ref)
    def test_mutated_original_evaluator_source_rejected(self):
        path=self.root/'evaluation.py';path.write_text('# original\n');proof=file_ref(path);path.write_text('# changed\n')
        with self.assertRaises(ValueError):validate_actual_evaluation_entry(str(path),self.root,['/new/entry.py','--original-entry',str(path)],dict(files={'evaluation.py':proof}),file_ref)


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(run_id='same-run',owner=dict(uid=1001));self.ref=dict(path='/config',sha256=CONFIG_SHAS['G_POS']);self.meta=failed_metadata()
    def test_exact_fifteen_failure_is_only_metadata_acceptance(self):
        self.assertIs(validate_observed_failure(self.meta,self.cfg,self.ref),self.meta)
    def reject(self,key,value):
        self.meta[key]=value
        with self.assertRaises(ValueError):validate_observed_failure(self.meta,self.cfg,self.ref)
    def test_old_zero_guard_cannot_be_used(self):self.reject('actual_optimizer_updates',0)
    def test_wrong_previous_outer_count(self):self.reject('completed_outers',14)
    def test_wrong_pair_exposure(self):self.reject('cumulative_pair_exposure',4567)
    def test_unknown_parent_wait_rejected(self):self.reject('original_evaluator_actual_parent_wait_returncode','UNKNOWN')
    def test_success_parent_wait_cannot_replace_failure(self):self.reject('original_evaluator_actual_parent_wait_returncode',0)
    def test_unknown_controller_wait_rejected(self):self.reject('original_controller_actual_supervisor_wait_returncode','UNKNOWN')
    def test_unknown_supervisor_wait_rejected(self):self.reject('original_driver_actual_same_pane_exit_returncode','UNKNOWN')
    def test_wrong_cp_rejected(self):
        self.meta['raw_checkpoint']['sha256']='b'*64
        with self.assertRaises(ValueError):validate_observed_failure(self.meta,self.cfg,self.ref)
    def test_partial_failure_must_not_become_formal_acceptance(self):self.reject('native_endpoint_accepted',True)
    def test_missing_wave_wait_must_not_be_rewritten(self):self.reject('missing_original_wave1_actual_wait_evidence',False)
    def test_wrong_pane_status_rejected(self):
        self.meta['same_server_socket_and_pane']['pane']['pane_dead_status']='0'
        with self.assertRaises(ValueError):validate_observed_failure(self.meta,self.cfg,self.ref)
    def test_old_unknown_worker_waits_cannot_be_zeroed(self):
        self.meta['eight_old_worker_processes_actual_absent'][0]['wait_returncode']=0
        with self.assertRaises(ValueError):validate_observed_failure(self.meta,self.cfg,self.ref)
    def test_still_running_worker_is_not_absent(self):
        self.meta['eight_old_worker_processes_actual_absent'][0]['state']='LIVE'
        with self.assertRaises(ValueError):validate_observed_failure(self.meta,self.cfg,self.ref)
    def test_wrong_failure_cause_rejected(self):self.reject('filtered_failure_stack',['OOM'])
    def test_fresh_old_process_check_refuses_live_identity(self):
        gates=types.SimpleNamespace(process=lambda pid:dict(pid=pid),same_process=lambda a,b:True)
        with self.assertRaises(ValueError):prove_old_absent(self.meta,gates)
    def test_fresh_actual_absence_never_implies_wait0(self):
        def missing(pid):raise FileNotFoundError()
        identities=prove_old_absent(self.meta,types.SimpleNamespace(process=missing))
        self.assertEqual(len(identities),11);self.assertEqual(self.meta['eight_old_worker_processes_actual_absent'][4]['wait_returncode'],'UNKNOWN')


class PreservationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def setup_tree(self):
        source=self.root/'failed';(source/'execution/worker-0').mkdir(parents=True)
        (source/'execution/worker-0/trace.json').write_text('{"old_absolute_ref":"/failed-original-only"}\n')
        return source,self.root/'preserved/failed15',self.root/'receipt.json'
    def test_all_failed_bytes_preserved_and_mapped_without_rewrite(self):
        source,dest,out=self.setup_tree();before=(source/'execution/worker-0/trace.json').read_bytes()
        ref=archive_tree(source,dest,out,dict(failure_metadata={'sha256':'a'*64}));value=json.loads(Path(ref['path']).read_bytes())
        self.assertFalse(source.exists());self.assertEqual((dest/'execution/worker-0/trace.json').read_bytes(),before)
        self.assertEqual(value['files'][0]['original']['sha256'],value['files'][0]['preserved']['sha256'])
        self.assertFalse(value['formal_endpoint_accepted']);self.assertTrue(value['old_absolute_references_must_resolve_with_this_mapping'])
    def test_existing_destination_is_never_overwritten(self):
        source,dest,out=self.setup_tree();dest.mkdir(parents=True)
        with self.assertRaises(ValueError):archive_tree(source,dest,out,{})
        self.assertTrue(source.exists())
    def test_symlink_failed_asset_rejected(self):
        source,dest,out=self.setup_tree();(source/'link').symlink_to(source/'execution/worker-0/trace.json')
        with self.assertRaises(ValueError):archive_tree(source,dest,out,{})
    def test_existing_manifest_rejected(self):
        source,dest,out=self.setup_tree();out.write_text('old proof')
        with self.assertRaises(ValueError):archive_tree(source,dest,out,{})
        self.assertTrue(source.exists())
    def test_real_root_symlink_rejected_before_resolve(self):
        source,dest,out=self.setup_tree();link=self.root/'link';link.symlink_to(source,target_is_directory=True)
        with self.assertRaises(ValueError):archive_tree(link,dest,out,{})
        self.assertTrue(source.exists())
    def test_mutable_status_metadata_saved_exactly_before_resume(self):
        source,dest,out=self.setup_tree();status=save(self.root/'old/status.json',dict(phase='FAILED',actual_wait=1))
        original=Path(status['path']).read_bytes();ref=archive_tree(source,dest,out,{},[status])
        value=json.loads(Path(ref['path']).read_bytes());row=value['metadata_files'][0]
        self.assertEqual(Path(row['preserved']['path']).read_bytes(),original)
        Path(status['path']).write_text('{"phase":"NEW_RESUME"}')
        self.assertEqual(Path(row['preserved']['path']).read_bytes(),original)
        self.assertEqual(value['metadata_file_count'],1)
    def test_mutated_status_ref_refused_without_rename(self):
        source,dest,out=self.setup_tree();status=save(self.root/'old/status.json',dict(phase='FAILED'))
        Path(status['path']).write_text('{"phase":"changed"}')
        with self.assertRaises(ValueError):archive_tree(source,dest,out,{},[status])
        self.assertTrue(source.exists())
    def test_duplicate_metadata_reference_refused(self):
        source,dest,out=self.setup_tree();status=save(self.root/'old/status.json',dict(phase='FAILED'))
        with self.assertRaises(ValueError):archive_tree(source,dest,out,{},[status,status])
        self.assertTrue(source.exists())


class FutureArmsTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def configs(self):
        return {v:(dict(variant=v,paths=dict(output_root=str(self.root/v/'runs'))),{}) for v in CONFIG_SHAS}
    def test_future_config_receipts_alone_do_not_mean_started(self):
        for v in CONFIG_SHAS:(self.root/v/'receipts').mkdir(parents=True)
        untouched_future_arms(self.configs())
    def test_any_future_arm_output_means_another_recovery_scope(self):
        for v in ('G_NEG','H_POS','H_NEG'):
            with self.subTest(variant=v):
                path=self.root/v/'runs';path.mkdir(parents=True)
                with self.assertRaises(ValueError):untouched_future_arms(self.configs())
                path.rmdir()


if __name__=='__main__':unittest.main()
