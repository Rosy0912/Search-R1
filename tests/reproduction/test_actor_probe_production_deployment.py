"""Tiny file/metadata fixtures; no GPU, model, remote action or native claim."""
import contextlib
import ast
import copy
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.a800_common import deploy_actor_probes as controller
from reproduction.a800_common import actor_probe_production_review as review
from reproduction.a800_common import prepare_actor_probe_production as prepare
from reproduction.a800_common.predecessor_admission import EXPECTED_PREDECESSOR
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import experiment_fields
from search_g1.actor_probe_recovery import identity,IDENTITY_KEYS


def documents():
    binding=dict(EXPECTED_PREDECESSOR,full_acceptance_sha256='a'*64,cleanup_sha256='b'*64)
    rows=[dict(target=name,source='/absent-local-QA/'+name,bytes=19,sha256='c'*64)
        for name in sorted(prepare.TARGETS)]
    for row in rows:
        if row['target']=='data/manifest.json':row['sha256']=prepare.SHARED_SHA
        if row['target']=='preparation/official-model-manifest-original.json':row['sha256']=prepare.OFFICIAL_SHA
    parent=dict(role='fresh_a800_common_deployment_intent',host='baai-a800',
        base=binding['base'],source_commit=binding['source_commit'],inputs=rows)
    stage=dict(role='a800_published_source_staging_only',source_commit='d'*40,
        base='/data/shaoshitong/Search-G1-actor-probes-source-'+'e'*32,
        source_sha256={'code.py':'f'*64},exact_published_source_verified=True,git_tree_clean=True,
        production_admission=False,services_started=False,trainer_started=False)
    return parent,stage,binding


def make_intent():
    parent,stage,binding=documents()
    with patch.object(controller.deploy,'published_source',return_value=stage['source_sha256']), \
         patch.object(controller.subprocess,'check_output',side_effect=[stage['source_commit']+'\n','',stage['source_commit']+'\trefs/heads/actor-probes-ten\n']):
        return controller.create_intent(parent,stage,binding,Path('/absent-fixture-code'),stage['source_commit'])


def fresh(root):
    base=Path(root).resolve()
    for name in ('logs','artifacts','checkpoints'):(base/name).mkdir()
    run={key:'fixture' for key in IDENTITY_KEYS}
    run.update(run_id='1'*32,execution_id='2'*32,source_commit='3'*40,source_sha256={'code.py':'4'*64},
        reward_schedule='g1_through100_then_binary',**experiment_fields('g1_through100_then_binary'),
        completed_updates=0,initial_probe_version=None,train_log=str(base/'logs/training.log'))
    return base,run


def write(path,value):path.write_text(json.dumps(value))


class ProductionDeploymentTests(unittest.TestCase):
    def test_metadata_intent_never_requires_local_QA_or_inherits_training_state(self):
        intent=make_intent()
        self.assertEqual(len(intent['inputs']),8)
        self.assertTrue(all('source' not in row for row in intent['inputs']))
        self.assertNotIn('initial_probes_source_base',intent)
        parent,stage,binding=documents()
        raw=(json.dumps(stage,sort_keys=True,allow_nan=False)+'\n').encode()
        self.assertEqual(intent['source_staging_receipt_sha256'],hashlib.sha256(raw).hexdigest())

    def test_changed_reward_parent_input_route_or_stage_hash_rejected(self):
        original=make_intent()
        for defect in ('reward','parent','route','local_source','stage_hash','duplicate','world'):
            with self.subTest(defect=defect):
                d=copy.deepcopy(original)
                if defect=='reward':d['reward_schedule']='g1_through50_then_binary'
                elif defect=='parent':d['predecessor_completion']['run_id']='0'*32
                elif defect=='route':d['inputs'][0]['remote_source']='/another-run/file'
                elif defect=='local_source':d['inputs'][0]['source']='/local/QA'
                elif defect=='stage_hash':d['source_staging_receipt_sha256']='bad'
                elif defect=='duplicate':d['inputs'][1]=d['inputs'][0]
                else:d['training_world_size']=8
                with self.assertRaises((AssertionError,ValueError)):prepare.validate_intent(d)

    def test_remote_copy_is_exclusive_unlinked_and_rechecks_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root).resolve();src=root/'source';dst=root/'target';src.write_bytes(b'tiny scalar fixture')
            expected=dict(bytes=src.stat().st_size,sha256=digest(src))
            prepare.copy_owned_input(src,dst,expected)
            self.assertEqual(dst.read_bytes(),src.read_bytes());self.assertNotEqual(src.stat().st_ino,dst.stat().st_ino)
            with self.assertRaises(AssertionError):prepare.copy_owned_input(src,dst,expected)
            dst.unlink();src.write_bytes(b'changed source fixture')
            with self.assertRaises(AssertionError):prepare.copy_owned_input(src,dst,expected)
            self.assertFalse(dst.exists())
            link=root/'link';link.symlink_to(src)
            with self.assertRaises(AssertionError):prepare.copy_owned_input(link,dst,dict(bytes=src.stat().st_size,sha256=digest(src)))

    def test_prepare_timeout_claim_prevents_duplicate_even_without_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            handle=Path(root)/'handle.json';write(handle,make_intent())
            with contextlib.redirect_stdout(io.StringIO()),patch.object(controller.deploy,'ssh_command',return_value=['fixture-only']), \
                 patch.object(controller.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture-only',1000)) as ssh:
                self.assertEqual(controller.perform(handle,'prepare'),1)
                with self.assertRaises(FileExistsError):controller.perform(handle,'prepare')
                self.assertEqual(ssh.call_count,1)
            claim=json.loads(handle.with_name('handle.prepare-once.json').read_text());self.assertTrue(claim['no_automatic_retry'])

    def test_missing_review_never_reaches_actual_config_read_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=fresh(root);write(base/review.PREPARED,{})
            with patch.object(review,'actual_preparation') as actual:
                with self.assertRaises((AssertionError,FileNotFoundError)):review.verify(base,run)
                actual.assert_not_called()

    def test_claimed_pipeline_entry_accepts_receipt_publication_race_but_requires_claim(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=fresh(root);review.unlaunched(base,run)
            with self.assertRaises((AssertionError,FileNotFoundError)):review.unlaunched(base,run,pipeline_entry=True)
            write(base/'logs/prepared-launch-claim.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
                runtime_source_commit=run['source_commit'],no_automatic_retry=True))
            (base/'logs/deployment-pipeline.log').write_text('')
            review.unlaunched(base,run,pipeline_entry=True)
            with self.assertRaises(AssertionError):review.unlaunched(base,run)
            write(base/'logs/training-launch-claim.json',{})
            with self.assertRaises(AssertionError):review.unlaunched(base,run,pipeline_entry=True)

    def test_actual_review_binds_preparation_source_and_identity_and_cannot_claim_native(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=fresh(root);sha='a'*64;source=review.document_sha256(run['source_sha256'])
            prepared=dict(identity=identity(run),source_map_sha256=source)
            accepted=dict(role='independent_actual_actor_probe_production_deployment_review',accepted=True,
                identity=identity(run),prepared_receipt_sha256=sha,source_map_sha256=source,
                independent_initialization_without_parent_context=True,actual_prepared_deployment_reviewed=True,
                actual_source_reviewed=True,reviewer_task='/root/independent_fixture',review_summary='Static fixture only.',
                native_algorithm_accepted=False)
            review.verify_documents(accepted,prepared,run,sha)
            for key,value in [('prepared_receipt_sha256','b'*64),('source_map_sha256','c'*64),
                ('native_algorithm_accepted',True),('actual_source_reviewed',False),('review_summary','')]:
                bad=dict(accepted,**{key:value})
                with self.assertRaises(AssertionError):review.verify_documents(bad,prepared,run,sha)

    def test_disk_receipt_change_after_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=fresh(root);actual={'fixture':True}
            disk=base/'logs/checkpoint-disk-admission.json';write(disk,dict(run_id=run['run_id'],accepted=True,required_free_bytes=0))
            prepared=dict(preparation=actual,disk_admission_sha256=digest(disk),required_free_bytes=0)
            write(base/review.PREPARED,prepared);write(base/review.REVIEW,{})
            with patch.object(review,'verify_documents'),patch.object(review,'actual_preparation',return_value=actual):
                review.verify(base,run)
                write(disk,dict(run_id=run['run_id'],accepted=True,required_free_bytes=1))
                with self.assertRaises(AssertionError):review.verify(base,run)

    def test_fresh_entry_rejects_absent_review_before_lock_or_native_allocation(self):
        from reproduction.a800_common import run_pipeline as pipeline
        with tempfile.TemporaryDirectory() as root,contextlib.ExitStack() as stack:
            base,run=fresh(root);run.update(execution_backend='ssh_local',slurm_job_id=None,execution_host=pipeline.socket.gethostname())
            stack.enter_context(patch.object(pipeline,'validate_base'));stack.enter_context(patch.object(pipeline,'assert_formal_run'))
            stack.enter_context(patch.object(pipeline,'bound_ray_tmp_dir',return_value='/fixture-private-ray'))
            stack.enter_context(patch.object(pipeline,'supported_execution_role',return_value='production'))
            stack.enter_context(patch.dict(os.environ,RAY_TMPDIR='/fixture-private-ray',SEARCH_G1_TRAINING_WORLD_SIZE='4',CUDA_VISIBLE_DEVICES='0,1,2,3'))
            allocation=stack.enter_context(patch('reproduction.common8.native_allocation.admit'))
            with self.assertRaises((FileNotFoundError,AssertionError)):pipeline.launch(base,run)
            allocation.assert_not_called();self.assertFalse((base/'gpu-job.lock').exists())

    def test_remote_launch_checks_review_then_leaves_repo_before_existing_process_scan(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=fresh(root);repo=base/'repo';module=repo/controller.MODULE
            module.parent.mkdir(parents=True);module.write_text('# tiny scalar entry fixture\n')
            intent=make_intent();intent['source_sha256'][controller.MODULE]=digest(module)
            run.update(source_commit=intent['source_commit'],source_sha256=intent['source_sha256'])
            payload=dict(intent=intent,action='launch',fixture_launcher=1)
            entry=Path(controller.__file__);old_cwd=Path.cwd();marker=base/'launcher-fixture.txt'
            # No real launcher/native work occurs: the stub asserts the exact
            # cwd condition required by the existing /proc duplicate scan.
            script="import os,json,sys\np=json.load(sys.stdin)\nassert os.getcwd()==p['intent']['base']\nwith open(os.path.join(os.getcwd(),'launcher-fixture.txt'),'x') as f:f.write('checked')"
            intent['base']=str(base)
            try:
                with patch.object(controller,'__file__',str(module)),patch.object(controller,'validate_intent',return_value=intent), \
                     patch.object(review,'read',side_effect=[(run,{}),(intent,{})]),patch.object(review,'verify') as verify, \
                     patch.object(controller,'launch_payload',return_value={'fixture_launcher':1}), \
                     patch.object(controller.deploy,'REMOTE_LAUNCH_PREPARED',script):
                    controller.remote_action(payload);verify.assert_called_once_with(base,run)
            finally:os.chdir(old_cwd)
            self.assertEqual(marker.read_text(),'checked');self.assertEqual(Path(controller.__file__),entry)

    def test_legacy_constructor_and_mutations_reject_actor_variant_before_local_QA_or_SSH(self):
        with self.assertRaises(AssertionError):controller.deploy.create_intent(SimpleNamespace(reward_schedule='g1_through100_then_binary'))
        with tempfile.TemporaryDirectory() as root:
            handle=Path(root)/'handle.json';write(handle,make_intent())
            for action in ('execute','launch_prepared','retry_unclaimed'):
                with self.subTest(action=action),patch.object(controller.deploy,'ssh_command') as ssh:
                    args=SimpleNamespace(handle=handle,execute=False,launch_prepared=False,retry_unclaimed=False)
                    setattr(args,action,True)
                    with self.assertRaises(AssertionError):controller.deploy.perform(args)
                    ssh.assert_not_called()
            self.assertEqual([p.name for p in Path(root).iterdir()],['handle.json'])

    def test_shared_popen_gate_rejects_missing_review_in_exact_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=fresh(root)
            with patch.object(review.sys,'prefix','/data/search-g1/env'),patch.object(review,'actual_preparation') as actual:
                with self.assertRaises((AssertionError,FileNotFoundError)):review.verify_before_popen(base,run)
                actual.assert_not_called()

    def test_actual_embedded_receipt_writer_publishes_complete_exclusive_bytes(self):
        tree=ast.parse(controller.deploy.REMOTE_LAUNCH_PREPARED)
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='write_once')
        ns=dict(os=os,json=json,time=time)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-shared-launcher-writer>','exec'),ns)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'deployment-launch-receipt.json';value={'actual_fixture':True,'pid':101}
            original=os.link
            def at_publication(src,dst,**kwargs):
                self.assertFalse(path.exists())
                self.assertEqual(json.loads(Path(src).read_text()),value)
                return original(src,dst,**kwargs)
            with patch.object(os,'link',side_effect=at_publication):ns['write_once'](path,value)
            self.assertEqual(json.loads(path.read_text()),value)
            with self.assertRaises(FileExistsError):ns['write_once'](path,{'overwrite':True})
            self.assertEqual(json.loads(path.read_text()),value)
            self.assertEqual(list(Path(root).iterdir()),[path])


if __name__=='__main__':unittest.main()
