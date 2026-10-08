"""Tiny local code tar/metadata/submit mocks; no Hydra/Slurm/assets/GPU."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
C=load('fresh200_dispatch_common');S=load('submit_tree_fresh200_once')

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(C,'ROOT',self.root);self.patch.start();self.modelpatch=patch.object(C,'MODEL',self.root/'instruct300/assets/qwen2.5-3b-instruct-aa8e725');self.modelpatch.start()
        (self.root/'instruct300/candidates').mkdir(parents=True);(self.root/'instruct300/transfers').mkdir()
        self.external=self.root/'tools/external.py';self.external.parent.mkdir();self.external.write_text('# synthetic external code\n')
        names=('supervisor.py','launch.sbatch','CONFIGURATION_OVERRIDES.json','calibration_contract.py',
            'source/repo/search_r1/baselines/sampling_consistency.py','source/repo/verl/workers/fsdp_workers.py',
            'source/repo/verl/trainer/ppo/ray_trainer.py','source/repo/verl/trainer/config/ppo_tree_fresh200.yaml')
        self.contents={name:b'# synthetic source only\n' for name in names}
        self.manifest=dict(stage=C.STAGE,training_protocol=C.PROTOCOL,code_commit=C.COMMIT,
            scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',official_model_path=str(C.MODEL),
            model_identity=dict(sha256=C.MODEL_ID,files={}),common_data_identity=C.DATA_ID,
            production_thresholds_accepted=False,actual_3b_training_accepted=False,long_training_ready=False,
            original_sampler_actor_gate_accepted=False,trial_thresholds=dict(target_old_policy_kl=.01,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),
            probability_calibration=dict(environment=C.ENVIRONMENT,source_interface_reviewed=True),
            files={name:C.digest(data) for name,data in self.contents.items()},external_files={str(self.external):C.file_sha(self.external)})
    def tearDown(self):self.modelpatch.stop();self.patch.stop();self.tmp.cleanup()
    def archive(self,extra=(),manifest=None):
        contents=dict(self.contents);raw=json.dumps(manifest or self.manifest).encode();contents['manifest.json']=raw
        output=self.root/'instruct300/transfers/package.tar'
        with tarfile.open(output,'w') as tar:
            for name,data in contents.items():
                member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
            for member,data in extra:tar.addfile(member,io.BytesIO(data) if data is not None else None)
        return output,C.file_sha(output),C.digest(raw)
    def extract(self,extra=(),manifest=None,max_bytes=2**20):
        archive,asha,msha=self.archive(extra,manifest);target=self.root/'instruct300/candidates/new'
        return C.safe_extract(archive,asha,target,msha,max_bytes),target
    def mocked_submit(self,stdout='777',rc=0,stderr='',exception=None,queue=''):
        calls=[]
        def run(command,**kwargs):
            calls.append(command)
            if command[0]=='squeue':return N(returncode=0,stdout=queue,stderr='')
            if exception:raise exception
            return N(returncode=rc,stdout=stdout,stderr=stderr)
        with patch.object(C,'protected_processes',lambda:None):
            try:result=S.submit_once(C,self.root/'instruct300/candidates/new','a'*64,'b'*64,{},run=run);error=None
            except Exception as exc:result=None;error=exc
        return result,error,calls
    def journal(self):return json.loads((self.root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-aaaaaaaaaaaa.json').read_text())

    def test_exact_regular_code_archive_extracts_and_existing_target_refuses(self):
        manifest,target=self.extract();self.assertEqual(manifest,self.manifest)
        archive,asha,msha=self.archive()
        with self.assertRaisesRegex(ValueError,'no-clobber'):C.safe_extract(archive,asha,target,msha,2**20)

    def test_absolute_traversal_symlink_duplicate_and_asset_reject_before_extract(self):
        cases=[]
        for name in ('/tmp/outside.py','../outside.py','data.parquet','supervisor.py'):
            item=tarfile.TarInfo(name);item.size=1;cases.append((item,b'x'))
        item=tarfile.TarInfo('link.py');item.type=tarfile.SYMTYPE;item.linkname='/tmp/outside';cases.append((item,None))
        for extra in cases:
            with self.subTest(member=extra[0].name),self.assertRaises(ValueError):self.extract(extra=(extra,))
            self.assertFalse((self.root/'instruct300/candidates/new').exists())

    def test_missing_full_inventory_unbound_member_and_size_bound_refuse(self):
        bad=copy.deepcopy(self.manifest);bad['files'].pop('launch.sbatch')
        with self.assertRaisesRegex(ValueError,'incomplete'):self.extract(manifest=bad)
        item=tarfile.TarInfo('unsigned.py');item.size=1
        with self.assertRaisesRegex(ValueError,'inventory'):self.extract(extra=((item,b'x'),))
        with self.assertRaisesRegex(ValueError,'size'):self.extract(max_bytes=1)

    def test_archive_and_json_sha_tamper_refuse(self):
        archive,asha,msha=self.archive()
        with self.assertRaisesRegex(ValueError,'Archive SHA'):C.safe_extract(archive,'0'*64,self.root/'instruct300/candidates/new',msha,2**20)
        with self.assertRaisesRegex(ValueError,'manifest SHA'):C.safe_extract(archive,asha,self.root/'instruct300/candidates/new','0'*64,2**20)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):C.read_bound(self.external,'0'*64)

    def test_old_stage_parent_or_training_claim_refuse(self):
        for mutate in (lambda m:m.update(stage='calibration110'),lambda m:m.update(parent_manifest={}),
                       lambda m:m.update(production_thresholds_accepted=True),lambda m:m.update(code_commit='a'*40)):
            bad=copy.deepcopy(self.manifest);mutate(bad)
            with self.subTest(bad=bad),self.assertRaises(ValueError):C.validate_manifest(bad)

    def test_same_node_paths_and_recovery_environment_failclosed(self):
        with self.assertRaisesRegex(ValueError,'remain'):C.local_path('/tmp/unrelated.py')
        with patch.object(C.socket,'gethostname',lambda:'h100-16'),self.assertRaisesRegex(ValueError,'h100-9'):C.assert_host()
        with patch.object(C.socket,'gethostname',lambda:'h100-9'),patch.dict(os.environ,{'TREE_RECOVERY_MANIFEST':'old'}),self.assertRaisesRegex(ValueError,'recovery'):C.assert_host()

    def test_full_original_pid_start_synthetic_proc_and_replacement_refusal(self):
        proc=self.root/'proc'
        for pid,start in C.PROTECTED.items():
            path=proc/pid;path.mkdir(parents=True);(path/'stat').write_text(pid+' (synthetic pid) '+' '.join(['S']+['0']*18+[start]))
        C.protected_processes(proc)
        changed=proc/next(iter(C.PROTECTED))/'stat';changed.write_text(changed.read_text().replace('1502352181','0'))
        with self.assertRaisesRegex(ValueError,'PID/start'):C.protected_processes(proc)

    def test_queue_empty_or_other_task_accepted_own_or_malformed_refused(self):
        C.queue_clear('');C.queue_clear('42|other-user|RUNNING|h100-9\n')
        for value in ('42|tree-fresh200-mu-zero|PENDING|h100-9','not a parsable queue line','42||RUNNING|h100-9'):
            with self.subTest(value=value),self.assertRaises(ValueError):C.queue_clear(value)

    def review_fixture(self):
        def save(name,value):
            path=self.root/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=C.file_sha(path))
        interface=dict(status='FRESH_CALIBRATION_MODE_AND_DIRECT_RANK_SCHEDULER_SOURCE_CPU_IMPLEMENTED',source_files={
            name:self.manifest['files']['source/repo/'+name] for name in
            ('search_r1/baselines/sampling_consistency.py','verl/workers/fsdp_workers.py')})
        audit=dict(git_head=C.COMMIT,source_files={name:dict(sha256=value,matches_git_head=True) for name,value in interface['source_files'].items()},
                   wrapper_files={'supervisor.py':dict(sha256=self.manifest['files']['supervisor.py'],matches=True)})
        audit_entry=save('audit.json',audit)
        context=dict(reviewer_context=dict(fork_turns='none',inherited_parent_conversation=False),
            source=dict(audited_clean_head=C.COMMIT,initial_clean=True),source_interface_reviewed=True,
            stages={'preparation':dict(accepted=True),C.STAGE:dict(source_review_completed=True,source_review_accepted=False)},
            execution_scope=dict(same_h100_9_cpu_only_preparation_allowed=True),artifacts={'BYTE_AUDIT.json':audit_entry['sha256']})
        return dict(source_interface_receipt=save('interface.json',interface),context_free_review=save('context.json',context),context_byte_audit=audit_entry)

    def test_actual_blocked_review_schema_allows_only_cpu_prep_preserving_false(self):
        inputs=self.review_fixture();result=C.validate_reviews(inputs,self.manifest)
        self.assertIs(result['source_review_accepted'],False)
        with self.assertRaisesRegex(ValueError,'remains blocked'):C.validate_reviews(inputs,self.manifest,for_dispatch=True)

    def test_byte_audit_or_review_missing_sha_and_changed_source_refuse(self):
        inputs=self.review_fixture();broken=copy.deepcopy(inputs);broken['context_byte_audit']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):C.validate_reviews(broken,self.manifest)
        broken=copy.deepcopy(self.manifest);broken['files']['source/repo/verl/workers/fsdp_workers.py']='0'*64
        with self.assertRaisesRegex(ValueError,'Source-interface bytes'):C.validate_reviews(inputs,broken)

    def test_submit_mock_success_records_once_and_duplicate_cannot_call_sbatch(self):
        result,error,calls=self.mocked_submit(stdout='777;cluster')
        self.assertIsNone(error);self.assertEqual(result['job_id'],777);self.assertEqual(self.journal()['status'],'SUBMITTED')
        result,error,calls=self.mocked_submit();self.assertIsInstance(error,FileExistsError)
        self.assertTrue(all(command[0]!='sbatch' for command in calls))

    def test_unknown_timeout_retains_journal_and_blocks_retrial(self):
        result,error,calls=self.mocked_submit(exception=TimeoutError('synthetic unknown'))
        self.assertIsInstance(error,ValueError);self.assertEqual(self.journal()['status'],'SUBMISSION_UNKNOWN_DO_NOT_RETRY')
        result,error,calls=self.mocked_submit();self.assertIsInstance(error,FileExistsError)
        self.assertTrue(all(command[0]!='sbatch' for command in calls))

    def test_invalid_success_stdout_is_unknown(self):
        result,error,calls=self.mocked_submit(stdout='777;cluster\nwarning')
        self.assertIsInstance(error,ValueError);self.assertEqual(self.journal()['status'],'SUBMISSION_UNKNOWN_DO_NOT_RETRY')

    def test_nonzero_submit_and_active_queue_never_automatic_retry(self):
        result,error,calls=self.mocked_submit(rc=1,stderr='synthetic denied')
        self.assertIsInstance(error,ValueError);self.assertEqual(self.journal()['status'],'SUBMISSION_FAILED_NO_AUTOMATIC_RETRY')
        result,error,calls=self.mocked_submit(queue='99|tree-probability-probe-unknown|RUNNING|h100-9')
        self.assertIsInstance(error,ValueError);self.assertTrue(all(command[0]!='sbatch' for command in calls))

if __name__=='__main__':unittest.main(verbosity=2)
