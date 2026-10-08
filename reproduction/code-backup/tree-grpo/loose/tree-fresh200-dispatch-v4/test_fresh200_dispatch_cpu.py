"""Tiny local code tar/metadata/submit mocks; no Hydra/Slurm/assets/GPU."""
import copy
import hashlib
import importlib.util
import io
import json
import os
import multiprocessing
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

def lock_probe(root, pipe):
    C.ROOT=Path(root)
    try:
        with S.task_submit_lock(C):pipe.send('LOCK_ACQUIRED')
    except ValueError as exc:pipe.send(str(exc))
    finally:pipe.close()

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(C,'ROOT',self.root);self.patch.start();self.modelpatch=patch.object(C,'MODEL',self.root/'instruct300/assets/qwen2.5-3b-instruct-aa8e725');self.modelpatch.start()
        (self.root/'instruct300/candidates').mkdir(parents=True);(self.root/'instruct300/transfers').mkdir()
        synthetic_external={}
        for name in C.EXTERNAL_RELATIVE_FILES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic external code '+name+'\n')
            synthetic_external[name]=C.file_sha(path)
        self.external=self.root/'tools/tree_shared_owned_runtime_fixed.py'
        self.closurepatch=patch.object(C,'EXTERNAL_RELATIVE_FILES',synthetic_external);self.closurepatch.start()
        names=tuple(C.WRAPPER_FILES)+tuple('source/repo/'+name for name in C.SOURCE_INTERFACE_FILES)
        self.contents={name:b'# synthetic source only\n' for name in names}
        # The true V7 hashes remain pinned in production. This explicit expected
        # mapping models toy wrapper bytes only; no real wrapper is executed.
        self.wrapperpatch=patch.object(C,'WRAPPER_HASHES',{name:C.digest(self.contents[name]) for name in C.WRAPPER_FILES})
        self.wrapperpatch.start()
        self.manifest=dict(stage=C.STAGE,training_protocol=C.PROTOCOL,code_commit=C.COMMIT,
            scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',official_model_path=str(C.MODEL),
            model_identity=dict(sha256=C.MODEL_ID,files={}),common_data_identity=C.DATA_ID,
            production_thresholds_accepted=False,actual_3b_training_accepted=False,long_training_ready=False,
            original_sampler_actor_gate_accepted=False,trial_thresholds=dict(target_old_policy_kl=.01,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),
            probability_calibration=dict(environment=C.ENVIRONMENT,source_interface_reviewed=True),
            wrapper_inventory_sha256=C.WRAPPER_INVENTORY_ID,
            wrapper_files={name:C.digest(self.contents[name]) for name in C.WRAPPER_FILES},
            files={name:C.digest(data) for name,data in self.contents.items()},external_files={str(self.root/name):value for name,value in synthetic_external.items()})
    def tearDown(self):self.wrapperpatch.stop();self.closurepatch.stop();self.modelpatch.stop();self.patch.stop();self.tmp.cleanup()
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
        with self.assertRaises(ValueError):self.extract(manifest=bad)
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
                       lambda m:m.update(production_thresholds_accepted=True),lambda m:m.update(code_commit='8a3bc24db05ee1fd67660e2f89088d6f56bc0875')):
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
        interface=dict(status='FRESH200_REPAIRED_SOURCE_INTERFACE_CPU_OWNER_VERIFIED',
            code_commit=C.COMMIT,source_clean=True,actual_gpu_executed=False,actual_remote_executed=False,training_release=False,source_files={
            name:self.manifest['files']['source/repo/'+name] for name in C.SOURCE_INTERFACE_FILES})
        audit=dict(git_head=C.COMMIT,source_files={name:dict(sha256=value,matches_git_head=True) for name,value in interface['source_files'].items()},
                   wrapper_files={name:dict(sha256=self.manifest['files'][name],matches=True) for name in C.WRAPPER_FILES})
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
        # Extend this existing contract case; reruns/subtests are not new test methods.
        interface_path=Path(inputs['source_interface_receipt']['path']);original=json.loads(interface_path.read_text())
        def write_interface(value):
            interface_path.write_text(json.dumps(value));inputs['source_interface_receipt']['sha256']=C.file_sha(interface_path)
        old=copy.deepcopy(original);old['status']='FRESH_CALIBRATION_MODE_AND_DIRECT_RANK_SCHEDULER_SOURCE_CPU_IMPLEMENTED'
        write_interface(old)
        with self.assertRaisesRegex(ValueError,'Wrong source-interface'):C.validate_reviews(inputs,self.manifest)
        for key,value in (('code_commit','1e06c6aa6f54eee5de8e0a29dd50efea76111c10'),('source_clean',False),('source_clean','true')):
            changed=copy.deepcopy(original);changed[key]=value;write_interface(changed)
            with self.subTest(source_commit_flag=key,value=value),self.assertRaisesRegex(ValueError,'final clean'):C.validate_reviews(inputs,self.manifest)
        for key in ('actual_gpu_executed','actual_remote_executed','training_release'):
            for value in (True,None,'false'):
                changed=copy.deepcopy(original);changed[key]=value;write_interface(changed)
                with self.subTest(flag=key,value=value),self.assertRaisesRegex(ValueError,'CPU-only'):C.validate_reviews(inputs,self.manifest)
            changed=copy.deepcopy(original);changed.pop(key);write_interface(changed)
            with self.subTest(missing=key),self.assertRaisesRegex(ValueError,'CPU-only'):C.validate_reviews(inputs,self.manifest)
        for name in sorted(C.SOURCE_INTERFACE_FILES):
            changed=copy.deepcopy(original);changed['source_files'].pop(name);write_interface(changed)
            with self.subTest(missing_source=name),self.assertRaisesRegex(ValueError,'all14'):C.validate_reviews(inputs,self.manifest)
            changed=copy.deepcopy(original);changed['source_files'][name]='0'*64;write_interface(changed)
            with self.subTest(changed_source=name),self.assertRaisesRegex(ValueError,'Source-interface bytes'):C.validate_reviews(inputs,self.manifest)
        write_interface(original)
        # Owner coverage never replaces the separately signed context-free audit.
        audit_path=Path(inputs['context_byte_audit']['path']);audit=json.loads(audit_path.read_text())
        audit['source_files'].pop('search_r1/baselines/gradient_norm.py')
        audit_path.write_text(json.dumps(audit));inputs['context_byte_audit']['sha256']=C.file_sha(audit_path)
        context_path=Path(inputs['context_free_review']['path']);context=json.loads(context_path.read_text())
        context['artifacts']['BYTE_AUDIT.json']=inputs['context_byte_audit']['sha256']
        context_path.write_text(json.dumps(context));inputs['context_free_review']['sha256']=C.file_sha(context_path)
        with self.assertRaisesRegex(ValueError,'all14'):C.validate_reviews(inputs,self.manifest)

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

    def test_real_index_and_jsonl_asset_names_in_signed_inventory_still_refuse(self):
        for name in ('source/repo/search_r1/baselines/rollout.py',
                     'source/repo/verl/workers/actor/dp_actor.py','source/repo/setup.py',
                     'source/repo/LICENSE','source/repo/verl/version/version'):
            self.assertEqual(str(C.relative_file(name)),name)
        for name in ('source/repo/e5_Flat.index', 'source/repo/wiki-18.jsonl',
                     'source/repo/wiki-18.jsonl.gz', 'source/repo/embeddings.npz',
                     'source/repo/reproduction/a800/old-audit.json',
                     'source/repo/reproduction/baselines/periodic/trajectories.json',
                     'source/repo/public/chart.json','source/repo/tests/fixture.json','source/repo/model.json'):
            with self.subTest(name=name):
                self.contents[name]=b'synthetic asset counterexample, never real data'
                self.manifest['files'][name]=C.digest(self.contents[name])
                with self.assertRaisesRegex(ValueError,'Code-only allowlist|Runtime source path closure'):self.extract()
                self.assertFalse((self.root/'instruct300/candidates/new').exists())
                del self.contents[name];del self.manifest['files'][name]

    def test_dummy_external_or_missing_actual_import_closure_refuses(self):
        for changed in ('only-dummy','missing-real','changed-sha'):
            m=copy.deepcopy(self.manifest)
            if changed=='only-dummy':m['external_files']={str(self.root/'tools/external.py'):'a'*64}
            elif changed=='missing-real':m['external_files'].pop(str(self.external))
            else:m['external_files'][str(self.external)]='a'*64
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'closure'):C.validate_manifest(m)

    def test_actual_two_process_whole_task_lock_refuses_concurrent_manifest(self):
        ctx=multiprocessing.get_context('spawn');receive,send=ctx.Pipe(duplex=False)
        with S.task_submit_lock(C):
            process=ctx.Process(target=lock_probe,args=(str(self.root),send));process.start();send.close()
            self.assertTrue(receive.poll(10));result=receive.recv();process.join(10)
            self.assertEqual(process.exitcode,0)
            self.assertIn('whole-task lock',result)
        receive.close()
        with S.task_submit_lock(C):pass

    def test_unknown_other_manifest_blocks_even_when_queue_is_empty(self):
        self.mocked_submit(exception=TimeoutError('unknown first intent'))
        calls=[]
        def run(command,**kwargs):calls.append(command);return N(returncode=0,stdout='',stderr='')
        with patch.object(C,'protected_processes',lambda:None),self.assertRaisesRegex(ValueError,'reconciliation'):
            S.submit_once(C,self.root/'instruct300/candidates/new','c'*64,'d'*64,{},run=run)
        self.assertTrue(all(cmd[0]!='sbatch' for cmd in calls))

    def test_different_manifest_needs_actual_prior_terminal_state_not_empty_queue(self):
        self.mocked_submit(stdout='777')
        for state in ('RUNNING','COMPLETED'):
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                if command[0]=='squeue':return N(returncode=0,stdout='',stderr='')
                if command[0]=='sacct':return N(returncode=0,stdout='777|'+state+'|0:0\n',stderr='')
                return N(returncode=0,stdout='778',stderr='')
            with patch.object(C,'protected_processes',lambda:None):
                if state=='RUNNING':
                    with self.assertRaisesRegex(ValueError,'not terminal'):
                        S.submit_once(C,self.root/'instruct300/candidates/new','c'*64,'d'*64,{},run=run)
                    self.assertTrue(all(cmd[0]!='sbatch' for cmd in calls))
                else:
                    result=S.submit_once(C,self.root/'instruct300/candidates/new','c'*64,'d'*64,{},run=run)
                    self.assertEqual(result['previous_terminal_evidence'][0]['scheduler_state'],'COMPLETED')

    def test_missing_or_unparseable_prior_terminal_is_unknown(self):
        self.mocked_submit(stdout='777')
        for answer in ('','777|COMPLETED|0:0|\n','wrong|COMPLETED|0:0\n'):
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                return N(returncode=0,stdout=answer if command[0]=='sacct' else '',stderr='')
            with patch.object(C,'protected_processes',lambda:None),self.assertRaises(ValueError):
                S.submit_once(C,self.root/'instruct300/candidates/new','c'*64,'d'*64,{},run=run)
            self.assertTrue(all(cmd[0]!='sbatch' for cmd in calls))

    def test_real_fifteen_item_audit_cannot_pass_old_nine_file_package(self):
        inputs=self.review_fixture()
        old=copy.deepcopy(self.manifest)
        for name in ('PLAN.md','SOURCE_CPU_RECEIPT.json','build_package_draft.py',
                     'render_dispatch.py','test_runtime_contract_cpu.py','tokenizer_raw_preflight.py'):
            old['files'].pop(name)
        with self.assertRaisesRegex(ValueError,'complete15'):C.validate_manifest(old)
        with self.assertRaises((KeyError,ValueError)):C.validate_reviews(inputs,old)
        changed=copy.deepcopy(self.manifest)
        # Repeating the approved inventory's label cannot approve altered bytes.
        changed['wrapper_files']['supervisor.py']='0'*64
        changed['files']['supervisor.py']='0'*64
        with self.assertRaisesRegex(ValueError,'complete15'):C.validate_manifest(changed)

if __name__=='__main__':unittest.main(verbosity=2)
