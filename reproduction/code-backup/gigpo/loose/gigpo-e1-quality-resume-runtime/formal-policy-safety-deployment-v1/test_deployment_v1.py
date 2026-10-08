"""CPU closure/install/once controls only. No target, Ray, GPU or real Slurm.

Cached original proof bytes feed record translation fixtures; no fabricated
production/native GPU acceptance is tested as real. Submission uses explicit
process fixtures and temporary local O_EXCL files, never sbatch execution.
"""
import copy
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import deployment_contract_v1 as c
import prepare_candidate_v1 as p
import install_package_v1 as i
import freeze_ready_v1 as f
import submit_once_v1 as s
import verify_physical_v1 as v


def payload_sha(data):return hashlib.sha256(data).hexdigest()


class BoundBytesCPU(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
    def test_original_JSON_bytes_SHA_actual_read(self):
        x=self.root/'x.json';x.write_text('{"v":1}');self.assertEqual(c.read_bound(c.record(x)),{'v':1})
    def test_same_size_changed_SHA_refuses(self):
        x=self.root/'x.json';x.write_text('{"v":1}');r=c.record(x);x.write_text('{"v":2}')
        with self.assertRaises(ValueError):c.read_bound(r)
    def test_duplicate_closure_refuses(self):
        r=dict(path=str(self.root/'x'),bytes=0,sha256='a'*64)
        with self.assertRaises(ValueError):c.closure([r,r])
    def test_invalid_path_bool_size_hash_or_extra_keys_refuse(self):
        r=dict(path=str(self.root/'x'),bytes=0,sha256='a'*64)
        for bad in (dict(r,path='../x'),dict(r,path='/tmp/../x'),dict(r,bytes=True),dict(r,sha256='zz'),dict(r,extra='x')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):c.valid_record(bad)
    def test_symlink_original_refuses(self):
        x=self.root/'x.json';x.write_text('{}');link=self.root/'link';link.symlink_to(x)
        with self.assertRaises(ValueError):c.read_bound(c.record(link))
    def test_atomic_claim_has_one_winner(self):
        x=self.root/'claim'
        def claim(n):
            try:c.write_once(x,dict(n=n));return True
            except FileExistsError:return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:r=list(pool.map(claim,(1,2)))
        self.assertEqual(r.count(True),1);self.assertIn(json.loads(x.read_text())['n'],(1,2))
    def test_claim_symlink_not_followed(self):
        x=self.root/'existing';x.write_text('original');claim=self.root/'claim';claim.symlink_to(x)
        with self.assertRaises(FileExistsError):c.write_once(claim,{})
        self.assertEqual(x.read_text(),'original')


class ActualInheritedClosureCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original,_=p.draft()
        # Resolve the four exact cached original proofs by SHA. All original
        # source/model/data/env paths stay as record metadata, not asset reads.
        names=('FORMAL-PROTECTION-DRAFT-IMPLEMENTATION-v4.json','FORMAL-PROTECTION-DRAFT-INDEPENDENT-REVIEW-v4.json',
            'CALIBRATION-CANDIDATE-52BEFE9-v9.json','policy-safety-calibration-52befe9-reviewed-binding-v9.json')
        cls.local={c.sha(p.OUT/name):p.OUT/name for name in names}
    def setUp(self):
        self.x=copy.deepcopy(self.original)
        def cached(item):
            c.valid_record(item);path=self.local[item['sha256']]
            self.assertEqual(path.stat().st_size,item['bytes']);self.assertEqual(c.sha(path),item['sha256'])
            return json.loads(path.read_text())
        self.cached=cached;self.patch=patch.object(c,'read_bound',side_effect=cached);self.patch.start();self.addCleanup(self.patch.stop)
    def test_template_has_exact_inherited42327_new29_control7(self):
        c.validate_candidate(self.x);self.assertEqual(len(self.x['files']),42363)
        self.assertEqual(self.x['inherited_file_count'],42327);self.assertEqual(len(self.x['evidence']),378)
        self.assertIsNone(self.x['production_artifact']);self.assertIsNone(self.x['probability_identity']['tolerances'])
    def test_changed_inherited_model_SHA_refuses(self):
        path=next(r['path'] for r in self.x['files'] if r['path'].endswith('/model-00001-of-00002.safetensors'))
        next(r for r in self.x['files'] if r['path']==path)['sha256']='a'*64
        with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_removed_original_common_manifest_refuses(self):
        self.x['files']=[r for r in self.x['files'] if not r['path'].endswith('/common-data/manifest.json')]
        with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_changed_frozen_formal_runtime_refuses(self):
        self.x['runtime_helpers'][0]['sha256']='a'*64
        with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_duplicate_runtime_alias_refuses(self):
        self.x['runtime_helpers'][-1]=self.x['runtime_helpers'][0]
        with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_old_calibration_or_native_helper_directory_refuses(self):
        for target in (str(c.ROOT/'tools/policy-safety-calibration-52befe9-v9'),str(c.ROOT/'tools/policy-safety-native-52befe9-v2')):
            self.x['helpers']=target
            with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_changed_guard_or_native_acceptance_refuses(self):
        self.x['guard_identity']['resource_guard']['sha256']='a'*64
        with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_other_source_node_budget_or_extra_AB_refuses(self):
        for key,value in (('code_commit','x'*40),('node','h100-11'),('formal_budget',dict(c.BUDGET,stop300=500)),('extra_AB_diagnostic_Adam_required',20)):
            x=copy.deepcopy(self.original);x[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_candidate(x)
    def test_gate30_moved_or_new_training_claim_refuses(self):
        for key,value in (('first_formal_hook_acceptance_stage','AFTER300'),('training_submitted',True),('runtime_full_SHA_before_GPU_required',False)):
            x=copy.deepcopy(self.original);x[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_candidate(x)
    def test_omitted_or_changed_old_evidence_refuses(self):
        self.x['evidence'].pop(0)
        with self.assertRaises(ValueError):c.validate_candidate(self.x)
    def test_proposal_current_null_never_imports_production_or_writes_READY(self):
        with self.assertRaisesRegex(ValueError,'still pending'):c.proposal(self.x)
    def test_physical_current_null_stops_before_target_import_asset_or_subprocess(self):
        item=dict(path='/tmp/candidate',bytes=1,sha256='a'*64)
        def values(r):return self.x if r==item else self.cached(r)
        with patch.object(c,'read_bound',side_effect=values),patch.object(v,'read_bound',side_effect=values),patch.object(v,'metadata',side_effect=AssertionError('No target read')),patch.object(v.subprocess,'check_output',side_effect=AssertionError('No subprocess')):
            with self.assertRaisesRegex(ValueError,'still pending'):v.verify(item,'/tmp/never-written')
    def test_freeze_current_null_cannot_write_any_READY(self):
        item=dict(path='/tmp/candidate',bytes=1,sha256='a'*64)
        with patch.object(f,'read_bound',return_value=self.x),patch.object(f,'write_once',side_effect=AssertionError('No READY write')):
            with self.assertRaisesRegex(ValueError,'still pending'):f.freeze(item,item,item)
    def test_same_source_static_READY_is_not_actual_calibration_acceptance(self):
        self.assertEqual(self.x['inherited_launch_READY_not_actual_calibration_acceptance']['sha256'],c.PARENT_READY_SHA)
        self.x['preparation_only']=False;self.x['actual_v9_calibration_acceptance']=self.x['inherited_launch_READY_not_actual_calibration_acceptance']
        with self.assertRaisesRegex(ValueError,'still pending'):c.proposal(self.x)


class ArchiveCPU(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
    def tar(self,name='runtime/x.py',kind=tarfile.REGTYPE,duplicate=False,size=None):
        path=self.root/'x.tar.gz';data=b'x';info=tarfile.TarInfo(name);info.type=kind;info.size=1 if size is None else size
        info.linkname='../../escape';members=[info,copy.copy(info)] if duplicate else [info]
        with tarfile.open(path,'w:gz') as stream:
            for item in members:stream.addfile(item,io.BytesIO(data) if item.isfile() and item.size==1 else None)
        return path,[dict(member=name,bytes=1,sha256=payload_sha(data))]
    def test_signed_regular_file_only(self):
        path,rows=self.tar();self.assertEqual(i.read_archive(path,c.sha(path),rows),{'runtime/x.py':b'x'})
    def test_absolute_traversal_normalization_and_unknown_path_refuse(self):
        for name in ('/runtime/x.py','runtime/../x.py','runtime//x.py','./runtime/x.py','runtime/x.py/'):
            with self.subTest(name=name):
                path,rows=self.tar(name)
                with self.assertRaises(ValueError):i.read_archive(path,c.sha(path),rows)
    def test_symlink_hardlink_directory_fifo_refuse(self):
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE,tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                path,rows=self.tar(kind=kind)
                with self.assertRaises(ValueError):i.read_archive(path,c.sha(path),rows)
    def test_duplicate_tar_members_refuse(self):
        path,rows=self.tar(duplicate=True)
        with self.assertRaises(ValueError):i.read_archive(path,c.sha(path),rows)
    def test_package_SHA_or_member_SHA_mismatch_refuse(self):
        path,rows=self.tar()
        with self.assertRaises(ValueError):i.read_archive(path,'a'*64,rows)
        rows[0]['sha256']='a'*64
        with self.assertRaises(ValueError):i.read_archive(path,c.sha(path),rows)
    def test_missing_or_unknown_inventory_refuses(self):
        path,rows=self.tar();rows.append(dict(member='runtime/y.py',bytes=1,sha256='a'*64))
        with self.assertRaises(ValueError):i.read_archive(path,c.sha(path),rows)
    def test_write_new_never_overwrites_or_follows_link(self):
        x=self.root/'x';x.write_bytes(b'old')
        with self.assertRaises(FileExistsError):i.write_new(x,b'new')
        link=self.root/'link';link.symlink_to(x)
        with self.assertRaises(FileExistsError):i.write_new(link,b'new')
        self.assertEqual(x.read_bytes(),b'old')


class InstallPreflightCPU(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
        (self.root/'tools').mkdir();(self.root/'logs').mkdir()
        values=dict(ROOT=self.root,HELPERS=self.root/'tools/new-runtime',CONTROLS=self.root/'tools/new-controls',INSTALL_INTENT=self.root/'logs/install-intent',INSTALL_RECEIPT=self.root/'logs/install-receipt')
        self.patch=patch.multiple(i,**values);self.patch.start();self.addCleanup(self.patch.stop)
        self.host=patch.object(i.socket,'gethostname',return_value='h100-8');self.host.start();self.addCleanup(self.host.stop)
        self.plan=dict(required_existing_evidence=[],members=[])
    def test_new_directories_not_created_by_preflight(self):
        self.assertEqual(i.preflight(self.plan),[]);self.assertFalse(i.HELPERS.exists());self.assertFalse(i.CONTROLS.exists())
    def test_existing_new_runtime_or_controls_refuse(self):
        i.HELPERS.mkdir()
        with self.assertRaises(ValueError):i.preflight(self.plan)
    def test_existing_or_symlink_install_claim_refuse(self):
        i.INSTALL_INTENT.symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):i.preflight(self.plan)
    def test_parent_symlink_or_unreadable_not_directory_refuse(self):
        (self.root/'tools').rmdir();(self.root/'tools').symlink_to(self.root/'logs')
        with self.assertRaises(ValueError):i.preflight(self.plan)
    def test_other_host_refuses_before_any_new_directory(self):
        with patch.object(i.socket,'gethostname',return_value='h100-11'),self.assertRaises(ValueError):i.preflight(self.plan)
    def test_exact_same_existing_proof_allowed_but_changed_or_link_refuse(self):
        proof=self.root/'logs/x';proof.write_bytes(b'old');r=dict(c.record(proof),reuse_same_bytes_allowed=True)
        self.plan['members']=[r];self.assertEqual(i.preflight(self.plan),[str(proof)])
        proof.write_bytes(b'new')
        with self.assertRaises(ValueError):i.preflight(self.plan)
    def test_runtime_existing_file_is_not_reusable(self):
        proof=self.root/'logs/x';proof.write_bytes(b'x');self.plan['members']=[dict(c.record(proof),reuse_same_bytes_allowed=False)]
        with self.assertRaises(ValueError):i.preflight(self.plan)
    def test_required_existing_evidence_changed_refuses(self):
        x=self.root/'logs/old';x.write_bytes(b'x');self.plan['required_existing_evidence']=[c.record(x)];x.write_bytes(b'y')
        with self.assertRaises(ValueError):i.preflight(self.plan)


class SingleSubmissionCPU(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
        self.binding=self.root/'ready.json';self.binding.write_text('{}');self.b=c.record(self.binding)
        self.r=self.root/'review.json';self.r.write_text(json.dumps(dict(status='PASS',scope=c.SCOPE,binding_sha256=self.b['sha256'],submit_helper_sha256=c.sha(Path(s.__file__)),extra_AB_diagnostic_Adam_required=0)))
        self.review=c.record(self.r);self.value=dict(files=[],evidence=[],repo=str(c.REPO),helpers=str(c.HELPERS))
        patcher=patch.multiple(s,READY=self.binding,INTENT=self.root/'intent',RECEIPT=self.root/'receipt');patcher.start();self.addCleanup(patcher.stop)
        self.host=patch.object(s.socket,'gethostname',return_value='h100-8');self.host.start();self.addCleanup(self.host.stop)
        self.admit=patch.object(s,'validate_ready',return_value=self.value);self.admit.start();self.addCleanup(self.admit.stop)
        def read(r):return self.value if r==self.b else c.read_bound(r)
        self.reader=patch.object(s,'read_bound',side_effect=read);self.reader.start();self.addCleanup(self.reader.stop)
        self.query=patch.object(s.subprocess,'check_output',return_value='');self.query_mock=self.query.start();self.addCleanup(self.query.stop)
        self.command=patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(stdout='1234\n',stderr='',returncode=0));self.spawn=self.command.start();self.addCleanup(self.command.stop)
    def test_single_positive_process_fixture_then_no_repeat(self):
        self.assertEqual(s.main(self.b,self.review)['job_id'],'1234');self.assertEqual(self.spawn.call_count,1)
        with self.assertRaises(ValueError):s.main(self.b,self.review)
        self.assertEqual(self.spawn.call_count,1)
    def test_busy_queue_or_gpu_creates_no_claim(self):
        for value in ('123 another RUNNING','999 otherGPU 12'):
            with patch.object(s.subprocess,'check_output',return_value=value),self.assertRaises(ValueError):s.main(self.b,self.review)
            self.assertFalse(s.INTENT.exists());self.spawn.assert_not_called()
    def test_unknown_resource_query_creates_no_claim(self):
        with patch.object(s.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,'nvidia-smi')),self.assertRaises(subprocess.CalledProcessError):s.main(self.b,self.review)
        self.assertFalse(s.INTENT.exists());self.spawn.assert_not_called()
    def test_other_host_or_old_binding_refuses_before_submit(self):
        with patch.object(s.socket,'gethostname',return_value='h100-11'),self.assertRaises(ValueError):s.main(self.b,self.review)
        with self.assertRaises(ValueError):s.main(dict(self.b,path=str(self.root/'old')),self.review)
        self.spawn.assert_not_called()
    def test_missing_actual_READY_refuses_before_resource_query(self):
        self.admit.stop()
        with self.assertRaises((ValueError,KeyError)):s.main(self.b,self.review)
        self.query_mock.assert_not_called();self.spawn.assert_not_called();self.assertFalse(s.INTENT.exists())
    def test_existing_symlink_or_uncertain_claim_blocks(self):
        s.INTENT.symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):s.main(self.b,self.review)
        self.spawn.assert_not_called()
    def test_ambiguous_or_nonpositive_job_preserves_claim_and_receipt(self):
        self.spawn.return_value=types.SimpleNamespace(stdout='0\n',stderr='',returncode=0)
        with self.assertRaises(ValueError):s.main(self.b,self.review)
        self.assertTrue(s.INTENT.exists());self.assertEqual(json.loads(s.RECEIPT.read_text())['status'],'UNCERTAIN_OR_FAILED_NO_RETRY')
        with self.assertRaises(ValueError):s.main(self.b,self.review)
        self.assertEqual(self.spawn.call_count,1)
    def test_submission_timeout_preserves_unreusable_claim(self):
        self.spawn.side_effect=subprocess.TimeoutExpired('sbatch',60)
        with self.assertRaises(subprocess.TimeoutExpired):s.main(self.b,self.review)
        self.assertTrue(s.INTENT.exists());self.assertEqual(json.loads(s.RECEIPT.read_text())['status'],'UNCERTAIN_OR_FAILED_NO_RETRY')
    def test_wrong_review_or_extra_AB_refuses(self):
        x=json.loads(self.r.read_text());x['extra_AB_diagnostic_Adam_required']=20;self.r.write_text(json.dumps(x));r=c.record(self.r)
        with self.assertRaises(ValueError):s.main(self.b,r)
        self.spawn.assert_not_called();self.assertFalse(s.INTENT.exists())
    def test_two_submitters_only_one_process_fixture(self):
        def call(_):
            try:return s.main(self.b,self.review)['status']
            except (ValueError,FileExistsError):return 'REFUSED'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(call,(1,2)))
        self.assertEqual(result.count('SUBMITTED_EXACTLY_ONCE'),1);self.assertEqual(self.spawn.call_count,1)


if __name__=='__main__':unittest.main()

class RealDraftPackageOrchestrationCPU(unittest.TestCase):
    """Entire package plan path uses actual frozen29+new7 bytes, only CPU review fixture."""
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
        self.candidate,self.proofs=p.draft();self.target=self.root/c.CANDIDATE.name
        c.write_once(self.target,self.candidate)
        manifest=json.loads((p.OUT/p.FORMAL_PROOFS[0]).read_text())
        sources={'runtime/'+Path(r['path']).name:Path(r['path']) for r in manifest['files']}
        sources.update({'control/'+name:p.HERE/name for name in p.CONTROL_NAMES})
        sources.update({'proof/'+name:local for name,local in self.proofs.items()})
        sources['candidate/'+c.CANDIDATE.name]=self.target
        self.archive=self.root/'package.tar.gz';p.archive(self.archive,sources)
        members=[]
        for name,path in sorted(sources.items()):
            category,filename=name.split('/',1)
            target=(c.HELPERS/filename if category=='runtime' else c.CONTROLS/filename if category=='control' else c.ROOT/'logs'/filename)
            members.append(dict(c.record(path),path=str(target),member=name,reuse_same_bytes_allowed=category in ('proof','candidate')))
        self.plan=dict(status='DRAFT_IMMUTABLE_INSTALL_PLAN_NOT_PRODUCTION_READY',scope=c.SCOPE,source_commit=c.SOURCE,ready=False,submitted=False,
            runtime_manifest_sha256=c.RUNTIME_MANIFEST_SHA,runtime_static_review_sha256=c.RUNTIME_REVIEW_SHA,
            package=c.record(self.archive),candidate=c.record(self.target),candidate_target=str(c.CANDIDATE),members=members,
            required_existing_evidence=[r for r in self.candidate['evidence'] if r['path'] not in {item['path'] for item in members}])
        self.plan_path=self.root/'plan.json';c.write_once(self.plan_path,self.plan);self.plan_record=c.record(self.plan_path)
        self.review_path=self.root/'CPU-fixture-review.json'
        self.review=dict(status='PASS',scope='GIGPO_FORMAL_IMMUTABLE_INSTALL_ONLY_NOT_PRODUCTION',plan_sha256=self.plan_record['sha256'],
            package_sha256=c.sha(self.archive),installer_sha256=c.sha(Path(i.__file__)),actual_formal_GPU_accepted=False)
        c.write_once(self.review_path,self.review);self.review_record=c.record(self.review_path)
    def invoke(self):return i.installation_plan(self.archive,self.plan_record,self.review_record)
    def test_actual_runtime29_and_new_control7_package_can_be_inspected_not_READY(self):
        plan,payloads=self.invoke();self.assertEqual(len(payloads),44);self.assertFalse(plan['ready'])
        self.assertIsNone(json.loads(payloads['candidate/'+c.CANDIDATE.name])['production_artifact'])
    def test_unsigned_install_or_GPU_scope_review_refuses(self):
        self.review['actual_formal_GPU_accepted']=True;self.review_path.unlink();c.write_once(self.review_path,self.review)
        with self.assertRaises(ValueError):i.installation_plan(self.archive,self.plan_record,c.record(self.review_path))
    def test_package_inventory_destination_changed_refuses(self):
        self.plan['members'][0]['path']=str(c.ROOT/'repo-policy-safety-52befe9/overwrite.py')
        self.plan_path.unlink();c.write_once(self.plan_path,self.plan);self.plan_record=c.record(self.plan_path)
        self.review['plan_sha256']=self.plan_record['sha256'];self.review_path.unlink();c.write_once(self.review_path,self.review)
        with self.assertRaises(ValueError):i.installation_plan(self.archive,self.plan_record,c.record(self.review_path))
    def test_installer_cannot_label_plan_production_READY(self):
        self.plan['ready']=True;self.plan_path.unlink();c.write_once(self.plan_path,self.plan)
        with self.assertRaises(ValueError):i.installation_plan(self.archive,c.record(self.plan_path),self.review_record)


class FreezeClosureOrchestrationCPU(unittest.TestCase):
    """Stub only mature upstream production acceptance; execute full freeze checks on CPU."""
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name).resolve()
        self.target=self.root/'READY-never-submitted.json'
        artifact=self.root/'CPU-upstream-fixture.json';artifact.write_text(json.dumps(dict(actual_proofs=dict(calibration=dict(sha256='a'*64)))))
        preview=self.root/'CPU-preview-fixture.json';preview.write_text(json.dumps(dict(resolved=dict(mode='production'))))
        self.candidate=dict(files=[dict(path='/example/only-metadata.py',bytes=1,sha256='b'*64)],evidence=[dict(path='/example/proof',bytes=2,sha256='c'*64)],
            production_artifact=c.record(artifact),expected_config_preview=c.record(preview),guard_identity={'scope':'CPU-fixture'},
            probability_identity=dict(pad_token_id=1,eos_token_id=2),native_acceptance=dict(review_sha256='d'*64),runtime_manifest=dict(sha256=c.RUNTIME_MANIFEST_SHA))
        x=self.root/'candidate.json';c.write_once(x,self.candidate);self.cr=c.record(x)
        self.physical=dict(status='PHYSICAL_FORMAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',candidate_sha256=self.cr['sha256'],code_commit=c.SOURCE,
            cuda_initialized=False,actual_formal_GPU_accepted=False,files=self.candidate['files'],evidence=self.candidate['evidence'],resolved_config=dict(mode='production'),
            config_preview_sha256=self.candidate['expected_config_preview']['sha256'],guard_identity=self.candidate['guard_identity'],pad_token_id=1,eos_token_id=2)
        self.ph=self.root/'physical.json';c.write_once(self.ph,self.physical);self.pr=c.record(self.ph)
        self.review=dict(status='PASS',scope=c.SCOPE,candidate_sha256=self.cr['sha256'],physical_sha256=self.pr['sha256'],freeze_helper_sha256=c.sha(Path(f.__file__)),source_commit=c.SOURCE,
            native_acceptance_sha256='d'*64,calibration_acceptance_sha256='a'*64,runtime_manifest_sha256=c.RUNTIME_MANIFEST_SHA,actual_formal_GPU_accepted=False,extra_AB_diagnostic_Adam_required=0)
        self.rv=self.root/'review.json';c.write_once(self.rv,self.review);self.rr=c.record(self.rv)
        self.ready=patch.object(f,'READY',self.target);self.ready.start();self.addCleanup(self.ready.stop)
        self.upstream=patch.object(f,'proposal',side_effect=lambda candidate:dict(candidate,status='REVIEWED_FOR_PRODUCTION'));self.upstream.start();self.addCleanup(self.upstream.stop)
        self.final=patch.object(f,'validate_ready',return_value=None);self.final.start();self.addCleanup(self.final.stop)
    def invoke(self):return f.freeze(self.cr,self.pr,self.rr,self.target)
    def test_full_freeze_orchestration_only_under_explicit_upstream_fixture(self):
        result=self.invoke();self.assertFalse(result['submitted']);self.assertFalse(result['actual_formal_GPU_accepted']);self.assertTrue(self.target.is_file())
    def test_late_record_same_path_different_SHA_refuses_without_output(self):
        self.physical['files'][0]['sha256']='e'*64
        self.ph.unlink();c.write_once(self.ph,self.physical);self.pr=c.record(self.ph)
        with self.assertRaises(ValueError):self.invoke()
        self.assertFalse(self.target.exists())
    def test_duplicate_physical_proof_refuses_without_output(self):
        self.physical['evidence']+=self.physical['evidence']
        self.ph.unlink();c.write_once(self.ph,self.physical);self.pr=c.record(self.ph)
        with self.assertRaises(ValueError):self.invoke()
        self.assertFalse(self.target.exists())
    def test_actual_PAD_EOS_or_config_difference_refuses(self):
        self.physical['eos_token_id']=3;self.ph.unlink();c.write_once(self.ph,self.physical);self.pr=c.record(self.ph)
        with self.assertRaises(ValueError):self.invoke()
        self.assertFalse(self.target.exists())
    def test_wrong_review_source_native_or_calibration_original_refuses(self):
        self.review['calibration_acceptance_sha256']='f'*64;self.rv.unlink();c.write_once(self.rv,self.review);self.rr=c.record(self.rv)
        with self.assertRaises(ValueError):self.invoke()
        self.assertFalse(self.target.exists())
