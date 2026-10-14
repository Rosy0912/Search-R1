"""CPU-only checks for the prospective control boundary; no SSH or model load."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from reproduction.a800_common import deploy_qprobe as control
from reproduction.a800_common import precompute_qprobe as features


def intent():
    identifier = '1' * 32
    return dict(role=control.ROLE, host='baai-a800', deployment_id=identifier,
        source_commit='2' * 40, source_sha256={'one.py':'3' * 64},
        baseline_intent_sha256='4' * 64, published_ref='refs/heads/codex/new-qprobe',
        base='/data/shaoshitong/Search-G1-a800-fresh200-' + identifier,
        source_base='/data/shaoshitong/Search-G1-qprobe-source-' + identifier,
        baseline_base=control.BASELINE, baseline_commit=control.BASELINE_COMMIT,
        initial_probes_source_base=control.INITIAL_PROBES, teacher_warmup_cache=control.TEACHER_CACHE,
        launch_services=False, target_updates=200)


class ProspectiveControlTests(unittest.TestCase):
    def test_embedded_remote_programs_compile(self):
        for name in ('REMOTE_STAGE','REMOTE_PREPARE','REMOTE_BIND','REMOTE_INSPECT','REMOTE_LAUNCH_GATE'):
            ast.parse(getattr(control, name), filename=name)

    def test_independent_review_cannot_approve_other_source_or_partial_contract(self):
        original = intent()
        review = dict(role='a800_qprobe_independent_code_review', approved=True, independent=True,
            reviewer='review-agent', implementation_actor='implementation-agent',
            source_commit=original['source_commit'], source_sha256=original['source_sha256'],
            checks={key:True for key in control.REVIEW_CHECKS}, reviewed_unix=time.time()-1)
        self.assertIs(control.validate_review(review, original), review)
        for patch in ({'source_commit':'5'*40},{'reviewer':'implementation-agent'},
                      {'checks':dict(review['checks'], frozen_teacher_5120_x_5=False)},
                      {'reviewed_unix':float('nan')}):
            changed = dict(review, **patch)
            with self.assertRaises(AssertionError):
                control.validate_review(changed, original)

    def test_feature_plan_never_uses_training_gpu_or_foreign_assets(self):
        plan = control.feature_plan(intent())
        self.assertFalse(plan['gpu_job_started'])
        self.assertEqual(plan['required_environment']['CUDA_VISIBLE_DEVICES'], '6')
        self.assertIn('reproduction.a800_common.precompute_qprobe', plan['argv'])
        self.assertTrue(plan['progress_path'].startswith(intent()['base']+'/'))
        with self.assertRaises(AssertionError):
            control.feature_plan(intent(), 0)

    def test_shared_teacher_cache_is_immutable_intent_and_rejects_controller_paths(self):
        first = intent();second = intent()
        second['deployment_id'] = '6'*32
        second['base'] = '/data/shaoshitong/Search-G1-a800-fresh200-' + second['deployment_id']
        second['source_base'] = '/data/shaoshitong/Search-G1-qprobe-source-' + second['deployment_id']
        self.assertEqual(control.validate_intent(first)['teacher_warmup_cache'],
                         control.validate_intent(second)['teacher_warmup_cache'])
        for path in ('/Users/shaoshitong/cache','/data/cache/../other','relative-cache'):
            with self.assertRaises(AssertionError):
                control.validate_teacher_cache(path)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ):
            namespace = {};exec(control.REMOTE_COMMON,namespace)
            cache = Path(directory).resolve()/'shared'/'teacher-v2'
            selection = dict(teacher_warmup_cache=str(cache))
            namespace['teacher_storage'](selection,create_parent=True)
            cache.mkdir();marker = cache/'identity.json';marker.write_text('accepted-existing-cache')
            namespace['teacher_storage'](selection,create_parent=True)
            self.assertEqual(marker.read_text(),'accepted-existing-cache')
            alias = Path(directory)/'cache-alias';alias.symlink_to(cache,target_is_directory=True)
            with self.assertRaises(AssertionError):
                namespace['teacher_storage'](dict(teacher_warmup_cache=str(alias)))

    def test_feature_cuda_uuid_must_equal_admitted_physical_uuid(self):
        admitted = 'GPU-00000000-0000-0000-0000-000000000005'
        other = 'GPU-00000000-0000-0000-0000-000000000006'
        with mock.patch.dict(os.environ), mock.patch.object(features.sys,'modules',{}):
            reader = mock.Mock(return_value=[admitted])
            self.assertEqual(features.bind_cuda_uuid(admitted,reader),[admitted])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],admitted)
            reader.assert_called_once_with(1)
            with self.assertRaisesRegex(AssertionError,'Actual CUDA UUID differs'):
                features.bind_cuda_uuid(admitted,lambda count:[other])

    def test_launch_pins_clean_source_and_rejects_cached_modified_deployer(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL,text=True).strip()
            git('init');git('config','user.name','test');git('config','user.email','test@example.invalid')
            original_repo = Path(control.__file__).resolve().parents[2]
            for relative in ('reproduction/a800_common/deploy.py','reproduction/a800_common/deploy_qprobe.py',
                             'verl/version/version','verl/single_controller/version/version'):
                target = repo/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((original_repo/relative).read_bytes())
            git('add','.');git('commit','-m','published fixture')
            commit = git('rev-parse','HEAD')
            selected = dict(source_commit=commit,source_sha256=control.published_source(repo,commit))
            with mock.patch.object(control.original_deploy,'__file__',str(repo/'reproduction/a800_common/deploy.py')):
                self.assertEqual(control.verified_launcher(repo,selected)[0],commit)
                with mock.patch.object(control.original_deploy,'REMOTE_LAUNCH_PREPARED',control.REMOTE_LAUNCH_PREPARED+'\n# dirty cached import'):
                    with self.assertRaises(AssertionError):
                        control.verified_launcher(repo,selected)
                with (repo/'reproduction/a800_common/deploy.py').open('a') as stream:
                    stream.write('\n# uncommitted imported deployer\n')
                with self.assertRaises(AssertionError):
                    control.verified_launcher(repo,selected)

    def test_timeout_consumes_claim_and_preserves_inspection_path(self):
        with tempfile.TemporaryDirectory() as directory:
            handle = Path(directory)/'handle.json'
            route = ['ssh','baai-a800','python3 -c pass']
            with mock.patch.object(control,'ssh_command',return_value=route), \
                 mock.patch.object(control.subprocess,'run',side_effect=subprocess.TimeoutExpired('ssh',1)) as call:
                result = control.invoke(handle,intent(),'prepare','pass',b'{}',1)
                self.assertEqual(result['failure_type'],'TimeoutExpired')
                self.assertIsNone(result['exit_code'])
                self.assertTrue(handle.with_name('handle.prepare-once.json').is_file())
                self.assertTrue(call.call_args.args[0][-1].startswith('/data/search-g1/env/bin/python -c '))
                with self.assertRaises(FileExistsError):
                    control.invoke(handle,intent(),'prepare','pass',b'{}',1)
                self.assertEqual(call.call_count,1)

    def test_bundle_contains_exact_generic_ref_with_baseline_prerequisite(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL,text=True).strip()
            git('init');git('config','user.name','test');git('config','user.email','test@example.invalid')
            (repo/'one.py').write_text('original = 1\n');git('add','one.py');git('commit','-m','baseline')
            baseline = git('rev-parse','HEAD')
            (repo/'one.py').write_text('original = 2\n');git('add','one.py');git('commit','-m','prospective')
            commit = git('rev-parse','HEAD');ref = 'refs/heads/codex/new-qprobe';git('update-ref',ref,commit)
            bundle = Path(directory)/'source.bundle'
            with mock.patch.object(control,'BASELINE_COMMIT',baseline):
                control.create_bundle(repo,commit,ref,bundle)
            self.assertEqual(subprocess.check_output(['git','bundle','list-heads',str(bundle)],text=True).strip(),commit+' '+ref)
            verify = subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle)],capture_output=True,text=True,check=True)
            self.assertIn(baseline, verify.stdout)
            self.assertLess(bundle.stat().st_size,control.MAX_BUNDLE)


if __name__ == '__main__':
    unittest.main()
