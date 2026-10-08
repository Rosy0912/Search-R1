"""Constructed local stdlib tests of the external runner; no remote/GPU execution."""
import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('target_v5', HERE / 'test_behavior_consistency_target_v5.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_repo(base):
    base.mkdir()
    for name in m.VERSION_FILES:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('0.1.0\n')
    (base / 'module.py').write_text('value = 1\n')
    (base / 'evidence').mkdir()
    (base / 'evidence/ignored.json').write_text('{}')
    for command in (['init','-q'],['add','.'],
                    ['-c','user.name=CPU Runner Test','-c','user.email=fixture@example.invalid',
                     'commit','-qm','constructed test fixture']):
        subprocess.run(['git','-C',str(base),*command],check=True,capture_output=True)
    return subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD']).decode().strip()


def payload():
    return dict(schema_version=1,base='/dataset/shaoshitong/g1-common8-behavior-cpu-v5-'+'a'*12,
                source_commit='b'*40,source_sha256={'a.py':'c'*64},source_map_sha256='d'*64,
                modules=list(m.MODULES),scope=m.SCOPE)


def proof(p):
    summary=dict(tests_run=264,skipped=[],failures=0,errors=0,successful=True,
                 cuda_available=False,cuda_initialized_before=False,cuda_initialized_after=False,
                 torch_version='2.4.0+cu121',vllm_module_version='dev',
                 distribution_versions=dict(torch='2.4.0+cu121',transformers='4.46.3',
                                            vllm='0.6.3',tensordict='0.5.0'),
                 expected_distribution_versions_matched=True,output='constructed local result only')
    return dict(**{key:p[key] for key in ('schema_version','base','source_commit','source_map_sha256',
                                         'source_sha256','modules','scope')},
                source_before_tests_sha256=p['source_sha256'],source_unchanged=True,
                expected_source_count=354,actual_source_count=354,actual_production_updates=0,
                production_admission=False,healthy_checkpoint_accepted=False,target_python=m.TARGET_PYTHON,
                cuda_visible_devices='',returncode=0,observation_timed_out=False,
                shell_syntax_returncode=0,shell_syntax_checks=[
                    dict(path=n,returncode=0,stdout='',stderr='') for n in
                    ('reproduction/common8/train.sh','reproduction/common8/launch_calibration_native.sh')],
                stdout=json.dumps(summary))


class CommittedPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.repo=Path(self.tmp.name)/'repo'
        self.commit=make_repo(self.repo)
        self.base='/dataset/shaoshitong/g1-common8-behavior-cpu-v5-'+'a'*12
    def tearDown(self):
        self.tmp.cleanup()
    def build(self,commit=None,base=None,count=3):
        return m.build_payload(self.repo,commit or self.commit,base or self.base,expected_count=count)
    def test_complete_committed_map_and_canonical_hash(self):
        p=self.build()
        self.assertEqual(set(p['files']),set(m.VERSION_FILES)|{'module.py'})
        self.assertEqual(p['source_commit'],self.commit)
        self.assertEqual(p['source_map_sha256'],m.sha(m.canonical(p['source_sha256'])))
    def test_uncommitted_runtime_edit_rejected(self):
        (self.repo/'module.py').write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'changed after commit'):self.build()
    def test_untracked_runtime_file_rejected(self):
        (self.repo/'extra.py').write_text('')
        with self.assertRaisesRegex(ValueError,'file set differs'):self.build()
    def test_missing_runtime_version_rejected(self):
        (self.repo/m.VERSION_FILES[0]).unlink()
        with self.assertRaises(ValueError):self.build()
    def test_short_commit_rejected(self):
        with self.assertRaisesRegex(ValueError,'full committed'):self.build(commit=self.commit[:7])
    def test_wrong_head_rejected(self):
        with self.assertRaisesRegex(ValueError,'current reviewed HEAD'):self.build(commit='f'*40)
    def test_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'count'):self.build(count=354)
    def test_unowned_namespace_rejected(self):
        with self.assertRaisesRegex(ValueError,'namespace'):self.build(base='/tmp/not-owned')
    def test_empty_committed_source_is_permitted(self):
        (self.repo/'empty.py').write_bytes(b'')
        subprocess.run(['git','-C',str(self.repo),'add','empty.py'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=CPU Runner Test',
                        '-c','user.email=fixture@example.invalid','commit','-qm','empty'],check=True)
        c=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD']).decode().strip()
        p=self.build(commit=c,count=4)
        self.assertEqual(p['source_sha256']['empty.py'],m.sha(b''))
    def test_runtime_symlink_rejected(self):
        (self.repo/'module.py').unlink()
        (self.repo/'module.py').symlink_to('verl/version/version')
        subprocess.run(['git','-C',str(self.repo),'add','module.py'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=CPU Runner Test',
                        '-c','user.email=fixture@example.invalid','commit','-qm','link'],check=True)
        c=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD']).decode().strip()
        with self.assertRaisesRegex(ValueError,'symlink'):self.build(commit=c)


class ProofTests(unittest.TestCase):
    def test_actual_distribution_and_module_labels_can_differ(self):
        p=payload();s=m.validate_proof(proof(p),p)
        self.assertEqual(s['vllm_module_version'],'dev')
        self.assertEqual(s['distribution_versions']['vllm'],'0.6.3')
    def test_all_source_and_identity_slots_required(self):
        p=payload()
        for key,bad in [('base','/tmp/x'),('source_commit','e'*40),('source_map_sha256','e'*64),
                        ('source_sha256',{}),('source_before_tests_sha256',{}),('source_unchanged',False),
                        ('expected_source_count',353),('actual_source_count',353),('modules',[]),
                        ('actual_production_updates',1),('production_admission',True),
                        ('healthy_checkpoint_accepted',True),('scope','healthy'),
                        ('target_python','python'),('cuda_visible_devices','0'),('returncode',1),
                        ('observation_timed_out',True),('shell_syntax_returncode',1)]:
            with self.subTest(key=key):
                r=proof(p);r[key]=bad
                with self.assertRaises(ValueError):m.validate_proof(r,p)
    def test_no_skip_failure_zero_tests_or_cuda(self):
        p=payload()
        for key,bad in [('tests_run',0),('skipped',[['test','missing']]),('failures',1),('errors',1),
                        ('successful',False),('cuda_available',True),('cuda_initialized_before',True),
                        ('cuda_initialized_after',True),('expected_distribution_versions_matched',False)]:
            with self.subTest(key=key):
                r=proof(p);s=json.loads(r['stdout']);s[key]=bad;r['stdout']=json.dumps(s)
                with self.assertRaises(ValueError):m.validate_proof(r,p)
    def test_installed_distribution_versions_independently_checked(self):
        p=payload();r=proof(p);s=json.loads(r['stdout'])
        s['distribution_versions']['vllm']='dev';r['stdout']=json.dumps(s)
        with self.assertRaisesRegex(ValueError,'distribution'):m.validate_proof(r,p)
    def test_immutable_receipt_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json';m.exclusive_json(path,{'first':1});before=path.read_bytes()
            with self.assertRaises(FileExistsError):m.exclusive_json(path,{'second':2})
            self.assertEqual(path.read_bytes(),before)
    def test_remote_and_runner_code_compile_and_include_real_parent_suite(self):
        compile(m.REMOTE_CODE,'targetCPU','exec');compile(m.CPU_RUNNER,'CPU runner','exec')
        self.assertEqual(len(m.MODULES),16)
        self.assertIn('reproduction.common8.test_sampling_capture',m.MODULES)
        self.assertIn("subprocess.run(['bash','-n',filename]",m.REMOTE_CODE)
        p=payload()
        for index in (0,1):
            r=proof(p);r['shell_syntax_checks'][index]['returncode']=2
            with self.assertRaisesRegex(ValueError,'Each shell file'):m.validate_proof(r,p)
        r=proof(p);r['shell_syntax_checks'].pop()
        with self.assertRaisesRegex(ValueError,'Each shell file'):m.validate_proof(r,p)
    def test_drycheck_never_claims_or_routes(self):
        p=payload();p['files']={'a.py':''}
        with patch.object(m,'build_payload',return_value=p),patch.object(m,'exclusive_json') as write,\
             patch.object(m.subprocess,'run') as rpc,patch('builtins.print'):
            self.assertEqual(m.main(['--source-commit','b'*40]),0)
            write.assert_not_called();rpc.assert_not_called()
    def test_prior_claim_prevents_remote_route(self):
        p=payload();p['files']={'a.py':''}
        with tempfile.TemporaryDirectory() as d,patch.object(m,'PREP',Path(d)),\
             patch.object(m,'build_payload',return_value=p),patch.object(m.subprocess,'run') as rpc:
            (Path(d)/(m.STEM+'-claim.json')).write_text('{}')
            with self.assertRaisesRegex(ValueError,'same handle'):
                m.main(['--source-commit','b'*40,'--execute'])
            rpc.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
