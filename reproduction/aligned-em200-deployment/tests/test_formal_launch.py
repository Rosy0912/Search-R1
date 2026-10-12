"""Local boundary fixtures only; no scheduler, GPU, or real acceptance created."""
import contextlib, importlib.util, io, json, os, pathlib, subprocess, sys, tempfile, types, unittest
from unittest.mock import patch
TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import formal_admission as admission
import submit_formal as submit


class FormalLaunchTests(unittest.TestCase):
    def test_unbound_candidate_never_contacts_scheduler(self):
        self.assertIsNone(admission.HARDWARE_REVIEW_SHA256)
        self.assertIsNone(admission.FORMAL_AUTHORIZATION_SHA256)
        with patch.object(submit.subprocess, 'check_output') as read, patch.object(submit.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'unbound'):
                submit.main()
            read.assert_not_called(); run.assert_not_called()

    def test_old_or_submission_receipts_are_not_hardware_acceptance(self):
        for delta in [dict(status='PASS_CPU'), dict(source='60f566'), dict(job='3315'), dict(scope='ACTUAL_HARDWARE_SUBMISSION_NOT_TRAINING')]:
            value = dict(status='PASS_INDEPENDENT_ALIGNED_HARDWARE_REVIEW', scope='ACTUAL_8RANK_SYNTHETIC_AND_LIVE_RETRIEVAL_NOT_TRAINING', source=admission.SOURCE, root=str(admission.ROOT), run=str(admission.HARDWARE_RUN), job='3345')
            value.update(delta)
            with self.subTest(delta=delta), patch.object(admission, 'bound_json', return_value=value):
                with self.assertRaises(ValueError): admission.validate_hardware()

    def test_unknown_submit_and_old_artifacts_are_not_retried(self):
        for case in ('success', 'timeout', 'nonzero', 'malformed', 'intent', 'receipt', 'response', 'active_queue'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory).resolve(); head=root/'scheduler-head'; head.mkdir()
                if case in ('intent', 'receipt', 'response'):
                    name='formal-sbatch-response.json' if case=='response' else 'formal-submit-'+case+'.json'
                    (head/name).write_text('{}')
                calls=[]
                def fake_run(command, **kwargs):
                    self.assertTrue((head/'formal-submit-intent.json').is_file()); calls.append(command)
                    if case=='timeout': raise subprocess.TimeoutExpired(command,45)
                    return types.SimpleNamespace(returncode=1 if case=='nonzero' else 0, stdout='garbled' if case=='malformed' else '12345\n', stderr='')
                with patch.object(submit,'ROOT',root), patch.object(submit,'validate_admission'), patch.object(submit.subprocess,'check_output',return_value='3345|own|BAPO_ALIGNED_EM200:hardware:x|RUNNING\n' if case=='active_queue' else ''), patch.object(submit.subprocess,'run',side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                    if case=='success':
                        submit.main(); self.assertEqual(len(calls),1); self.assertEqual(json.loads((head/'formal-submit-receipt.json').read_text())['job'],'12345')
                    elif case in ('timeout','nonzero','malformed'):
                        with self.assertRaises((ValueError,subprocess.TimeoutExpired)): submit.main()
                        self.assertEqual(len(calls),1)
                        with self.assertRaises(ValueError): submit.main()
                        self.assertEqual(len(calls),1); self.assertFalse((head/'formal-submit-receipt.json').exists())
                    else:
                        with self.assertRaises(ValueError): submit.main()
                        self.assertFalse(calls)

    def test_actual_source_claim_once_before_source_record(self):
        source = TOOLS.parents[2] / 'bapo-baseline-alignment-20261009/search_r1/baselines/aligned_startup.py'
        if not source.exists():
            source = TOOLS.parents[3] / 'bapo-baseline-alignment-20261009/search_r1/baselines/aligned_startup.py'
        self.assertTrue(source.is_file())
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory).resolve(); repo=root/'repo/search_r1/baselines'; repo.mkdir(parents=True)
            copied=repo/'aligned_startup.py'; copied.write_bytes(source.read_bytes())
            spec=importlib.util.spec_from_file_location('review_actual_startup',copied); actual=importlib.util.module_from_spec(spec); spec.loader.exec_module(actual)
            package=types.ModuleType('search_r1.baselines'); package.aligned_startup=actual
            run=root/'runs/formal-aligned-em200-fixture-99'
            with patch.object(admission,'ROOT',root),patch.object(admission,'validate_admission'),patch.object(admission,'verify_submission'),patch.dict(os.environ,SLURM_JOB_ID='99'),patch.dict(sys.modules,{'search_r1.baselines':package}):
                admission.claim(run)
                self.assertTrue((run/'.aligned-startup-claim.json').is_file()); self.assertEqual((run/'source-commit.txt').read_text().strip(),admission.SOURCE)
                with self.assertRaises(ValueError): admission.claim(run)
                prior=root/'runs/formal-aligned-em200-prior-99'; prior.mkdir(); (prior/'train.log').write_text('prior')
                with self.assertRaises(ValueError): admission.claim(prior)

    def test_submission_receipt_identity_required_before_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory).resolve();(root/'scheduler-head').mkdir()
            run=root/'runs/formal-aligned-em200-token-99'
            value=dict(source=admission.SOURCE,root=str(root),job='99',run=str(run),token='token',hardware_review_sha256=admission.HARDWARE_REVIEW_SHA256,formal_authorization_sha256=admission.FORMAL_AUTHORIZATION_SHA256,files=admission.FORMAL_FILES,scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION')
            path=root/'scheduler-head/formal-submit-receipt.json'
            with patch.object(admission,'ROOT',root),patch.dict(os.environ,SLURM_JOB_ID='99'):
                with self.assertRaisesRegex(ValueError,'unknown'):admission.verify_submission(run,timeout=0)
                path.write_text(json.dumps(value));admission.verify_submission(run,timeout=0)
                for key,bad in [('job','3315'),('source','60'),('token','other'),('scope','PASS_GPU'),('formal_authorization_sha256','wrong')]:
                    path.write_text(json.dumps(dict(value,**{key:bad})))
                    with self.subTest(key=key),self.assertRaises(ValueError):admission.verify_submission(run,timeout=0)

    def test_ray_head_foreign_birth_and_port_swaps_refused(self):
        import formal_service_proof as proof
        cli=dict(pid=10,ppid=1,start_ticks=100,uid=os.getuid(),command=[str(admission.ROOT/'env/bin/python'),'-B',str(admission.ROOT/'tools/deployment/ray_cli_compat.py'),'start','--head','--block','--port=16383'],executable={'path':'python'})
        owner=dict(pid=11,ppid=10,start_ticks=101,uid=os.getuid(),command=['gcs_server'],executable={'path':'/actual/gcs_server'})
        with patch.dict(os.environ,BAPO_RAY_CLI_PID='10'):
            with patch.object(proof,'own_identity',side_effect=lambda p:cli if p==10 else owner),patch.object(proof,'port_owner',return_value=owner):
                self.assertEqual(proof.ray_head()['listener'],owner)
            foreign=dict(owner,ppid=1)
            with patch.object(proof,'own_identity',side_effect=lambda p:cli if p==10 else foreign),patch.object(proof,'port_owner',return_value=foreign):
                with self.assertRaises(ValueError):proof.ray_head()
            with patch.object(proof,'own_identity',side_effect=lambda p:cli if p==10 else owner),patch.object(proof,'port_owner',side_effect=[owner,dict(owner,start_ticks=102)]):
                with self.assertRaisesRegex(ValueError,'changed'):proof.ray_head()

    def test_click_sentinel_feature_detection_and_identity(self):
        import copy, enum, ray_cli_compat as shim
        with patch.object(shim.importlib,'import_module',return_value=types.SimpleNamespace()):
            self.assertFalse(shim.prepare_click())
        error=ModuleNotFoundError("No module named 'click._utils'",name='click._utils')
        with patch.object(shim.importlib,'import_module',side_effect=error):self.assertFalse(shim.prepare_click())
        sentinel=enum.Enum('SyntheticSentinel',['ONE','TWO'])
        with patch.object(shim.importlib,'import_module',return_value=types.SimpleNamespace(Sentinel=sentinel)),patch.dict(copy._deepcopy_dispatch):
            self.assertTrue(shim.prepare_click())
            for value in sentinel:self.assertIs(copy.deepcopy(value),value)
        self.assertNotIn(sentinel,copy._deepcopy_dispatch)

    def test_worker_claim_precedes_every_nonbootstrap_output_and_service(self):
        worker=(TOOLS/'formal_worker.sh').read_text(); inner=(TOOLS/'formal_inner.sh').read_text()
        self.assertEqual(worker.count('formal_admission.py" claim'),1)
        self.assertLess(worker.index('formal_admission.py" claim'),worker.index('cleanup()'))
        self.assertLess(worker.index('formal_admission.py" claim'),worker.index('formal-supervisor.log'))
        self.assertNotIn('aligned_startup',inner); self.assertNotIn('run.sh',inner.replace('# The worker has exclusively claimed RUN; do not invoke source run.sh again.',''))
        self.assertIn('"$ROOT/env/bin/python" "$ROOT/tools/resource_watch.py"',worker)
        self.assertLess(inner.index('formal_admission.py" environment'),inner.index('ray_cli_compat.py'))
        driver=(TOOLS/'formal_driver.py').read_text()
        self.assertIn('ppo_bapo_baseline_aligned_production',driver);self.assertIn("'-m', 'search_r1.baselines.preflight'",driver)
        self.assertNotIn('resume_',driver);self.assertNotIn('fresh200_gate_enabled=false',driver)


if __name__ == '__main__': unittest.main()
