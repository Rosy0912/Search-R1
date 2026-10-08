"""Pure helper-function CPU fixtures; never invoke main, a route or subprocess.

The FAILED allocation and run-result bytes are previously captured job3314
metadata. Disabled-sacct and live-runner returns are injected fixture results;
these tests are not actual controller inquiries or dispatch acceptance.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
CAPTURE = WORK/'tree-fresh200-freeze-02b1-v11/actual-zero-dispatch'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = load('terminal_submit_v12', HERE/'submit_tree_fresh200_once.py')


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.c = load('terminal_common_v12', HERE/'fresh200_dispatch_common.py')
        self.c.ROOT = Path(self.temp.name)
        (self.c.ROOT/'instruct300').mkdir()
        self.prior = json.loads((CAPTURE/'ZERO_SUBMIT_JOURNAL_COPY.json').read_bytes())
        self.prior['reviewed_launch_path_provenance_only'] = str(
            self.c.ROOT/'instruct300/candidates/fresh200-mu-02b1-v11/launch.sbatch')
        self.journal = self.c.ROOT/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-old.json'
        self.current = self.c.ROOT/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-new.json'
        self.run_path = self.c.ROOT/'runs/tree-fresh200-3314/run-result.json'
        self.run_path.parent.mkdir(parents=True)
        self.run_value = json.loads((CAPTURE/'terminal-native/run-result.json').read_bytes())
        self.control = json.loads((CAPTURE/'TERMINAL3314_AUDIT.json').read_bytes())['scontrol']
        self.sacct = NS(returncode=1, stdout='', stderr='Slurm accounting storage is disabled\n')
        self.live_queue = NS(returncode=0, stdout='', stderr='')
        self.previous = {}
        self.queries = []
        self.save()

    def save(self):
        self.journal.write_text(json.dumps(self.prior))
        self.run_path.write_text(json.dumps(self.run_value))

    def runner(self, argv, **kwargs):
        self.queries.append(argv)
        self.assertEqual(kwargs, dict(env={}, text=True, capture_output=True, timeout=30))
        if argv[0] == 'sacct':
            self.assertEqual(argv, ['sacct','--local','-X','-n','-P','-j','3314',
                                   '-o','JobIDRaw,Cluster,State,ExitCode'])
            return self.sacct
        if argv[0] == 'squeue':
            self.assertEqual(argv, ['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%N'])
            return self.live_queue
        self.assertEqual(argv, ['scontrol','--local','show','job','3314'])
        return NS(**self.control)

    def check(self):
        self.save()
        return submit.previous_journals_clear(self.c, self.current, self.runner, {}, self.previous)

    def retained_proof(self):
        self.save()
        audit = json.loads((CAPTURE/'TERMINAL3314_AUDIT.json').read_bytes())
        proof = dict(status='RETAINED_POSITIVE_OWN_ZERO_TERMINAL_PROOF', job_id=3314,
                     cluster_name='wuhan-h100', stage=self.prior['stage'], code_commit=self.prior['code_commit'],
                     manifest_sha256=self.prior['manifest_sha256'],
                     journal_sha256=hashlib.sha256(self.journal.read_bytes()).hexdigest(),
                     run_result_sha256=hashlib.sha256(self.run_path.read_bytes()).hexdigest(),
                     scontrol=audit['scontrol'], squeue=audit['squeue'])
        path = self.c.ROOT/'instruct300/RETAINED_TERMINAL3314_PROOF.json'
        path.write_text(json.dumps(proof))
        self.previous = {'3314':dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())}
        self.control = dict(returncode=1,stdout='',stderr='slurm_load_jobs error: Invalid job id specified\n')
        return path,proof

    def refused(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.check()

    def test_captured3314_failed_scope_and_cleanup_accept(self):
        result = self.check()
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertEqual(item['scheduler_state'], 'FAILED')
        self.assertEqual(item['exit_code'], '1:0')
        self.assertEqual(item['failed_run']['sha256'], hashlib.sha256(self.run_path.read_bytes()).hexdigest())
        self.assertFalse(item['actual_zero_or_training_accepted'])
        self.assertEqual([q[0] for q in self.queries], ['sacct','scontrol'])

    def test_normal_sacct_original_path_accept_without_fallback(self):
        self.sacct = NS(returncode=0, stdout='3314|wuhan-h100|FAILED|1:0\n', stderr='')
        result = self.check()
        self.assertEqual(result[0]['scheduler_state'], 'FAILED')
        self.assertEqual([q[0] for q in self.queries], ['sacct'])
        self.assertNotIn('failed_run', result[0])

    def test_explicit_disabled_stdout_diagnostic_accept(self):
        self.sacct = NS(returncode=1, stdout='Slurm accounting storage is disabled\n', stderr='')
        self.assertEqual(self.check()[0]['scheduler_state'], 'FAILED')

    def test_unrelated_sacct_error_refuses_without_fallback(self):
        self.sacct = NS(returncode=1, stdout='', stderr='controller unavailable')
        self.refused()
        self.assertEqual([q[0] for q in self.queries], ['sacct'])

    def test_mixed_disabled_diagnostic_refuses_without_fallback(self):
        self.sacct = NS(returncode=1, stdout='', stderr='Slurm accounting storage is disabled\nunknown controller')
        self.refused()
        self.assertEqual([q[0] for q in self.queries], ['sacct'])

    def test_running_unknown_completed_and_duplicate_fields_refuse(self):
        original = self.control['stdout']
        for value in ('RUNNING','UNKNOWN','COMPLETED'):
            with self.subTest(value=value):
                self.control['stdout'] = original.replace('JobState=FAILED', 'JobState='+value)
                self.refused()
        self.control['stdout'] = original+'\n JobId=3314\n'
        self.refused()

    def test_wrong_job_node_gpu_and_exit_refuse(self):
        original = self.control['stdout']
        for before, after in (('JobId=3314','JobId=3315'), ('NodeList=h100-9','NodeList=h100-8'),
                              ('TresPerNode=gres:gpu:8','TresPerNode=gres:gpu:4'),
                              ('ExitCode=1:0','ExitCode=0:0'), ('JobName=tree-fresh200-mu-zero','JobName=foreign')):
            with self.subTest(after=after):
                self.control['stdout'] = original.replace(before, after)
                self.refused()

    def test_expired_or_nonzero_controller_refuses(self):
        self.control = dict(returncode=1, stdout='', stderr='Invalid job id specified')
        self.refused()

    def test_exact_expired_with_bound_positive_proof_and_live_empty_queue_accept(self):
        self.retained_proof()
        result = self.check()[0]
        self.assertEqual(result['evidence_source'], 'sha_bound_retained_positive_terminal_live_expired_and_empty_queue')
        self.assertEqual(result['current_scontrol_returncode'], 1)
        self.assertEqual(result['scheduler_state'], 'FAILED')
        self.assertFalse(result['actual_zero_or_training_accepted'])
        self.assertEqual([q[0] for q in self.queries], ['sacct','scontrol','squeue'])

    def test_exact_expired_without_bound_positive_proof_refuses(self):
        self.control = dict(returncode=1,stdout='',stderr='slurm_load_jobs error: Invalid job id specified\n')
        self.refused()

    def test_expired_unknown_error_does_not_use_retained_proof(self):
        self.retained_proof()
        self.control['stderr'] = 'controller unavailable'
        self.refused()
        self.assertEqual([q[0] for q in self.queries], ['sacct','scontrol'])

    def test_expired_wrong_proof_sha_job_or_journal_refuses(self):
        path,proof = self.retained_proof()
        self.previous['3314']['sha256'] = '0'*64
        self.refused()
        for key,value in (('job_id',3315), ('journal_sha256','0'*64), ('manifest_sha256','0'*64)):
            with self.subTest(key=key):
                changed = copy.deepcopy(proof); changed[key] = value; path.write_text(json.dumps(changed))
                self.previous['3314']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.refused()

    def test_expired_live_active_unknown_or_failed_queue_refuses(self):
        self.retained_proof()
        for value in (NS(returncode=0,stdout='3314|tree-fresh200-mu-zero|RUNNING|h100-9\n',stderr=''),
                      NS(returncode=0,stdout='3314|other|RUNNING|h100-9\n',stderr=''),
                      NS(returncode=0,stdout='unknown',stderr=''),
                      NS(returncode=1,stdout='',stderr='controller unavailable')):
            with self.subTest(queue=value.stdout):
                self.live_queue = value
                self.refused()

    def test_expired_current_native_run_bytes_must_equal_positive_proof(self):
        self.retained_proof()
        self.run_value['error'] = 'different failed-run bytes'
        self.refused()

    def test_missing_controller_fields_refuse(self):
        self.control['stdout'] = 'JobId=3314 JobState=FAILED ExitCode=1:0\n'
        self.refused()

    def test_wrong_prior_stage_launch_and_nonzero_budget_refuse(self):
        original = copy.deepcopy(self.prior)
        for key,value in (('stage','other'), ('expected_optimizer_writes',1),
                          ('actual_submit_stdin_sha256','0'*64), ('old_checkpoint_inherited',True)):
            with self.subTest(key=key):
                self.prior = copy.deepcopy(original); self.prior[key] = value
                self.refused()

    def test_unknown_prior_and_foreign_cluster_refuse(self):
        original = copy.deepcopy(self.prior)
        for key,value in (('status','SUBMISSION_UNKNOWN_DO_NOT_RETRY'), ('cluster_name','foreign')):
            with self.subTest(key=key):
                self.prior = copy.deepcopy(original); self.prior[key] = value
                self.refused()

    def test_failed_run_job_source_status_and_allocation_mismatch_refuse(self):
        original = copy.deepcopy(self.run_value)
        for key,value in (('job_id','3315'), ('code_commit','0'*40), ('status','COMPLETED'),
                          ('allocated_gpus',4), ('allocated_cpus',32)):
            with self.subTest(key=key):
                self.run_value = copy.deepcopy(original); self.run_value[key] = value
                self.refused()

    def test_missing_cleanup_or_protected_identity_refuses(self):
        original = copy.deepcopy(self.run_value)
        for key,value in (('cleanup_residual',None), ('cleanup_residual',{'unexpected_pid':1}),
                          ('original_eight_pid_start_times_preserved',False)):
            with self.subTest(key=key):
                self.run_value = copy.deepcopy(original); self.run_value[key] = value
                self.refused()

    def test_missing_failed_run_evidence_refuses(self):
        self.run_path.unlink()
        with self.assertRaises(FileNotFoundError):
            submit.previous_journals_clear(self.c,self.current,self.runner,{})


if __name__ == '__main__':
    unittest.main(verbosity=2)
