import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .contracts import sha
from .repeats import inspect_repeat
from . import test_repeats


class RepeatRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = test_repeats.RepeatTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.previous = fixture.plan
        self.root = Path(self.previous['output']).parent
        self.root.mkdir(parents=True)
        self.pp = Path(fixture.tmp.name)/'failed-repeat.json'
        self.pp.write_text(json.dumps(self.previous))
        self.sub = {k:self.previous[k] for k in
                    ('job','output','checkpoint_identity','group','outer')}
        self.sub.update(job_id='139021',plan=str(self.pp),plan_sha256=sha(self.pp))
        self.sp = self.root/'submission.json';self.sp.write_text(json.dumps(self.sub))
        self.out = self.root/'output';self.out.mkdir()
        (self.root/'exit.code').write_text('1\n')
        (self.out/'entry-exit.json').write_text(json.dumps(dict(status='FAILED',
            actual_job_id='139021',job=self.previous['job'])))
        self.plan = copy.deepcopy(self.previous)
        self.plan.update(evaluation_code_commit='9'*40,evaluation_run_id='8'*32,
            job=self.previous['job']+'-startup-r1',
            output=str(self.root/('startup-recovery-'+'9'*12)/'output'),
            repeat_recovery=dict(previous_plan=str(self.pp),previous_plan_sha256=sha(self.pp),
                previous_submission=str(self.sp),previous_submission_sha256=sha(self.sp),
                reason='failed_startup_zero_predictions'))
        self.sacct = patch('reproduction.seven_dataset.repeat_recovery.subprocess.check_output',
                          return_value='139021|jmzhang|FAILED\n').start()
        self.queue = patch('reproduction.seven_dataset.repeat_recovery.subprocess.run',
            return_value=SimpleNamespace(returncode=0,stdout='',stderr='')).start()
        self.addCleanup(patch.stopall)

    def test_zero_prediction_failed_repeat_has_separate_namespace_same_slot(self):
        self.assertEqual(inspect_repeat(self.plan),Path(self.plan['output']).parent)
        self.assertEqual(self.plan['repeat'],self.previous['repeat'])
        self.assertNotEqual(self.plan['output'],self.previous['output'])

    def test_live_or_ambiguous_previous_is_rejected(self):
        for rows in ['139021|jmzhang|RUNNING\n','139021|jmzhang|COMPLETED\n',
                     '', '139021|someone|FAILED\n','139021|jmzhang|FAILED\n'*2]:
            self.sacct.return_value=rows
            with self.subTest(rows=rows),self.assertRaises(ValueError):inspect_repeat(self.plan)

    def test_prediction_or_progress_or_summary_prevents_reinference(self):
        for name,content in [('predictions.jsonl','{}\n'),('progress.json','{}'),('summary.json','{}')]:
            p=self.out/name;p.write_text(content)
            with self.subTest(name=name),self.assertRaises(ValueError):inspect_repeat(self.plan)
            p.unlink()
        (self.out/'predictions.jsonl').touch()
        inspect_repeat(self.plan)

    def test_science_repeat_slot_extra_fields_and_reused_identity_rejected(self):
        for key,value in [('model_files',{}),('training_config_sha256','0'*64),
            ('repeat',dict(self.plan['repeat'],index=3)),('science_override',True),
            ('evaluation_run_id',self.previous['evaluation_run_id']),
            ('evaluation_run_id','c'*32),('job',self.previous['job']),('output',self.previous['output'])]:
            bad=copy.deepcopy(self.plan);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):inspect_repeat(bad)

    def test_tamper_and_recursive_retry_rejected(self):
        self.pp.write_text(self.pp.read_text()+' ')
        with self.assertRaises(ValueError):inspect_repeat(self.plan)
        prior=dict(self.previous,repeat_recovery={});self.pp.write_text(json.dumps(prior))
        self.plan['repeat_recovery']['previous_plan_sha256']=sha(self.pp)
        with self.assertRaises(ValueError):inspect_repeat(self.plan)

    def test_submission_and_exit_identity_rejected(self):
        self.sub['job_id']='999';self.sp.write_text(json.dumps(self.sub))
        self.plan['repeat_recovery']['previous_submission_sha256']=sha(self.sp)
        with self.assertRaises(ValueError):inspect_repeat(self.plan)

    def test_uncertain_squeue_rejected_only_exact_absent_error_allowed(self):
        for r in [SimpleNamespace(returncode=0,stdout='139021|RUNNING',stderr=''),
                  SimpleNamespace(returncode=1,stdout='',stderr='connection failed')]:
            self.queue.return_value=r
            with self.assertRaises(ValueError):inspect_repeat(self.plan)
        self.queue.return_value=SimpleNamespace(returncode=1,stdout='',stderr='Invalid job id specified')
        inspect_repeat(self.plan)

    def test_entry_runs_replacement_namespace(self):
        from . import entry
        pf=self.root/'replacement.json';pf.write_text(json.dumps(self.plan))
        modules={'omegaconf':SimpleNamespace(OmegaConf=None),
                 'search_r1.baselines.preflight':SimpleNamespace(validate_gpu_lease=None)}
        with patch.dict('sys.modules',modules),patch('sys.argv',['entry','--plan',str(pf)]),\
                patch.object(entry,'inspect_inputs',return_value=(None,[])),patch.object(entry,'run_frozen') as run:
            entry.main()
        self.assertEqual(run.call_args.args[-1],Path(self.plan['output']).parent)


if __name__ == '__main__':unittest.main()
