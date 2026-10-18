import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .contracts import sha
from .searchr1_contracts import validate_searchr1_plan
from .test_searchr1_contracts import SearchR1Contracts


class SearchR1RecoveryTests(unittest.TestCase):
    def setUp(self):
        f=SearchR1Contracts();f.setUp();self.addCleanup(f.doCleanups)
        self.previous=f.plan;self.template=f.template
        self.root=Path(self.previous['output']).parent;self.root.mkdir(parents=True)
        self.pp=self.root/'previous-plan.json';self.pp.write_text(json.dumps(self.previous))
        self.sub={k:self.previous[k] for k in ('job','output','checkpoint_identity','group','outer')}
        self.sub.update(job_id='139030',plan=str(self.pp),plan_sha256=sha(self.pp))
        self.sp=self.root/'submission.json';self.sp.write_text(json.dumps(self.sub))
        self.out=self.root/'output';self.out.mkdir();(self.root/'exit.code').write_text('1\n')
        (self.out/'entry-exit.json').write_text(json.dumps(dict(status='FAILED',actual_job_id='139030',job=self.previous['job'])))
        self.plan=dict(self.previous,evaluation_code_commit='9'*40,evaluation_run_id='8'*32,
            job=self.previous['job']+'-startup-r1',output=str(self.root/('startup-recovery-'+'9'*12)/'output'),
            repeat_recovery=dict(previous_plan=str(self.pp),previous_plan_sha256=sha(self.pp),previous_submission=str(self.sp),previous_submission_sha256=sha(self.sp),reason='failed_startup_zero_predictions'))
        self.sacct=patch('reproduction.seven_dataset.repeat_recovery.subprocess.check_output',return_value='139030|jmzhang|FAILED\n').start()
        self.queue=patch('reproduction.seven_dataset.repeat_recovery.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')).start()
        self.addCleanup(patch.stopall)

    def test_same_slot_new_namespace(self):
        self.assertEqual(validate_searchr1_plan(self.plan,self.template),Path(self.plan['output']).parent)
        self.assertEqual(self.plan['repeat_index'],1)

    def test_science_or_repeat_changes_rejected(self):
        for k,v in [('model_revision','other'),('model_files',{}),('repeat_index',2),('repeat_count',5),('protocol_identity','0'*64),('science_override',True),('evaluation_code_commit',self.previous['evaluation_code_commit']),('evaluation_run_id',self.previous['evaluation_run_id']),('job',self.previous['job']),('output',self.previous['output'])]:
            p=copy.deepcopy(self.plan);p[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):validate_searchr1_plan(p,self.template)

    def test_live_completed_or_ambiguous_job_rejected(self):
        for s in ['139030|jmzhang|RUNNING\n','139030|jmzhang|COMPLETED\n','139030|other|FAILED\n','', '139030|jmzhang|FAILED\n'*2]:
            self.sacct.return_value=s
            with self.subTest(s=s),self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_any_predictions_progress_or_summary_rejected(self):
        for name in ['predictions.jsonl','progress.json','summary.json']:
            p=self.out/name;p.write_text('{}\n')
            with self.subTest(name=name),self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)
            p.unlink()

    def test_recursive_recovery_and_tampering_rejected(self):
        self.pp.write_text(self.pp.read_text()+' ')
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)
        self.pp.write_text(json.dumps(dict(self.previous,repeat_recovery={})))
        self.plan['repeat_recovery']['previous_plan_sha256']=sha(self.pp)
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_submission_and_exit_mismatch_rejected(self):
        self.sub['job_id']='999';self.sp.write_text(json.dumps(self.sub))
        self.plan['repeat_recovery']['previous_submission_sha256']=sha(self.sp)
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_queue_uncertainty_rejected(self):
        self.queue.return_value=SimpleNamespace(returncode=1,stdout='',stderr='scheduler unavailable')
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)


if __name__=='__main__':unittest.main()
