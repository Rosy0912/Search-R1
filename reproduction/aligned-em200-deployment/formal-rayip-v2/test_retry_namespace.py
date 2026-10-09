"""Zero-update retry namespace and immutable predecessor boundary tests."""
from pathlib import Path
import json, os, unittest, sys
from unittest.mock import patch
import formal_admission as a

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'evidence-tools'))
import aligned_first5_cpu_evidence as cpu
class RetryNamespace(unittest.TestCase):
    def receipt(self):
        return dict(source=a.SOURCE,root=str(a.ROOT),job='9999',run=str(a.ROOT/'runs'/('formal-aligned-em200-'+'a'*32+'-9999')),token='a'*32,hardware_review_sha256=a.HARDWARE_REVIEW_SHA256,formal_authorization_sha256=a.FORMAL_AUTHORIZATION_SHA256,files=a.FORMAL_FILES,scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION',deployment_directory=str(a.ROOT/'tools/deployment/formal-rayip-v2'),submission_namespace='formal-rayip-v2',previous_failed_job='3348',comment='BAPO_ALIGNED_EM200:formal:'+a.SOURCE+':'+'a'*32)
    def validate(self,r):
        with patch.dict(os.environ,SLURM_JOB_ID='9999'),patch.object(a.Path,'exists',return_value=True),patch.object(a,'raw_file',return_value=json.dumps(r).encode()):
            a.verify_submission(Path(self.receipt()['run']))
    def test_current_namespace_accepted(self):self.validate(self.receipt())
    def test_old_receipt_rejected(self):
        r=self.receipt();r.pop('submission_namespace')
        with self.assertRaises(ValueError):self.validate(r)
    def test_changed_namespace_rejected(self):
        r=self.receipt();r['deployment_directory']=str(a.ROOT/'tools/deployment')
        with self.assertRaises(ValueError):self.validate(r)
    def test_wrong_predecessor_rejected(self):
        r=self.receipt();r['previous_failed_job']='3315'
        with self.assertRaises(ValueError):self.validate(r)
    def test_tool_binding_changed_rejected(self):
        r=self.receipt();r['files']={}
        with self.assertRaises(ValueError):self.validate(r)
    def test_cpu_consumer_new_receipt(self):
        self.assertEqual(cpu.identity(self.receipt())['job'],'9999')
    def test_cpu_consumer_old_namespace_rejected(self):
        r=self.receipt();r.pop('submission_namespace')
        with self.assertRaises(ValueError):cpu.identity(r)
    def test_cpu_consumer_predecessor_rejected(self):
        r=self.receipt();r['job']='3348';r['run']=str(a.ROOT/'runs'/('formal-aligned-em200-'+'a'*32+'-3348'))
        with self.assertRaises(ValueError):cpu.identity(r)
    def test_both_global_locks_remain_shared(self):
        submit=(HERE/'submit_formal.py').read_text();worker=(HERE/'formal_worker.sh').read_text()
        self.assertIn("lock=ROOT/'scheduler-head/submission-global.lock'",submit)
        self.assertIn('$ROOT/scheduler-head/execution-global.lock',worker)
        self.assertNotIn('formal-rayip-v2/execution-global.lock',worker)
    def test_all_receipt_consumers_namespaced(self):
        for name in ('aligned_first5_cpu_evidence.py','publish_first5_release.py','aligned_final_evidence.py','aligned_final_phase2.py'):
            s=(HERE/'evidence-tools'/name).read_text()
            self.assertIn('scheduler-head/formal-rayip-v2/formal-submit-receipt.json',s)
            self.assertNotIn('scheduler-head/formal-submit-receipt.json',s)
    def test_failure_inventory_is_complete_and_small(self):
        self.assertEqual(len(a.PREVIOUS_FAILURE_FILES),33)
        self.assertTrue(all(len(h)==64 for h in a.PREVIOUS_FAILURE_FILES.values()))
        self.assertIn(str(a.ROOT/'scheduler-head/formal-submit-receipt.json'),a.PREVIOUS_FAILURE_FILES)
if __name__=='__main__':unittest.main()
