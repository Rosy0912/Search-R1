import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from .inspect_run import inspect, read_jsonl, verify_rank


class RuntimeEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)
        (self.run/'receipts').mkdir();(self.run/'output').mkdir()
        self.identity=dict(job_id='1',run_id='run',code_commit='source',config_sha256='config',data_identity='data')
        (self.run/'receipts/configuration.json').write_text(json.dumps(dict(run_id='run',source_commit='source',config_sha256='config',data_identity='data')))
        (self.run/'receipts/train-start.json').write_text(json.dumps(dict(job='1')))
        (self.run/'output/fresh-initialization.json').write_text(json.dumps(self.identity))
        self.log=self.run/'log';self.log.write_text('TREE_TIMING={"unfinished":')

    def test_no_training_cannot_pass_acceptance_and_partial_log_is_ignored(self):
        result=inspect(self.run,self.log)
        self.assertFalse(result['initial_acceptance_verified'])
        self.assertFalse(result['native5']['all_native_files_hash_verified'])
        self.assertEqual(result['completed_rounds'],0)

    def test_partial_audit_line_is_not_evidence(self):
        path=self.run/'audit.jsonl';path.write_text('{"complete":true}\n{"incomplete":')
        self.assertEqual(read_jsonl(path),[{'complete':True}])

    def test_direct_rank_byte_change_is_rejected(self):
        root=self.run/'output/fresh-state';root.mkdir()
        path=root/'rank.json';path.write_text('{}')
        expected=hashlib.sha256(path.read_bytes()).hexdigest();path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'modified'):
            verify_rank(dict(rank=0,direct_rank_record=dict(path=str(path),sha256=expected)),self.run,self.identity,1)

    def test_nonmilestone_uses_real_ledger_without_demanding_disabled_detailed_audit(self):
        root=self.run/'output/fresh-state';root.mkdir()
        path=root/'rank.json'
        ledger=[dict(index=i,actual_writes=1,rows=16,local_row_indices=list(range((i-1)*16,i*16)),
                     status='SUCCESS_ADAM_WRITE',gradient_preclip=.25) for i in range(1,21)]
        physical=dict(self.identity,status='DIRECT_RANK_COMPLETE_OUTER_OBSERVED',rank=0,world_size=8,
            actual_optimizer_calls=20,scheduler_step_before=0,scheduler_step_after=1,
            optimizer_opportunity_ledger=ledger,attempt_id=1)
        path.write_text(json.dumps(physical))
        record=dict(rank=0,optimizer_calls=20,total_optimizer_calls=20,scheduler_before=0,scheduler_after=1,
            direct_rank_record=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        result=verify_rank(record,self.run,self.identity,1)
        self.assertEqual(result['actual_adam_writes'],20)
        self.assertEqual(result['finite_grad_norms'],20)
        self.assertIsNone(result['all_observed_loss_finite'])
        # Unknown microbatch loss must not be reported as observed finite loss.
        self.assertEqual(result['observed_loss_scalars'],0)


if __name__=='__main__':unittest.main()
