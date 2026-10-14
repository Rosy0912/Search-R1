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


if __name__=='__main__':unittest.main()
