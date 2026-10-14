"""Real head/runtime publisher -> common checkpoint consumer regression.

The head is genuinely fitted and serialized on CPU. Actor/HF states use the
existing small four-rank fixture; this does not claim actual GPU recovery.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from search_g1.test_qprobe_checkpoint import PairFixture
from search_g1.test_qprobe_runtime import runtime,stage_pending
from reproduction.common8.runtime import checkpoint_complete


class HeadPublisherConsumerTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads=torch.get_num_threads();torch.set_num_threads(1)

    def tearDown(self):torch.set_num_threads(self.previous_threads)

    def fixture(self,folder):
        fixture=PairFixture(folder,outer=5,size=4)
        value=runtime(folder)
        for outer in range(1,6):
            attempt='publisher-%d'%outer
            stage_pending(value,outer=outer,attempt_id=attempt)
            value.complete_outer(outer,fixture.run_id+':'+str(outer),dict(attempt_id=attempt))
        # Use the public real publisher, without manually manufacturing head
        # bytes/SHA as the older PairFixture did.
        proof=value.save_checkpoint(fixture.state/'actual-runtime',5,fixture.stream,
                                    fixture.run['initial_probe_version'],fixture.run_id+':5')
        manifest=json.loads((fixture.state/'manifest.json').read_text())
        manifest['question_probe']=proof
        (fixture.state/'manifest.json').write_text(json.dumps(manifest))
        return fixture,proof,manifest

    def consume(self,fixture):
        hf=dict(run_id=fixture.run_id,step=5,finite_all_saved_tensors=True,
                validated_tensor_count=1,files_sha256={})
        with patch.dict(os.environ,SEARCH_G1_BASE=str(fixture.base)),\
             patch('reproduction.common8.runtime.hf_checkpoint_complete',return_value=copy.deepcopy(hf)):
            return checkpoint_complete(5,fixture.cp)

    def test_real_publisher_matches_four_rank_checkpoint_consumer(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture,proof,_=self.fixture(folder)
            head=proof['head'];path=Path(head['path'])
            self.assertEqual(head['bytes'],path.stat().st_size)
            self.assertEqual(head['sha256'],head['checkpoint_sha'])
            result=self.consume(fixture)
            self.assertEqual(result['question_probe']['completed_outer'],5)
            self.assertEqual(result['files_sha256'][str(path.relative_to(fixture.cp))],head['sha256'])

    def test_missing_legacy_head_metadata_and_changed_head_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture,proof,manifest=self.fixture(folder)
            for key in ('bytes','sha256'):
                missing=copy.deepcopy(manifest);missing['question_probe']['head'].pop(key)
                (fixture.state/'manifest.json').write_text(json.dumps(missing))
                with self.assertRaises(KeyError):self.consume(fixture)
            (fixture.state/'manifest.json').write_text(json.dumps(manifest))
            with Path(proof['head']['path']).open('ab') as f:f.write(b'changed')
            with self.assertRaises(AssertionError):self.consume(fixture)


if __name__=='__main__':unittest.main()
