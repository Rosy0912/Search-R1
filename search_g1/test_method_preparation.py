import json
import tempfile
import time
import unittest
from pathlib import Path
from reproduction.common8.prepare_run import write_method
from reproduction.common8.formal_inline_safety import verify_method

class MethodPreparationTests(unittest.TestCase):
    def test_method_is_frozen_before_start_and_bound_to_actual_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source={'runtime.py':'a'*64};commit='b'*40
            method=write_method(Path(temp),commit,source)
            run={'source_commit':commit,'source_sha256':source,'started_unix':time.time()}
            verify_method(method,run)
            self.assertEqual(json.loads((Path(temp)/'preparation/formal-inline-method.json').read_text()),method)
            self.assertFalse(method['healthy_checkpoint_accepted'])
            with self.assertRaises(FileExistsError):write_method(Path(temp),commit,source)
            with self.assertRaises(AssertionError):verify_method(method,dict(run,source_commit='c'*40))

if __name__=='__main__':unittest.main()
