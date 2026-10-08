"""CPU checks for actual frozen scorer import and answer nesting regressions."""
import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from .checkpoint import exclusive_json
    from .scoring import verify_scoring, QA_RELATIVE, INIT_RELATIVE, QA_SHA256, EMPTY_SHA256
except ImportError:
    from checkpoint import exclusive_json
    from scoring import verify_scoring, QA_RELATIVE, INIT_RELATIVE, QA_SHA256, EMPTY_SHA256


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.old_modules={k:sys.modules.pop(k) for k in ('search_r1.qa_protocol','search_r1') if k in sys.modules}
        self.old_path=list(sys.path)

    def tearDown(self):
        for name in ('search_r1.qa_protocol','search_r1'): sys.modules.pop(name,None)
        sys.modules.update(self.old_modules);sys.path[:]=self.old_path

    def fixture(self, root):
        # The repository scorer and frozen upstream scorer have the same accepted bytes.
        runtime=os.environ.get('OPSD_ONLINE_TEST_RUNTIME')
        source=(Path(runtime)/QA_RELATIVE if runtime else
                Path(__file__).resolve().parents[4]/'search_r1'/'qa_protocol.py')
        target=root/QA_RELATIVE;target.parent.mkdir(parents=True)
        shutil.copyfile(source,target);(root/INIT_RELATIVE).write_bytes(b'')
        manifest=exclusive_json(root/'runtime-manifest.json',{QA_RELATIVE:QA_SHA256,INIT_RELATIVE:EMPTY_SHA256})
        return dict(paths={'runtime':str(root)},runtime_manifest=manifest)

    def test_nested_upstream_import_and_original_strict_score(self):
        with tempfile.TemporaryDirectory() as directory:
            config=self.fixture(Path(directory));proof=verify_scoring(config)
            self.assertTrue(proof['accepted'])
            qa=importlib.import_module('search_r1.qa_protocol')
            self.assertEqual(qa.extract_final_answer(['<think>reason</think><answer>target</answer>']),'target')
            for text in ('<search><answer>target</answer></search>',
                         '<think><answer>target</answer>',
                         '<information><answer>target</answer></information>',
                         '<answer>target</answer><search>query</search>'):
                self.assertIsNone(qa.extract_final_answer([text]))
            self.assertIsNone(qa.extract_final_answer(['<answer>target</answer>','<search>query</search>']))

    def test_changed_scorer_file_is_refused_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=self.fixture(root)
            (root/QA_RELATIVE).write_text('raise RuntimeError("must not import changed scorer")')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):verify_scoring(config)

    def test_cached_other_scorer_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config=self.fixture(Path(directory))
            class Other:
                __file__='/other/scorer.py'
            with patch('importlib.import_module',return_value=Other()),self.assertRaisesRegex(ValueError,'Actual imported'):
                verify_scoring(config)


if __name__=='__main__': unittest.main()
