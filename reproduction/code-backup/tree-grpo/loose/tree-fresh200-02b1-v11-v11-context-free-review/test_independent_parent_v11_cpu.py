"""Narrow independent V11 parent authority checks using only temporary tiny bytes."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import calibration_contract as contract
import test_tree_behavior_mu_data_cpu as fixture_module
from test_tree_raw_tokens_cpu import Tokenizer
from search_r1.baselines import tree_raw_tokens as raw


class IndependentParentAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.BehaviorMuDataTests()
        self.fixture.setUp()
        self.path = Path(self.fixture.tmp.name).resolve()
        for name in contract.PARENT_TOKENIZER_FILES:
            if not (self.path / name).exists():
                (self.path / name).write_bytes(b'{"tiny_independent_file":true}')
        self.package = {
            'official_model_path': str(self.path),
            'model_identity': {'files': {
                name: hashlib.sha256((self.path / name).read_bytes()).hexdigest()
                for name in contract.PARENT_TOKENIZER_FILES
            }},
        }
        raw._VALIDATION_TOKENIZERS.clear()
        contract._PARENT_VALIDATION_TOKENIZERS.clear()
        self.calls = []
        self.loaded = self.fixture.tok
        utils = ModuleType('verl.utils')

        def loader(path, **kwargs):
            self.calls.append((path, kwargs))
            return self.loaded

        utils.hf_tokenizer = loader
        self.patches = [
            patch.dict(sys.modules, {'verl.utils': utils}),
            patch.object(contract, 'OFFICIAL_TOKENIZER_PATH', self.path),
            patch.object(contract, 'OFFICIAL_TOKENIZER_SHA', self.package['model_identity']['files']['tokenizer.json']),
            patch.object(contract.socket, 'gethostname', return_value='h100-9'),
        ]
        for current in self.patches:
            current.start()

    def tearDown(self):
        for current in reversed(self.patches):
            current.stop()
        contract._PARENT_VALIDATION_TOKENIZERS.clear()
        raw._VALIDATION_TOKENIZERS.clear()
        self.fixture.tearDown()

    def test_all_four_pinned_file_mutations_fail_before_cached_reuse(self):
        actual = contract.initialize_parent_validation_tokenizer(self.package)
        self.assertIs(actual, self.fixture.tok)
        for name in contract.PARENT_TOKENIZER_FILES:
            path = self.path / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b' altered')
                with self.assertRaisesRegex(ValueError, 'bytes differ'):
                    contract.initialize_parent_validation_tokenizer(self.package)
                self.assertEqual(len(self.calls), 1)
            finally:
                path.write_bytes(original)
        self.assertIs(contract.initialize_parent_validation_tokenizer(self.package), actual)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], (str(self.path), {'local_files_only': True, 'trust_remote_code': False}))

    def test_same_bytes_symlink_refused_before_loader(self):
        path = self.path / 'tokenizer_config.json'
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / path.name
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'regular file, not symlink'):
                contract.initialize_parent_validation_tokenizer(self.package)
            self.assertEqual(self.calls, [])

    def test_wrong_loaded_actual_path_refused_despite_same_tokenizer_sha(self):
        with tempfile.TemporaryDirectory() as other:
            other_path = Path(other)
            (other_path / 'tokenizer.json').write_bytes((self.path / 'tokenizer.json').read_bytes())
            self.loaded = Tokenizer(other_path)
            with self.assertRaisesRegex(ValueError, 'actual loaded tokenizer authority differs'):
                contract.initialize_parent_validation_tokenizer(self.package)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(contract._PARENT_VALIDATION_TOKENIZERS, {})


if __name__ == '__main__':
    unittest.main()
