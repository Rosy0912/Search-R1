from pathlib import Path
import os
import tempfile
import unittest
from .contracts import sha
from .model_view import TOKENIZER_FILES,create_view,validate_view


class View(unittest.TestCase):
    def test_export_only_optional_tokenizer_files_are_not_carried_forward(self):
        from .model_view import view_files
        p=dict(model_files={n:'export' for n in TOKENIZER_FILES},
            canonical_tokenizer_files={n:'training' for n in TOKENIZER_FILES-{'added_tokens.json','special_tokens_map.json'}})
        self.assertNotIn('added_tokens.json',view_files(p))
        self.assertNotIn('special_tokens_map.json',view_files(p))

    def test_weights_remain_exact_and_training_tokenizer_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);canonical=root/'canonical';weights=root/'weights'
            canonical.mkdir();weights.mkdir()
            for name in TOKENIZER_FILES:
                (canonical/name).write_text('training '+name)
                (weights/name).write_text('serialized export '+name)
            (weights/'weight.safetensors').write_bytes(b'learned weights')
            plan=dict(canonical_tokenizer=str(canonical),model=str(weights),inference_model=str(root/'view'),
                model_files={p.name:sha(p) for p in weights.iterdir()},
                canonical_tokenizer_files={p.name:sha(p) for p in canonical.iterdir()})
            expected=plan['canonical_tokenizer_files']['tokenizer.json']
            before={p.name:(p.stat().st_ino,p.stat().st_ctime_ns,p.stat().st_nlink,sha(p))
                    for p in canonical.iterdir()}
            create_view(plan,expected)
            after={p.name:(p.stat().st_ino,p.stat().st_ctime_ns,p.stat().st_nlink,sha(p))
                   for p in canonical.iterdir()}
            self.assertEqual(before,after)
            for name in TOKENIZER_FILES:
                self.assertFalse(os.path.samefile(root/'view'/name,canonical/name))
            self.assertTrue(os.path.samefile(root/'view/weight.safetensors',weights/'weight.safetensors'))
            self.assertEqual((root/'view/weight.safetensors').read_bytes(),b'learned weights')
            self.assertEqual((weights/'tokenizer.json').read_text(),'serialized export tokenizer.json')
            with self.assertRaises(FileExistsError):create_view(plan,expected)
            with self.assertRaises(ValueError):validate_view(plan,'wrong training tokenizer hash')
            (root/'view/weight.safetensors').unlink();(root/'view/weight.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_view(plan,expected)

if __name__=='__main__':unittest.main()
