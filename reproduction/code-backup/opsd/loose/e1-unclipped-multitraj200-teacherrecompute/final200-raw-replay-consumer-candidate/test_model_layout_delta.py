"""Two synthetic layout-only tests; no model/checkpoint qualification or main."""
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent


def source_module(path, name):
    raw = path.read_bytes()
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    return value


c = source_module(HERE / 'replay_final200_raw.py', 'layout_delta_current')
old = source_module(HERE / 'retained-705-v2' / 'replay_final200_raw.py', 'layout_delta_retained705')


def fixture(parent, role='actual_hf200'):
    directory = parent / 'model'; directory.mkdir()
    values = {'config.json': b'{"synthetic_layout_only":true}', 'tokenizer.json': b'{}',
              'tokenizer_config.json': b'{}', 'model-00001-of-00001.safetensors': b'NOT_REAL_MODEL_WEIGHTS'}
    for name, raw in values.items():
        (directory / name).write_bytes(raw)
    pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in values.items()}
    return directory, dict(model_role=role, model_directory=str(directory), model_files=pins)


class ModelLayoutDeltaTests(unittest.TestCase):
    def test_actual_hf_layout_only_keeps_training_state_separate_and_regular_files_pinned(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, provenance = fixture(Path(folder).resolve())
            state = directory / 'training_state'; state.mkdir()
            state_file = state / 'rank0.pt'; state_file.write_bytes(b'SYNTHETIC_NOT_CHECKPOINT')
            state_before = state_file.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Actual complete model membership differs'):
                old.verify_model(old.Originals(os.getuid()), provenance)
            reader = c.Originals(os.getuid()); c.verify_model(reader, provenance)
            self.assertEqual(set(reader.records), {str(directory / name) for name in provenance['model_files']})
            self.assertNotIn(str(state_file), reader.records)
            self.assertEqual(state_file.read_bytes(), state_before)
            (directory / 'config.json').write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'full SHA differs'):
                c.verify_model(c.Originals(os.getuid()), provenance)

    def test_unknown_symlink_special_wrong_owner_and_official_base_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder).resolve()
            for index, case in enumerate(('extra_directory', 'symlink_state', 'symlink_file', 'fifo', 'wrong_owner', 'official_base')):
                with self.subTest(case=case):
                    item = parent / str(index); item.mkdir()
                    directory, provenance = fixture(item, 'official_base' if case == 'official_base' else 'actual_hf200')
                    if case == 'symlink_state':
                        target = item / 'real_state'; target.mkdir(); (directory / 'training_state').symlink_to(target, target_is_directory=True)
                    else:
                        (directory / 'training_state').mkdir()
                    if case == 'extra_directory': (directory / 'unknown').mkdir()
                    elif case == 'symlink_file': (directory / 'unknown.json').symlink_to(directory / 'config.json')
                    elif case == 'fifo': os.mkfifo(directory / 'unknown.fifo')
                    uid = os.getuid() + 1 if case == 'wrong_owner' else os.getuid()
                    with self.assertRaises(ValueError):
                        c.verify_model(c.Originals(uid), provenance)


if __name__ == '__main__':
    unittest.main(verbosity=2)
