"""Inspect actual generated remote code dependencies without executing collection."""
from pathlib import Path
import builtins
import dis
import hashlib
import json
import types
import unittest

H = Path(__file__).resolve().parent


def module(path):
    raw = path.read_bytes()
    m = types.ModuleType('candidate_generated_namespace')
    m.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), m.__dict__)
    return m


def codes(code):
    yield code
    for child in code.co_consts:
        if isinstance(child, types.CodeType):
            yield from codes(child)


def unresolved(m):
    namespace = {}
    # Exact library bytes and exact generated constants; no altered function body.
    exec(compile(m.reader_constants() + m.REMOTE_LIBRARY, 'generated-library', 'exec'), namespace)
    actual_program = compile(m.remote_program({}), 'actual-generated-remote-program', 'exec')
    available = set(namespace) | set(vars(builtins))
    return {i.argval for code in codes(actual_program) for i in dis.get_instructions(code)
            if i.opname == 'LOAD_GLOBAL' and i.argval not in available}


class NamespaceDelta(unittest.TestCase):
    def test_actual_failed_198a_generated_namespace_missing_only_trajectory_constant(self):
        old = module(H / 'collect_first5_originals_candidate-v5-198a-missing-generated-constant.original.py')
        self.assertEqual(unresolved(old), {'TRAJECTORY_NAMES'})
        failed = json.loads((H / 'job3304-first5-complete-originals-v2/transport-exit.original.json').read_bytes())
        self.assertEqual(failed['returncode'], 1)
        self.assertIn("NameError: name 'TRAJECTORY_NAMES' is not defined", (H / 'job3304-first5-complete-originals-v2/transport.stderr.original').read_text())

    def test_new_actual_generated_program_recursive_globals_are_closed(self):
        current = module(H / 'collect_first5_originals_candidate.py')
        self.assertEqual(unresolved(current), set())
        namespace = {}
        exec(compile(current.reader_constants() + current.REMOTE_LIBRARY, 'generated-library', 'exec'), namespace)
        self.assertEqual(namespace['TRAJECTORY_NAMES'], current.TRAJECTORY_NAMES)
        self.assertEqual(len(namespace['TRAJECTORY_NAMES']), 20)
        self.assertEqual(len(namespace['CHECKPOINT_MAP']), 141)

    def test_only_new_constant_binding_changed(self):
        current = (H / 'collect_first5_originals_candidate.py').read_bytes()
        prior = (H / 'collect_first5_originals_candidate-v5-198a-missing-generated-constant.original.py').read_bytes()
        self.assertEqual(hashlib.sha256(prior).hexdigest(), '198af97f5541a29506588daea3975f105e264e95a81ceaa10f8f35e812a67c71')
        self.assertEqual(current.replace(b"'OPTIONAL','TRAJECTORY_NAMES','CHECKPOINT_MAP'", b"'OPTIONAL','CHECKPOINT_MAP'"), prior)


if __name__ == '__main__':
    unittest.main(verbosity=2)
