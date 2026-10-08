"""Local exact-byte JSON/JSONL EOF regressions; no remote or GPU qualification."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RAW5 = HERE.parent / 'job3304-first5-complete-originals-v3/raw'
ACCEPTED5 = '30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
ACTUAL = {
    'questions.json': '3698de193fbee993a0f1657581513aa2979bdc6498482da16e07fb65b9ce45ca',
    'paired-manifest.json': '9421c2b0c345d5aa682b86d42b1040de4a7d21aa12440e6c4026e5f4a7036206',
    'student-manifest.json': 'd682b7f6d069abac6fd22f349cdc84b8080dade0bf081b9f36deb743f7c2af5c',
    'teacher-rank-0.manifest.json': '33a3e00aedcb5ed707a1816831296a051c16970d0fe3b8ca3b3fe0ce04818839',
}


def load(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


c = load('current_collector_eof_regression', HERE / 'collect_outer26.py')
old = load('old_collector_eof_counterexample', HERE / 'historical-before-json-eof-repair-v1/collect_outer26.py')
a = load('unchanged_analyzer_eof_regression', HERE / 'analyze_observed26.py')
k = a.load(HERE.parent / 'paired-trajectory-cpu-analysis-candidate/analyze_outer.py', a.KERNEL_SHA,
           'unchanged_outer1_kernel_eof_regression')


def local_opened(path):
    """CPU fixture reader only: real FD/stats, local UID rather than production root."""
    path = Path(path)
    if path.resolve(strict=True) != path:
        raise ValueError('Fixture path is not physical')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
        raise ValueError('Fixture owner/type/link differs')
    handle = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb')
    if c.sig(os.fstat(handle.fileno())) != c.sig(before):
        handle.close()
        raise ValueError('Fixture open race')
    return before, handle


class DocumentEOF(unittest.TestCase):
    def inspect(self, raw, name='questions.json', implementation=c, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / name
            path.write_bytes(raw)
            path.chmod(0o444)
            wanted = expected or dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            with patch.object(implementation, 'opened', local_opened):
                return implementation.hash_member(path, wanted)

    def actual(self, name):
        manifest_path = RAW5 / 'full_state/global_step_5/manifest.json'
        raw = manifest_path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ACCEPTED5)
        manifest = json.loads(raw)
        self.assertEqual(manifest['files']['trajectory_outer_000001/' + name]['sha256'], ACTUAL[name])
        value = (RAW5 / 'trajectories/outer_000001' / name).read_bytes()
        self.assertEqual(hashlib.sha256(value).hexdigest(), ACTUAL[name])
        self.assertFalse(value.endswith(b'\n'))
        return value

    def test_actual_questions_no_lf_old_fails_new_passes(self):
        raw = self.actual('questions.json')
        with self.assertRaisesRegex(ValueError, 'Full member digest/bytes/EOF differs'):
            self.inspect(raw, implementation=old)
        self.assertEqual(self.inspect(raw)['sha256'], ACTUAL['questions.json'])

    def test_actual_other_documents_no_lf(self):
        for name in ('paired-manifest.json', 'student-manifest.json', 'teacher-rank-0.manifest.json'):
            with self.subTest(name=name):
                raw = self.actual(name)
                self.assertEqual(self.inspect(raw, name)['sha256'], ACTUAL[name])

    def test_actual_unchanged_analyzer_json_reader_accepts_no_lf(self):
        path = RAW5 / 'trajectories/outer_000001/questions.json'
        self.actual('questions.json')
        value, record = k.json_member(path, ACTUAL['questions.json'])
        self.assertEqual(len(value['questions']), 512)
        self.assertEqual(record['sha256'], ACTUAL['questions.json'])

    def test_whole_json_with_or_without_lf(self):
        for raw in (b'{"x":1}', b'{"x":1}\n', b'{"x":1}\n\n'):
            self.assertEqual(self.inspect(raw)['bytes'], len(raw))

    def test_extra_json_or_garbage_rejected(self):
        for raw in (b'{"x":1}{"x":2}', b'{"x":1}\n{}', b'{"x":1} trailing'):
            with self.assertRaises(json.JSONDecodeError):
                self.inspect(raw)

    def test_partial_json_rejected(self):
        for raw in (b'{"x":', b'{"x":1', b''):
            with self.assertRaises(json.JSONDecodeError):
                self.inspect(raw)

    def test_duplicate_or_nonfinite_json_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError):
                self.inspect(raw)

    def test_jsonl64_requires_lf(self):
        self.assertEqual(self.inspect(b'{}\n' * 64, 'teacher-rank-0.jsonl')['bytes'], 192)
        with self.assertRaisesRegex(ValueError, 'JSONL EOF/row count differs'):
            self.inspect(b'{}\n' * 63 + b'{}', 'teacher-rank-0.jsonl')

    def test_jsonl64_exact_count(self):
        for count in (63, 65):
            with self.assertRaisesRegex(ValueError, 'JSONL EOF/row count differs'):
                self.inspect(b'{}\n' * count, 'teacher-rank-0.jsonl')

    def test_student_jsonl2560_requires_lf_and_count(self):
        self.inspect(b'{}\n' * 2560, 'student.jsonl')
        for raw in (b'{}\n' * 2559, b'{}\n' * 2559 + b'{}', b'{}\n' * 2561):
            with self.assertRaisesRegex(ValueError, 'JSONL EOF/row count differs'):
                self.inspect(raw, 'student.jsonl')

    def test_wrong_sha_or_length_rejected_before_parse(self):
        for expected in (dict(sha256='0' * 64, bytes=7),
                         dict(sha256=hashlib.sha256(b'{"x":1}').hexdigest(), bytes=8)):
            with self.assertRaisesRegex(ValueError, 'Full member digest/bytes differs'):
                self.inspect(b'{"x":1}', expected=expected)


if __name__ == '__main__':
    unittest.main()
