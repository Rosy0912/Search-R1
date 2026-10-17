import hashlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from .return_copy import copy_file
from .pj_return_dispatch_recovery import validate_retirement


class CopyTests(unittest.TestCase):
    def exercise(self, bad_sha=False, always_fail=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = b'full-model-adam-rng' * 1000
            (root / 'payload').write_bytes(data)
            script = root / 'reader.py'
            script.write_text("import pathlib,sys\np=pathlib.Path(__file__).parent\no=int(sys.argv[1].split()[-1])\nb=(p/'payload').read_bytes()\n(p/'offsets').open('a').write(str(o)+'\\n')\n" +
                              ("sys.exit(255)\n" if always_fail else "sys.stdout.buffer.write(b[o:o+731] if o==0 else b[o:])\nsys.exit(255 if o==0 else 0)\n"))
            item = dict(path='/source/rank-00.pt', bytes=len(data),
                        sha256='0' * 64 if bad_sha else hashlib.sha256(data).hexdigest())
            events = []
            args = ('generation-2', item, root, [sys.executable, str(script)],
                    lambda k, **v: events.append(v), threading.Event(), time.monotonic() + 30)
            if bad_sha or always_fail:
                with self.assertRaises(ValueError if bad_sha else RuntimeError):
                    copy_file(*args, max_attempts=2, retry_delay=0)
                self.assertFalse((root / 'rank-00.pt').exists())
                self.assertTrue((root / 'rank-00.pt.part').exists())
            else:
                result = copy_file(*args, max_attempts=2, retry_delay=0)
                self.assertEqual(result['state'], 'ACTUAL_SHA_VERIFIED')
                self.assertEqual((root / 'rank-00.pt').read_bytes(), data)
                self.assertEqual((root / 'offsets').read_text().splitlines(), ['0', '731'])
                copy_file(*args, max_attempts=2, retry_delay=0)
                self.assertEqual((root / 'offsets').read_text().splitlines(), ['0', '731'])
    def test_connection_failure_resumes_exact_offset_and_existing_final(self):
        self.exercise()
    def test_sha_failure_preserves_partial_and_never_renames(self):
        self.exercise(bad_sha=True)
    def test_connection_retries_are_bounded(self):
        self.exercise(always_fail=True)
    def test_dispatch_recovery_requires_empty_retired_attempt(self):
        proof = dict(status='INSPECTED_NETWORK_FAILURE_OLD_TRANSPORT_RETIRED', exact_scheduler_names_absent=True)
        old = dict(pid=4131830, status='NEEDS_INSPECTION_NO_AUTOMATIC_REPLAY', groups={})
        validate_retirement(proof, old, False, False)
        for p, o, live, submitted in [({}, old, False, False), (proof, old, True, False),
                                      (proof, old, False, True), (proof, dict(old, groups={'budget-5': 'job'}), False, False)]:
            with self.assertRaises(ValueError):
                validate_retirement(p, o, live, submitted)


if __name__ == '__main__':
    unittest.main()
