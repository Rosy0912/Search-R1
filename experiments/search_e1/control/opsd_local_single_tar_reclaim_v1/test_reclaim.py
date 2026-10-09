"""One tiny fixture; only temporary files can be removed."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reclaim as r


class SingleFileBoundary(unittest.TestCase):
    def test_incomplete_identity_open_and_single_unlink(self):
        p = r.primitives()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); target = root/'original.tar'; target.write_bytes(b'GNU bytes')
            keep = root/'archive.tar.xz'; keep.write_bytes(b'preserved xz')
            row = p.fileproof(target); compressed = p.fileproof(keep)
            partial = dict(compressed, path=str(root/'archive.tar.xz.partial'))
            complete = dict(schema='local-accepted-exact-gnu-tar-xz-v1', accepted=True,
                actual_child_returncode=0, original=row, compressed=compressed,
                original_tar_retained=True, no_deletion=True)
            worker = dict(schema='local-exact-tar-xz-worker-v1', accepted=True, actual_pid=7, uid=501,
                original_tar_retained=True, member_hashes_repeated=False,
                compression=dict(original_sha256=row['sha256'], original_bytes=row['bytes'], input=row),
                recovery=dict(decoded_sha256=row['sha256'], decoded_bytes=row['bytes'], check='CRC64',
                    single_stream=True, independent_disk_read=True, no_extracted_payload_written=True,
                    compressed=partial))
            terminal = dict(actual_returncode=0, status_not_inferred_from_absence=True, actual_pid=7)
            completion_ref = dict(path=str(r.COMPLETION), sha256='2'*64)
            parent_start = dict(actual_pid=41057, uid=501)
            parent = dict(schema='local-recompress-parent-terminal-root-observation-v1',
                actual_pid=41057, uid=501, actual_returncode=0, status_not_inferred_from_absence=True,
                start=dict(path=str(r.PARENT_START), sha256=r.PARENT_START_SHA), completion=completion_ref)
            r.validate_parent_terminal(parent, parent_start, completion_ref)
            with self.assertRaises(RuntimeError):
                r.validate_parent_terminal(dict(parent, actual_returncode=2), parent_start, completion_ref)
            with patch.multiple(r, ARCHIVE=target, XZ=keep, XZ_DIR=root,
                                RAW_SHA=row['sha256'], RAW_BYTES=row['bytes']):
                r.validate_recovery(complete, worker, terminal)
                for bad in ['incomplete', 'wrong_sha']:
                    c = copy.deepcopy(complete); w = copy.deepcopy(worker)
                    if bad == 'incomplete': c['accepted'] = False
                    else: w['recovery']['decoded_sha256'] = '0'*64
                    with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                        r.validate_recovery(c, w, terminal)
                intended = dict(plan=dict(path='fixture', sha256='1'*64), file=row)
                pending = root/'pending.json'
                with patch.object(p, 'no_open_files', side_effect=RuntimeError('open')):
                    with self.assertRaises(RuntimeError):
                        r.unlink_one(p, row, pending, intended, [compressed])
                self.assertFalse(pending.exists()); self.assertTrue(target.exists())
                changed = dict(row, inode=row['inode']+1)
                with patch.object(p, 'no_open_files', return_value=None), self.assertRaises(RuntimeError):
                    r.unlink_one(p, changed, pending, intended, [compressed])
                self.assertTrue(target.exists())
                with patch.object(p, 'no_open_files', return_value=None):
                    outcome = r.unlink_one(p, row, pending, intended, [compressed])
                self.assertEqual(outcome, 'exact_planned_original_tar_unlinked')
                self.assertFalse(target.exists()); self.assertTrue(keep.exists())
                self.assertEqual(p.checked_json(pending, p.fileproof(pending)['sha256'])[0], intended)
                with patch.object(p, 'no_open_files', return_value=None):
                    self.assertEqual(r.unlink_one(p, row, pending, intended, [compressed]),
                                     'absent_after_durable_pending_actual_unlink_not_inferred')


if __name__ == '__main__':
    unittest.main()
