"""Byte-copy/relocation CPU tests; fixtures are text, no actual model states."""
import copy
from pathlib import Path
import tempfile
import unittest

from reproduction.hpc3_qprobe.archive_completed_pair import copy_checked, destination, relocate, file_records
from search_g1.qprobe_features import sha_file


class ArchiveTests(unittest.TestCase):
    def test_two_independent_copies_and_actual_receipt_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            path=source/'cp/model.txt';path.parent.mkdir();path.write_bytes(b'synthetic-file-content')
            st=path.stat();row=dict(path=str(path),relative_path='model.txt',inode=st.st_ino,
                bytes=st.st_size,mtime_ns=st.st_mtime_ns,full_read_sha256=sha_file(path))
            receipt=dict(source_base=str(source),checkpoint=str(path.parent),question_probe_state_path=str(path),
                head_path=str(path),files=[row],reward_files=[copy.deepcopy(row)],actual_cuda_restore_accepted=False)
            for label in ('archive','target'):
                target=root/label;target.mkdir();dest=destination(path,source,target)
                copy_checked(path,dest,row['full_read_sha256'],row['bytes'])
                rebound=relocate(receipt,target)
                self.assertEqual(rebound['files'][0]['path'],str(dest))
                self.assertNotEqual(rebound['files'][0]['inode'],row['inode'])
                self.assertEqual(rebound['files'][0]['full_read_sha256'],row['full_read_sha256'])
                self.assertEqual(dest.read_bytes(),path.read_bytes())
                self.assertEqual(len(list(file_records(rebound))),2)
                self.assertFalse(rebound['actual_cuda_restore_accepted'])
            self.assertEqual(receipt['files'][0],row)

    def test_source_tamper_existing_copy_and_target_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'source.txt';p.write_bytes(b'original')
            sha=sha_file(p);size=p.stat().st_size
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):copy_checked(p,root/'bad.txt',sha,size)
            self.assertFalse((root/'bad.txt').exists())
            p.write_bytes(b'original');copy_checked(p,root/'copy.txt',sha,size)
            with self.assertRaises(ValueError):copy_checked(p,root/'copy.txt',sha,size)

    def test_external_encoder_shards_keep_one_parent_and_safe_names(self):
        source=Path('/data/own/source');target=Path('/data/own/archive')
        a=destination('/data/shared/model/index.json',source,target)
        b=destination('/data/shared/model/shard.safetensors',source,target)
        self.assertEqual(a.parent,b.parent);self.assertTrue(a.is_relative_to(target))
        self.assertNotEqual(a.parent,destination('/data/other/model/index.json',source,target).parent)


if __name__=='__main__':unittest.main()
