"""Only the two c5a review deltas; never production main or actual final evidence."""
import copy
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import test_raw_replay as old

c=old.c


def save_npz(path,arrays):
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)


def replace_member(path,key,data):
    with zipfile.ZipFile(path) as archive:members={name:archive.read(name) for name in archive.namelist()}
    members[key+'.npy']=data
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in members.items():archive.writestr(name,raw)


def header(shape,descr='<i8',body=b''):
    stream=io.BytesIO();np.lib.format.write_array_header_1_0(stream,dict(shape=shape,descr=descr,fortran_order=False))
    return stream.getvalue()+body


class ReviewDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.actual_loop_fixture=old.fixture(1,3)

    def test_source_real_small_npz_and_scores_still_match(self):
        records,rows,tools,arrays,calls=copy.deepcopy(self.actual_loop_fixture)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'actual-small.npz';save_npz(path,arrays)
            result=c.npz_arrays(c.Originals(os.getuid()),str(path),hashlib.sha256(path.read_bytes()).hexdigest())
            for name in c.SCORING_KEYS:self.assertTrue(np.array_equal(result[name],arrays[name]))
            self.assertEqual(old.legacy.score_arrays(result,records,old.TOK),old.legacy.score_arrays(arrays,records,old.TOK))
            c.extra_raw_checks(records,rows,tools,result,old.TOK,'official_base',old.INVALID)
            old.final.validate_raw_batch(records,rows,tools,old.TOK)

    def test_tiny_giant_shape_object_and_wrong_dim_header_rejected_before_load(self):
        arrays=self.actual_loop_fixture[3]
        examples=[header((64,10**12)),header((1,10),descr='|O'),header((1,1,1)),header((True,10)),header((65,1))]
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder).resolve()
            for index,bad in enumerate(examples):
                path=parent/('bad-'+str(index)+'.npz');save_npz(path,arrays);replace_member(path,'responses',bad)
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                with mock.patch.object(c.np,'load',side_effect=AssertionError('NPZ allocation must not begin')) as loader:
                    with self.assertRaises(ValueError):c.npz_arrays(c.Originals(os.getuid()),str(path),sha)
                    loader.assert_not_called()
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),sha)

    def test_valid_shape_missing_body_and_cross_field_offsets_rejected_before_load(self):
        arrays=self.actual_loop_fixture[3]
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder).resolve()
            cases=[('short',arrays,header(arrays['responses'].shape)),
                   ('offset',dict(arrays,position_ids=arrays['position_ids'][:,:-1].copy()),None)]
            for name,values,bad in cases:
                path=parent/(name+'.npz');save_npz(path,values)
                if bad is not None:replace_member(path,'responses',bad)
                with mock.patch.object(c.np,'load',side_effect=AssertionError('NPZ allocation must not begin')) as loader:
                    with self.assertRaises(ValueError):c.npz_arrays(c.Originals(os.getuid()),str(path),hashlib.sha256(path.read_bytes()).hexdigest())
                    loader.assert_not_called()

    def test_partial_scores_are_fsynced_before_original_error_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder).resolve();path=parent/'partial.jsonl';synced=[];real_fsync=os.fsync
            def spy(fd):
                value=os.fstat(fd);synced.append((value.st_dev,value.st_ino,stat.S_ISDIR(value.st_mode)));return real_fsync(fd)
            with mock.patch.object(c.os,'fsync',side_effect=spy):
                with self.assertRaisesRegex(RuntimeError,'exact fixture failure'):
                    with c.derived_stream(path) as stream:
                        stream.write(b'{"scope":"CPU_FIXTURE_PARTIAL"}\n');raise RuntimeError('exact fixture failure')
            self.assertEqual(path.read_bytes(),b'{"scope":"CPU_FIXTURE_PARTIAL"}\n')
            f=path.stat();d=parent.stat();self.assertEqual(synced,[(f.st_dev,f.st_ino,False),(d.st_dev,d.st_ino,True)])
            with self.assertRaises(FileExistsError):
                with c.derived_stream(path):pass

    def test_success_scores_file_and_parent_sync_on_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder).resolve();path=parent/'scores.jsonl';seen=[];real=os.fsync
            with mock.patch.object(c.os,'fsync',side_effect=lambda fd:(seen.append(os.fstat(fd).st_ino),real(fd))[1]):
                with c.derived_stream(path) as stream:stream.write(b'{"fixture":true}\n')
            self.assertEqual(seen,[path.stat().st_ino,parent.stat().st_ino]);self.assertFalse(old.torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
