"""Real CPU spawn/join lifecycle and proven flat serialization tests; no CUDA."""
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_phase_proxy import SpawnedRuntime, process_identity, join_owned_process
from flat_memmap import derive_flat_layout, validated_prefix_mmap, verify_unchanged


def cpu_worker(connection, configuration):
    connection.send(dict(status='READY',identity=process_identity()))
    while True:
        request = connection.recv()
        if request['op']=='release':
            connection.send(dict(status='RELEASED')); connection.close(); return
        if request['op']=='fault':
            connection.send(dict(status='FAULT',error='scripted programming fault'))
            connection.close(); return
        connection.send(dict(status='OK',pid=os.getpid(),value=request['value']))


def serialize(d,count):
    return b'IxFI'+struct.pack('<IQQQ',d,count,17,count*d)+np.arange(count*d,dtype='<f4').tobytes()


class Lifecycle(unittest.TestCase):
    def test_join_never_blocks_on_live_native_threads(self):
        class NativeThreadPending:
            def __init__(self): self.polls=0; self.joined=False
            def is_alive(self):
                self.polls+=1
                return self.polls<3
            def join(self, timeout):
                if self.polls<3:raise AssertionError('Live join may block in native teardown')
                self.joined=True
        process=NativeThreadPending()
        join_owned_process(process,1)
        self.assertTrue(process.joined)

    def test_lazy_reuse_actual_join_then_new_pid(self):
        runtime=SpawnedRuntime(cpu_worker,{},timeout=5)
        self.assertIsNone(runtime.process)
        first=runtime.request(dict(op='scan',value=1))
        second=runtime.request(dict(op='scan',value=2))
        self.assertEqual(first['pid'],second['pid'])
        process=runtime.process
        receipt=runtime.release()
        self.assertEqual(receipt['exitcode'],0)
        self.assertTrue(receipt['graceful'])
        self.assertFalse(process.is_alive())
        self.assertIsNone(runtime.process)
        third=runtime.request(dict(op='scan',value=3))
        self.assertNotEqual(first['pid'],third['pid'])
        self.assertEqual(runtime.release()['exitcode'],0)

    def test_fault_cleanup_does_not_look_like_successful_scan(self):
        runtime=SpawnedRuntime(cpu_worker,{},timeout=5)
        runtime.ensure_started(); process=runtime.process
        with self.assertRaisesRegex(RuntimeError,'scripted programming fault'):
            runtime.request(dict(op='fault'))
        self.assertFalse(process.is_alive()); self.assertIsNone(runtime.process)

    def test_absent_release_no_child_started(self):
        runtime=SpawnedRuntime(cpu_worker,{},timeout=5)
        self.assertEqual(runtime.release()['status'],'NO_GPU_CHILD')
        self.assertIsNone(runtime.process)


class Memmap(unittest.TestCase):
    def test_proven_layout_readonly_no_vector_hash_and_metadata(self):
        layout=derive_flat_layout(serialize,7)
        with tempfile.TemporaryDirectory() as base:
            path=Path(base)/'flat.index'; path.write_bytes(serialize(7,9))
            host,proof=validated_prefix_mmap(path,layout,hash_selected_rows=0)
            self.assertFalse(host.flags.writeable)
            self.assertEqual(host.shape,(9,7))
            self.assertEqual(host[8,6],62)
            self.assertEqual(proof['selected_vector_hashed_bytes'],0)
            self.assertIsNone(proof['selected_vector_prefix_sha256'])
            self.assertFalse(proof['full_index_file_sha_checked'])
            verify_unchanged(proof)
            path.write_bytes(path.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'metadata changed'): verify_unchanged(proof)

    def test_refuses_unproven_header_or_size(self):
        layout=derive_flat_layout(serialize,7)
        with tempfile.TemporaryDirectory() as base:
            path=Path(base)/'flat.index'
            for value in (serialize(7,9)[:-1],b'BAD!'+serialize(7,9)[4:],serialize(8,9)):
                path.write_bytes(value)
                with self.assertRaises(ValueError): validated_prefix_mmap(path,layout,hash_selected_rows=0)

    def test_rejects_unknown_probe_variability(self):
        def varying(d,count):
            data=bytearray(serialize(d,count)); data[20]=count+20; return data
        with self.assertRaises(ValueError): derive_flat_layout(varying,7)


if __name__=='__main__': unittest.main()
