import hashlib,sys,tempfile,unittest
from pathlib import Path
from .stream_copy import transfer
class StreamCopyTests(unittest.TestCase):
 def test_exact_bytes_without_intermediate_archive(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'target'
   transfer([sys.executable,'-c','import sys;sys.stdout.buffer.write(bytes(range(256))*8192)'],[sys.executable,'-c','import sys;open(sys.argv[1],"xb").write(sys.stdin.buffer.read())',str(p)],10)
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),hashlib.sha256(bytes(range(256))*8192).hexdigest())
   self.assertEqual(list(Path(d).iterdir()),[p])
 def test_source_failure_not_reported_success(self):
  with self.assertRaises(RuntimeError):transfer([sys.executable,'-c','raise SystemExit(9)'],[sys.executable,'-c','import sys;sys.stdin.buffer.read()'],10)
 def test_target_failure_does_not_replay(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'partial'
   with self.assertRaises(RuntimeError):transfer([sys.executable,'-c','print("data")'],[sys.executable,'-c','import sys;open(sys.argv[1],"xb").write(b"partial");raise SystemExit(7)',str(p)],10)
   self.assertEqual(p.read_bytes(),b'partial')
if __name__=='__main__':unittest.main()
