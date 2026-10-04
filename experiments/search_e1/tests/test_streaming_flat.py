import sys,unittest
from pathlib import Path
import numpy as np,torch,faiss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
from torch_streaming_flat import TorchStreamingFlatIP
class StreamingTests(unittest.TestCase):
 def test_complete_scan_and_partial_tiles(self):
  rng=np.random.default_rng(7);x=rng.normal(size=(137,31)).astype('float32');q=rng.normal(size=(69,31)).astype('float32')
  idx=faiss.IndexFlatIP(31);idx.add(x)
  for rows in [3,17,32768]:
   backend=TorchStreamingFlatIP(idx,'cpu',vector_rows=rows,query_micro=13)
   cs,ci=idx.search(q,4);s,i=backend.search(q,4)
   np.testing.assert_array_equal(ci,i);np.testing.assert_allclose(cs,s,atol=1e-5,rtol=0)
   self.assertEqual(backend.search(q[:0],4)[0].shape,(0,4))
   with self.assertRaises(ValueError):backend.search(np.full((2,31),np.nan),4)
   with self.assertRaises(ValueError):backend.search(q,138)
if __name__=='__main__':unittest.main()
