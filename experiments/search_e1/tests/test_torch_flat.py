import sys,unittest
from pathlib import Path
import numpy as np,faiss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
from torch_flat import TorchShardedFlatIP
class FlatTests(unittest.TestCase):
 def test_real_faiss_global_ids_and_bounded_query_chunks(self):
  rng=np.random.default_rng(42);x=rng.normal(size=(4097,32)).astype(np.float32);q=rng.normal(size=(137,32)).astype(np.float32)
  index=faiss.IndexFlatIP(32);index.add(x);reference_scores,reference_ids=index.search(q,7)
  resident=TorchShardedFlatIP(index,['cpu']*4,query_micro=16)
  scores,ids=resident.search(q,7)
  np.testing.assert_array_equal(ids,reference_ids);np.testing.assert_allclose(scores,reference_scores,rtol=1e-5,atol=1e-5)
  self.assertEqual(resident.bounds[0][0],0);self.assertEqual(resident.bounds[-1][1],4097)
  self.assertTrue(all(a[1]==b[0] for a,b in zip(resident.bounds,resident.bounds[1:])))
  resident.pool.shutdown()
 def test_ties_remain_valid_topk_without_bitwise_id_claim(self):
  x=np.ones((17,4),np.float32);index=faiss.IndexFlatIP(4);index.add(x)
  resident=TorchShardedFlatIP(index,['cpu']*4);scores,ids=resident.search(np.ones((1,4),np.float32),3)
  np.testing.assert_array_equal(scores,np.full((1,3),4,np.float32));self.assertEqual(len(set(ids[0])),3)
  resident.pool.shutdown()
if __name__=='__main__':unittest.main()
