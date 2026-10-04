import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
from virtual_dp import stripe_partitions
class PartitionTests(unittest.TestCase):
 def test_full_2560_rows_and_ten_minibatches(self):
  parts=[list(range(v*640,(v+1)*640)) for v in range(4)];order=stripe_partitions(parts)
  self.assertEqual(sorted(order),list(range(2560)))
  for v in range(4):
   a=order[(2*v)*320:(2*v+1)*320];b=order[(2*v+1)*320:(2*v+2)*320]
   for mini in range(10):
    self.assertEqual(sorted(a[mini*32:(mini+1)*32]+b[mini*32:(mini+1)*32]),parts[v][mini*64:(mini+1)*64])
    for micro in range(4):
     off=mini*32+micro*8
     self.assertEqual(a[off:off+8]+b[off:off+8],parts[v][mini*64+micro*16:mini*64+(micro+1)*16])
if __name__=='__main__':unittest.main()
