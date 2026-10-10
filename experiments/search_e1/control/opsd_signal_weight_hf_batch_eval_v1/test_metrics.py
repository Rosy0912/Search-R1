import copy,unittest
from metrics import aggregate,paired
class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.questions=[dict(question_id=str(i),question='q'+str(i),answers=['a'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
        self.rows=[dict(q,sibling=0,reward=int(i%2==0),search_calls=1,protocol=dict(protocol_valid_final_action=True,failure_class=None if i%2==0 else 'valid_gold_mismatch')) for i,q in enumerate(self.questions)]
    def test_full_denominator(self):
        s=aggregate(self.rows,self.questions);self.assertEqual(s['overall']['strict_correct'],256);self.assertEqual(s['overall']['denominator'],512);self.assertEqual(s['datasets']['nq']['denominator'],256)
    def test_missing_rejected(self):
        with self.assertRaises(ValueError):aggregate(self.rows[:-1],self.questions)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):aggregate(self.rows[:-1]+[self.rows[0]],self.questions)
    def test_wrong_gold_rejected(self):
        r=copy.deepcopy(self.rows);r[0]['answers']=['changed']
        with self.assertRaises(ValueError):aggregate(r,self.questions)
    def test_paired_gains_losses_and_failures(self):
        r=copy.deepcopy(self.rows);r[0]['reward']=0;r[0]['protocol']=dict(protocol_valid_final_action=False,failure_class='final_invalid');r[1]['reward']=1;r[1]['protocol']['failure_class']=None
        p=paired(r,self.rows);self.assertEqual(p['gained'],['1']);self.assertEqual(p['lost'],['0']);self.assertEqual(p['lost_final_invalid'],1);self.assertEqual(p['net_hits'],0)
    def test_rank_partition_unchanged(self):
        partitions=[self.questions[r::8] for r in range(8)];self.assertTrue(all(len(x)==64 for x in partitions));self.assertEqual(len({q['question_id'] for x in partitions for q in x}),512)
    def test_pair_coverage_rejected(self):
        with self.assertRaises(ValueError):paired(self.rows[:-1],self.rows)
if __name__=='__main__':unittest.main()
