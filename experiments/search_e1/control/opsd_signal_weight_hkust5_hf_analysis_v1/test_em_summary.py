import copy
from pathlib import Path
import sys
import unittest
from em_summary import convert, ENDPOINTS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'opsd_signal_weight_v1'))
from result_analysis import analyze

class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.q=[dict(question_id=str(i),question='q'+str(i),answers=['a'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
        self.rows={e:[dict(q,sibling=0,reward=0,search_calls=1,termination='answer',protocol=dict(protocol_valid_final_action=True,failure_class='valid_gold_mismatch')) for q in self.q] for e in ENDPOINTS}
    def test_same_cohort_primary_sign_and_failure_counts(self):
        for endpoint in ['G_POS-25','H_POS-25']:
            for r in self.rows[endpoint][:16]:r['reward']=1;r['protocol']['failure_class']=None
        self.rows['G_NEG-25'][20]['protocol']=dict(protocol_valid_final_action=False,failure_class='final_invalid')
        s=analyze(convert(self.rows,self.q),repetitions=10)
        self.assertEqual(s['formal_predictions'],4608)
        self.assertEqual(s['models']['G_POS/25']['datasets']['nq']['questions'],256)
        p=s['paired']['G_NEG/25->G_POS/25']
        self.assertEqual(p['net_hits'],16);self.assertEqual(p['delta_percentage_points'],3.125)
        self.assertEqual(s['primary_family_size'],2)
        self.assertEqual(s['models']['G_NEG/25']['overall']['final_invalid'],1)
    def test_missing_duplicate_and_changed_gold_rejected(self):
        for change in ('missing','duplicate','gold'):
            rr=copy.deepcopy(self.rows)
            if change=='missing':rr['G_POS-15'].pop()
            elif change=='duplicate':rr['G_POS-15'][-1]=rr['G_POS-15'][0]
            else:rr['G_POS-15'][0]['answers']=['other']
            with self.assertRaises(ValueError):convert(rr,self.q)
    def test_invalid_answer_cannot_be_correct(self):
        r=self.rows['G_POS-25'][0];r['reward']=1;r['protocol']=dict(protocol_valid_final_action=False,failure_class=None)
        with self.assertRaises(ValueError):convert(self.rows,self.q)

if __name__=='__main__':unittest.main()
