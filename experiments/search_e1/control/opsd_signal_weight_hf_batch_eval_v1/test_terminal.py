import unittest,copy
from evaluate import completed_training_terminal
class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.status={'phase':'FOUR_GROUPS_TRAIN_COMPLETE_TESTS_PENDING'}
        self.receipt={'accepted':True,'completed_groups':4,'completed_outers':100,'full_training_em_rows':100,'actual_cpu_supervisor':{'pid':2297236,'uid':1001,'starttick':693190113,'run_id':'signal-study-port-resume-84064393f92b476ea78f7b0b13b52889','argv':['/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/port-resume-0146350/opsd_signal_weight_port_resume_v1/entry.py','10d94f6a413924c4f869846989e8fdb7cd34274d9ca8f0a6661111949781f828']}}
    def test_completed_bound_owner(self):self.assertEqual(completed_training_terminal(self.status,self.receipt,'2297233|1|0')['pid'],2297236)
    def test_still_running_and_failed_pane_rejected(self):
        for pane in ['2297233|0|','2297233|1|1','1861244|1|0']:
            with self.assertRaises(ValueError):completed_training_terminal(self.status,self.receipt,pane)
    def test_old_or_incomplete_native_record_rejected(self):
        for field,value in [('completed_outers',99),('completed_groups',3),('full_training_em_rows',99)]:
            r=copy.deepcopy(self.receipt);r[field]=value
            with self.assertRaises(ValueError):completed_training_terminal(self.status,r,'2297233|1|0')
        r=copy.deepcopy(self.receipt);r['actual_cpu_supervisor']['pid']=1861244
        with self.assertRaises(ValueError):completed_training_terminal(self.status,r,'2297233|1|0')
if __name__=='__main__':unittest.main()
