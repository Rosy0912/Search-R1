import copy
import unittest
from .retirement import validate_retirement
from reproduction.recursive_pj.resume_profiles import profile


class RetirementTests(unittest.TestCase):
    def test_exact_terminal_evidence(self):
        e=dict(job='old',authority='human_rename_and_resume_20261017',snapshots=[
            dict(at=1,state='NOT_FOUND_404',active_pods=0),dict(at=2,state='NOT_FOUND_404',active_pods=0)])
        self.assertTrue(validate_retirement(e,'old'))
        for field,value in [('state','RUNNING'),('state','PENDING'),('active_pods',1),('at',1)]:
            b=copy.deepcopy(e);b['snapshots'][1][field]=value
            with self.assertRaises(ValueError):validate_retirement(b,'old')
        with self.assertRaises(ValueError):validate_retirement(e,'another')
        b=copy.deepcopy(e);b['snapshots'].pop()
        with self.assertRaises(ValueError):validate_retirement(b,'old')

    def test_exact_resume_boundaries(self):
        b=profile('budget5-ttt53-v7');d=profile('depth2-ttt108-v3')
        self.assertEqual((b['step'],d['step']),(53,108))
        self.assertTrue(b['storage_protected_parent']);self.assertTrue(d['scheduler_deleted_parent'])
        self.assertEqual(d['parent_source'],'19b9493fd19184593c7f1c98ef1fc74c06cb67bc')
        for p in [b,d]:
            self.assertTrue(p['target_job'].startswith('zhangjunming-ttt-'))
            self.assertLessEqual(len(p['target_job']),32)


if __name__=='__main__':unittest.main()
