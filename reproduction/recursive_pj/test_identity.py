import unittest
from search_r1.baselines.execution_identity import job_id,provenance_keys
from search_r1.baselines.tree_guard_continue import execution_profile_bound

class Identity(unittest.TestCase):
    def test_real_scheduler_keys_are_distinct(self):
        self.assertEqual(job_id({'SLURM_JOB_ID':'123'}),'123')
        pt=dict(SEARCH_R1_EXECUTION_BACKEND='pj_pt',SEARCH_R1_JOB_ID='real-pt-name')
        self.assertEqual(job_id(pt),'real-pt-name')
        self.assertEqual(provenance_keys(pt,('a','SLURM_JOB_ID')),('a','SEARCH_R1_EXECUTION_BACKEND','SEARCH_R1_JOB_ID'))
        with self.assertRaises(ValueError):job_id(dict(pt,SLURM_JOB_ID='123'))
        with self.assertRaises(ValueError):job_id({'SEARCH_R1_EXECUTION_BACKEND':'pj_pt'})
    def test_progress_requires_bound_profile(self):
        env=dict(SEARCH_R1_EXECUTION_BACKEND='pj_pt',SEARCH_R1_JOB_ID='pt-name',
          RECURSIVE_A_PJ_FRESH_PROFILE='1',SEARCH_R1_BASELINE_RUN_ID='run',SEARCH_R1_BASELINE_CONFIG_SHA256='a'*64)
        self.assertTrue(execution_profile_bound(env))
        self.assertFalse(execution_profile_bound(dict(env,SLURM_JOB_ID='fake')))
        self.assertFalse(execution_profile_bound(dict(env,SEARCH_R1_BASELINE_CONFIG_SHA256='')))
if __name__=='__main__':unittest.main()
