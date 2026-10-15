import copy,unittest
from unittest.mock import patch
from .resume_transport import validate_inputs,validate_gate,PARENT_JOB,PARENT_SOURCE,RUN_ID,CONFIG_SHA
class NativeTransport(unittest.TestCase):
    def fixture(self):
        record=dict(group='budget-5',run_id=RUN_ID,source_commit=PARENT_SOURCE,config_sha256=CONFIG_SHA,data_identity='data')
        ident=dict(job_id=PARENT_JOB,code_commit=PARENT_SOURCE,config_sha256=CONFIG_SHA,data_identity='data',completed_step=2,actual_update_limit=150)
        workers=[dict(identity=dict(ident,rank=i,world_size=8),actual_total_optimizer_calls=40,scheduler_last_epoch=2,full_worker_state_finite=True,path=f'/old/rank{i}.pt',sha256=str(i)) for i in range(8)]
        manifest=dict(identity=ident,status='COMPLETE_SAME_WORLD_STATE_SAVED',worker_states=workers,driver=dict(path='/old/driver.pt',sha256='driver',next_step=3))
        binding=dict(parent_job=PARENT_JOB,completed_outer=2,members=[dict(name=w['path'].split('/')[-1],original=w['path'],sha256=w['sha256']) for w in workers+[manifest['driver']]])
        return binding,record,manifest
    def test_full_exact_boundary(self):self.assertTrue(validate_inputs(*self.fixture()))
    def test_reject_changed_optimizer_counter(self):
        b,r,m=self.fixture();m['worker_states'][3]['actual_total_optimizer_calls']=0
        with self.assertRaises(ValueError):validate_inputs(b,r,m)
    def test_reject_missing_rank(self):
        b,r,m=self.fixture();m['worker_states'].pop()
        with self.assertRaises(ValueError):validate_inputs(b,r,m)
    def test_reject_different_run_or_config(self):
        for key in ['run_id','config_sha256','source_commit']:
            b,r,m=self.fixture();r[key]='different'
            with self.assertRaises(ValueError):validate_inputs(b,r,m)
    def test_reject_changed_member(self):
        b,r,m=self.fixture();b['members'][0]['sha256']='bad'
        with self.assertRaises(ValueError):validate_inputs(b,r,m)
    def test_cutover_rejects_live_parent_and_stale_boundary(self):
        binding=dict(manifest_sha256='m',parent_entry_sha256='e')
        probe=dict(job='pt',status='INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED',training_updates_executed=0,completed_outer=2,parent_manifest_sha256='m',ranks=[dict(rank=i,actual_state_equal=True,observed_adam_steps=[40]) for i in range(8)])
        gate=dict(parent_job=PARENT_JOB,parent_state='CANCELLED',completed_outer=2,latest_durable_outer=2,manifest_sha256='m',parent_entry_sha256='e',target_job='pt',probe_sha256='p')
        with patch('reproduction.recursive_pj.resume_transport.sha',return_value='p'):
            validate_gate(gate,binding,probe)
            for key,value in [('parent_state','RUNNING'),('latest_durable_outer',3),('target_job','other'),('probe_sha256','bad')]:
                with self.assertRaises(ValueError):validate_gate(dict(gate,**{key:value}),binding,probe)
if __name__=='__main__':unittest.main()
