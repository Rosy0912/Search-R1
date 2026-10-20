import copy,json,unittest
from .contract import BASE,STEPS,SCIENCE,ALLOCATION,validate,project_config,job
class ContractTests(unittest.TestCase):
 def pair(self):
  g='C1-root-only';parent=dict(job='old',condition={'variant':'root_only'},model='m',model_files={'f':'a'},data_identity='d',run_id='r')
  m={'identity':{'code_commit':SCIENCE,'job_id':'old','actual_update_limit':150,'completed_step':33},'driver':{'next_step':34},'worker_states':[{'identity':{'rank':i}} for i in range(8)]}
  p=dict(group=g,job=job(g),root=str(BASE/'runs'/g),source_commit=SCIENCE,allocation=ALLOCATION,actual_update_limit=150,scheduler_scale=200,
   protocol='sevenb_ablation_full_native_nfs_recovery_v1',fresh_from_common_initialization=False,no_auto_retry=True,no_hostname_affinity=True,
   condition=parent['condition'],model='m',model_files={'f':'a'},data_identity='d',parent={'step':33,'manifest':m,'uid':'u','state':'FAILED'},superseded_uid='u',
   entry_source_commit='f'*40,source=str(BASE/'sources'/('f'*40)),run_id='r',entry_run_id='a'*32,output=str(BASE/'runs'/g/'output'/('attempt-'+'a'*32)))
  return p,parent
 def test_positive(self):p,r=self.pair();validate(p,r)
 def test_reject_science_and_identity(self):
  for k,v in [('group','x'),('job','x'),('root','/tmp/x'),('source_commit','f'*40),('actual_update_limit',200),('scheduler_scale',150),('fresh_from_common_initialization',True),('no_auto_retry',False),('model','wrong'),('data_identity','wrong'),('superseded_uid','wrong'),('entry_source_commit',SCIENCE),('output','x')]:
   p,r=self.pair();p[k]=v
   with self.assertRaises((AssertionError,KeyError,ValueError)):validate(p,r)
 def test_all_eight_ranks(self):
  p,r=self.pair();p['parent']['manifest']['worker_states'].pop()
  with self.assertRaises(AssertionError):validate(p,r)
 def test_saved_next_step(self):
  p,r=self.pair();p['parent']['manifest']['driver']['next_step']=35
  with self.assertRaises(AssertionError):validate(p,r)
 def test_projection_only_runtime_locations(self):
  c={'trainer':{'default_local_dir':'old','actual_update_limit':150},'baselines':{'local_runtime_host':'old','seed':42},'actor':{'lr':1e-6}};old=copy.deepcopy(c);v=project_config(c,'new','node');self.assertEqual(c,old);v['trainer']['default_local_dir']='old';v['baselines']['local_runtime_host']='old';self.assertEqual(v,c)
if __name__=='__main__':unittest.main()
