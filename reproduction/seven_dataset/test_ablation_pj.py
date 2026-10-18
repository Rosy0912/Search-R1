import copy,unittest
from .ablation_pj import BASE,validate_export,validate_projection
from .contracts import digest
class AblationPJTests(unittest.TestCase):
 def fixture(self):
  parent=dict(group='budget-20',outer=150,backend='slurm_lumia',protocol='tree_seven_milestone_v1',base='/lumia',model='/old/model',inference_model='/old/view',canonical_tokenizer='/old/tokenizer',data_root='/old/data',retriever_receipt='/old/retriever',retriever_receipt_sha256='a'*64,training_config='/old/config',configuration_receipt='/old/receipt',checkpoint_manifest='/old/manifest',job='parent',output='/old/output',evaluation_run_id='1'*32,evaluation_code_commit='a'*40,checkpoint_identity='b'*64,training_config_sha256='c'*64,model_files={'weight':'d'*64},max_turns_marker=4)
  e=dict(status='LUMIA_FULL_INPUTS_VERIFIED_UNSUBMITTED',questions=51713,plan=parent,plan_sha256=digest(parent),config={'max_turns':4},config_digest=digest({'max_turns':4}),exclusive_claim=dict(group='budget-20',outer=150,destination='pj_pt',repeats=3,exact_scheduler_matches=[],origin_output_absent=True))
  p=copy.deepcopy(parent);stage=BASE+'/ablation-imports/'+'b'*64
  p.update(base=BASE,backend='pj_pt',data_root=BASE+'/common-data',evaluation_run_id='2'*32,evaluation_code_commit='f'*40,repeat_index=1,repeat_count=3,pj_export=stage+'/export.json',pj_export_sha256='f'*64,model=stage+'/model',inference_model=stage+'/view',canonical_tokenizer=stage+'/tokenizer',checkpoint_manifest=stage+'/manifest.json',training_config=stage+'/training.yaml',configuration_receipt=stage+'/configuration.json',retriever_receipt=BASE+'/receipts/pj-retriever-ready.json',retriever_receipt_sha256='e'*64,output=BASE+'/ablation-seven-evaluations/'+'b'*64+'/r1/output',job='zhangjunming-ttt-e-b20-150-r1-bbbbbbbb-pj')
  return p,e
 def test_exact_projection(self):
  p,e=self.fixture();self.assertTrue(str(validate_projection(p,e)).endswith('/r1'))
 def test_changed_identity(self):
  for k,v in [('training_config_sha256','x'),('model_files',{}),('checkpoint_identity','a'*64),('max_turns_marker',5)]:
   p,e=self.fixture();p[k]=v
   with self.subTest(k=k),self.assertRaises(ValueError):validate_projection(p,e)
 def test_path_escape(self):
  for k in ['model','inference_model','canonical_tokenizer','checkpoint_manifest','training_config','configuration_receipt','pj_export','retriever_receipt','data_root','base','output']:
   p,e=self.fixture();p[k]='/unexpected'
   with self.subTest(k=k),self.assertRaises(ValueError):validate_projection(p,e)
 def test_three_runs(self):
  for v in [0,4,True,'1']:
   p,e=self.fixture();p['repeat_index']=v
   with self.subTest(v=v),self.assertRaises(ValueError):validate_projection(p,e)
 def test_duplicate_origin(self):
  for k,v in [('exact_scheduler_matches',['139999']),('origin_output_absent',False),('destination','slurm_lumia'),('repeats',5)]:
   p,e=self.fixture();e['exclusive_claim'][k]=v
   with self.subTest(k=k),self.assertRaises(ValueError):validate_projection(p,e)
 def test_export_binding(self):
  for k in ['config','plan']:
   p,e=self.fixture();e[k]['tamper']=1
   with self.subTest(k=k),self.assertRaises(ValueError):validate_projection(p,e)
 def test_only_future_final(self):
  for g,o in [('original',200),('generation-2',150),('budget-5',100)]:
   p,e=self.fixture();e['plan'].update(group=g,outer=o);e['plan_sha256']=digest(e['plan'])
   with self.subTest(g=g),self.assertRaises(ValueError):validate_export(e)
 def test_name_schema(self):
  for k,v in [('job','other'),('repeat_count',5),('evaluation_run_id','1'*32),('unknown',True)]:
   p,e=self.fixture();p[k]=v
   with self.subTest(k=k),self.assertRaises(ValueError):validate_projection(p,e)
if __name__=='__main__':unittest.main()
