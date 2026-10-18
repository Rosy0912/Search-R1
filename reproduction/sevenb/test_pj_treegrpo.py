import copy,json,unittest
from pathlib import Path
from .pj_treegrpo import BASE,COMMON,JOB,REVISION,ALLOCATION,VERSIONS,COMPILER,FAILED_PLAN_SHA,scientific_overrides,validate_spec,validate_params,validate_gate
from .contract import models
from reproduction.official_treegrpo.contract import UPSTREAM,MANIFEST_SHA,JOB as TEMPLATE_JOB,recipe
class Tests(unittest.TestCase):
 def setUp(self):
  self.script='''algorithm.adv_estimator=tree \\
actor_rollout_ref.rollout.tree_search=true \\
actor_rollout_ref.actor.optim.lr=$lr \\
trainer.total_training_steps=180 \\
max_turns=3 \\
'''
  files={f['file']:f['sha256'] for f in models()['base']['files']}
  self.plan=dict(protocol='official_treegrpo_7b_pj_replacement_v1',job=JOB,root=BASE+'/runs/treegrpo-v1',model=BASE+'/models/base/'+REVISION,model_revision=REVISION,model_files=files,data=COMMON+'/common-data',data_manifest_sha256=MANIFEST_SHA,upstream_source=COMMON+'/official-treegrpo/deployment-pj-v1/upstream',upstream_commit=UPSTREAM,allocation=ALLOCATION.copy(),required_versions=VERSIONS,compiler=COMPILER,no_hostname_affinity=True,no_auto_retry=True,actual_iterations_expected=179,old_failure=dict(job='139102',completed_iterations=9,native_checkpoint_absent=True,status='FAILED_CUDA_OOM_9_COMPLETE_PARTIAL10_NO_NATIVE'),old_plan_sha256=FAILED_PLAN_SHA,retriever='http://shared:8000')
  self.plan['overrides']=scientific_overrides(self.script,self.plan)
  self.old=dict(kind='treegrpo',job='zhangjunming-7b-treegrpo-0',model_revision=REVISION,model_files=files,required_versions={k:VERSIONS[k] for k in ['torch','vllm','transformers','ray','flash-attn']},model='/old/model',data_root='/old/data',root='/old/run',retriever='http://old:8000')
  self.old['overrides']=[s.replace('='+TEMPLATE_JOB,'='+self.old['job']) for s in recipe(self.script,self.old['model'],self.old['data_root'],self.old['root'],self.old['retriever'])]
 def test_same_official_recipe_on_7b80gb(self):self.assertEqual(validate_spec(self.plan,self.script,self.old),Path(self.plan['root']))
 def test_threeb_initialization_rejected(self):
  p=copy.deepcopy(self.plan);p['model_revision']='aa8e72537993ba99e69dfaafa59ed015b17504d1'
  with self.assertRaises(AssertionError):validate_spec(p,self.script,self.old)
 def test_smaller_gpu_cpu_or_memory_rejected(self):
  for k,v in [('gpu_memory_gib',48),('memory_gib',512),('cpus',60),('gpus',4)]:
   p=copy.deepcopy(self.plan);p['allocation'][k]=v
   with self.assertRaises(AssertionError):validate_spec(p,self.script,self.old)
 def test_scientific_change_or_extra_iteration_rejected(self):
  p=copy.deepcopy(self.plan);p['overrides']=[x.replace('lr=1e-6','lr=2e-6') for x in p['overrides']]
  with self.assertRaises(AssertionError):validate_spec(p,self.script,self.old)
  p=copy.deepcopy(self.plan);p['actual_iterations_expected']=180
  with self.assertRaises(AssertionError):validate_spec(p,self.script,self.old)
 def test_old_different_recipe_rejected(self):
  old=copy.deepcopy(self.old);old['overrides']=[x.replace('lr=1e-6','lr=2e-6') for x in old['overrides']]
  with self.assertRaises(AssertionError):validate_spec(self.plan,self.script,old)
 def test_missing_failure_or_resumable_checkpoint_rejected(self):
  p=copy.deepcopy(self.plan);p['old_failure']['native_checkpoint_absent']=False
  with self.assertRaises(AssertionError):validate_spec(p,self.script,self.old)
 def test_exact_external_gate_uid_plan_and_resources(self):
  sub=dict(job=JOB,uid='exact',plan_sha256='sha');gate=dict(plan_sha256='sha',actual=dict(name=JOB,uid='exact',state='RUNNING',resources=ALLOCATION.copy()))
  validate_gate(self.plan,sub,gate)
  for a in [dict(gate['actual'],uid='other'),dict(gate['actual'],state='PENDING'),dict(gate['actual'],resources=dict(ALLOCATION,memory_gib=512))]:
   with self.assertRaises(AssertionError):validate_gate(self.plan,sub,dict(gate,actual=a))
 def test_no_hostname_affinity_or_retry(self):
  p=dict(job_name=JOB,num_nodes=1,gpus_per_task=8,cpus_per_task=64,memory_per_task=600,retry=0,no_env=True);validate_params(self.plan,p)
  for p2 in [dict(p,include='ecp-pcluster-c480'),dict(p,retry=1),dict(p,memory_per_task=512)]:
   with self.assertRaises(AssertionError):validate_params(self.plan,p2)
if __name__=='__main__':unittest.main()
