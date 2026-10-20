"""Synthetic lineage only. No saved production weights/predictions are accepted."""
import copy,hashlib,json,unittest
from pathlib import Path
from . import scope
from .lineage import EVIDENCE,ROUTES,validate
from .contract import validate_plan,CANONICAL_FILES,SCIENCE_FILES,DATA_MANIFEST,SCIENCE,PROFILE,COMMON,checkpoint_root

def h(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def fixture(group='budget-5',step=50,repeat=1):
 declared=scope.slot(group,step,repeat);base=scope.BASE/'fresh-runs'/group;job='zhangjunming-7b-abl-'+scope.GROUPS[group]+'-lfresh1'
 p=dict(group=group,source_commit=scope.SCIENCE,entry_source_commit=scope.SOURCE,fresh_from_common_initialization=True,parent_checkpoint=None,parent_step=0,actual_update_limit=150,scheduler_scale=200,warmup_rounds=57,root=str(base),job=job,entry_run_id='a'*32,run_id='CPU_SYNTHETIC_ONLY',data_identity={'synthetic':True})
 sub=dict(group=group,job_id='139584',job=job,source=scope.SOURCE,science_source=scope.SCIENCE,fresh_full150=True,root=str(base),plan_sha256=h(p))
 native=base/'checkpoints/training_state'/('global_step_'+str(step))/'post-evaluation';hf=declared['model'];hfidentity={'files':{'config.json':'d'*64,'generation_config.json':'e'*64,'model.safetensors':'f'*64}}
 identity=dict(job_id=sub['job_id'],completed_step=step,actual_update_limit=150,code_commit=scope.SCIENCE,data_identity=p['data_identity'])
 states=[dict(identity=dict(identity,rank=r,world_size=8),path=str(native/('rank-%02d.pt'%r)),sha256=h({'CPU_rank':r}),bytes=100,full_worker_state_finite=True) for r in range(8)]
 manifest=dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',identity=identity,worker_states=states,driver=dict(path=str(native/'driver.pt'),next_step=step+1),hf_checkpoint=hf,hf_identity=hfidentity)
 n=dict(status='NEW_FRESH_ABLATION_FINAL150_FULL_NATIVE_CPU_ACCEPTED' if step==150 else 'NEW_FRESH_ABLATION_MILESTONE_FULL_NATIVE_CPU_ACCEPTED',group=group,job_id=sub['job_id'],entry_source=scope.SOURCE,science_source=scope.SCIENCE,completed_actual_rounds=step,plan_sha256=sub['plan_sha256'],manifest_sha256=h(manifest),local_raw_sha_verified=True,workers=[{'rank':r} for r in range(8)],physical_rank_files={'rank-%02d.pt'%r:{k:states[r][k] for k in ('path','bytes','sha256')} for r in range(8)},hf_identity=hfidentity,not51713_prediction_acceptance=True,post_terminal_recheck_unchanged=True,post_parent_and_payload_recheck_passed=True,actual_training_limit_remains=150,native_boundary='post-evaluation')
 hfproof=dict(status='NEW_FRESH_ABLATION_FINAL150_HF_NATIVE_PROJECTION_AND_FULL339_CPU_RELOAD_ACCEPTED' if step==150 else 'NEW_FRESH_ABLATION_MILESTONE_HF_NATIVE_PROJECTION_AND_FULL339_CPU_RELOAD_ACCEPTED',group=group,job_id=sub['job_id'],entry_source=scope.SOURCE,science_source=scope.SCIENCE,plan_sha256=sub['plan_sha256'],native_manifest_sha256=h(manifest),native_local_acceptance_sha256=h(n),hf=hf,hf_identity=hfidentity,local_raw_sha_verified=True,post_terminal_recheck_unchanged=True,post_parent_and_payload_recheck_passed=True,not51713_prediction_acceptance=True,reload={'logical_parameters':339,'all_tensors_finite_bf16_and_exact_saved_weight_bf16_projection':True},native_projection={'logical_parameters':339,'native_ranks':8,'all_logical_elements_exact_native_BF16_projection':True},canonical_encoder_and_vocab_verified=True,step=step,completed_actual_rounds=step,actual_training_limit_remains=150,native_boundary='post-evaluation')
 nc=dict(group=group,job_id=sub['job_id'],step=step,manifest_sha256=h(manifest),route_sha256=ROUTES['native_final' if step==150 else 'native_milestone'])
 hc=dict(group=group,job_id=sub['job_id'],step=step,native_local_acceptance_sha256=h(n),route_sha256=ROUTES['HF_final' if step==150 else 'HF_milestone'])
 values=dict(zip(EVIDENCE,[n,hfproof,nc,hc,manifest,p,sub]));hashes={k:h(v) for k,v in values.items()}
 cp=checkpoint_root(group,step);files={k:v for k,v in hfidentity['files'].items()};files.update(CANONICAL_FILES)
 plan=dict(group=group,actual_training_iterations=step,repeat=repeat,repeat_count=declared['repeat_count'],profile=PROFILE,science=SCIENCE,kind='trained-ablation-fresh',job=declared['job_name'],root=declared['root'],training_job_id=sub['job_id'],training_source=scope.SOURCE,training_science_source=scope.SCIENCE,training_limit_remains=150,native_boundary='post-evaluation',accepted_model=hf,hf_files=hfidentity['files'],model=str(cp/'model-view'),model_revision=None,model_identity=scope.model_binding(declared,sub,hfproof,hashes['hf.json']),model_files=files,data=str(cp/'final.jsonl'),data_sha256=scope.DATA_SHA,id_gold_identity=scope.ID_GOLD,data_manifest_sha256=DATA_MANIFEST,retriever_receipt=COMMON+'/receipts/gpu-retriever-ready.json',gpus=8,cpus=64,memory_gib=600,no_auto_retry=True,source_commit='b'*40,source=str(scope.BASE/'sources'/('b'*40)),source_files=dict(SCIENCE_FILES),engine_seed=0,greedy=True,questions=51713)
 return plan,values,hashes

class ContractTests(unittest.TestCase):
 def test_all23_unique_slots_and_fifteen_checkpoints(self):
  seen=[];models=set()
  for row in scope.all_slots():
   plan,v,hs=fixture(row['group'],row['actual_training_updates'],row['repeat_index']);self.assertEqual(validate_plan(plan,v,hs),Path(row['root']));seen.append(row['job_name']);models.add(row['model'])
  self.assertEqual(len(set(seen)),23);self.assertEqual(len(models),15)
 def test_fixed3_exact_scope(self):self.assertEqual(sum(scope.repeat_count(g,s) for g in scope.GROUPS for s in scope.STEPS),23)
 def test_no_fourth_repeat(self):
  with self.assertRaises(AssertionError):scope.slot('budget-5',150,4)
 def test_n1_variance_null(self):self.assertIsNone(scope.statistic_scope('C1-root-only',50)['sample_SD'])
 def test_cross_step_model_identity_differs(self):
  a=fixture(step=50)[0];b=fixture(step=100)[0];self.assertNotEqual(a['model_identity'],b['model_identity'])

plan_mutations={
 'wrong_group':lambda p:p.update(group='C1-root-only'),
 'wrong_step':lambda p:p.update(actual_training_iterations=150),
 'wrong_repeat':lambda p:p.update(repeat=2),
 'wrong_repeat_count':lambda p:p.update(repeat_count=3),
 'wrong_name':lambda p:p.update(job='old-ablation'),
 'old_root':lambda p:p.update(root='/old/r1'),
 'wrong_training_job':lambda p:p.update(training_job_id='old-UID'),
 'wrong_source':lambda p:p.update(training_source='a'*40),
 'wrong_science':lambda p:p.update(training_science_source='a'*40),
 'wrong_target':lambda p:p.update(training_limit_remains=200),
 'rolling':lambda p:p.update(native_boundary='rolling'),
 'wrong_model':lambda p:p.update(accepted_model='/old/final'),
 'wrong_view':lambda p:p.update(model='/old/model-view'),
 'wrong_identity':lambda p:p.update(model_identity='a'*64),
 'wrong_weights':lambda p:p.update(model_files={}),
 'wrong_data':lambda p:p.update(data_sha256='a'*64),
 'wrong_IDgold':lambda p:p.update(id_gold_identity='a'*64),
 'wrong_retriever':lambda p:p.update(retriever_receipt='/other'),
 'wrong_GPU':lambda p:p.update(gpus=4),
 'wrong_memory':lambda p:p.update(memory_gib=900),
 'retry':lambda p:p.update(no_auto_retry=False),
 'seed':lambda p:p.update(engine_seed=1),
 'nongreedy':lambda p:p.update(greedy=False),
 'subset':lambda p:p.update(questions=512),
 'changed_worker':lambda p:p['source_files'].update({'reproduction/official_eval_v2/driver.py':'a'*64}),
}
proof_mutations={
 'old_native':lambda v:v['native.json'].update(status='OLD_CPU_ACCEPTED'),
 'wrong_native_step':lambda v:v['native.json'].update(completed_actual_rounds=150),
 'wrong_HF_step':lambda v:v['hf.json'].update(step=100),
 'wrong_native_route':lambda v:v['native-claim.json'].update(route_sha256='a'*64),
 'wrong_HF_route':lambda v:v['hf-claim.json'].update(route_sha256='a'*64),
 'no_native_originals':lambda v:v['native.json'].update(local_raw_sha_verified=False),
 'no_HF_originals':lambda v:v['hf.json'].update(local_raw_sha_verified=False),
 'missing_rank':lambda v:v['native.json'].update(workers=[]),
 'wrong_HF_projection':lambda v:v['hf.json']['native_projection'].update(logical_parameters=338),
 'wrong_HF_reload':lambda v:v['hf.json']['reload'].update(all_tensors_finite_bf16_and_exact_saved_weight_bf16_projection=False),
}
for name,mutate in plan_mutations.items():
 def case(self,mutate=mutate):
  p,v,h=fixture();mutate(p)
  with self.assertRaises((AssertionError,KeyError,ValueError)):validate_plan(p,v,h)
 setattr(ContractTests,'test_reject_plan_'+name,case)
for name,mutate in proof_mutations.items():
 def case(self,mutate=mutate):
  p,v,h=fixture();mutate(v)
  with self.assertRaises((AssertionError,KeyError,ValueError)):validate_plan(p,v,h)
 setattr(ContractTests,'test_reject_proof_'+name,case)
