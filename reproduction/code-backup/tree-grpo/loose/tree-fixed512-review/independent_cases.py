"""Independent CPU checks of fixed512 post-update semantics and I/O cost scope."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

WORK=Path(__file__).resolve().parents[1]
REPO=WORK/'tree-safety-five-acceptance'
sys.path.insert(0,str(REPO))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
T=module('owner_fixture',REPO/'tests/baselines/test_fixed512_acceptance.py')
O=module('actual_cost_observer',WORK/'tree-safety110-a632ba9-v2/validation_cost_hooks_instruct300.py')
checks=[];observations=[]

def fixture():
    case=T.Fixed512AcceptanceTests();case.setUp();return case
def refusal(case,trainer,message,**kwargs):
    try:case.execute(trainer,**kwargs)
    except (RuntimeError,ValueError,OSError) as exc:
        assert message in str(exc),(message,str(exc));return str(exc)
    raise AssertionError('Expected refusal')

case=fixture()
try:
    t=case.trainer();t.validation_metrics['val/f1/all']=.3
    refusal(case,t,'quality or coverage gate')
    score=case.score(t)
    assert t.actual_outer_updates==115 and t.global_steps==115
    assert score['actual_updates_this_run']==5 and score['actual_counter']==115
    assert score['status']=='FIXED512_POSTUPDATE_VALIDATION_REFUSED'
    assert not (case.root/'acceptance-step115').exists()
    observations.append({'case':'quality_floor_rejection_after_update','actual_counter':t.actual_outer_updates,
        'step':t.global_steps,'completed_new_updates':score['actual_updates_this_run'],'score_published':True,'gate_released':False})
    checks.append('quality_rejection_preserves_115_counter_five_committed_updates_no_release')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer();t.actual_outer_updates=109
    refusal(case,t,'actual update counter')
    assert t.actual_outer_updates==110 and not t.events
    assert not (case.root/'fixed512-validation').exists()
    checks.append('stale110_counter_cannot_label_score_as115')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer();t.validation_metrics['val/qa_count/nq']=255
    refusal(case,t,'coverage gate');score=case.score(t)
    assert score['count']==512 and score['qa_source_counts']['nq']==255
    assert 'fixed512_evaluated_coverage' in score['refusal_reasons']
    assert t.actual_outer_updates==115
    checks.append('full_reward_count_cannot_hide_incomplete_answer_scoring_count')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer();t.validation_metrics['val/evaluated_count/nq']=True
    refusal(case,t,'coverage gate');score=case.score(t)
    assert score['count'] is None and score['evaluated_source_counts']['nq'] is None
    json.dumps(score,allow_nan=False)
    checks.append('bool_coverage_count_refused_and_finite_json_retained')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer();t._tree_fixed512_split_identity['validation_path']=str(case.root/'other.parquet')
    Path(t._tree_fixed512_split_identity['validation_path']).write_text('changed source')
    refusal(case,t,'split changed');assert not t.events
    checks.append('validation_file_identity_changed_refused_before_evaluation')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer();Path(t._tree_fixed512_split_identity['manifest_path']).write_text('changed manifest')
    refusal(case,t,'split changed');assert not t.events
    checks.append('split_manifest_identity_changed_refused_before_evaluation')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer()
    def invalid_validate():
        t.actual_outer_updates=116
        return dict(t.validation_metrics)
    t._validate=invalid_validate
    refusal(case,t,'training sampler cursor')
    assert not (case.root/'fixed512-validation').exists()
    checks.append('validation_counter_mutation_refused_without_success_score')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer(step=120,enabled=False);t.validation_metrics['val/f1/all']=.3
    refusal(case,t,'Recovery fixed512 evaluation quality gate')
    assert t.events[0]==('validate',120,120) and t.actual_outer_updates==120
    assert not (case.root/'fixed512-validation').exists()
    checks.append('disabled_new_acceptance_keeps_original_recovery_periodic_quality_refusal')
finally:case.doCleanups()

case=fixture()
try:
    t=case.trainer();case.execute(t);path=case.root/'fixed512-validation/outer-000115.json';original=path.read_bytes()
    t.actual_outer_updates=114
    try:case.execute(t)
    except FileExistsError:pass
    else:raise AssertionError('Existing score overwritten')
    assert path.read_bytes()==original and t.actual_outer_updates==115
    assert len([e for e in t.events if e[0]=='validate'])==2
    checks.append('same_step_reentry_cannot_overwrite_evidence_and_does_not_pretend_no_update')
    observations.append({'case':'same_step_reentry','single_trigger_calls_per_invocation':1,
        'separate_repeat_invocation_rejected_on_publish':True,'additional_validation_cost_requires_retention':True})
finally:case.doCleanups()

# Exercise the existing real cost observer on a CPU RPC double, then force only
# the new score-publication I/O to fail after validation returns. No model is run.
case=fixture()
try:
    template=case.trainer()
    class WG:
        def generate_sequences(self,batch):return SimpleNamespace(batch={'responses':torch.ones(512,1,dtype=torch.int64)})
    class Manager:
        def __init__(self,wg):self.actor_rollout_wg=wg;self.config=SimpleNamespace(num_gpus=8)
        def run_llm_loop(self,gen_batch,initial_input_ids):return self._generate_with_gpu_padding(gen_batch)
        def _generate_with_gpu_padding(self,active_batch):return self.actor_rollout_wg.generate_sequences(active_batch)
        def _batch_search(self,queries):return {'result':[]}
    class Trainer:
        def _prepare_final_validation(self):pass
        def _validate(self):
            self.events.append(('validate',self.global_steps,self.actual_outer_updates))
            batch=SimpleNamespace(batch={'input_ids':torch.ones(512,1,dtype=torch.int64),'attention_mask':torch.ones(512,1,dtype=torch.int64)})
            Manager(self.actor_rollout_wg).run_llm_loop(batch,batch.batch['input_ids'])
            return dict(self.validation_metrics)
    t=Trainer();t.__dict__.update({k:v for k,v in vars(template).items() if k!='_validate'})
    t.config.do_search=True;t.tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2);t.actor_rollout_wg=WG()
    cost_root=case.root/'validation-cost'
    O.wrap_trainer(Trainer,Manager,cost_root,final_expected_rows=51713)
    with patch('search_r1.baselines.training_state.atomic_save',side_effect=OSError('independent score I/O failure')):
        refusal(case,t,'independent score I/O failure')
    reports=list(cost_root.glob('*.json'));assert len(reports)==1
    cost=json.loads(reports[0].read_bytes())
    assert cost['status']=='VALIDATION_COST_COMPLETE' and cost['global_step']==115
    assert cost['evaluated_rows']==512 and cost['counters']['returned_rows']==512
    assert cost['counters']['generation_attempts']==1
    assert t.actual_outer_updates==115 and t.global_steps==115
    assert not (case.root/'fixed512-validation').exists()
    observations.append({'case':'score_publication_IO_failure','actual_counter':t.actual_outer_updates,
        'validation_cost_status':cost['status'],'validation_rows':cost['evaluated_rows'],
        'validation_wall_seconds':cost['wall_seconds'],'generation_attempts':cost['counters']['generation_attempts'],
        'score_success_published':False,'observer_phase':cost['phase'],
        'phase_limit':'Historical cost observer says periodic; acceptance trigger requires separate context, not fabricated score success'})
    checks.append('real_external_observer_retains_completed512_cost_before_score_IO_failure')
finally:case.doCleanups()

# Compare actual source methods/configured protocol values, not fixture booleans.
import ast
old=subprocess.check_output(['git','show','a632ba91840fb116aacfc0e770681a860647b295:verl/trainer/ppo/ray_trainer.py'],cwd=REPO,text=True)
new=(REPO/'verl/trainer/ppo/ray_trainer.py').read_text()
def methods(source):
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
a,b=methods(old),methods(new)
for name in ['_validate','_prepare_final_validation','_save_checkpoint']:
    assert a[name]==b[name],name
checks.append('original_validate_final51713_and_save_implementation_unchanged')

result=dict(status='FIXED512_INDEPENDENT_CPU_AND_SOURCE_REVIEW_PASS',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    checks=checks,check_count=len(checks),observations=observations,actual_model_assets_loaded=False,remote_calls=False,
    no_candidate_changes=True,cuda_initialized=torch.cuda.is_initialized(),scope='Fabricated CPU RPC/dataframe fixtures and source control flow only; no actual 115 score or five-outer acceptance')
print(json.dumps(result,sort_keys=True,allow_nan=False))
