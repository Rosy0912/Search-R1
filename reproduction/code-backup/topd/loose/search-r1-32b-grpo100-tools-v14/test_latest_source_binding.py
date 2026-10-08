import copy,json,pathlib,ast
import pytest
from source_binding import check,validate_native_result,SOURCE,SOURCE_SHA,UPSTREAM,NATIVE
HERE=pathlib.Path(__file__).resolve().parent

def config():return {'candidate':str(SOURCE),'source_manifest':{'path':str(SOURCE/'r1-files.json'),'sha256':SOURCE_SHA},'source_upstream_commit':UPSTREAM,'source_native_cpu_evidence':copy.deepcopy(NATIVE),'training_command':['python','-m','verl.trainer.main_ppo','+trainer.weight_continuation_step=0']}
def test_actual_native3200_result_matches_latest_source():
 r={'job':'3200','tests':26,'skips':0,'failures':0,'exit_code':0,'cuda_initialized':False,'source_manifest_sha256':SOURCE_SHA}
 assert validate_native_result(r)==r and check(config())==SOURCE_SHA
 for k,v in [('job','3199'),('tests',22),('cuda_initialized',True),('source_manifest_sha256','9040')]:
  wrong=dict(r);wrong[k]=v
  with pytest.raises(ValueError):validate_native_result(wrong)
@pytest.mark.parametrize('key,value',[('source_upstream_commit','0d8a921'),('source_native_cpu_evidence',None)])
def test_old_source_or_unbound_native_rejected(key,value):
 c=config();c[key]=value
 with pytest.raises(ValueError):check(c)
def test_production_callers_fresh_read_native_proof():
 for filename in ('supervise.py','train_entry.py','prepare.py'):
  tree=ast.parse((HERE/filename).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('check','check_source_v4')]
  assert any(any(k.arg=='read_native' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in calls),filename

def test_prepare_fresh_base_and_original_geometry():
 from prepare import derive
 from test_standard_r1 import prior
 c=derive(prior(),'profile2','/dataset/shaoshitong/search-r1-rl/topd/logs/latest-v10')
 args={x.split('=',1)[0].lstrip('+'):x.split('=',1)[1] for x in c['training_command'][3:]}
 assert args['trainer.weight_continuation_step']=='0' and not any('actor_checkpoint_path' in x for x in args)
 assert args['data.train_batch_size']=='512' and args['actor_rollout_ref.actor.ppo_micro_batch_size']=='64' and args['actor_rollout_ref.actor.ppo_epochs']=='1'
 assert c['source_native_cpu_evidence']==NATIVE and c['source_upstream_commit']==UPSTREAM
