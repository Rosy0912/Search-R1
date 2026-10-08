import sys,json,contextlib,io
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
sys.path.insert(0,str(Path('tests/reproduction').resolve()))
from test_final_checkpoint import real_iteration_tail,FakeTrainer
from verl import DataProto
run=real_iteration_tail();t=FakeTrainer(1,501);t.config.baselines=SimpleNamespace(method='tree_grpo');t.config.trainer.acceptance_step=5
logs=[];t.logger=SimpleNamespace(log=lambda data,step:logs.append((step,data.copy())))
def update(batch):
 t.weight+=1;t.events.append(('update',t.global_steps,t.weight))
 return SimpleNamespace(meta_info={'metrics':{'actor/optimizer_steps':12}})
t.actor_rollout_wg.update_actor=update
with contextlib.redirect_stdout(io.StringIO()):
 for _ in range(500):run(t)
assert t.global_steps==501 and t.actual_outer_updates==500 and t.weight==500
assert [e[1] for e in t.events if e[0]=='save']==[5,100,200,300,400,500]
assert [v['training/actual_outer_updates'] for _,v in logs if 'training/actual_outer_updates' in v]==list(range(1,501))
saves=[e for e in t.events if e[0]=='save']; final=[e for e in t.events if e[0]=='validate'][-1]
assert saves[-1][2]==final[2]==500
# Old verl retains first rank metadata, rather than averaging meta dictionaries.
ranks=[DataProto(meta_info={'metrics':{'actor/optimizer_steps':12}}) for _ in range(8)]
gathered=DataProto.concat(ranks)
assert gathered.meta_info['metrics']['actor/optimizer_steps']==12
print(json.dumps({'sha':'de9ef3d53b9fb5a3c6b66e24fce1d81304db5c5b','outer_updates':t.actual_outer_updates,'global_steps':t.global_steps,'optimizer_steps_per_outer':12,'saved_actual_updates':[e[1] for e in saves],'final_policy_update':final[2],'actual_update_logs_contiguous':True,'8_rank_metadata_optimizer_steps':12,'rank_metadata_aggregation':'first rank meta_info; equal optimizer step counts required by synchronized actor loop'},indent=2))
