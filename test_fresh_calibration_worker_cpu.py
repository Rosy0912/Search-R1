"""Affected fresh zero and direct rank scheduler boundaries; synthetic CPU only."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from search_r1.baselines import sampling_consistency as source
from search_r1.baselines import policy_safety_runtime as runtime_module
from test_calibration_actor_mode_cpu import FakeRuntime, flags, worker_fixture

ENV=dict(TREE_PPO_CALIBRATION_ONLY='1',SEARCH_R1_BASELINE_COMMIT='a'*40,
 SEARCH_R1_BASELINE_CONFIG_SHA256='b'*64,SEARCH_R1_BASELINE_DATA_IDENTITY='c'*64,
 SEARCH_R1_MODEL_IDENTITY='d'*64,SEARCH_R1_BASELINE_RUN_ID='synthetic-fresh-cpu',SLURM_JOB_ID='0')
PROVENANCE=dict(calibration_profile='fresh_base_zero_v1',training_protocol='tree_fresh200_five_leaf_v1',
 initialized_global_steps=0,driver_completed_outer=0,parent_completed_step=0,parent_cursor=0,checkpoint_loaded=False)

class C(N):
    def get(self,key,default=None):return getattr(self,key,default)

def load_actual_helpers():
    tree=ast.parse(Path('verl/workers/fsdp_workers.py').read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
           ('inspect_fresh_worker_state','fresh200_scheduler_after_actor')]
    assert len(nodes)==2
    namespace=dict(torch=torch,os=os)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-fsdp-worker-helpers','exec'),namespace)
    return namespace['inspect_fresh_worker_state'],namespace['fresh200_scheduler_after_actor']

INSPECT,SCHEDULE=load_actual_helpers()

def fresh_worker(calls=20):
    param=torch.nn.Parameter(torch.tensor([1.]))
    opt=torch.optim.AdamW([param],lr=1e-6)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(step/57,1.))
    assert scheduler.last_epoch==0 and scheduler.get_last_lr()==[0.]
    for _ in range(calls):
        param.grad=torch.ones_like(param);opt.step();opt.zero_grad()
    worker=N(_is_actor=True,actor_optimizer=opt,actor_lr_scheduler=scheduler,
             config=C(actor=C(fresh200_protocol='tree_fresh200_five_leaf_v1'),model=C(path='/synthetic/official-model')))
    return worker,param


def gloo_guard_worker(rank,rendezvous,output,scenario):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        calls=(19 if rank==0 else 21) if scenario=='uneven' else 20
        worker,param=fresh_worker(calls)
        env=dict(ENV,SEARCH_R1_BASELINE_OUTPUT=output)
        with patch.dict(os.environ,env):
            io=(patch('search_r1.baselines.training_state.atomic_save',side_effect=OSError('synthetic rank1 IO'))
                if scenario=='io' and rank==1 else patch('search_r1.baselines.training_state.atomic_save',wraps=source.atomic_save))
            with io:
                try:
                    SCHEDULE(worker,N(meta_info=dict(baseline_outer_step=1)),
                             {'actor/optimizer_steps':calls,'actor/skipped_empty_minibatches':0},0,0.)
                    raise AssertionError('Both ranks must refuse')
                except RuntimeError as error:
                    assert worker.actor_lr_scheduler.last_epoch==(1 if scenario=='io' else 0)
                    actual=int(next(iter(worker.actor_optimizer.state.values()))['step'].item())
                    assert actual==calls
                    (Path(output)/f'gloo-{scenario}-rank{rank}.json').write_text(json.dumps(dict(
                        rank=rank,calls=actual,scheduler=worker.actor_lr_scheduler.last_epoch,
                        refused=True,error=str(error)),allow_nan=False))
    finally:dist.destroy_process_group()


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.env=patch.dict(os.environ,dict(ENV,SEARCH_R1_BASELINE_OUTPUT=self.tmp.name));self.env.start()
        for key in list(os.environ):
            if key.startswith('TREE_RECOVERY_'):os.environ.pop(key)
        FakeRuntime.calls=0;FakeRuntime.fail=False
    def tearDown(self):self.env.stop();self.tmp.cleanup()
    def actual_record(self):return json.loads((Path(self.tmp.name)/'fresh-state/outer-000001-rank00.json').read_text())
    def fresh_calibration(self):
        worker,data=worker_fixture()
        worker.config.actor=C(policy_safety=worker.config.actor.policy_safety,fresh200_protocol='tree_fresh200_five_leaf_v1')
        worker.actor_optimizer.state={};worker.actor_lr_scheduler.last_epoch=0
        data.meta_info=dict(calibration_profile='fresh_base_zero_v1',calibration_fresh_initialization=copy.deepcopy(PROVENANCE),
                            calibration_parent_outer=0,baseline_outer_step=1)
        return worker,data
    def rank_mocks(self):
        return (patch.object(source.dist,'get_rank',return_value=0),patch.object(source.dist,'get_world_size',return_value=1),
                patch.object(source.dist,'all_reduce',lambda *a,**k:None))

    def test_fresh_calibration_actual_entry_empty_state_no_recovery_env_train_restore(self):
        worker,data=self.fresh_calibration();before=flags(worker.actor.actor_module)
        a,b,c=self.rank_mocks()
        with a,b,c,patch.object(runtime_module,'PolicySafetyRuntime',FakeRuntime):record=source.calibrate_actor(worker,data)
        self.assertEqual(record['restored_optimizer_steps'],[]);self.assertEqual(record['fresh_post_monitor_state']['optimizer_state_entries'],0)
        self.assertEqual(record['scheduler_step'],0);self.assertEqual(record['observed_outer'],1)
        self.assertEqual(flags(worker.actor.actor_module),before);self.assertEqual(worker.actor_optimizer.state,{})
        self.assertTrue(record['actor_calibration_mode']['body_returned']);self.assertEqual(FakeRuntime.calls,1)

    def test_fresh_nonempty_step0_state_rejected_before_monitor(self):
        worker,data=self.fresh_calibration();worker.actor_optimizer.state={0:{'step':torch.tensor(0)}}
        with self.assertRaisesRegex(ValueError,'entire empty Adam'):source.calibration_worker_state(worker,data)
        self.assertEqual(FakeRuntime.calls,0)

    def test_fresh_observed0_or_completed1_or_restore_env_rejected(self):
        worker,data=self.fresh_calibration()
        for key,value in (('baseline_outer_step',0),('calibration_parent_outer',1)):
            broken=copy.deepcopy(data);broken.meta_info[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):source.calibration_worker_state(worker,broken)
        with patch.dict(os.environ,{'TREE_RECOVERY_MANIFEST':'synthetic-old'}),self.assertRaisesRegex(ValueError,'historical recovery'):
            source.calibration_worker_state(worker,data)

    def test_fresh_mutated_adam_during_monitor_refuses_without_final_success(self):
        worker,data=self.fresh_calibration()
        original=FakeRuntime.measure
        def mutate(*args,**kwargs):
            answer=original(*args,**kwargs);worker.actor_optimizer.state[0]={'step':torch.tensor(0)};return answer
        a,b,c=self.rank_mocks()
        with a,b,c,patch.object(runtime_module,'PolicySafetyRuntime',FakeRuntime),patch.object(FakeRuntime,'measure',mutate):
            with self.assertRaisesRegex(RuntimeError,'empty Adam'):source.calibrate_actor(worker,data)
        self.assertFalse((Path(self.tmp.name)/'probability-calibration/rank-00.json').exists())
        self.assertTrue(worker._last_calibration_mode_evidence['original_module_modes_restored'])

    def test_rpc_actual_empty_state_scheduler0_path_and_scope(self):
        worker,_=fresh_worker(0)
        with patch.object(dist,'get_rank',return_value=0),patch.object(dist,'get_world_size',return_value=1):
            record=INSPECT(worker,'/synthetic/official-model')
        self.assertEqual(record['optimizer_state_entries'],0);self.assertEqual(record['scheduler_step'],0)
        self.assertFalse(record['complete_model_parameter_bytes_verified'])
        for badpath in ('/synthetic/another-model','relative-model'):
            with self.subTest(path=badpath),self.assertRaisesRegex(RuntimeError,'path'):INSPECT(worker,badpath)

    def test_rpc_nonempty_adam_even_step0_or_advanced_scheduler_refused(self):
        worker,param=fresh_worker(1);worker.actor_optimizer.state[param]['step'].zero_()
        with self.assertRaisesRegex(RuntimeError,'entire state mapping'):INSPECT(worker,'/synthetic/official-model')
        worker,_=fresh_worker(0);worker.actor_lr_scheduler.last_epoch=1
        with self.assertRaisesRegex(RuntimeError,'scheduler'):INSPECT(worker,'/synthetic/official-model')

    def test_real_adam20_first_lr0_advance_once_direct_rank_json_retains_zero_delta(self):
        worker,param=fresh_worker(20);metrics={'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0}
        receipt=SCHEDULE(worker,N(meta_info=dict(baseline_outer_step=1)),metrics,0,0.)
        record=self.actual_record();self.assertEqual(int(worker.actor_optimizer.state[param]['step']),20)
        self.assertEqual(param.item(),1.);self.assertEqual(record['actual_optimizer_calls'],20)
        self.assertEqual((record['scheduler_step_before'],record['scheduler_step_after']),(0,1))
        self.assertTrue(record['first_outer_lr_zero']);self.assertFalse(record['parameter_delta_verified_by_this_record'])
        self.assertEqual(record['learning_rates_after'],[1e-6/57]);self.assertEqual(source.file_sha(receipt['path']),receipt['sha256'])

    def test_partial19_or_skipped1_refuses_before_scheduler_keeps_writes(self):
        worker,param=fresh_worker(19)
        with self.assertRaisesRegex(RuntimeError,'retained'):
            SCHEDULE(worker,N(meta_info=dict(baseline_outer_step=1)),{'actor/optimizer_steps':19,'actor/skipped_empty_minibatches':1},0,0.)
        record=self.actual_record();self.assertEqual(worker.actor_lr_scheduler.last_epoch,0)
        self.assertEqual(record['actual_optimizer_calls'],19);self.assertEqual(record['learning_rates_before'],[0.])
        self.assertEqual(int(worker.actor_optimizer.state[param]['step']),19);self.assertFalse(record['scheduler_step_invoked'])

    def test_scheduler_before_mismatch_refuses_without_advancing(self):
        worker,_=fresh_worker(20);worker.actor_lr_scheduler.last_epoch=3
        with self.assertRaisesRegex(RuntimeError,'retained'):
            SCHEDULE(worker,N(meta_info=dict(baseline_outer_step=1)),{'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0},3,0.)
        self.assertEqual(worker.actor_lr_scheduler.last_epoch,3);self.assertFalse(self.actual_record()['scheduler_step_invoked'])

    def test_scheduler_after_mismatch_retains_actual_after_without_completed_credit(self):
        worker,_=fresh_worker(20)
        def wrongstep():worker.actor_lr_scheduler.last_epoch=2
        with patch.object(worker.actor_lr_scheduler,'step',wrongstep),self.assertRaisesRegex(RuntimeError,'retained'):
            SCHEDULE(worker,N(meta_info=dict(baseline_outer_step=1)),{'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0},0,0.)
        record=self.actual_record();self.assertEqual(record['scheduler_step_after'],2)
        self.assertEqual(record['status'],'SCHEDULER_AFTER_REFUSAL_WRITES_RETAINED')

    def test_evidence_io_failure_refuses_after_true20_and_scheduler1_without_rollback(self):
        worker,param=fresh_worker(20)
        with patch('search_r1.baselines.training_state.atomic_save',side_effect=OSError('synthetic direct IO')),self.assertRaisesRegex(RuntimeError,'direct IO'):
            SCHEDULE(worker,N(meta_info=dict(baseline_outer_step=1)),{'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0},0,0.)
        self.assertEqual(worker.actor_lr_scheduler.last_epoch,1);self.assertEqual(int(worker.actor_optimizer.state[param]['step']),20)
        self.assertFalse((Path(self.tmp.name)/'fresh-state/outer-000001-rank00.json').exists())

    def test_real_two_rank_19_21_mean20_refuses_all_before_scheduler(self):
        rendezvous=str(Path(self.tmp.name)/'uneven-rendezvous')
        mp.spawn(gloo_guard_worker,args=(rendezvous,self.tmp.name,'uneven'),nprocs=2,join=True)
        records=[json.loads((Path(self.tmp.name)/f'gloo-uneven-rank{r}.json').read_text()) for r in range(2)]
        self.assertEqual([record['calls'] for record in records],[19,21]);self.assertEqual([record['scheduler'] for record in records],[0,0])
        direct=[json.loads((Path(self.tmp.name)/f'fresh-state/outer-000001-rank{r:02d}.json').read_text()) for r in range(2)]
        self.assertTrue(all(record['all_rank_before_admitted'] is False for record in direct))

    def test_real_two_rank_one_io_failure_refuses_all_retains20_and_scheduler1(self):
        rendezvous=str(Path(self.tmp.name)/'io-rendezvous')
        mp.spawn(gloo_guard_worker,args=(rendezvous,self.tmp.name,'io'),nprocs=2,join=True)
        records=[json.loads((Path(self.tmp.name)/f'gloo-io-rank{r}.json').read_text()) for r in range(2)]
        self.assertTrue(all(record['calls']==20 and record['scheduler']==1 and record['refused'] for record in records))

    def test_source_rpc_one_to_all_and_fresh_branch_precedes_legacy(self):
        tree=ast.parse(Path('verl/workers/fsdp_workers.py').read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker')
        rpc=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='inspect_fresh_initialization')
        self.assertIn('Dispatch.ONE_TO_ALL',ast.unparse(rpc.decorator_list[0]))
        update=ast.unparse(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='update_actor'))
        self.assertIn("elif metrics.get('actor/optimizer_steps', 0) > 0",update)
        self.assertLess(update.index('metrics = self.actor.update_policy'),update.index('fresh200_scheduler_after_actor'))

if __name__=='__main__':unittest.main(verbosity=2)
