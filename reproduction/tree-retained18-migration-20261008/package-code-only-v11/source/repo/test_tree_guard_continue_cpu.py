"""Real tiny Adam/two-rank Gloo and actual actor/worker/driver source; no GPU claims."""
import ast
import copy
import json
import itertools
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import test_tree_behavior_mu_actor_cpu as fixture
import test_partial_worker_state_cpu as worker_fixture
from test_tree_partial_driver_cpu import ENV
from search_r1.baselines import partial_worker_state as partial
from search_r1.baselines import policy_safety_runtime as safety
from search_r1.baselines import gradient_norm
from search_r1.baselines.tree_guard_continue import (PROTOCOL, begin_attempt, accounting,
    validate_completed_result, record_completed_result, validate_ledger)
from search_r1.baselines.tree_partial_state import partial_context, stop_on_partial_actor_result
from search_r1.baselines.training_state import (CheckpointBatchSampler, driver_snapshot, restore_driver,
    worker_snapshot, digest, file_sha, atomic_save)
from search_r1.baselines.tree_fresh200 import validate_completed_outer

ROOT = Path(__file__).resolve().parent


class Sharding:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def preprocess_data(self, data): return data
    def postprocess_data(self, data): return data


class SyntheticFSDPLinear(torch.nn.Linear):
    """Only layout/lazy-init are synthetic; clip source and Gloo run unchanged."""
    def __init__(self):
        super().__init__(1,1,bias=False)
        self.weight.data.fill_(-.5)
        self._is_root=True;self._all_handles=[];self.compute_device=torch.device('cpu')
        self.process_group=dist.group.WORLD;self.cpu_offload=N(offload_params=False)
    def _assert_state(self,state):pass


def cfg():
    value = fixture.config(target_old_policy_kl=.2)
    value['ppo_mini_batch_size']=16;value['ppo_micro_batch_size']=16
    value.policy_safety['monitor_rows_per_rank']=32
    value.update(fresh200_protocol='tree_fresh200_five_leaf_v1', guard_continue_protocol=PROTOCOL)
    return value


def new_data(factory, config, attempt):
    # New immutable mu/emission/tree/leaf bindings on every attempt; this CPU
    # factory stands in for model sampling, and is not a native vLLM proof.
    out = factory.output([factory.state(call=f'new-attempt-{attempt}-row-{i}').branch() for i in range(320)])
    data = fixture.Proto(out.batch, out.non_tensor_batch, out.meta_info)
    width = data.batch['responses'].shape[1]
    data.batch['old_log_probs'] = torch.full((320,width), -.5)
    fixture.trainer['prepare_behavior_mu_old_policy'](data, config)
    data.batch['loss_mask'] = data.batch['info_mask'][:,-width:].clone()
    data.batch['advantages'] = torch.ones((320,width))
    mask = (data.batch['loss_mask']*data.batch['attention_mask'][:,-width:]).float()
    data.batch['policy_weights'] = mask/mask.sum(-1,keepdim=True).clamp_min(1)
    data.meta_info.update(temperature=1., global_token_num=int(mask.sum()))
    data.to = lambda *args: data
    return data


def make_worker(config):
    actor = fixture.Actor(config)
    actor._fresh_total_committed_optimizer_calls=0
    actor.actor_module=SyntheticFSDPLinear()
    actor.actor_optimizer=torch.optim.Adam(actor.actor_module.parameters(),lr=.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(actor.actor_optimizer, lambda step:min(step/57,1.))
    worker = N(actor=actor,actor_optimizer=actor.actor_optimizer,actor_lr_scheduler=schedule,
        config=fixture.Config(actor=config),_is_actor=True,_is_offload_param=False,
        _is_offload_optimizer=False,_is_offload_grad=False,world_size=2,
        ulysses_sharding_manager=Sharding(),flops_counter=N(estimate_flops=lambda *args:(1.,1.)))
    return worker


def actual_update():
    scope = dict(torch=torch,os=os,DataProto=worker_fixture.WorkerProto,Timer=worker_fixture.FakeTimer,
        logger=None,log_gpu_memory_usage=lambda *args,**kwargs:None,
        fresh200_scheduler_after_actor=worker_fixture.SCHEDULE)
    return worker_fixture.load_actual_update_actor(scope)


def actual_audit():
    path=ROOT/'search_r1/baselines/tree_rank_audit.py'
    source=ast.parse(path.read_text())
    klass=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='TreeRankAudit')
    scope=dict(torch=torch,hashlib=hashlib,json=json,math=math,os=os,time=time,Path=Path,
               __file__=str(path),__package__='search_r1.baselines')
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    return scope['TreeRankAudit']


def tiny_trainer(run, config):
    sampler = CheckpointBatchSampler(2048,512,42,'synthetic-fresh-data')
    return N(global_steps=1,actual_outer_updates=0,checkpoint_sampler=sampler,
        config=N(actor_rollout_ref=N(actor=config),trainer=N(default_local_dir=str(run)),
            baselines=N(training_protocol='tree_fresh200_five_leaf_v1')),
        actor_rollout_wg=N(world_size=2),baseline_controller=N(method='tree_grpo',tree_rng=random.Random(42),
            stage=N(state_dict=lambda:{'stage':'training'},load_state_dict=lambda value:None)))


def gloo_worker(rank, rendezvous, run, mode):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    factory=fixture.collector_fixture.BehaviorMuDataTests();factory.setUp()
    run=Path(run);config=cfg();worker=make_worker(config);update=actual_update()
    trainer=tiny_trainer(run,config);sampler_iter=iter(trainer.checkpoint_sampler)
    env=dict(SEARCH_R1_BASELINE_OUTPUT=str(run/'output'),SEARCH_R1_TREE_RANK_AUDIT_DIR=str(run/'events'))
    observed=[]
    saved_receipts=[]
    try:
        with patch.dict(os.environ,env),patch.object(torch.cuda,'empty_cache',lambda:None), \
                patch.object(gradient_norm,'FSDP',SyntheticFSDPLinear), \
                patch('torch.distributed.fsdp._runtime_utils._lazy_init',lambda *args:None), \
                patch.object(fixture.rank_audit,'enabled',lambda:True), \
                patch.object(fixture.rank_audit,'TreeRankAudit',actual_audit()):
            attempts = 3 if mode=='repeated_allskip' else (2 if mode in ('allskip','gradient') else 1)
            prior_trajectories=set()
            for attempt in range(1,attempts+1):
                consumed=next(sampler_iter);metadata=begin_attempt(trainer)
                data=new_data(factory,config,attempt);data.meta_info.update(metadata)
                data.meta_info['tree_partial_context']=partial_context(trainer)
                trajectories=set(data.non_tensor_batch['trajectory_uid'])
                assert not (trajectories & prior_trajectories);prior_trajectories |= trajectories
                native_measure=safety.PolicySafetyRuntime.measure
                def measure(runtime,phase,publish=True):
                    active=(attempt<=2 if mode=='repeated_allskip' else attempt==1)
                    should=(mode in ('post1','disk','nonfinite') and phase=='after_optimizer' and runtime.committed==1
                        or mode=='pre2' and phase=='before_optimizer' and runtime.committed==1
                        or mode in ('allskip','repeated_allskip') and phase=='before_optimizer' and runtime.committed==0
                        or mode=='lastpost' and phase=='after_optimizer' and runtime.committed==20)
                    if active and should:
                        if mode=='nonfinite' and rank==1:worker.actor.actor_module.weight.data.fill_(float('nan'))
                        runtime.sync_valid(rank==1,'synthetic_matched_numeric_guard')
                    return native_measure(runtime,phase,publish=publish)
                native_save=partial.atomic_save
                def save(value,path,**kwargs):
                    if mode=='disk' and rank==1:raise OSError('synthetic rank1 partial snapshot IO')
                    return native_save(value,path,**kwargs)
                hook=None
                if mode in ('gradient','gradient_after1') and attempt==1:
                    hook_calls=[0]
                    def corrupt_gradient(gradient):
                        hook_calls[0]+=1
                        bad=rank==1 and (mode=='gradient' or hook_calls[0]>=2)
                        return torch.full_like(gradient,float('nan')) if bad else gradient
                    hook=worker.actor.actor_module.weight.register_hook(corrupt_gradient)
                if mode=='ordinary':
                    worker.actor._forward_micro_batch=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('ordinary actor error'))
                try:
                    with patch.object(safety.PolicySafetyRuntime,'measure',measure),patch.object(partial,'atomic_save',save):
                        output=update(worker,data)
                except RuntimeError as error:
                    assert mode=='ordinary' and not isinstance(error,safety.PolicySafetyRefusal)
                    assert worker.actor._actual_committed_optimizer_calls==0
                    assert worker.actor_lr_scheduler.last_epoch==0
                    observed.append(dict(attempt=attempt,ordinary_error=str(error),credit=0));break
                finally:
                    if hook is not None:hook.remove()
                envelope=output.meta_info.get('tree_partial_state')
                rank_path=run/'events'/f'step-{trainer.global_steps:06d}-attempt-{attempt:06d}-rank-{rank:02d}.json'
                actor_rank=json.loads(rank_path.read_text())
                assert actor_rank['successful_optimizer_calls']==worker.actor._actual_committed_optimizer_calls
                assert actor_rank['finite_grad_norms']==worker.actor._actual_committed_optimizer_calls
                if envelope:
                    own=envelope['worker_states'][rank]
                    if own['status']=='PARTIAL_WORKER_STATE_SAVED':
                        saved=torch.load(own['path'],map_location='cpu',weights_only=False)
                        assert file_sha(own['path'])==own['sha256'] and digest(saved)==own['logical_state_sha256']
                        assert saved['optimizer_opportunity_ledger']==worker.actor._fresh_opportunity_ledger
                        saved_receipts.append((own['path'],own['sha256']))
                        if mode in ('gradient','gradient_after1') and rank==1:assert torch.isnan(saved['gradients']['weight']).all()
                    if mode in ('disk','nonfinite'):
                        assert envelope['continuation_decision']=='FAULT_STOP'
                    else:assert envelope['continuation_decision']=='CONTINUE_FRESH_ROLLOUT'
                common={k:v for k,v in output.meta_info.items() if k!='metrics'}
                gathered=[None,None];dist.all_gather_object(gathered,common);assert gathered[0]==gathered[1]
                if rank==0:
                    if mode in ('disk','nonfinite'):
                        try:stop_on_partial_actor_result(trainer,data,output)
                        except safety.PolicySafetyRefusal:pass
                        else:raise AssertionError('unsafe/incomplete state cannot continue')
                        assert trainer.actual_outer_updates==0
                    else:
                        retry=stop_on_partial_actor_result(trainer,data,output)
                        if retry:
                            assert trainer.actual_outer_updates==0 and trainer.global_steps==1
                        else:
                            before=copy.deepcopy(getattr(trainer,'completed_optimizer_calls_per_rank',{}))
                            validate_completed_result(trainer,output);assert before==getattr(trainer,'completed_optimizer_calls_per_rank',{})
                            metrics=dict(output.meta_info['metrics'])
                            validate_completed_outer(metrics,trainer.global_steps)
                            record_completed_result(trainer,output)
                            trainer.actual_outer_updates+=1;trainer.global_steps+=1
                    observed.append(dict(attempt=attempt,consumed_question_indices=consumed,
                        actor_result=common,accounting=accounting(trainer),sampler_cursor=trainer.checkpoint_sampler.cursor,
                        scheduler_step=worker.actor_lr_scheduler.last_epoch))
                # Share only driver counters needed to construct the next real
                # worker identity. Driver/model/Adam/gradients are never reloaded.
                sync=[dict(global_steps=trainer.global_steps,actual_outer_updates=trainer.actual_outer_updates,
                    training_attempts=trainer.training_attempts) if rank==0 else None]
                dist.broadcast_object_list(sync,src=0)
                if rank!=0:
                    for key,value in sync[0].items():setattr(trainer,key,value)
                if mode in ('disk','nonfinite'):break
            if rank==0:
                if mode not in ('disk','nonfinite','ordinary'):
                    for path,sha in saved_receipts:assert file_sha(path)==sha
                    state=accounting(trainer)
                    identity=dict(completed_step=trainer.actual_outer_updates,guard_continue_accounting=state)
                    trainer.global_steps=trainer.actual_outer_updates
                    saved=driver_snapshot(trainer,identity);before=digest(saved)
                    restore_driver(trainer,saved,identity);assert digest(driver_snapshot(trainer,identity))==before
                    observed.append(dict(complete_checkpoint_accounting=state,
                        actual_adam_steps=sorted(int(value['step']) for value in worker.actor_optimizer.state.values()),
                        checkpoint_cursor=saved['sampler']['cursor'],exact_tiny_driver_reload=True))
                (run/'cpu-proof.json').write_text(json.dumps(observed,sort_keys=True,allow_nan=False))
    finally:
        factory.tearDown();dist.destroy_process_group()


class GuardContinueTests(unittest.TestCase):
    def test_saved_private_nonfinite_gradient_is_not_masked_by_public_zero_grad(self):
        worker=make_worker(cfg());parameter=worker.actor.actor_module.weight
        parameter.grad=torch.ones_like(parameter)
        parameter._saved_grad_shard=torch.full_like(parameter,float('nan'))
        saved=partial.raw_partial_worker_snapshot(worker,{'cpu_fixture':True},False)
        self.assertTrue(saved['model_adam_finite']);self.assertFalse(saved['gradients_finite'])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'raw.pt';sha=atomic_save(saved,path)
            before=parameter.detach().clone()
            result=partial.reset_saved_guard_gradients(worker,{'continuation_decision':'CONTINUE_FRESH_ROLLOUT'})
            self.assertEqual(result['continuation_decision'],'FAULT_STOP')
            self.assertTrue(torch.equal(parameter,before));self.assertIsNone(parameter.grad)
            self.assertTrue(torch.isnan(parameter._saved_grad_shard).all());self.assertEqual(file_sha(path),sha)
            self.assertFalse(result['gradient_reset_after_raw_persistence']['ranks'][0]['private_gradients_after_reset_finite'])

    def test_actual_concat_takes_rank_zero_meta_requires_pre_gathered_protocol(self):
        # Execute the actual DataProto.concat AST with tiny storage only; this
        # validates its API without claiming an installed Ray execution.
        source=ast.parse((ROOT/'verl/protocol.py').read_text())
        klass=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='DataProto')
        concat=copy.deepcopy(next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='concat'))
        concat.decorator_list=[]
        class Proto:
            def __init__(self,batch=None,non_tensor_batch=None,meta_info=None):
                self.batch=batch;self.non_tensor_batch=non_tensor_batch or {};self.meta_info=meta_info or {}
        scope=dict(DataProto=Proto,List=list,torch=torch,
            list_of_dict_to_dict_of_list=lambda list_of_dict:{})
        exec(compile(ast.Module(body=[concat],type_ignores=[]),'actual-DataProto-concat','exec'),scope)
        first=Proto(meta_info={'result':'rank0'});second=Proto(meta_info={'result':'rank1'})
        self.assertEqual(scope['concat']([first,second]).meta_info,{'result':'rank0'})
        common={'protocol':'tree_fresh_actor_result_v1','ranks':[{'rank':0},{'rank':1}]}
        first.meta_info=copy.deepcopy(common);second.meta_info=copy.deepcopy(common)
        self.assertEqual(scope['concat']([first,second]).meta_info,common)

    def test_strict_real_local_ledger_rejects_row_and_boolean_counterexamples(self):
        ledger=[dict(index=i+1,rows=16,local_row_indices=list(range(i*16,(i+1)*16)),
                     status='SUCCESS_ADAM_WRITE',actual_writes=1) for i in range(20)]
        validate_ledger(ledger,20)
        for field,value in (('rows',1),('index',True),('actual_writes',True),('local_row_indices',[0])):
            modified=copy.deepcopy(ledger);modified[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_ledger(modified,20)
        with self.assertRaises(ValueError):validate_ledger(ledger[:19],19)

    def test_actual_fresh_epoch_expression_real_sampler_consumes_six_hundred_attempts(self):
        # Only guard/credit outcomes are simulated here. The source loop epoch
        # expression and the seeded production sampler execute unmodified.
        source=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        fit=next(node for node in ast.walk(source) if isinstance(node,ast.FunctionDef) and node.name=='fit')
        loop=next(node for node in ast.walk(fit) if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                  and node.target.id=='epoch')
        expression=ast.Expression(body=copy.deepcopy(loop.iter));ast.fix_missing_locations(expression)
        epochs=eval(compile(expression,'actual-fit-epoch-expression','eval'),
                    dict(itertools=itertools,fresh200=True,self=N(config=N(trainer=N(total_epochs=1)))))
        sampler=CheckpointBatchSampler(169028,512,42,'unchanged-split')
        trainer=N(global_steps=1,actual_outer_updates=0)
        boundaries=[]
        for epoch in epochs:
            for batch in sampler:
                begin_attempt(trainer)
                if trainer.training_attempts<=400:
                    continue
                trainer.actual_outer_updates+=1;trainer.global_steps+=1
                if trainer.actual_outer_updates in (5,50,100,150,200):
                    boundaries.append((trainer.actual_outer_updates,trainer.training_attempts))
                if trainer.actual_outer_updates==200:break
            if trainer.actual_outer_updates==200:break
        self.assertEqual(boundaries,[(5,405),(50,450),(100,500),(150,550),(200,600)])
        self.assertEqual((trainer.global_steps,trainer.training_attempts,sampler.epoch,sampler.cursor),
                         (201,600,1,138240))

    def test_real_two_rank_guard_opportunities_and_fault_boundaries(self):
        modes=('post1','pre2','allskip','repeated_allskip','lastpost','gradient','gradient_after1','disk','nonfinite','ordinary')
        requested=os.environ.get('TREE_GUARD_CONTINUE_CPU_MODE')
        if requested:modes=(requested,)
        for mode in modes:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temporary:
                run=Path(temporary)/'run';run.mkdir()
                mp.spawn(gloo_worker,args=(str(run/'gloo-init'),str(run),mode),nprocs=2,join=True)
                proof=json.loads((run/'cpu-proof.json').read_text())
                directory=os.environ.get('TREE_GUARD_CONTINUE_CPU_PROOF_DIRECTORY')
                if directory:
                    target=Path(directory);target.mkdir(parents=True,exist_ok=True)
                    (target/f'gloo-{mode}.json').write_text(json.dumps(proof,sort_keys=True,allow_nan=False))
                if mode in ('post1','pre2','lastpost','gradient_after1'):
                    rank=proof[0]['actor_result']['tree_fresh_actor_result']['ranks'][0]
                    self.assertEqual(rank['opportunities_processed'],20)
                    self.assertEqual(rank['optimizer_calls'],20 if mode=='lastpost' else 1)
                    self.assertEqual(rank['scheduler_after'],1)
                    self.assertEqual(rank['opportunities_skipped'],0 if mode=='lastpost' else 19)
                if mode in ('allskip','gradient'):
                    self.assertEqual([row['sampler_cursor'] for row in proof[:-1]],[512,1024])
                    self.assertEqual([row['accounting']['completed_outer_updates'] for row in proof[:-1]],[0,1])
                    self.assertEqual(proof[-1]['actual_adam_steps'],[20])
                    self.assertEqual(proof[-1]['complete_checkpoint_accounting']['training_attempts'],2)
                if mode=='repeated_allskip':
                    self.assertEqual([row['sampler_cursor'] for row in proof[:-1]],[512,1024,1536])
                    self.assertEqual([row['accounting']['completed_outer_updates'] for row in proof[:-1]],[0,0,1])
                    self.assertEqual(proof[-1]['actual_adam_steps'],[20])


if __name__=='__main__':unittest.main(verbosity=2)
