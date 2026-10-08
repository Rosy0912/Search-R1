"""Real two-rank Gloo failure convergence; constructed CPU Adam, not eight GPUs."""
import json
import tempfile
import unittest
from pathlib import Path


def _rank(rank, root, scenario):
    import torch
    import torch.distributed as dist
    from reproduction.common8.scheduler import BASE_LR, OuterSchedulerEvidence, fixed_outer_warmup_scheduler
    root=Path(root)
    dist.init_process_group('gloo',rank=rank,world_size=2,
                            init_method='file://'+str(root/'rendezvous'))
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    optimizer=torch.optim.AdamW([p],lr=BASE_LR)
    if scenario=='initialize':
        # Execute the actual worker's optimizer/scheduler caller AST with only
        # model parameters and CPU backend constructed. No FSDP/model claim.
        import ast,os
        from types import SimpleNamespace
        source=Path(__file__).resolve().parents[2]/'verl/workers/fsdp_workers.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_model_optimizer')
        clause=next(n for n in method.body if isinstance(n,ast.If) and
                    isinstance(n.test,ast.Attribute) and n.test.attr=='_is_actor')
        wrapper=ast.parse('def initialize():\n    pass\n').body[0]
        wrapper.body=[clause,ast.Return(value=ast.Tuple(elts=[ast.Name(id='actor_optimizer',ctx=ast.Load()),
                            ast.Name(id='actor_lr_scheduler',ctx=ast.Load())],ctx=ast.Load()))]
        module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
        class Config(dict):
            def __getattr__(self,key):return self[key]
        initialized=[]
        def adam_factory(*args,**kwargs):
            actual=torch.optim.AdamW(*args,**kwargs);initialized.append(actual);return actual
        namespace=dict(os=os,optim=SimpleNamespace(AdamW=adam_factory),self=SimpleNamespace(_is_actor=True),
            actor_module_fsdp=SimpleNamespace(parameters=lambda:[p]),
            optim_config=Config(lr=BASE_LR,lr_warmup_steps=56 if rank==1 else 57,
                              lr_warmup_steps_ratio=0.,completed_updates=0))
        exec(compile(module,'actual initializer caller CPU fixture','exec'),namespace)
        os.environ['SEARCH_G1_PROFILE']='common8';os.environ['SEARCH_G1_CALIBRATION_ROLE']='production'
        failure=None;failure_detail=None;failure_cause=None
        try:namespace['initialize']()
        except BaseException as error:
            failure=type(error).__name__;failure_detail=str(error)
            failure_cause=type(error.__cause__).__name__ if error.__cause__ else None
        # Both ranks must converge before attempting the following collective.
        reached=torch.tensor([1],dtype=torch.int64);dist.all_reduce(reached)
        (root/f'result-{rank}.json').write_text(json.dumps(dict(rank=rank,failure=failure,
            real_cpu_adam_writes=0,next_collective_reached=int(reached.item()),failure_detail=failure_detail,
            failure_cause=failure_cause,optimizer_factory_calls=len(initialized),
            actual_initialized_lrs=[[float(group['lr']) for group in opt.param_groups] for opt in initialized])))
        dist.destroy_process_group();return
    scheduler=fixed_outer_warmup_scheduler(optimizer,warmup_steps=57)
    # Extract the worker's actual update caller. Only CUDA/offload setup is
    # outside this CPU fixture; the sharding block, collectives and fatal
    # receipt exception handling are the real source, not copied mechanics.
    import ast,os
    from types import SimpleNamespace
    from unittest.mock import patch
    source=Path(__file__).resolve().parents[2]/'verl/workers/fsdp_workers.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_actor')
    block=next(n for n in method.body if isinstance(n,ast.With) and
               isinstance(n.items[0].context_expr,ast.Attribute) and
               n.items[0].context_expr.attr=='ulysses_sharding_manager')
    wrapper=ast.parse('def call(data):\n    pass\n').body[0];wrapper.body=[block]
    module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
    class Sharding:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def preprocess_data(self,*,data):return data
        def postprocess_data(self,*,data):return data
    class Timer:
        last=1.
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
    class Data:
        def __init__(self,meta_info):self.meta_info=meta_info
        def to(self,*args):return self
    actor=SimpleNamespace(_optimizer_writes=0)
    def update_policy(*,data):
        for _ in range(20):
            p.grad=torch.ones_like(p);optimizer.step();actor._optimizer_writes+=1
        return {}
    actor.update_policy=update_policy
    worker=SimpleNamespace(actor=actor,actor_optimizer=optimizer,actor_lr_scheduler=scheduler,
        ulysses_sharding_manager=Sharding(),world_size=8,
        config=SimpleNamespace(actor=SimpleNamespace(ppo_epochs=1)),
        flops_counter=SimpleNamespace(estimate_flops=lambda *_:(1.,1.)))
    # Metadata declares the real production topology, while all convergence
    # collectives below still execute in the actual two-process Gloo group.
    namespace=dict(self=worker,os=os,torch=SimpleNamespace(distributed=SimpleNamespace(
        get_rank=lambda:rank,get_world_size=lambda:8)),Timer=Timer,DataProto=Data,
        logger=SimpleNamespace(error=lambda *_:None),log_gpu_memory_usage=lambda *args,**kwargs:None)
    exec(compile(module,'actual update_actor sharding caller CPU fixture','exec'),namespace)
    os.environ['SEARCH_G1_PROFILE']='common8';os.environ['SEARCH_G1_BASE']=str(root)
    os.environ['SLURM_JOB_ID']='12345'
    injected=[];failure=None;failure_detail=None;failure_cause=None
    original=OuterSchedulerEvidence._append;original_fsync=os.fsync
    syncing=None
    def append(recorder,row):
        nonlocal syncing
        syncing=(recorder,row['kind'])
        try:return original(recorder,row)
        finally:syncing=None
    def fsync(fd):
        if (rank==1 and scenario in ('prepare','commit') and syncing is not None and
                fd==syncing[0].fd and syncing[1]==(
                'COMMIT_PREPARED' if scenario=='prepare' else 'COMPLETE')):
            # This replaces the syscall at its real boundary, after the actual
            # append wrote bytes. Readability does not prove durable success.
            injected.append(syncing[1])
            raise OSError('constructed one-rank fsync failure after receipt write')
        return original_fsync(fd)
    try:
        if scenario=='begin' and rank==1:
            (root/'logs/scheduler-rank-1.jsonl').write_text('preexisting evidence, never overwrite\n')
        with patch('reproduction.common8.scheduler._start_ticks',return_value=67890+rank), \
             patch.object(OuterSchedulerEvidence,'_append',append), \
             patch('reproduction.common8.scheduler.os.fsync',fsync):
            namespace['call'](Data(dict(logical_update=1,global_token_num=1,
                                       behavior_probability_gate=dict(sha256='d'*64))))
    except BaseException as error:
        failure=type(error).__name__;failure_detail=str(error)
        failure_cause=type(error.__cause__).__name__ if error.__cause__ else None
    finally:
        result=dict(rank=rank,failure=failure,failure_detail=failure_detail,failure_cause=failure_cause,
                    injected_receipts=injected,real_cpu_adam_writes=actor._optimizer_writes,
                    last_epoch=scheduler.last_epoch,step_count=scheduler._step_count,
                    actual_rejected=getattr(worker,'_outer_scheduler_evidence',None).rejected
                        if hasattr(worker,'_outer_scheduler_evidence') else None,
                    actual_learning_rates=[float(group['lr']) for group in optimizer.param_groups],
                    actual_adam_step=int(optimizer.state[p]['step']) if p in optimizer.state else 0)
        (root/f'result-{rank}.json').write_text(json.dumps(result))
        if hasattr(worker,'_outer_scheduler_evidence'):worker._outer_scheduler_evidence.close()
        dist.destroy_process_group()


class CollectiveSchedulerTests(unittest.TestCase):
    def scenario(self, scenario):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            run=dict(profile='common8',contract_kind='fresh200_warmup57',run_id='a'*32,
                source_commit='b'*40,source_sha256={'constructed.py':'c'*64},completed_updates=0,target_updates=200,
                budget=dict(contract_kind='fresh200_warmup57',target_actual_outer_updates=200,lr_warmup_steps=57,
                            lr_warmup_steps_ratio=0.,learning_rate=1e-6,world_size=8,optimizer_steps_per_outer=20))
            (root/'logs/run-manifest.json').write_text(json.dumps(run))
            (root/'logs/native-allocation-admission.json').write_text(json.dumps(
                dict(accepted=True,run_id='a'*32,slurm_job_id='12345',scope='constructed CPU fixture')))
            context=mp.spawn(_rank,args=(folder,scenario),nprocs=2,join=False)
            # A bounded real collective test fails rather than hang the suite.
            import time
            deadline=time.monotonic()+60
            while not context.join(timeout=1):
                if time.monotonic()>deadline:
                    for process in context.processes:process.terminate()
                    for process in context.processes:process.join(timeout=5)
                    self.fail('Scheduler ranks failed to converge in60s')
            results=[json.loads((root/f'result-{rank}.json').read_text()) for rank in range(2)]
            rows=[]
            for rank in range(2):
                path=root/f'logs/scheduler-rank-{rank}.jsonl'
                rows.append([json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
                            if path.exists() else [])
            return results,rows
    def test_real_initializer_one_rank_scheduler_error_converges_before_next_collective(self):
        results,rows=self.scenario('initialize')
        self.assertTrue(all(r['failure']=='SchedulerError' for r in results))
        self.assertTrue(all(r['real_cpu_adam_writes']==0 for r in results))
        self.assertTrue(all(r['next_collective_reached']==2 for r in results))
        self.assertEqual([r['optimizer_factory_calls'] for r in results],[1,1])
        self.assertEqual([r['actual_initialized_lrs'] for r in results],[[[0.]],[[1e-6]]])
        self.assertEqual([r['failure_cause'] for r in results],[None,'SchedulerError'])
        self.assertIn('exactly57',results[1]['failure_detail'])
        self.assertIn('peer failure',results[0]['failure_detail'])
        self.assertEqual(rows,[[],[]])
    def test_one_rank_begin_io_failure_prevents_all_adam_writes(self):
        results,rows=self.scenario('begin')
        self.assertTrue(all(r['failure']=='SchedulerError' for r in results))
        self.assertEqual([r['real_cpu_adam_writes'] for r in results],[0,0])
        self.assertEqual([r['actual_adam_step'] for r in results],[0,0])
        self.assertEqual([r['last_epoch'] for r in results],[0,0])
        self.assertEqual([r['step_count'] for r in results],[1,1])
        self.assertEqual(rows[0][-1]['kind'],'REJECTED')
        self.assertEqual(rows[1],[])
    def test_one_rank_prepare_io_failure_retains_twenty_writes_without_advance(self):
        results,rows=self.scenario('prepare')
        self.assertTrue(all(r['failure']=='SchedulerError' for r in results))
        self.assertEqual([r['real_cpu_adam_writes'] for r in results],[20,20])
        self.assertEqual([r['actual_adam_step'] for r in results],[20,20])
        self.assertEqual([r['last_epoch'] for r in results],[0,0])
        self.assertEqual([r['step_count'] for r in results],[1,1])
        self.assertEqual([r['actual_learning_rates'] for r in results],[[0.],[0.]])
        self.assertEqual([r['injected_receipts'] for r in results],[[],['COMMIT_PREPARED']])
        self.assertEqual([r['failure_cause'] for r in results],[None,'OSError'])
        self.assertTrue(all('scheduler prepare rejected' in r['failure_detail'] for r in results))
        self.assertIn('constructed one-rank fsync failure after receipt write',results[1]['failure_detail'])
        self.assertTrue(any(row['kind']=='COMMIT_PREPARED' for row in rows[1]))
        self.assertTrue(all(r[-1]['kind']=='REJECTED' for r in rows))
    def test_one_rank_post_commit_fsync_failure_is_fatal_despite_visible_complete_bytes(self):
        results,rows=self.scenario('commit')
        self.assertTrue(all(r['failure']=='SchedulerError' for r in results))
        self.assertEqual([r['real_cpu_adam_writes'] for r in results],[20,20])
        self.assertEqual([r['actual_adam_step'] for r in results],[20,20])
        self.assertEqual([r['last_epoch'] for r in results],[1,1])
        self.assertEqual([r['step_count'] for r in results],[2,2])
        self.assertEqual([r['actual_learning_rates'] for r in results],[[1e-6/57],[1e-6/57]])
        self.assertTrue(all(r['actual_rejected'] is True for r in results))
        self.assertEqual([r['injected_receipts'] for r in results],[[],['COMPLETE']])
        self.assertEqual([r['failure_cause'] for r in results],[None,'OSError'])
        self.assertTrue(all('scheduler commit rejected' in r['failure_detail'] for r in results))
        self.assertTrue(all(r[-1]['kind']=='POST_COMMIT_FAILURE' for r in rows))
        self.assertTrue(all(r[-1]['actual_scheduler_last_epoch']==1 for r in rows))
        for rank_rows in rows:
            fatal=rank_rows[-1]
            self.assertEqual(fatal['actual_scheduler_step_count'],2)
            self.assertEqual(fatal['writes_after'],20)
            self.assertEqual(fatal['actual_learning_rates'],[1e-6/57])
            self.assertIs(fatal['fatal_no_automatic_resume'],True)
            self.assertIs(fatal['commit_started'],True)
        # The attempted COMPLETE bytes exist even though their fsync failed.
        # The following fatal receipt invalidates this outer on both ranks.
        self.assertEqual(sum(row['kind']=='COMPLETE' for row in rows[1]),1)
        self.assertIn('constructed one-rank fsync failure after receipt write',results[1]['failure_detail'])


if __name__=='__main__':unittest.main()
