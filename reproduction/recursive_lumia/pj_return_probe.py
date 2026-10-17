"""Independent eight-GPU full-state restore probe; never trains or generates rollouts."""
from datetime import timedelta
import json,os,pathlib,sys,time
from pathlib import Path
from .pj_return_contracts import BASE, PROFILES, PARENT_SOURCE
from .pj_return import load, validate_allocation
from reproduction.recursive_lumia.restore_probe import isolate_rank_device
from release.recursive_bran_a.ablation150_v1.contracts import sha

def run_probe():
    binding_path=Path(os.environ['RECURSIVE_RETURN_BINDING'])
    binding,record,manifest=load(binding_path)
    job=validate_allocation(binding,record)
    directory=binding_path.parent; profile=PROFILES[binding['group']]
    STEP=profile['step']; PARENT_JOB=profile['job']
    inputs_sha=sha(binding_path)
    # Import actual frozen training implementations, not an approximated model layout.
    runtime = BASE / 'sources' / os.environ['RECURSIVE_SOURCE_COMMIT']
    sys.path.insert(0, str(runtime))
    rank, physical_device = isolate_rank_device(os.environ)
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    import random
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from search_r1.baselines import training_state as state
    from search_r1.baselines.bapo import BoundaryStage
    for module in (sys.modules[ActorRolloutRefWorker.__module__], state):
        if not Path(module.__file__).resolve().is_relative_to(runtime):
            raise ValueError('The model/state implementation is not the frozen training source')
    if torch.cuda.device_count() != 1:
        raise ValueError('Each rank must see its one allocated GPU, as in the Ray training worker')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    if torch.cuda.get_device_properties(0).total_memory < 44*1024**3:
        raise ValueError('Unexpected diagnostic hardware')
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    started = time.time()
    cfg = OmegaConf.load(record['config'])
    os.environ.update(SEARCH_R1_EXECUTION_BACKEND='slurm_lumia',
        RECURSIVE_A_LUMIA_FRESH_PROFILE='1', RECURSIVE_A_GUARD_ROUND_PROGRESS='1',
        SEARCH_R1_BASELINE_RUN_ID=record['run_id'], SEARCH_R1_BASELINE_CONFIG_SHA256=record['config_sha256'])

    def load_member(member):
        return state.verified_state_file(member)

    try:
        worker = SimpleNamespace(config=cfg.actor_rollout_ref, rank=rank,
            _is_actor=True, _is_ref=False, _is_rollout=True, ulysses_sequence_parallel_size=1,
            device_mesh=init_device_mesh('cuda', mesh_shape=(8,), mesh_dim_names=['fsdp']))
        c = worker.config
        module, optimizer, scheduler, _ = ActorRolloutRefWorker._build_model_optimizer(worker,
            model_path=c.model.path, fsdp_config=c.actor.fsdp_config, optim_config=c.actor.optim,
            override_model_config=OmegaConf.to_container(c.model.get('override_config', OmegaConf.create())),
            use_remove_padding=c.model.get('use_remove_padding', False),
            enable_gradient_checkpointing=c.model.get('enable_gradient_checkpointing', False),
            trust_remote_code=c.model.get('trust_remote_code', False))
        if optimizer.state or scheduler.last_epoch != 0:
            raise ValueError('Cold diagnostic optimizer is not empty')
        expected = next(x for x in manifest['worker_states'] if x['identity']['rank'] == rank)
        saved = load_member(expected)
        if state.digest(saved) != expected['logical_state_sha256']:
            raise ValueError('Actual saved worker logical digest differs')
        state.prepare_cold_fsdp_layout(module, c.actor.fsdp_config.param_offload,
                                      c.actor.fsdp_config.get('grad_offload', False))
        state.restore_worker(module, optimizer, scheduler, saved, expected['identity'], cold=True)
        del saved
        actual = state.worker_snapshot(module, optimizer, scheduler, expected['identity'], cuda=True)
        observed = state.digest(actual)
        if observed != expected['logical_state_sha256']:
            raise ValueError('Live restored full model/Adam/scheduler/RNG differs from the saved state')
        steps = sorted({int(x['step']) for x in optimizer.state.values()})
        if steps != [expected['actual_total_optimizer_calls']] or scheduler.last_epoch != STEP:
            raise ValueError('Live restored optimizer/scheduler count differs')
        del actual
        local = dict(rank=rank, parent_sha256=expected['sha256'], live_logical_sha256=observed,
            actual_state_equal=True, observed_adam_steps=steps, scheduler_step=scheduler.last_epoch,
            training_updates_executed=0, cuda_device=0, allocated_visible_device=physical_device)
        results = [None] * 8
        dist.all_gather_object(results, local)
        if rank == 0:
            saved = load_member(manifest['driver'])
            if state.digest(saved) != manifest['driver']['logical_state_sha256']:
                raise ValueError('Saved driver digest differs')
            sampler = saved['sampler']
            trainer = SimpleNamespace(config=cfg, actor_rollout_wg=SimpleNamespace(world_size=8),
                baseline_controller=SimpleNamespace(method=saved['algorithm']['method'],
                    stage=BoundaryStage(), tree_rng=random.Random()),
                checkpoint_sampler=state.CheckpointBatchSampler(sampler['rows'], sampler['batch_size'],
                    42, sampler['identity'], shuffle=sampler['shuffle']))
            state.restore_driver(trainer, saved, manifest['identity'])
            driver_digest = state.digest(state.driver_snapshot(trainer, manifest['identity']))
            if driver_digest != manifest['driver']['logical_state_sha256']:
                raise ValueError('Live restored driver RNG/cursor/accounting differs')
            if (len(results) != 8 or {x['rank'] for x in results} != set(range(8))
                    or len({x['allocated_visible_device'] for x in results}) != 8):
                raise ValueError('Incomplete live restored rank coverage')
            value = dict(at=time.time(), started=started, job=job,
                probe_source=os.environ['RECURSIVE_SOURCE_COMMIT'], parent_runtime_source=PARENT_SOURCE,
                status='INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED', parent_job=PARENT_JOB,
                parent_manifest_sha256=binding['metadata_sha256']['parent-manifest.json'], input_binding_sha256=inputs_sha,
                ranks=results, driver_logical_sha256=driver_digest,
                driver_cursor=trainer.checkpoint_sampler.cursor, completed_outer=trainer.actual_outer_updates,
                training_updates_executed=0,
                scope='Independent cold FSDP model/Adam/scheduler/gradient/buffer/RNG and driver restore equality. No rollout, optimizer step, continued training, migration-entry acceptance, vLLM bitwise identity, or end-to-end speed claim.')
            with (directory / ('return-probe-' + job + '.json')).open('x') as f:
                json.dump(value, f, indent=2, allow_nan=False)
            print(json.dumps(value), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':run_probe()
