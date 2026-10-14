"""Independent eight-GPU cold-state audit; never advances or resumes a study run."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

BASE = Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1')
RUNTIME = '2dd059cc8bcf58ad378584cde8528482aea8a9ff'
CONFIG = '22f44835a5f50d560fa2f6563113601f1b421aeebf1ff9c52c3732e188329664'
PARENT_JOB = '138654'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def isolate_rank_device(environ):
    """Match Ray: one distinct visible physical GPU, process-local CUDA index0."""
    rank = int(environ['RANK'])
    devices = environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if (int(environ['WORLD_SIZE']) != 8 or rank != int(environ['LOCAL_RANK'])
            or not 0 <= rank < 8 or len(devices) != 8 or len(set(devices)) != 8
            or any(not x.strip() for x in devices)):
        raise ValueError('Require eight distinct allocated GPUs before per-rank isolation')
    environ['CUDA_VISIBLE_DEVICES'] = devices[rank]
    return rank, devices[rank]


def validate_parent(record, manifest):
    identity = manifest['identity']
    workers = manifest['worker_states']
    if (record['group'] != 'C1-root-only' or record['source_commit'] != RUNTIME
            or record['config_sha256'] != CONFIG
            or manifest['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or identity['job_id'] != PARENT_JOB or identity['code_commit'] != RUNTIME
            or identity['config_sha256'] != CONFIG or identity['data_identity'] != record['data_identity']
            or identity['actual_update_limit'] != 150 or identity['completed_step'] != 1
            or len(workers) != 8 or {x['identity']['rank'] for x in workers} != set(range(8))):
        raise ValueError('Probe must use the exact completed C1 round1 eight-rank state')
    for worker in workers:
        if (any(worker['identity'][k] != v for k, v in identity.items())
                or worker['identity']['world_size'] != 8
                or worker['actual_total_optimizer_calls'] != 20
                or worker['scheduler_last_epoch'] != 1):
            raise ValueError('Worker boundary differs from the actual completed round')


def stage(destination):
    run = BASE / 'runs/C1-root-only'
    path = run / 'checkpoints/native-rolling/global_step_1/manifest.json'
    record_path = run / 'receipts/configuration.json'
    record = json.loads(record_path.read_text())
    manifest = json.loads(path.read_text())
    validate_parent(record, manifest)
    if sha(record['config']) != CONFIG:
        raise ValueError('Frozen configuration bytes changed')
    destination = Path(destination).resolve()
    if destination.parent != BASE / 'diagnostics' or destination.exists():
        raise ValueError('Use a new independent task-owned diagnostics directory')
    destination.mkdir(parents=True)
    members = []
    for member in manifest['worker_states'] + [manifest['driver']]:
        original = Path(member['path'])
        if original.parent != path.parent or original.is_symlink() or not original.is_file():
            raise ValueError('Source native member is missing or outside this boundary')
        target = destination / original.name
        # Hard links preserve existing bytes if the running job later rotates its cache.
        # Do not chmod these links: their source inode is shared with the live run.
        os.link(original, target)
        members.append(dict(original=str(original), snapshot=str(target), sha256=member['sha256']))
    (destination / 'configuration.json').write_bytes(record_path.read_bytes())
    (destination / 'resolved.yaml').write_bytes(Path(record['config']).read_bytes())
    (destination / 'manifest.json').write_bytes(path.read_bytes())
    value = dict(at=time.time(), parent_manifest=str(path), parent_manifest_sha256=sha(path),
        configuration_sha256=sha(record_path), config_sha256=CONFIG, runtime_source=RUNTIME,
        members=members, scope='Independent read-only hard-linked inputs; does not authorize a training restart.')
    (destination / 'inputs.json').write_text(json.dumps(value, indent=2) + '\n')
    return dict(directory=str(destination), inputs_sha256=sha(destination / 'inputs.json'))


def run_probe(directory, inputs_sha):
    directory = Path(directory).resolve()
    if (directory.parent != BASE / 'diagnostics' or os.getuid() != 1066
            or os.environ.get('SLURM_JOB_NODELIST') != 'node2'
            or socket.gethostname().split('.')[0] != 'gpu-3090-1'
            or not os.environ.get('SLURM_JOB_ID', '').isdigit()
            or os.environ['SLURM_JOB_ID'] == PARENT_JOB
            or sha(directory / 'inputs.json') != inputs_sha):
        raise ValueError('Require the independently allocated node2 diagnostic and bound inputs')
    binding = json.loads((directory / 'inputs.json').read_text())
    record = json.loads((directory / 'configuration.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    validate_parent(record, manifest)
    if (sha(directory / 'manifest.json') != binding['parent_manifest_sha256']
            or sha(directory / 'configuration.json') != binding['configuration_sha256']
            or sha(directory / 'resolved.yaml') != CONFIG or binding['runtime_source'] != RUNTIME):
        raise ValueError('Diagnostic inputs changed')
    members = {x['original']: x for x in binding['members']}
    if len(members) != 9:
        raise ValueError('Need all eight workers and the driver')

    # Import actual frozen training implementations, not an approximated model layout.
    runtime = BASE / 'sources' / RUNTIME
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
    if torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Unexpected diagnostic hardware')
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    started = time.time()
    cfg = OmegaConf.load(directory / 'resolved.yaml')
    os.environ.update(SEARCH_R1_EXECUTION_BACKEND='slurm_lumia',
        RECURSIVE_A_LUMIA_FRESH_PROFILE='1', RECURSIVE_A_GUARD_ROUND_PROGRESS='1',
        SEARCH_R1_BASELINE_RUN_ID=record['run_id'], SEARCH_R1_BASELINE_CONFIG_SHA256=CONFIG)

    def load_member(member):
        mapping = members[member['path']]
        target = Path(mapping['snapshot'])
        if (target.parent != directory or target.is_symlink() or mapping['sha256'] != member['sha256']):
            raise ValueError('Native input binding differs')
        return state.verified_state_file(dict(member, path=str(target)))

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
        if steps != [20] or scheduler.last_epoch != 1:
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
            value = dict(at=time.time(), started=started, job=os.environ['SLURM_JOB_ID'],
                probe_source=os.environ['RECURSIVE_SOURCE_COMMIT'], frozen_runtime_source=RUNTIME,
                status='INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED', parent_job=PARENT_JOB,
                parent_manifest_sha256=binding['parent_manifest_sha256'], input_binding_sha256=inputs_sha,
                ranks=results, driver_logical_sha256=driver_digest,
                driver_cursor=trainer.checkpoint_sampler.cursor, completed_outer=trainer.actual_outer_updates,
                training_updates_executed=0,
                scope='Independent cold FSDP model/Adam/scheduler/gradient/buffer/RNG and driver restore equality. No rollout, optimizer step, continued training, migration-entry acceptance, vLLM bitwise identity, or end-to-end speed claim.')
            with (directory / ('result-' + os.environ['SLURM_JOB_ID'] + '.json')).open('x') as f:
                json.dump(value, f, indent=2, allow_nan=False)
            print(json.dumps(value), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['stage', 'run'])
    p.add_argument('--directory', required=True)
    p.add_argument('--inputs-sha')
    args = p.parse_args()
    if args.action == 'stage':
        print(json.dumps(stage(args.directory)), flush=True)
    else:
        run_probe(args.directory, args.inputs_sha)


if __name__ == '__main__':
    main()
