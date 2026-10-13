"""Actual frozen model/counter readback and isolated teacher sampling RNG."""
import hashlib
from pathlib import Path

from search_g1.qprobe_features import sha_json


def capture(worker, directory):
    import torch
    from search_g1.training_state import capture_worker
    digest = hashlib.sha256()
    total = 0
    for name, parameter in worker.actor.actor_module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        raw = memoryview(value.view(torch.uint8).numpy()).cast('B')
        for offset in range(0, len(raw), 8*1024*1024):
            digest.update(raw[offset:offset+8*1024*1024])
        total += value.numel()
    if total <= 0 or worker.actor_optimizer.state:
        raise ValueError('TEACHER_WARMUP_REQUIRES_REAL_FRESH_MODEL_AND_EMPTY_ADAM')
    proof = capture_worker(worker, directory)
    return dict(rank=worker.rank, parameter_sha256=digest.hexdigest(), local_parameter_count=total,
        optimizer_sha256=sha_json(worker.actor_optimizer.state_dict()),
        scheduler_sha256=sha_json(worker.actor_lr_scheduler.state_dict()),
        actual_optimizer_writes=getattr(worker.actor, '_optimizer_writes', 0),
        completed_outer=getattr(worker.actor, '_complete_outers', 0), rng_receipt=proof)


def restore_rng(worker, directory):
    import random
    import numpy as np
    import torch
    from search_g1.full_state_restore import equal_tree
    from search_g1.training_state import cpu_tree, worker_rng
    path = Path(directory)/('rank-%d.pt'%worker.rank)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['rank'] != worker.rank or saved['actual_optimizer_writes'] or saved['completed_outer']:
        raise ValueError('TEACHER_WARMUP_RNG_BOUNDARY_IS_NOT_FRESH')
    rng = saved['rng']
    manager = worker.rollout_sharding_manager
    engine = worker.rollout.inference_engine.llm_engine
    if engine.has_unfinished_requests():
        raise ValueError('TEACHER_WARMUP_CANNOT_RESTORE_BUSY_ENGINE_RNG')
    generators = engine.model_executor.worker.model_runner.get_generators()
    generators.clear()
    for key, value in rng['request_generators'].items():
        generator = torch.Generator(device=value['device'])
        generator.set_state(value['state']); generators[key] = generator
    manager.torch_random_states = cpu_tree(rng['actor_cuda_saved'])
    manager.gen_random_states = cpu_tree(rng['rollout_cuda_saved'])
    torch.set_rng_state(rng['torch_cpu'])
    torch.cuda.set_rng_state(rng['torch_cuda'], device=torch.cuda.current_device())
    np.random.set_state(rng['numpy']); random.setstate(rng['python'])
    equal_tree(worker_rng(worker), rng)
    return dict(rank=worker.rank, rng_equal=True, actual_optimizer_writes=0,
                completed_training_outers=0, source=str(path))
