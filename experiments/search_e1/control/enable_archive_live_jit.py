"""Apply existing validated Adam state staging at owned workers' RPC boundaries."""
import json
import time
from pathlib import Path
import ray

BASE = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
RUN = 'f04fe593830c4831918ede872812c016'
NAMESPACE = '24a0d72a-8903-44cb-a291-7ae4894a5546'


def enable(worker):
    import os
    import torch
    import types
    import time
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

    assert os.environ['E1_RUN_ID'] == 'f04fe593830c4831918ede872812c016'
    assert os.getpid() in {1437900, 1438381, 1438382, 1438383}
    assert worker.world_size == 4 and worker._is_actor
    assert worker.config.actor.fsdp_config.optimizer_offload is True
    assert worker.actor.actor_optimizer is worker.actor_optimizer
    assert os.environ['E1_PHYSICAL_MICRO'] == '1' and os.environ['E1_PHYSICAL_SCORE'] == '1'
    actor = worker.actor
    scheduler = worker.actor_lr_scheduler.state_dict()
    steps = sorted({float(v['step']) for v in worker.actor_optimizer.state.values() if 'step' in v})
    if not getattr(actor, '_e1_live_jit_enabled', False):
        assert worker._is_offload_optimizer
        original = actor._optimizer_step
        actor._e1_prior_optimizer_step = original
        worker._e1_prior_optimizer_offload = worker._is_offload_optimizer

        def staged_step(self):
            load_fsdp_optimizer(self.actor_optimizer, torch.cuda.current_device())
            try:
                return original()
            finally:
                offload_fsdp_optimizer(self.actor_optimizer)
                torch.cuda.synchronize()

        offload_fsdp_optimizer(worker.actor_optimizer)
        torch.cuda.synchronize()
        actor._optimizer_step = types.MethodType(staged_step, actor)
        # The original worker otherwise loads all moments before forward/backward.
        # The new step wrapper owns their transfers; logical config stays true.
        worker._is_offload_optimizer = False
        actor._e1_live_jit_enabled = True
        os.environ['E1_JIT_OPTIMIZER'] = '1'
    assert worker.actor_lr_scheduler.state_dict() == scheduler
    assert steps == sorted({float(v['step']) for v in worker.actor_optimizer.state.values() if 'step' in v})
    devices = sorted({str(v.device) for state in worker.actor_optimizer.state.values()
                      for v in state.values() if torch.is_tensor(v)})
    assert devices == ['cpu']
    return dict(rank=worker.rank, pid=os.getpid(), time=time.time(), scheduler=scheduler,
                optimizer_steps=steps, state_devices=devices, jit_enabled=True,
                logical_optimizer_offload=True, worker_pre_forward_load=False)


ray.init(address='10.123.4.11:59249', log_to_driver=False)
names = [n for n in ray.util.list_named_actors(all_namespaces=True)
         if n['namespace'] == NAMESPACE and n['name'].startswith('OhbVrpWorkerDict_0:')]
assert len(names) == 4
refs = [ray.get_actor(n['name'], namespace=NAMESPACE)
        .actor_rollout_execute_with_func_generator.remote(func=enable) for n in names]
print('OWNED_FOUR_JIT_CALLBACKS_QUEUED_AT_RPC_BOUNDARY', flush=True)
results = ray.get(refs)
assert sorted(r['rank'] for r in results) == list(range(4))
report = dict(run_id=RUN, time=time.time(), ranks=results, training_restarted=False,
              frozen_files_modified=False, loaded_execution_overlay='per-step Adam CPU/GPU state staging',
              equivalence_proof='fsdp-jit-optimizer-check.log: four-rank bitwise equivalence')
target = BASE / 'receipts/archive-live-jit-optimizer.json'
temp = target.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(target)
print(json.dumps(report), flush=True)
ray.shutdown()
