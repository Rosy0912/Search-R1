"""Reduce only owned live workers' read-only scoring pieces, without restarting."""
import json
import time
from pathlib import Path

import ray

BASE = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
RUN = 'f04fe593830c4831918ede872812c016'
NAMESPACE = '24a0d72a-8903-44cb-a291-7ae4894a5546'


def adjust(worker):
    import os
    import time

    assert os.environ['E1_RUN_ID'] == 'f04fe593830c4831918ede872812c016'
    assert os.getpid() in {1437900, 1438381, 1438382, 1438383}
    assert worker.world_size == 4 and worker._is_actor
    assert os.environ['E1_PHYSICAL_MICRO'] == '1'
    assert not worker.config.ref.log_prob_use_dynamic_bsz
    assert not worker.config.rollout.log_prob_use_dynamic_bsz
    old = os.environ['E1_PHYSICAL_SCORE']
    assert old in {'1', '2'}
    scheduler = worker.actor_lr_scheduler.state_dict()
    steps = sorted({float(v['step']) for v in worker.actor_optimizer.state.values() if 'step' in v})
    os.environ['E1_PHYSICAL_SCORE'] = '1'
    assert worker.actor_lr_scheduler.state_dict() == scheduler
    assert steps == sorted({float(v['step']) for v in worker.actor_optimizer.state.values() if 'step' in v})
    return dict(rank=worker.rank, pid=os.getpid(), previous=old, physical_score=1,
                scheduler=scheduler, optimizer_steps=steps,
                logical_ref_micro=worker.config.ref.log_prob_micro_batch_size,
                logical_rollout_micro=worker.config.rollout.log_prob_micro_batch_size,
                time=time.time())


ray.init(address='10.123.4.11:59249', log_to_driver=False)
names = [n for n in ray.util.list_named_actors(all_namespaces=True)
         if n['namespace'] == NAMESPACE and n['name'].startswith('OhbVrpWorkerDict_0:')]
assert len(names) == 4
refs = [ray.get_actor(n['name'], namespace=NAMESPACE)
        .actor_rollout_execute_with_func_generator.remote(func=adjust) for n in names]
print('OWNED_FOUR_SCORE_ADJUSTMENTS_QUEUED_AT_RPC_BOUNDARY', flush=True)
results = ray.get(refs)
assert sorted(r['rank'] for r in results) == list(range(4))
report = dict(run_id=RUN, time=time.time(), ranks=results,
              purpose='Reduce physical read-only scoring pieces from2 to1; logical config unchanged',
              training_restarted=False, frozen_runtime_modified=False)
target = BASE / 'receipts/archive-live-physical-score1.json'
temp = target.with_suffix('.tmp')
temp.write_text(json.dumps(report, indent=2))
temp.replace(target)
print(json.dumps(report), flush=True)
ray.shutdown()
