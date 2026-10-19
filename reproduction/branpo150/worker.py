"""The original8-rank native loader, followed by exact real-state observations."""
import json
import os
from pathlib import Path
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as OriginalWorker
from verl.single_controller.base.decorator import register, Dispatch
from .contract import sha, canonical
from .state import tensor_record, optimizer_record, rng_record


def verify_loaded_manager(manager, proof, audit_dir):
    import torch
    from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
    from verl.utils.fsdp_utils import get_fsdp_state_ctx
    rank = torch.distributed.get_rank()
    assert torch.distributed.get_world_size() == 8 and manager.world_size == 8
    expected = proof['native']
    assert expected['actual_updates'] == 140
    before = rng_record(manager.get_rng_state())
    assert before == expected['workers'][rank]['rng'], 'Saved RNG not restored exactly'
    assert manager.lr_scheduler.state_dict() == expected['workers'][rank]['scheduler']
    with get_fsdp_state_ctx(manager.model, StateDictType.SHARDED_STATE_DICT,
                           ShardedStateDictConfig(offload_to_cpu=True),
                           ShardedOptimStateDictConfig(offload_to_cpu=True)):
        actual_model = {k:tensor_record(v) for k,v in manager.model.state_dict().items()}
    assert actual_model == expected['model_shards'][rank], 'Actual loaded model differs from native'
    actual_optim = optimizer_record(manager.optimizer.state_dict(), None, 37 if proof['size']=='3b' else 29)
    assert actual_optim == expected['workers'][rank]['optimizer'], 'Actual Adam state differs from native'
    assert rng_record(manager.get_rng_state()) == before, 'Restore observation changed RNG'
    p = Path(audit_dir)/f'cold-restore-rank-{rank:02d}.json'
    with p.open('x') as stream:
        json.dump(dict(rank=rank,world_size=8,parent_actual_updates=140,
                       job_id=os.environ['SLURM_JOB_ID'],model_exact=True,adam_exact=True,
                       scheduler_exact=True,saved_rng_exact=True,observation_rng_unchanged=True,
                       proof_sha256=sha(os.environ['BRANPO150_NATIVE_PROOF'])),stream,indent=2)
    torch.distributed.barrier()


class AsyncActorRolloutRefWorker(OriginalWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert local_path and str(local_path).endswith('/global_step_240/actor')
        assert hdfs_path is None and del_local_after_load is False
        proof = json.loads(Path(os.environ['BRANPO150_NATIVE_PROOF']).read_text())
        original = self.checkpoint_manager.load_checkpoint
        def observed_load(*args, **kwargs):
            result = original(*args, **kwargs)
            verify_loaded_manager(self.checkpoint_manager, proof, os.environ['BRANPO_AUDIT_DIR'])
            return result
        self.checkpoint_manager.load_checkpoint = observed_load
        try:
            return super().load_checkpoint(local_path, hdfs_path, False)
        finally:
            self.checkpoint_manager.load_checkpoint = original
