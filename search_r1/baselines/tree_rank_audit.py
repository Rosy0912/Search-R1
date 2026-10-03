"""Opt-in detached Tree actor evidence; no RNG, collectives or parameter gathers."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch


def enabled():
    return bool(os.environ.get('SEARCH_R1_TREE_RANK_AUDIT_DIR'))


class TreeRankAudit:
    def __init__(self, model, batch, meta, config, actor_source):
        start = time.perf_counter()
        self.model = model
        self.step = int(meta['baseline_outer_step'])
        distributed = torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if distributed else 0
        self.world = torch.distributed.get_world_size() if distributed else 1
        self.rows = len(batch)
        self.mini = int(config.ppo_mini_batch_size)
        self.micro_size = int(config.ppo_micro_batch_size)
        self.epochs = int(config.ppo_epochs)
        self.source = hashlib.sha256(Path(actor_source).read_bytes()).hexdigest()
        self.grads = []
        from .runtime_audit import shard_sample
        self.layout, self.values = shard_sample(model)
        self.hook_seconds = time.perf_counter() - start

    def epoch(self):
        pass

    def micro(self, *args, **kwargs):
        pass

    def gradient(self, norm):
        start = time.perf_counter()
        value = float(norm.detach())
        if not math.isfinite(value):
            raise ValueError('Nonfinite existing Tree optimizer norm')
        self.grads.append(value)
        self.hook_seconds += time.perf_counter() - start

    def finish(self, updates, skips):
        start = time.perf_counter()
        if updates != len(self.grads):
            raise ValueError('Tree actual optimizer count differs from successful returns')
        from .runtime_audit import shard_sample, stats, tensor_hash
        layout, values = shard_sample(self.model)
        compatible = self.layout == layout
        root = Path(os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR'])
        root.mkdir(parents=True, exist_ok=True)
        target = root / f'step-{self.step:06d}-rank-{self.rank:02d}.json'
        record = dict(kind='tree_actor_rank', outer_step=self.step, rank=self.rank,
            world_size=self.world, pid=os.getpid(), job_id=os.environ.get('SLURM_JOB_ID'),
            code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
            config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
            data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'),
            run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
            actor_source_sha256=self.source, auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            local_rows=self.rows, local_minibatch=self.mini, local_microbatch=self.micro_size,
            epochs=self.epochs, successful_optimizer_calls=updates, skipped_empty_minibatches=skips,
            existing_grad_norms=self.grads, grad_norm_scope='existing optimizer FSDP global norm; recorded independently at each rank',
            finite_grad_norms=len(self.grads), nonzero_grad_norms=sum(v != 0 for v in self.grads),
            local_shard_sample_layout=layout, local_shard_sample_layout_unchanged=compatible,
            local_shard_sample_delta=stats(values-self.values) if compatible else None,
            local_shard_sample_before_sha256=tensor_hash(self.values),
            local_shard_sample_after_sha256=tensor_hash(values),
            local_shard_sample_scope='fixed up to8192 local coordinates, no gather; zero sample delta does not prove entire shard unchanged',
            hook_body_seconds_excluding_publication=self.hook_seconds+time.perf_counter()-start)
        pending = target.with_suffix('.pending')
        with pending.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, target)
        pending.unlink()
