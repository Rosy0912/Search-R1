"""Opt-in detached Tree actor evidence; no RNG, collectives or parameter gathers."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch


def enabled(step=None):
    from .recording_schedule import record_details
    return bool(os.environ.get('SEARCH_R1_TREE_RANK_AUDIT_DIR')) and (step is None or record_details(step))


class TreeRankAudit:
    def __init__(self, model, batch, meta, config, actor_source):
        start = time.perf_counter()
        self.model = model
        self.step = int(meta['baseline_outer_step'])
        from .tree_guard_continue import evidence_suffix
        self.attempt_suffix = evidence_suffix(meta)
        self.attempt_id = meta.get('baseline_attempt_id')
        distributed = torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if distributed else 0
        self.world = torch.distributed.get_world_size() if distributed else 1
        self.rows = len(batch)
        self.mini = int(config.ppo_mini_batch_size)
        self.micro_size = int(config.ppo_micro_batch_size)
        self.epochs = int(config.ppo_epochs)
        self.source = hashlib.sha256(Path(actor_source).read_bytes()).hexdigest()
        self.grads = []
        self.pending_microbatches = []
        from .runtime_audit import shard_sample
        self.layout, self.values = shard_sample(model)
        self.hook_seconds = time.perf_counter() - start

    def epoch(self):
        pass

    def micro(self, log_prob, old_log_prob, response_mask, loss, step_ids=None,
              advantages=None, loss_parts=None, ref_logp=None, diagnostic_data=None):
        # Detached references only: no CPU transfer, RNG, collective or graph
        # retention on the successful path. Keep every contributing microbatch
        # until clipping succeeds; a failing norm may come from an earlier one.
        values = dict(log_prob=log_prob, old_log_prob=old_log_prob,
                      response_mask=response_mask, loss=loss, step_ids=step_ids,
                      advantages=advantages, ref_log_prob=ref_logp)
        values.update(diagnostic_data or {})
        self.pending_microbatches.append(dict(
            tensors={k: v.detach() for k, v in values.items() if isinstance(v, torch.Tensor)},
            loss_parts={k: v.detach() for k, v in (loss_parts or {}).items()
                        if isinstance(v, torch.Tensor)}))

    def failure(self, error):
        """Publish original local loss inputs without hiding the primary error."""
        import warnings
        try:
            root = Path(os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR']) / 'failures'
            root.mkdir(parents=True, exist_ok=True)
            target = root / (f'step-{self.step:06d}{self.attempt_suffix}-rank-{self.rank:02d}-'
                             f'optimizer-{len(self.grads):03d}-pid-{os.getpid()}.pt')
            microbatches = [dict(tensors={k: v.cpu() for k, v in m['tensors'].items()},
                                loss_parts={k: v.cpu() for k, v in m['loss_parts'].items()})
                            for m in self.pending_microbatches]
            gradient_evidence = []
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                flat = parameter.grad.detach().reshape(-1)
                finite_count, max_abs, bad_indices, bad_values = 0, None, [], []
                offset = 0
                for chunk in flat.split(262144):
                    finite = torch.isfinite(chunk)
                    finite_count += int(finite.sum())
                    valid = chunk[finite]
                    if valid.numel():
                        maximum = float(valid.abs().max())
                        max_abs = maximum if max_abs is None else max(max_abs, maximum)
                    take = max(0, 16 - len(bad_indices))
                    if take:
                        indices = (~finite).nonzero().flatten()[:take]
                        bad_indices.extend((indices + offset).cpu().tolist())
                        bad_values.extend(chunk[indices].cpu().tolist())
                    offset += chunk.numel()
                gradient_evidence.append(dict(name=name, count=flat.numel(),
                    finite_count=finite_count, max_finite_abs=max_abs,
                    nonfinite_sample_indices=bad_indices, nonfinite_sample_values=bad_values))
            record = dict(outer_step=self.step, attempt_id=self.attempt_id, rank=self.rank, world_size=self.world,
                successful_optimizer_calls=len(self.grads), exception=str(error),
                source_sha256=self.source,
                code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
                run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
                microbatches=microbatches, local_gradient_evidence=gradient_evidence,
                scope='Local unmodified detached inputs for failed optimizer accumulation; not exact resume')
            # Exclusive publication preserves earlier failures; failed I/O is
            # secondary and must not replace the optimization exception.
            pending = target.with_suffix('.pending')
            with pending.open('xb') as stream:
                torch.save(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(pending, target)
            pending.unlink()
        except Exception as diagnostic_error:
            try:
                warnings.warn(f'Tree failure evidence could not be saved: {diagnostic_error}')
            except Exception:
                pass

    def gradient(self, norm):
        start = time.perf_counter()
        value = float(norm.detach())
        if not math.isfinite(value):
            raise ValueError('Nonfinite existing Tree optimizer norm')
        if os.environ.get('SEARCH_R1_TRAJECTORY_DIR'):
            from .training_state import atomic_save
            summaries=[]
            for micro in self.pending_microbatches:
                tensors=micro['tensors'];valid=tensors['response_mask'].bool()
                delta=(tensors['log_prob']-tensors['old_log_prob'])[valid].double()
                adv=tensors['advantages'][valid].double()
                negative=delta[adv<0]
                summaries.append(dict(valid_tokens=delta.numel(),
                    max_log_ratio=float(delta.max()) if delta.numel() else None,
                    min_log_ratio=float(delta.min()) if delta.numel() else None,
                    negative_advantage_max_log_ratio=float(negative.max()) if negative.numel() else None,
                    abs_log_ratio_over2=int((delta.abs()>2).sum()),
                    loss_parts={k:float(v) for k,v in micro['loss_parts'].items()}))
            path=Path(os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR'])/'ratio-tail'/f'step-{self.step:06d}{self.attempt_suffix}-rank-{self.rank:02d}-optimizer-{len(self.grads):02d}.json'
            atomic_save(dict(outer_step=self.step,attempt_id=self.attempt_id,rank=self.rank,optimizer_call=len(self.grads)+1,
                             preclip_norm=value,microbatches=summaries),path,json_format=True)
        self.grads.append(value)
        self.pending_microbatches.clear()
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
        target = root / f'step-{self.step:06d}{self.attempt_suffix}-rank-{self.rank:02d}.json'
        record = dict(kind='tree_actor_rank', outer_step=self.step, attempt_id=self.attempt_id, rank=self.rank,
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
        if hasattr(self, 'guard_ledger'):
            record.update(optimizer_opportunity_ledger=self.guard_ledger,
                opportunities_processed=len(self.guard_ledger),
                opportunities_skipped=sum(e['actual_writes']==0 for e in self.guard_ledger),
                gradient_norm_source='Observed successful-return preclip norms retained in actual opportunity ledger',
                rollout_braked=True)
        pending = target.with_suffix('.pending')
        with pending.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, target)
        pending.unlink()

    def guard_finish(self, updates, skips, ledger):
        """Report the real returned writes, including a brake after optimizer return."""
        self.guard_ledger = ledger
        self.grads = [entry['gradient_preclip'] for entry in ledger if entry['actual_writes']]
        if not all(value is not None and math.isfinite(value) for value in self.grads):
            raise ValueError('Guard ledger lacks an actual finite successful-write norm')
        self.finish(updates, skips)
