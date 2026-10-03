"""Opt-in, detached TOP-D runtime evidence. No collectives or model gathers."""
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import torch


def enabled():
    return bool(os.environ.get('SEARCH_R1_TOPD_AUDIT_DIR'))


def runtime_env_vars():
    if not enabled():
        return {}
    names = ('AUDIT_DIR', 'RUN_ID', 'COMMIT', 'CONFIG_SHA256', 'DATA_IDENTITY')
    result = {f'SEARCH_R1_TOPD_{name}': os.environ[f'SEARCH_R1_TOPD_{name}']
              for name in names if f'SEARCH_R1_TOPD_{name}' in os.environ}
    resource = 'SEARCH_R1_TOPD_TEACHER_CPU_THREADS'
    if resource in os.environ:
        result[resource] = str(teacher_cpu_threads())
    return result


def teacher_cpu_threads():
    """Optional driver-only thread budget; actor and retrieval budgets are separate."""
    value = os.environ.get('SEARCH_R1_TOPD_TEACHER_CPU_THREADS')
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 16:
        raise ValueError('TOP-D teacher CPU threads must be an integer in 1..16')
    return int(value)


def configure_teacher_cpu_threads():
    count = teacher_cpu_threads()
    if count is not None:
        # Called only in the TOP-D driver, before loading its frozen CPU teacher.
        torch.set_num_threads(count)
    return torch.get_num_threads()


def config_boundary(config, split_manifest):
    """Bind the actual post-holdout entry config, before trainer budget injection."""
    if not enabled():
        return
    from omegaconf import OmegaConf
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    encoded = OmegaConf.to_yaml(resolved).encode('utf-8')
    actual = hashlib.sha256(encoded).hexdigest()
    identity = split_manifest['content_identity']['sha256']
    if actual != os.environ.get('SEARCH_R1_TOPD_CONFIG_SHA256'):
        raise RuntimeError('Actual TOP-D entry configuration differs from frozen launch config')
    if identity != os.environ.get('SEARCH_R1_TOPD_DATA_IDENTITY'):
        raise RuntimeError('Actual TOP-D holdout identity differs from frozen launch data')
    root = Path(os.environ['SEARCH_R1_TOPD_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'entry-config.yaml').open('xb') as stream:
        stream.write(encoded)
    emit('config_boundary', dict(actual_config_sha256=actual, actual_data_identity=identity,
         config_scope='resolved Hydra + common holdout before trainer optim.total_training_steps injection',
         entry_total_training_steps=int(config.trainer.total_training_steps)), rank='entry')


def emit(kind, record, rank='driver'):
    directory = os.environ.get('SEARCH_R1_TOPD_AUDIT_DIR')
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = dict(record, kind=kind, rank=rank, pid=os.getpid(),
                  run_id=os.environ.get('SEARCH_R1_TOPD_RUN_ID', root.name),
                  commit=os.environ.get('SEARCH_R1_TOPD_COMMIT'),
                  config_sha256=os.environ.get('SEARCH_R1_TOPD_CONFIG_SHA256'),
                  data_identity=os.environ.get('SEARCH_R1_TOPD_DATA_IDENTITY'))
    # Rank files have one writer; driver records never contain reward_model/gold.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    with (root / f'{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(clean(record), allow_nan=False, sort_keys=True) + '\n')


def tensor_hash(tensor):
    value = tensor.detach()
    flat = value.view(-1) if value.is_contiguous() else None
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    # <=8 MiB CPU chunks, including FP32 teacher tensors; no full model copy.
    chunk = max(1, (8 * 1024 * 1024) // tensor.element_size())
    for start in range(0, value.numel(), chunk):
        if flat is not None:
            part = flat[start:start + chunk]
        else:
            # Avoid reshape copying a whole noncontiguous teacher parameter.
            indices = torch.arange(start, min(start + chunk, value.numel()), device=value.device)
            part = value[torch.unravel_index(indices, value.shape)]
        raw = part.to('cpu').contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def stats(tensor, mask=None):
    value = tensor.detach()
    if mask is not None:
        value = value[mask.bool()]
    finite = torch.isfinite(value)
    good = value[finite]
    return dict(count=value.numel(), finite=int(finite.sum()),
                nonzero=int((good != 0).sum()),
                minimum=float(good.min()) if good.numel() else None,
                maximum=float(good.max()) if good.numel() else None)


def fixed_hashes(batch):
    return {key: tensor_hash(batch[key]) for key in ('old_log_probs', 'advantages', 'policy_weights')}


def shard_sample(model, maximum=8192):
    """Sample evenly spaced local coordinates, never summon full FSDP parameters."""
    params = [(name, param.detach()) for name, param in model.named_parameters() if param.numel()]
    total = sum(param.numel() for _, param in params)
    count = min(maximum, total)
    coordinates = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long().tolist() if count else []
    values, layout = [], []
    offset = 0
    for name, param in params:
        indices = [i - offset for i in coordinates if offset <= i < offset + param.numel()]
        if indices:
            index = torch.tensor(indices, device=param.device)
            values.append(param.reshape(-1).index_select(0, index).float().cpu())
            layout.append(dict(name=name, local_shape=list(param.shape), indices=indices))
        offset += param.numel()
    return layout, torch.cat(values) if values else torch.empty(0)


class ActorAudit:
    def __init__(self, model, batch, meta, epochs):
        self.model, self.batch = model, batch
        self.step = int(meta['topd_outer_step'])
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.before = fixed_hashes(batch)
        self.layout, self.values = shard_sample(model)
        self.rows, self.epochs = len(batch), epochs
        self.grads, self.epoch_hashes, self.micro_stats = [], [], []

    def epoch(self):
        self.epoch_hashes.append(fixed_hashes(self.batch))

    def micro(self, logp, old, mask, loss):
        valid = mask.bool()
        self.micro_stats.append(dict(current_log_probs=stats(logp, valid),
                                     ratios=stats(torch.exp(logp.detach()[valid] - old.detach()[valid])),
                                     policy_loss=stats(loss)))

    def gradient(self, norm):
        value = float(norm.detach())
        self.grads.append(value if math.isfinite(value) else None)

    def finish(self, updates, skips):
        layout, values = shard_sample(self.model)
        compatible = layout == self.layout
        change = stats(values - self.values) if compatible else None
        after = fixed_hashes(self.batch)
        mask = self.batch.get('loss_mask', None)
        emit('actor_outer', dict(outer_step=self.step, rows=self.rows, epochs=self.epochs,
             optimizer_steps=updates, skipped_empty_minibatches=skips,
             micro_stats=self.micro_stats, grad_norm_scope='existing_optimizer_norm_FSDP_global',
             grad_norms=self.grads, grad_finite=sum(v is not None for v in self.grads),
             grad_nonzero=sum(v is not None and v != 0 for v in self.grads),
             fixed_before=self.before, fixed_epochs=self.epoch_hashes, fixed_after=after,
             fixed_unchanged=self.before == after and all(v == self.before for v in self.epoch_hashes),
             shard_layout=layout, shard_layout_unchanged=compatible, shard_delta=change,
             shard_sample_before_sha256=tensor_hash(self.values),
             shard_sample_after_sha256=tensor_hash(values),
             shard_sample_before=stats(self.values), shard_sample_after=stats(values),
             tensors={key: stats(self.batch[key], mask) for key in self.before}), self.rank)


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss is KiB; runtime is Linux. Label process envelope explicitly.
    try:
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, IndexError):
        rss = None
    return dict(wall=time.perf_counter(), user_cpu=value.ru_utime, system_cpu=value.ru_stime,
                driver_peak_rss_kib=value.ru_maxrss, driver_rss_bytes=rss)


def frozen_teacher_snapshot(teacher):
    digest = hashlib.sha256()
    requires = grads = count = parameter_bytes = 0
    devices, dtypes = set(), set()
    for name, param in teacher.model.named_parameters():
        digest.update(name.encode())
        digest.update(tensor_hash(param).encode())
        requires += int(param.requires_grad)
        grads += int(param.grad is not None)
        count += param.numel()
        parameter_bytes += param.numel() * param.element_size()
        devices.add(str(param.device))
        dtypes.add(str(param.dtype))
    record = dict(parameter_sha256=digest.hexdigest(),
                  parameters=count, parameter_bytes=parameter_bytes,
                  training_modules_count=sum(module.training for module in teacher.model.modules()),
                  requires_grad_parameters=requires, grad_parameters=grads,
                  training=teacher.model.training, devices=sorted(devices), dtypes=sorted(dtypes))
    return record


def teacher_boundary(teacher, step, phase):
    record=frozen_teacher_snapshot(teacher)
    record.update(outer_step=step,phase=phase)
    emit('teacher_boundary',record)
    return record


def driver_evidence(data, teacher_logp, before, after, tokens, privileged):
    batch = data.batch
    width = batch['responses'].shape[1]
    attention = batch['attention_mask'][:, -width:]
    info = batch['info_mask'][:, -width:]
    mask = attention * info
    # Prefer an actual observation-bearing response, then retain only one trajectory.
    candidates = ((attention.bool()) & (~info.bool())).any(-1).nonzero().flatten()
    row = int(candidates[0]) if candidates.numel() else 0
    sample = {key: batch[key][row].detach().cpu().tolist() for key in
              ('prompts', 'responses', 'old_log_probs', 'advantages', 'returns', 'policy_weights')}
    sample.update(uid=str(data.non_tensor_batch['uid'][row]), attention_mask=attention[row].cpu().tolist(),
                  info_mask=info[row].cpu().tolist(), loss_mask=mask[row].cpu().tolist(),
                  teacher_log_probs=teacher_logp[row].cpu().tolist(),
                  dense_rewards=batch['token_level_rewards'][row].cpu().tolist())
    emit('driver_outer', dict(outer_step=int(data.meta_info['topd_outer_step']), rows=len(data),
         unique_questions=len(set(map(str, data.non_tensor_batch['uid']))),
         teacher_targets_none=not privileged, teacher_scored_tokens_delta=int(tokens),
         teacher_scoring_wall_seconds=after['wall']-before['wall'],
         driver_scoring_user_cpu_seconds=after['user_cpu']-before['user_cpu'],
         driver_scoring_system_cpu_seconds=after['system_cpu']-before['system_cpu'],
         driver_rss_before=before, driver_rss_after=after, torch_threads=torch.get_num_threads(),
         action_tokens=int(mask.sum()), padding_tokens=int((~attention.bool()).sum()),
         observation_tokens=int((attention.bool() & ~info.bool()).sum()),
         tensors=dict(teacher_log_probs=stats(teacher_logp, mask), **{key: stats(batch[key], mask) for key in
                      ('old_log_probs', 'token_level_rewards', 'returns', 'advantages', 'policy_weights')}),
         sample=sample))


def require_actual_update(metrics):
    """Fail before checkpoint/global-step advancement, even with auditing disabled."""
    count = float(metrics.get('actor/optimizer_steps', 0))
    if not math.isfinite(count) or count <= 0:
        raise RuntimeError('TOP-D outer iteration performed zero actor optimizer steps or a nonfinite count; refusing budget/checkpoint advancement')
