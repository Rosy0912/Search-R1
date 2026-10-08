"""Explicit diagnostic constant LR and fresh200 outer-counted warmup.

Scheduler receipts describe real optimizer writes and actual torch counters.
They are not GPU, accuracy, checkpoint or full-run acceptance declarations.
"""
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

BASE_LR = 1e-6
WARMUP_OUTERS = 57
TARGET_OUTERS = 200
WRITES_PER_OUTER = 20
CONTRACT_KIND = 'fresh200_warmup57'
MEASUREMENT_ROLE = 'warmup57_measurement'
MEASUREMENT_OUTERS = 2


class SchedulerError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise SchedulerError(message)


def expected_lr(completed_outer):
    _need(type(completed_outer) is int and 0 <= completed_outer <= TARGET_OUTERS,
          'Scheduler completed-outer counter is outside fresh200')
    return BASE_LR * min(completed_outer / WARMUP_OUTERS, 1.)


def constant_scheduler(optimizer):
    """Legacy constant-LR diagnostic or synthetic mechanics only."""
    from torch.optim.lr_scheduler import LambdaLR
    assert optimizer.param_groups and all(group['lr'] == BASE_LR for group in optimizer.param_groups)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    assert all(group['lr'] == BASE_LR for group in optimizer.param_groups)
    scheduler._common8_schedule_kind = 'diagnostic_constant_lr'
    return scheduler


def fixed_outer_warmup_scheduler(optimizer, *, warmup_steps):
    """Fresh LambdaLR: completed0 consumes LR0; completed57 consumes base LR."""
    from torch.optim.lr_scheduler import LambdaLR
    _need(type(warmup_steps) is int and warmup_steps == WARMUP_OUTERS,
          'Formal warmup must be exactly57 explicitly counted outer updates')
    _need(optimizer.param_groups and all(group['lr'] == BASE_LR for group in optimizer.param_groups),
          'Wrong fresh optimizer base LR')
    _need(not optimizer.state and all('initial_lr' not in group for group in optimizer.param_groups),
          'Formal scheduler/Adam must initialize fresh')
    scheduler = LambdaLR(optimizer, lr_lambda=lambda completed: min(completed / WARMUP_OUTERS, 1.))
    _need(scheduler.last_epoch == 0 and scheduler._step_count == 1 and
          all(group['lr'] == 0. for group in optimizer.param_groups),
          'Actual torch constructor did not establish zero-index warmup')
    scheduler._common8_schedule_kind = CONTRACT_KIND
    return scheduler


def actor_scheduler(optimizer, config, *, execution_role):
    """Explicit production57 contract; old diagnostics remain constant0."""
    _need(type(config.get('completed_updates', 0)) is int and config.get('completed_updates', 0) == 0 and
          type(config.get('lr_warmup_steps_ratio', 0.)) in (int, float) and
          config.get('lr_warmup_steps_ratio', 0.) == 0.,
          'Fresh common8 cannot inherit counters or derive warmup from a ratio')
    if execution_role in ('production', MEASUREMENT_ROLE):
        return fixed_outer_warmup_scheduler(optimizer, warmup_steps=config.get('lr_warmup_steps'))
    _need(execution_role in ('policy_safety_calibration_discovery', 'policy_safety_calibration_confirmation') and
          config.get('lr_warmup_steps', 0) == 0,
          'Only original calibration roles may use diagnostic constant LR')
    return constant_scheduler(optimizer)


def collective_actor_initialization(optimizer_factory, config, *, execution_role):
    """Catch fresh optimizer/scheduler construction before rollout collectives."""
    optimizer, scheduler = None, None
    def initialize():
        nonlocal optimizer, scheduler
        optimizer = optimizer_factory()
        scheduler = actor_scheduler(optimizer, config, execution_role=execution_role)
        return dict(execution_role=execution_role, schedule_kind=scheduler._common8_schedule_kind,
                    learning_rates=_learning_rates(optimizer), last_epoch=scheduler.last_epoch,
                    step_count=scheduler._step_count, fresh_adam_entries=len(optimizer.state))
    collective_call(initialize, stage='optimizer_scheduler_initialization')
    return optimizer, scheduler


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def collective_call(callback, *, stage):
    """Agree on local I/O/check results before any following training collective.

    callback returns a JSON-compatible state common to all ranks. A local
    failure is broadcast to peers; no rank proceeds merely because its own
    receipt succeeded. This cannot recover a dead process or failed transport.
    """
    import torch
    import torch.distributed as dist
    _need(dist.is_initialized(), 'Scheduler admission requires a real process group')
    error, state, fingerprint = None, None, None
    try:
        state = callback()
        fingerprint = hashlib.sha256(_canonical(state)).hexdigest()
    except BaseException as caught:
        error = caught
    device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
    accepted = torch.tensor([int(error is None)], dtype=torch.int64, device=device)
    dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
    if not accepted.item():
        raise SchedulerError('Collective scheduler '+stage+' rejected; local='+
                             (type(error).__name__+': '+str(error) if error is not None else 'peer failure')) from error
    fingerprints = [None] * dist.get_world_size()
    dist.all_gather_object(fingerprints, fingerprint)
    _need(all(value == fingerprint for value in fingerprints),
          'Scheduler '+stage+' states differ across ranks')
    return state


def _start_ticks():
    text = Path('/proc/self/stat').read_text()
    return int(text[text.rfind(')') + 2:].split()[19])


def _learning_rates(optimizer):
    values = [float(group['lr']) for group in optimizer.param_groups]
    _need(values and all(math.isfinite(value) and value >= 0. for value in values),
          'Invalid actual optimizer learning rate')
    return values


def _identity(base, *, job_id):
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    budget = run['budget']
    role = run.get('policy_calibration_role', 'production')
    _need(role in ('production', MEASUREMENT_ROLE), 'Wrong fixed57 scheduler execution role')
    _need(run['profile'] == 'common8' and run.get('contract_kind', budget.get('contract_kind')) == CONTRACT_KIND and
          run['completed_updates'] == 0 and run['target_updates'] == TARGET_OUTERS and
          budget['target_actual_outer_updates'] == TARGET_OUTERS and
          type(budget['lr_warmup_steps']) is int and budget['lr_warmup_steps'] == WARMUP_OUTERS and
          budget['lr_warmup_steps_ratio'] == 0. and budget['learning_rate'] == BASE_LR and
          budget['world_size'] == 8 and budget['optimizer_steps_per_outer'] == WRITES_PER_OUTER,
          'Run is not the fresh200 fixed57 scientific scheduler contract')
    _need(re.fullmatch('[0-9a-f]{32}', run['run_id']) and
          re.fullmatch('[0-9a-f]{40}', run['source_commit']) and
          isinstance(run['source_sha256'], dict) and bool(run['source_sha256']) and
          all(isinstance(name, str) and isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha)
              for name, sha in run['source_sha256'].items()), 'Invalid run/source scheduler identity')
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    _need(allocation['accepted'] is True and allocation['run_id'] == run['run_id'] and
          allocation['slurm_job_id'] == job_id and re.fullmatch('[1-9][0-9]*', job_id),
          'Scheduler is not bound to this actual native allocation')
    identity = dict(run_id=run['run_id'], source_commit=run['source_commit'], slurm_job_id=job_id,
                source_map_sha256=hashlib.sha256(_canonical(run['source_sha256'])).hexdigest(),
                scheduler_contract_sha256=hashlib.sha256(_canonical(budget)).hexdigest())
    if role == MEASUREMENT_ROLE:
        from reproduction.common8.warmup_measurement_plan import load_plan
        plan, plan_sha = load_plan(base, run, expected_slurm_job_id=job_id)
        _need(plan['phase_budget']['outer_count'] == MEASUREMENT_OUTERS,
              'Wrong prospective measurement scheduler limit')
        identity.update(execution_role=role, warmup_measurement_plan_sha256=plan_sha)
    return identity


class OuterSchedulerEvidence:
    """One rank, one fresh scheduler, one exclusive append-only durable ledger."""
    def __init__(self, base, optimizer, scheduler, *, rank, world_size, job_id):
        _need(type(rank) is int and rank in range(8) and world_size == 8, 'Wrong real scheduler rank topology')
        self.base = Path(base).resolve()
        self.identity = _identity(self.base, job_id=job_id)
        self.outer_limit = MEASUREMENT_OUTERS if self.identity.get('execution_role') == MEASUREMENT_ROLE else TARGET_OUTERS
        _need(getattr(scheduler, '_common8_schedule_kind', None) == CONTRACT_KIND and
              scheduler.last_epoch == 0 and scheduler._step_count == 1 and
              _learning_rates(optimizer) == [0.] * len(optimizer.param_groups) and not optimizer.state,
              'Outer scheduler evidence must start with fresh LambdaLR and Adam')
        self.optimizer, self.scheduler = optimizer, scheduler
        self.rank, self.world_size = rank, world_size
        self.completed_outer, self.pending, self.rejected = 0, None, False
        self.prepared = None
        self.process_pid, self.process_start_ticks = os.getpid(), _start_ticks()
        _need(self.process_start_ticks > 0, 'Missing actual process birth ticks')
        self.path = self.base / 'logs' / ('scheduler-rank-%d.jsonl' % rank)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL |
                          getattr(os, 'O_NOFOLLOW', 0), 0o600)
        self.inode = os.fstat(self.fd).st_ino
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _append(self, row):
        _need(self.fd is not None and self.path.stat().st_ino == self.inode and not self.path.is_symlink(),
              'Exclusive scheduler ledger changed')
        header = dict(schema_version=1, role=('warmup57_measurement_outer_scheduler'
                                if self.identity.get('execution_role') == MEASUREMENT_ROLE else 'fresh200_outer_scheduler'),
                                contract_kind=CONTRACT_KIND, rank=self.rank, world_size=self.world_size,
                                process_pid=self.process_pid, process_start_ticks=self.process_start_ticks,
                                base_learning_rate=BASE_LR, warmup_steps=WARMUP_OUTERS,
                                observed_unix=time.time(),
                                **self.identity)
        for key in header.keys() & row.keys():
            _need(header[key] == row[key], 'Scheduler receipt attempted to replace immutable identity: '+key)
        header.update(row)
        data = _canonical(header) + b'\n'
        _need(os.write(self.fd, data) == len(data), 'Incomplete scheduler evidence write')
        os.fsync(self.fd)

    def _check_state(self):
        _need(_identity(self.base, job_id=self.identity['slurm_job_id']) == self.identity,
              'Scheduler run/source/contract identity changed during training')
        _need(self.scheduler.last_epoch == self.completed_outer and
              self.scheduler._step_count == self.completed_outer + 1 and
              _learning_rates(self.optimizer) == [expected_lr(self.completed_outer)] * len(self.optimizer.param_groups),
              'Scheduler/counter/LR changed outside a completed outer')

    def begin_outer(self, logical_update, *, optimizer_writes, batch_gate_sha256):
        _need(not self.rejected and self.pending is None and
              type(logical_update) is int and logical_update == self.completed_outer + 1 and
              logical_update <= self.outer_limit and type(optimizer_writes) is int and
              optimizer_writes == self.completed_outer * WRITES_PER_OUTER and
              re.fullmatch('[0-9a-f]{64}', batch_gate_sha256), 'Wrong sequential outer/write/behavior-gate binding')
        self._check_state()
        row = dict(logical_update=logical_update, writes_before=optimizer_writes,
                   writes_after=optimizer_writes, actual_optimizer_writes=0,
                   completed_outer_before=self.completed_outer, completed_outer_after=self.completed_outer,
                   consumed_learning_rates=_learning_rates(self.optimizer),
                   next_learning_rates=_learning_rates(self.optimizer),
                   scheduler_last_epoch_before=self.scheduler.last_epoch,
                   scheduler_last_epoch_after=self.scheduler.last_epoch,
                   scheduler_step_count_before=self.scheduler._step_count,
                   scheduler_step_count_after=self.scheduler._step_count,
                   batch_gate_sha256=batch_gate_sha256, scheduler_advances=0)
        self._append(dict(kind='BEGIN', **row))
        self.pending = row

    def agreement_state(self):
        """Exclude rank/process identity, include the actual shared counters."""
        return dict(enabled=True, **self.identity,
                    completed_outer=self.completed_outer, pending=self.pending,
                    prepared=self.prepared, last_epoch=self.scheduler.last_epoch,
                    step_count=self.scheduler._step_count,
                    learning_rates=_learning_rates(self.optimizer))

    def prepare_outer(self, *, optimizer_writes):
        _need(self.pending is not None and not self.rejected, 'No active accepted outer to complete')
        _need(self.prepared is None, 'Duplicate scheduler commit preparation')
        self._check_state()
        before = self.pending['writes_before']
        _need(type(optimizer_writes) is int and optimizer_writes - before == WRITES_PER_OUTER,
              'Exactly20 dynamically observed real optimizer writes required before scheduler advance')
        row = dict(self.pending)
        row.update(kind='COMMIT_PREPARED', writes_after=optimizer_writes,
                   actual_optimizer_writes=optimizer_writes-before)
        self._append(row)
        self.prepared = row
        return row

    def commit_prepared(self):
        """After collective preparation: advance once, then persist the fact.

        A failure here is fatal post-commit evidence, never a brake rejection
        with zero advance. Real Adam writes and scheduler state are retained.
        """
        _need(self.pending is not None and self.prepared is not None and not self.rejected,
              'No durable scheduler commit preparation')
        self._check_state()
        self.scheduler.step()
        after = self.completed_outer + 1
        _need(self.scheduler.last_epoch == after and self.scheduler._step_count == after + 1 and
              _learning_rates(self.optimizer) == [expected_lr(after)] * len(self.optimizer.param_groups),
              'Actual scheduler did not advance exactly once')
        row = dict(self.prepared)
        row.update(kind='COMPLETE', completed_outer_after=after,
                   next_learning_rates=_learning_rates(self.optimizer),
                   scheduler_last_epoch_after=self.scheduler.last_epoch,
                   scheduler_step_count_after=self.scheduler._step_count, scheduler_advances=1)
        self._append(row)
        self.completed_outer, self.pending, self.prepared = after, None, None
        return row

    def complete_outer(self, *, optimizer_writes):
        """Local mechanics helper; the real worker uses collective stages."""
        self.prepare_outer(optimizer_writes=optimizer_writes)
        return self.commit_prepared()

    def failure_snapshot(self, *, optimizer_writes, reason, commit_started=True):
        """Actual scalar fallback also usable when the durable ledger fails."""
        row = dict(kind='POST_COMMIT_FAILURE' if commit_started else 'PRE_COMMIT_EVIDENCE_FAILURE',
                   logical_update=self.pending['logical_update'] if self.pending else self.scheduler.last_epoch,
                   writes_after=optimizer_writes, reason=str(reason),
                   actual_scheduler_last_epoch=self.scheduler.last_epoch,
                   actual_scheduler_step_count=self.scheduler._step_count,
                   actual_learning_rates=_learning_rates(self.optimizer),
                   retained_pending=self.pending, retained_prepared=self.prepared,
                   fatal_no_automatic_resume=True, commit_started=commit_started, run_id=self.identity['run_id'],
                   rank=self.rank, process_pid=self.process_pid, process_start_ticks=self.process_start_ticks)
        return row

    def fail_after_commit(self, *, optimizer_writes, reason):
        """Best-effort fatal receipt after commit started, with actual counters."""
        row = self.failure_snapshot(optimizer_writes=optimizer_writes, reason=reason)
        self.rejected = True
        self._append(row)
        return row

    def reject_outer(self, *, optimizer_writes, reason):
        _need(self.pending is not None, 'No pending outer rejection to preserve')
        before = self.pending['writes_before']
        _need(type(optimizer_writes) is int and before <= optimizer_writes <= before + WRITES_PER_OUTER,
              'Invalid actual partial optimizer count')
        unchanged = (self.scheduler.last_epoch == self.pending['scheduler_last_epoch_before'] and
                     self.scheduler._step_count == self.pending['scheduler_step_count_before'] and
                     _learning_rates(self.optimizer) == self.pending['consumed_learning_rates'])
        row = dict(self.pending)
        row.update(kind='REJECTED', writes_after=optimizer_writes,
                   actual_optimizer_writes=optimizer_writes - before, reason=str(reason),
                   next_learning_rates=_learning_rates(self.optimizer),
                   scheduler_last_epoch_after=self.scheduler.last_epoch,
                   scheduler_step_count_after=self.scheduler._step_count,
                   scheduler_advances=self.scheduler._step_count - self.pending['scheduler_step_count_before'],
                   scheduler_unchanged_on_rejection=unchanged)
        self._append(row)
        self.rejected, self.pending, self.prepared = True, None, None
        _need(unchanged, 'Rejected outer had an unauthorized scheduler advance')
        return row

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
