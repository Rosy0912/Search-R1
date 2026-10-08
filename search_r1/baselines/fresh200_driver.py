"""Accepted-outer bookkeeping; retained writes are distinct from skipped attempts."""
from dataclasses import dataclass, field, asdict


@dataclass
class Fresh200Progress:
    baseline_aligned: bool = False
    attempts: int = 0
    completed_outers: int = 0
    actual_adam_writes_per_rank: list = field(default_factory=lambda: [0] * 8)
    skipped_opportunities_per_rank: list = field(default_factory=lambda: [0] * 8)
    incomplete_attempts: int = 0
    recorded_attempts: int = 0

    def begin_attempt(self, actual_outer):
        if self.attempts != self.recorded_attempts:
            raise ValueError('Previous attempt has no recorded outcome')
        if type(actual_outer) is not int or actual_outer != self.completed_outers + 1 or not 1 <= actual_outer <= 200:
            raise ValueError('Attempt must use the next accepted outer, within 1..200')
        self.attempts += 1
        return self.attempts

    def record(self, events, actual_outer):
        from .fresh200_contract import validate_rank_updates
        validate_rank_updates(events,actual_outer)
        if any(e.get('baseline_aligned', False) != self.baseline_aligned for e in events):
            raise ValueError('Progress profile differs from worker update profile')
        if self.attempts != self.recorded_attempts + 1 or actual_outer != self.completed_outers + 1:
            raise ValueError('Attempt outcome is duplicate or out of order')
        expected = 20 if self.baseline_aligned or actual_outer < 50 else 10
        if (len(events) != 8 or any(type(e.get('rank')) is not int for e in events)
                or {e.get('rank') for e in events} != set(range(8))):
            raise ValueError('Attempt ledger requires all eight unique ranks')
        common = None
        for event in events:
            if event.get('actual_outer') != actual_outer or event.get('attempt_id') != self.attempts:
                raise ValueError('Attempt/accepted-outer identity differs')
            status = event.get('status')
            if status not in ('COMPLETE', 'COMPLETE_WITH_SKIPS', 'SKIPPED', 'ABORTED'):
                raise ValueError('Unsupported attempt status')
            names = ('optimizer_steps', 'processed_opportunities', 'skipped_opportunities', 'unprocessed_opportunities')
            counts = [event.get(name) for name in names]
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError('Attempt counts must be nonnegative integers')
            writes, processed, skipped, unprocessed = counts
            if (event.get('expected_optimizer_steps') != expected
                    or event.get('optimizer_opportunities_total') != expected
                    or processed + unprocessed != expected or writes + skipped != processed):
                raise ValueError('Attempt opportunity/write accounting differs')
            complete = status in ('COMPLETE', 'COMPLETE_WITH_SKIPS')
            if any(type(event.get(key)) is not bool for key in ('postabort','update_complete','attempt_complete')):
                raise ValueError('Attempt completion/abort flags must be booleans')
            if complete != (processed == expected and writes > 0 and not event['postabort']):
                raise ValueError('Accepted outer requires all opportunities and at least one commit')
            if (event['update_complete'] is not complete
                    or event['attempt_complete'] != (processed == expected and not event['postabort'])):
                raise ValueError('Attempt completion flag differs')
            if status == 'COMPLETE' and skipped or status == 'COMPLETE_WITH_SKIPS' and not skipped:
                raise ValueError('Complete status must describe actual skips')
            if status == 'SKIPPED' and writes:
                raise ValueError('Whole-skipped attempt cannot claim retained writes')
            if (status == 'SKIPPED' and event['postabort']) or (status == 'ABORTED' and not event['postabort']):
                raise ValueError('Abort status must describe post-update abortion')
            signature = (status, writes, processed, skipped, unprocessed, event['postabort'])
            if common is not None and common != signature:
                raise ValueError('Collective ranks disagree on attempt outcome')
            common = signature
        # Validate all ranks before mutating any cumulative counters.
        for event in events:
            rank = event['rank']
            self.actual_adam_writes_per_rank[rank] += event['optimizer_steps']
            self.skipped_opportunities_per_rank[rank] += event['skipped_opportunities']
        complete = events[0]['update_complete']
        if complete:
            self.completed_outers += 1
        else:
            self.incomplete_attempts += 1
        self.recorded_attempts = self.attempts
        return complete

    def state_dict(self):
        return dict(asdict(self), no_skip_expected_writes_per_rank=4000 if self.baseline_aligned else 2490,
                    actual_adam_writes_all_ranks=sum(self.actual_adam_writes_per_rank))


def persist_first_sampling_batch(data, directory, safety, source, run, shared_bindings):
    """Immutable actual first formal batch, captured before any actor write."""
    import copy
    import json
    import os
    from pathlib import Path
    from .fresh200_contract import verify_sampling_policy_binding, critical_source_hashes
    from .full_state import atomic_save
    from .fresh200_gate import digest
    attempt=data.meta_info.get('attempt_id');outer=data.meta_info.get('actual_outer')
    if type(attempt) is not int or attempt<1 or type(outer) is not int or not 1<=outer<=200:
        raise ValueError('Actual formal attempt/candidate identity required for sampling proof')
    import hashlib
    binding=hashlib.sha256(str(data.meta_info.get('policy_snapshot_id','')).encode()).hexdigest()
    mask=data.batch['sampling_mask'].bool()
    import torch
    try:
        delta=(data.batch['frozen_hf_log_probs'][mask].double()-data.batch['sampling_log_probs'][mask].double()).abs()
        maximum=float(delta.max()) if delta.numel() and torch.isfinite(delta).all() else None
    except Exception:maximum=None
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    payload=dict(version=1,status='ACTUAL_FORMAL_FIRST_BATCH_CAPTURED_NOT_RELEASE',source_commit=source,
        source_sha256=critical_source_hashes(),run_root=str(Path(run).resolve()),driver_pid=os.getpid(),
        attempt_id=attempt,actual_outer=outer,shared_input_bindings=copy.deepcopy(shared_bindings),
        tensors={k:v.detach().cpu().clone() for k,v in data.batch.items()},
        meta_info=copy.deepcopy(data.meta_info),non_tensor_batch=copy.deepcopy(data.non_tensor_batch),
        sampling_binding_sha256=binding,max_selected_logprob_abs_diff=maximum,
        selected_tokens=int(mask.sum()),sampling_logprob_atol=float(safety.get('sampling_logprob_atol',1e-3)),
        probability_contract='ACTUAL_NATIVE_BEHAVIOR_V2_HF_DIAGNOSTIC_SEPARATE',
        hf_native_drift_is_identity_gate=False,
        behavior_policy_role=data.meta_info.get('behavior_policy_role'))
    path=directory/'formal-first-sampling-batch.pt';atomic_save(payload,path)
    with path.open('rb') as handle:os.fsync(handle.fileno())
    report={k:payload[k] for k in ('version','status','source_commit','source_sha256','run_root','driver_pid','attempt_id','actual_outer','sampling_binding_sha256','max_selected_logprob_abs_diff','selected_tokens','sampling_logprob_atol','probability_contract','hf_native_drift_is_identity_gate','behavior_policy_role')}
    report.update(path=str(path.resolve()),sha256=digest(path))
    admission_error=None
    try:verify_sampling_policy_binding(data,safety)
    except Exception as error:admission_error=error
    from .policy_safety import NonfiniteSamplingPolicy
    numeric_rejected=isinstance(admission_error,NonfiniteSamplingPolicy)
    report['sampling_verifier_status']=('PASS_ACTUAL_BATCH_BINDING_NOT_RELEASE' if admission_error is None else
        'NUMERIC_REJECTED_ACTUAL_BATCH_BINDING' if numeric_rejected else 'FAIL_ACTUAL_BATCH_BINDING')
    report['sampling_verifier_error']=None if admission_error is None else str(admission_error)
    target=directory/'formal-first-sampling-batch.json'
    with target.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False);handle.flush();os.fsync(handle.fileno())
    directory_fd=os.open(str(directory),os.O_RDONLY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    print('FRESH200_FIRST_SAMPLING_PROOF_JSON='+json.dumps(report,allow_nan=False),flush=True)
    if admission_error is not None and not numeric_rejected:raise ValueError('First actual sampler binding failed; immutable raw proof retained: '+str(admission_error)) from admission_error
    return report
