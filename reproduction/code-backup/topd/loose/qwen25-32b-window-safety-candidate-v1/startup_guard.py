"""Explicit parent-only wrapper around pinned multi-PID window safety; no child/source shadowing."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

SAFETY_SHA = '3316d920f76256c5b45c5bf5e3c16c5c89a26a95cdaf821eb12ab612dc7a7384'
IDENTITY_CHANGED = 'Process identities changed during SM window'
LOW_HIGH = 'Device SM increased after upper-bound sample'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def process_uid(pid):
    return (Path('/proc') / str(pid)).stat().st_uid


def held_leases():
    from search_r1.baselines.topd_local_gpu import lease_ownership
    leases = json.loads(os.environ.get('SEARCH_R1_GPU_LEASES', '[]'))
    require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT') == '1' and len(leases) == 8
            and len({r['uuid'] for r in leases}) == 8, 'Require8 actual inherited leases')
    for row in leases:
        lease_ownership(row)
    return {r['uuid'] for r in leases}


class StartupGuard:
    def __init__(self, sealed, *, expected_path, trace_path, startup_seconds=120, retry_call_seconds=30,
                 max_attempts=6, retry_pause=.25):
        path = Path(sealed.__file__).resolve()
        require(path == Path(expected_path).resolve(), 'Unexpected sealed module path')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == SAFETY_SHA, 'Pinned window safety bytes differ')
        require(0 < startup_seconds <= 120 and 0 < retry_call_seconds <= 30 and 1 <= max_attempts <= 6
                and 0 <= retry_pause <= 1, 'Invalid bounded startup budget')
        self.sealed, self.path = sealed, path
        self.original_gate, self.original_snapshot = sealed.active_gate, sealed.nvml_snapshot
        self.original_collect = sealed.collect_process_windows
        self.trace_path = Path(trace_path)
        self.local, self.write_lock = threading.local(), threading.Lock()
        self.deadline = None
        self.startup_seconds = startup_seconds
        self.retry_call_seconds = retry_call_seconds
        self.max_attempts, self.retry_pause = max_attempts, retry_pause
        self.closed = False
        self.ready_check = lambda: False
        self.uid_reader, self.lease_reader = process_uid, held_leases
        self.identity_reader = sealed.pid_identity
        self.owner_uid = os.getuid()
        require(not getattr(sealed, '_startup_guard_spy', False), 'Snapshot spy already installed')
        sealed._startup_guard_spy = True
        sealed.nvml_snapshot = self._snapshot_spy
        sealed.collect_process_windows = self._collect_spy

    def _snapshot_spy(self):
        original = self.original_snapshot()
        captures = getattr(self.local, 'captures', None)
        if captures is not None:
            record = dict(epoch=time.time(), raw_snapshot=copy.deepcopy(original), uids={}, uid_errors={})
            # The validator sees the same object; sidecar failure never guesses identity.
            for pid in original['identities']:
                try:
                    record['uids'][pid] = self.uid_reader(int(pid))
                    require(self.identity_reader(int(pid)) == original['identities'][pid],
                            'PID changed while reading UID sidecar')
                except Exception as exc:
                    record['uid_errors'][pid] = type(exc).__name__ + ': ' + str(exc)
            captures.append(record)
        return original

    def _collect_spy(self, snapshot, foreign):
        original = self.original_collect(snapshot, foreign)
        records = getattr(self.local, 'windows', None)
        if records is not None:
            records.append(dict(epoch=time.time(), snapshot=copy.deepcopy(snapshot),
                                foreign=[list(v) for v in sorted(foreign)],
                                windows=copy.deepcopy(original[0]), query_completed_us=original[1]))
        return original  # Exact original object; no timestamps filtered or modified.


    def _write(self, row):
        row.update(wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   sealed_path=str(self.path), sealed_sha256=SAFETY_SHA,
                   thread_id=threading.get_ident(), epoch=time.time())
        with self.write_lock:
            with self.trace_path.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def _validate_captures(self, captures, runtime, history):
        require(len(captures) >= 2, 'Missing actual internal snapshot pair')
        uuids = self.lease_reader()  # Revalidate held FDs before retry authorization.
        previous = None
        foreign_ids = set(runtime['baseline']['identities'])
        changed = False
        for cap in captures:
            snap = cap['raw_snapshot']
            self.sealed.accounting_gate(snap, runtime['baseline'], uuids,
                runtime['retriever_identity'], runtime['cgroup'], {}, False, validate_sm=False)
            require(not cap['uid_errors'], 'Unknown UID during actual internal snapshot')
            ids = snap['identities']
            require(set(ids) == set(cap['uids']), 'Incomplete UID sidecar')
            for pid in set(ids) - foreign_ids:
                require(cap['uids'][pid] == self.owner_uid, 'Own cgroup PID has foreign UID')
            if previous is not None:
                old = previous['raw_snapshot']['identities']
                for pid in set(ids) & set(old):
                    require(ids[pid] == old[pid] and cap['uids'][pid] == previous['uids'][pid],
                            'Same PID identity/UID changed')
                delta = set(ids) ^ set(old)
                require(not delta & foreign_ids, 'Protected baseline changed during window')
                changed = changed or bool(delta)
            for pid, identity in ids.items():
                actual = (identity, cap['uids'][pid])
                require(pid not in history or history[pid] == actual, 'Cross-attempt PID identity/UID changed')
                history[pid] = copy.deepcopy(actual)
            previous = cap
        return changed

    def _own_churn_only(self, captures, runtime, history):
        require(self._validate_captures(captures, runtime, history),
                'Identity error was not added/removed own PID churn')

    def _poll_ready(self):
        if self.ready_check():
            self.close_startup()

    def _stable_transition(self, captures, runtime, history):
        changed = self._validate_captures(captures, runtime, history)
        require(not changed, 'Telemetry transition changed process identity')
        first, last = [cap['raw_snapshot'] for cap in (captures[0], captures[-1])]
        initial = {g['uuid']: g for g in first['gpus']}
        transitioned = {g['uuid'] for g in last['gpus']
                        if initial[g['uuid']]['sm'] <= 15 and g['sm'] > 15}
        require(transitioned, 'No actual low-to-high GPU pair captured')
        known_foreign = {(p['uuid'], p['pid']) for p in runtime['baseline']['applications']}
        require(all(any(p['uuid']==uid and (p['uuid'], p['pid']) not in known_foreign
                        for p in last['applications']) for uid in transitioned),
                'No known own GPU owner for telemetry remeasurement')
        return transitioned

    def _fresh_sm_result(self, result, transitioned):
        gpus = {g['uuid']: g for g in result['raw_snapshot']['gpus']}
        for uid in transitioned:
            evidence = result['foreign_sm_evidence'][uid]
            bound = result['foreign_sm_aggregate_upper_bounds'][uid]
            require(0 <= bound <= 15 and evidence['upper_bound'] == bound,
                    'Fresh original foreign SM result absent/invalid')
            method = evidence['method']
            require(method in ('DEVICE_TOTAL_AGGREGATE_UPPER_BOUND',
                               'FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW'),
                    'Fresh original evidence unsupported')
            if method == 'DEVICE_TOTAL_AGGREGATE_UPPER_BOUND':
                require(gpus[uid]['sm'] <= 15, 'Fresh device total is not a safe aggregate bound')

    def active_gate(self, runtime, pre_model=False):
        require(not hasattr(self.local, 'captures'), 'Nested validator call')
        self._poll_ready()
        history = {}; did_retry = False; did_identity_retry = False
        call_deadline = None; transitioned = set(); previous_kind = None; measurement_identity = None
        def same_measurement_identity(captures):
            if measurement_identity is not None:
                for cap in captures:
                    require((cap['raw_snapshot']['identities'], cap['uids']) == measurement_identity,
                            'Measurement process identity/UID changed across attempts')
        def effective_deadline():
            return min(call_deadline, self.deadline) if did_identity_retry else call_deadline
        for attempt in range(1, self.max_attempts + 1):
            self._poll_ready()
            if attempt > 1:
                require(time.monotonic() < effective_deadline()
                        and (previous_kind != 'IDENTITY_CHURN' or not self.closed),
                        'Startup closed/expired before fresh retry')
            self.local.captures = []; self.local.windows = []
            try:
                result = self.original_gate(runtime, pre_model=pre_model)
            except Exception as exc:
                captures = self.local.captures; self._poll_ready()
                typed = type(exc) is RuntimeError and not pre_model and attempt < self.max_attempts
                identity_retry = typed and str(exc) == IDENTITY_CHANGED and not self.closed and (
                    self.deadline is None or time.monotonic() < self.deadline)
                measurement_retry = typed and str(exc) == LOW_HIGH
                eligible = identity_retry or measurement_retry
                reason = None; kind = None
                if eligible:
                    try:
                        same_measurement_identity(captures)
                        if identity_retry:
                            self._own_churn_only(captures, runtime, history)
                            self._poll_ready()
                            require(not self.closed, 'Readiness closed before identity retry authorization')
                            if self.deadline is None:
                                self.deadline = time.monotonic() + self.startup_seconds
                            did_identity_retry = True; kind = 'IDENTITY_CHURN'
                        else:
                            transitioned |= self._stable_transition(captures, runtime, history)
                            if measurement_identity is None:
                                measurement_identity = copy.deepcopy((captures[0]['raw_snapshot']['identities'], captures[0]['uids']))
                            kind = 'FRESH_LOW_TO_HIGH_MEASUREMENT'
                        if call_deadline is None:
                            call_deadline = time.monotonic() + self.retry_call_seconds
                        require(time.monotonic() < effective_deadline(), 'Fresh measurement budget exhausted')
                        did_retry = True; previous_kind = kind
                    except Exception as rejection:
                        eligible, reason = False, str(rejection)
                self._write(dict(attempt=attempt, accepted=False, retry_authorized=eligible,
                                 retry_kind=kind, error=str(exc), rejection=reason, captures=captures, original_windows=self.local.windows,
                                 foreign_sm_status='UNKNOWN_REJECTED_TRANSITION' if str(exc)==LOW_HIGH else 'NOT_ACCEPTED'))
                if not eligible: raise
            else:
                try:
                    self._poll_ready()
                    if did_retry:
                        require(time.monotonic() < effective_deadline(),
                                'Startup validator completed after bounded deadline')
                    self._validate_captures(self.local.captures, runtime, history)
                    same_measurement_identity(self.local.captures)
                    if transitioned: self._fresh_sm_result(result, transitioned)
                    if did_retry:
                        require(time.monotonic() < effective_deadline(),
                                'Startup acceptance exceeded bounded deadline')
                except Exception as rejection:
                    self._write(dict(attempt=attempt, accepted=False, retry_authorized=False,
                                     rejection=str(rejection), captures=self.local.captures))
                    raise
                self._write(dict(attempt=attempt, accepted=True, retry_authorized=False,
                                 validator='PINNED_MULTI_PID_WINDOW_FULL_SM_GATE', captures=self.local.captures, original_windows=self.local.windows,
                                 observed_foreign_sm_evidence=result.get('foreign_sm_evidence'),
                                 observed_foreign_sm_upper_bounds=result.get('foreign_sm_aggregate_upper_bounds'),
                                 fresh_measurement_uuids=sorted(transitioned),
                                 fresh_foreign_sm_evidence=result.get('foreign_sm_evidence') if transitioned else None,
                                 fresh_foreign_sm_upper_bounds=result.get('foreign_sm_aggregate_upper_bounds') if transitioned else None))
                return result
            finally:
                del self.local.captures;del self.local.windows
            time.sleep(self.retry_pause)
        raise RuntimeError('Bounded startup attempts exhausted')

    def close_startup(self):
        """Caller closes at model-ready boundary; later churn is never retried."""
        self.closed = True
