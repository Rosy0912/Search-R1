"""Explicit parent-only wrapper around sealed c7; no child/source shadowing."""
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time

C7_SHA = 'c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d'
IDENTITY_CHANGED = 'Process identities changed during SM window'


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
    def __init__(self, sealed, *, expected_path, trace_path, startup_seconds=20,
                 max_attempts=6, retry_pause=.25):
        path = Path(sealed.__file__).resolve()
        require(path == Path(expected_path).resolve(), 'Unexpected sealed module path')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == C7_SHA, 'Sealed c7 bytes differ')
        require(0 < startup_seconds <= 30 and 1 <= max_attempts <= 12
                and 0 <= retry_pause <= 1, 'Invalid bounded startup budget')
        self.sealed, self.path = sealed, path
        self.original_gate, self.original_snapshot = sealed.active_gate, sealed.nvml_snapshot
        self.trace_path = Path(trace_path)
        self.local, self.write_lock = threading.local(), threading.Lock()
        self.deadline = None
        self.startup_seconds = startup_seconds
        self.max_attempts, self.retry_pause = max_attempts, retry_pause
        self.closed = False
        self.uid_reader, self.lease_reader = process_uid, held_leases
        self.identity_reader = sealed.pid_identity
        self.owner_uid = os.getuid()
        require(not getattr(sealed, '_startup_guard_spy', False), 'Snapshot spy already installed')
        sealed._startup_guard_spy = True
        sealed.nvml_snapshot = self._snapshot_spy

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

    def _write(self, row):
        row.update(wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   sealed_path=str(self.path), sealed_sha256=C7_SHA,
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

    def active_gate(self, runtime, pre_model=False):
        require(not hasattr(self.local, 'captures'), 'Nested validator call')
        history = {}
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                require(not self.closed and time.monotonic() < self.deadline,
                        'Startup closed/expired before fresh retry')
            self.local.captures = []
            try:
                result = self.original_gate(runtime, pre_model=pre_model)
            except Exception as exc:
                captures = self.local.captures
                eligible = (type(exc) is RuntimeError and str(exc) == IDENTITY_CHANGED
                            and not pre_model and not self.closed
                            and (self.deadline is None or time.monotonic() < self.deadline) and attempt < self.max_attempts)
                reason = None
                if eligible:
                    try:
                        self._own_churn_only(captures, runtime, history)
                        if self.deadline is None:
                            self.deadline = time.monotonic() + self.startup_seconds
                    except Exception as rejection:
                        eligible, reason = False, str(rejection)
                self._write(dict(attempt=attempt, accepted=False, retry_authorized=eligible,
                                 error=str(exc), rejection=reason, captures=captures))
                if not eligible:
                    raise
            else:
                try:
                    self._validate_captures(self.local.captures, runtime, history)
                except Exception as rejection:
                    self._write(dict(attempt=attempt, accepted=False, retry_authorized=False,
                                     rejection=str(rejection), captures=self.local.captures))
                    raise
                self._write(dict(attempt=attempt, accepted=True, retry_authorized=False,
                                 validator='UNCHANGED_C7_FULL_SM_GATE', captures=self.local.captures))
                return result
            finally:
                del self.local.captures
            time.sleep(self.retry_pause)
        raise RuntimeError('Bounded startup attempts exhausted')

    def close_startup(self):
        """Caller closes at model-ready boundary; later churn is never retried."""
        self.closed = True
