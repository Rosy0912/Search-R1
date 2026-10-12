"""Matched /proc ownership reads, with held identity and explicit thread scope.

No process-name exemption. No NVML or policy thresholds in this module.
"""
import os
from pathlib import Path
import select
import time


class ProcBackend:
    def identity(self, pid):
        raw = Path('/proc', str(pid), 'stat').read_text()
        fields = raw.rsplit(')', 1)[1].split()
        return dict(state=fields[0], start=int(fields[19]), raw_stat=raw)

    def cgroup(self, pid):
        return Path('/proc', str(pid), 'cgroup').read_text()

    def status(self, pid):
        raw = Path('/proc', str(pid), 'status').read_text()
        values = dict(line.split(':', 1) for line in raw.splitlines() if ':' in line)
        return dict(tgid=int(values['Tgid'].strip()), threads=int(values['Threads'].strip()), raw_status=raw)

    def task_ids(self, pid):
        return sorted(int(p.name) for p in Path('/proc', str(pid), 'task').iterdir() if p.name.isdigit())

    def open_fd(self, pid):
        return os.pidfd_open(pid)

    def fd_info(self, fd):
        raw = Path('/proc/self/fdinfo', str(fd)).read_text()
        values = dict(line.split(':', 1) for line in raw.splitlines() if ':' in line)
        return dict(pid=int(values['Pid'].strip()), exited=bool(select.select([fd], [], [], 0)[0]), raw_fdinfo=raw)

    def close_fd(self, fd):
        os.close(fd)


class OwnershipObserver:
    def __init__(self, job, emit, backend=None):
        self.job = str(job)
        self.emit = emit
        self.backend = backend or ProcBackend()
        self.known_own = {}
        self.zombie_proofs = {}

    def has_job(self, text):
        return any('job_' + self.job in line.split(':', 2)[-1].strip().split('/') for line in text.splitlines())

    def _pair(self, pid):
        pair = dict(before={}, after={}, stable=False)
        try:
            pair['before'] = self.backend.identity(pid)
            pair['before']['raw_cgroup'] = self.backend.cgroup(pid)
            pair['after'] = self.backend.identity(pid)
            pair['after']['raw_cgroup'] = self.backend.cgroup(pid)
            a, b = pair['before'], pair['after']
            pair['stable'] = ((a['state'], a['start'], a['raw_cgroup']) ==
                              (b['state'], b['start'], b['raw_cgroup']))
            return pair
        except BaseException as exc:
            pair['read_error'] = repr(exc)
            exc.ownership_partial_pair = pair
            raise

    def _cached(self, pid):
        cached = self.known_own.get(pid)
        if cached is None:
            return None
        return dict(start=cached['start'], fd=cached['fd'], **self.backend.fd_info(cached['fd']))

    def _emit(self, pid, decision, reason, reads, cached, thread_scope=None):
        self.emit(dict(event='OWNERSHIP_IDENTITY', epoch=time.time(), pid=pid,
                       decision=decision, reason=reason, matched_reads=reads,
                       cached_identity=cached, known_own_cache_size=len(self.known_own),
                       thread_scope=thread_scope))

    def _live_group(self, pid, start, cached, reads):
        # A zombie leader alone is not a proof that the entire group exited.
        scope = dict(proof='current live thread membership, not historical zombie exemption', threads=[])
        try:
            tids = self.backend.task_ids(pid)
            status = self.backend.status(pid)
            scope.update(task_ids_before=tids, leader_status_before=status)
            if not tids or pid not in tids or len(tids) != status['threads'] or status['tgid'] != pid:
                return False, 'INCOMPLETE_LIVE_THREADGROUP_INVENTORY', scope
            active = []
            for tid in tids:
                pair = self._pair(tid)
                thread_status = self.backend.status(tid)
                scope['threads'].append(dict(tid=tid, reads=pair, status=thread_status))
                if not pair['stable'] or thread_status['tgid'] != pid:
                    return False, 'UNSTABLE_OR_REUSED_THREAD_IDENTITY', scope
                state = pair['after']['state']
                if state not in ('Z', 'X', 'x'):
                    if not self.has_job(pair['after']['raw_cgroup']):
                        return False, 'LIVE_THREAD_OUTSIDE_OWN_JOB', scope
                    active.append((tid, pair))
            if not active:
                return False, 'PIDFD_NOT_EXITED_WITHOUT_PROVEN_OWN_LIVE_THREADS', scope
            # Revalidate every live TID after the complete group inventory read.
            for tid, pair in active:
                final = self._pair(tid)
                scope['threads'].append(dict(tid=tid, final_reads=final))
                if (not final['stable'] or final['after']['start'] != pair['after']['start']
                        or final['after']['state'] in ('Z', 'X', 'x')
                        or not self.has_job(final['after']['raw_cgroup'])
                        or self.backend.status(tid)['tgid'] != pid):
                    return False, 'LIVE_THREAD_MEMBERSHIP_CHANGED_DURING_PROOF', scope
            ending = self._pair(pid)
            final_fd = self.backend.fd_info(cached['fd'])
            end_tids = self.backend.task_ids(pid)
            end_status = self.backend.status(pid)
            scope.update(task_ids_after=end_tids, leader_status_after=end_status, leader_reads_after=ending,
                         pidfd_after=final_fd)
            if (not ending['stable'] or ending['after']['start'] != start
                    or ending['after']['state'] != 'Z' or end_tids != tids
                    or end_status['threads'] != len(tids) or end_status['tgid'] != pid
                    or final_fd['pid'] != pid or final_fd['exited']):
                return False, 'THREADGROUP_IDENTITY_CHANGED_DURING_PROOF', scope
            return True, 'OWN_CURRENT_LIVE_THREADGROUP', scope
        except (FileNotFoundError, ProcessLookupError):
            return False, 'UNREADABLE_OR_EXITING_LIVE_THREADGROUP', scope

    def own(self, pid):
        reads = []
        cached = None
        self.zombie_proofs.pop(pid, None)
        try:
            for attempt in range(3):
                existing = self.known_own.get(pid)
                cached = dict(existing) if existing is not None else None
                if cached is not None:
                    cached.update(self.backend.fd_info(existing['fd']))
                pair = self._pair(pid)
                reads.append(pair)
                if not pair['stable']:
                    continue
                state, start = pair['after']['state'], pair['after']['start']
                matches_cache = cached is not None and cached['start'] == start and cached['pid'] == pid
                if state in ('Z', 'X', 'x') and not matches_cache:
                    self._emit(pid, False, 'UNCACHED_OR_REUSED_EXITED_IDENTITY', reads, cached)
                    return False
                if self.has_job(pair['after']['raw_cgroup']):
                    if not matches_cache:
                        fd = self.backend.open_fd(pid)
                        try:
                            ending = self._pair(pid)
                            opened = self.backend.fd_info(fd)
                            reads.append(ending)
                            if (not ending['stable'] or ending['after']['start'] != start
                                    or ending['after']['state'] in ('Z', 'X', 'x')
                                    or opened['pid'] != pid or opened['exited']
                                    or not self.has_job(ending['after']['raw_cgroup'])):
                                self.backend.close_fd(fd)
                                continue
                        except BaseException:
                            self.backend.close_fd(fd)
                            raise
                        old = self.known_own.get(pid)
                        self.known_own[pid] = dict(start=start, fd=fd)
                        if old is not None:
                            self.backend.close_fd(old['fd'])
                        cached = self._cached(pid)
                        self._emit(pid, True, 'OWN_CGROUP_IDENTITY_CACHED', reads, cached)
                        state = ending['after']['state']
                    if state == 'Z' and not cached['exited']:
                        decision, reason, scope = self._live_group(pid, start, cached, reads)
                        self._emit(pid, decision, reason, reads, cached, scope)
                        return decision
                    return True
                if state == 'Z' and matches_cache:
                    if not cached['exited']:
                        decision, reason, scope = self._live_group(pid, start, cached, reads)
                        self._emit(pid, decision, reason, reads, cached, scope)
                        return decision
                    ending = self._pair(pid)
                    fd_after = self.backend.fd_info(cached['fd'])
                    reads.append(ending)
                    if (not ending['stable'] or ending['after']['state'] != 'Z'
                            or ending['after']['start'] != start or fd_after['pid'] != pid
                            or not fd_after['exited']):
                        continue
                    proof = dict(pid=pid, state='Z', start_ticks=start, original_pidfd_pid=pid,
                                 pidfd_readable=True, scope='held same identity; all threadgroup exited before reap')
                    self.zombie_proofs[pid] = proof
                    self._emit(pid, True, 'OWN_EXITED_ZOMBIE_IDENTITY', reads, cached)
                    return True
                self._emit(pid, False, 'CURRENT_CGROUP_FOREIGN_OR_UNKNOWN_EXITED_IDENTITY', reads, cached)
                return False
            self._emit(pid, None, 'UNSTABLE_PROC_IDENTITY_AFTER_MATCHED_RETRIES', reads, cached)
            return None
        except (FileNotFoundError, ProcessLookupError) as exc:
            if hasattr(exc, 'ownership_partial_pair'):
                reads.append(exc.ownership_partial_pair)
            self._emit(pid, None, 'PROC_IDENTITY_DISAPPEARED', reads, cached)
            return None
        except BaseException as exc:
            if hasattr(exc, 'ownership_partial_pair'):
                reads.append(exc.ownership_partial_pair)
            self._emit(pid, None, 'OWNERSHIP_READ_FAILED:' + repr(exc), reads, cached)
            raise

    def close(self):
        for cached in self.known_own.values():
            self.backend.close_fd(cached['fd'])
        self.known_own.clear()
