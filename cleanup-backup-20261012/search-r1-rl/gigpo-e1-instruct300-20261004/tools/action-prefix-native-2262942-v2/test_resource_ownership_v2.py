"""CPU-only controlled ownership classes; no GPU and no real PID mutation."""
import copy
import unittest
from resource_ownership_v2 import OwnershipObserver


class Backend:
    def __init__(self):
        self.values = {100: dict(state='S', start=123, cgroup='1:freezer:/job_25/step_0\n', tgid=100, threads=1)}
        self.fds = {}
        self.next_fd = 10
        self.closed = []
        self.identity_queues = {}
        self.cgroup_queues = {}
        self.task_list = [100]
    def identity(self, pid):
        if pid not in self.values:
            raise FileNotFoundError(pid)
        queue = self.identity_queues.get(pid)
        value = queue.pop(0) if queue else self.values[pid]
        return dict(state=value['state'], start=value['start'], raw_stat=str(value))
    def cgroup(self, pid):
        if pid not in self.values:
            raise FileNotFoundError(pid)
        queue = self.cgroup_queues.get(pid)
        return queue.pop(0) if queue else self.values[pid]['cgroup']
    def status(self, pid):
        value = self.values[pid]
        return dict(tgid=value['tgid'], threads=value['threads'], raw_status=str(value))
    def task_ids(self, pid):
        return list(self.task_list)
    def open_fd(self, pid):
        fd = self.next_fd; self.next_fd += 1
        self.fds[fd] = dict(pid=pid, exited=False)
        return fd
    def fd_info(self, fd):
        return dict(self.fds[fd], raw_fdinfo=str(self.fds[fd]))
    def close_fd(self, fd):
        self.closed.append(fd)


class OwnershipClasses(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.events = []
        self.obs = OwnershipObserver('25', self.events.append, self.backend)
    def cache(self):
        self.assertIs(self.obs.own(100), True)
        return self.obs.known_own[100]['fd']
    def zombie(self, exited=True):
        fd = self.cache()
        self.backend.values[100].update(state='Z', cgroup='1:freezer:/\n')
        self.backend.fds[fd]['exited'] = exited
        return fd
    def live_threads(self):
        self.zombie(False)
        self.backend.values[100]['threads'] = 2
        self.backend.values[101] = dict(state='S', start=124, cgroup='1:freezer:/job_25/step_0\n', tgid=100, threads=2)
        self.backend.task_list = [100, 101]
    def test_live_membership_cached_with_held_identity(self):
        self.cache(); self.assertEqual(self.obs.known_own[100]['start'], 123)
    def test_component_job_name_is_exact(self):
        for group in ('/job_250/step_0', '/prefix_job_25/step_0'):
            self.backend.values[100]['cgroup'] = '1:freezer:' + group
            self.assertIs(self.obs.own(100), False)
    def test_live_move_is_not_historically_exempted(self):
        self.cache(); self.backend.values[100]['cgroup'] = '1:freezer:/job_99/step_0\n'
        self.assertIs(self.obs.own(100), False)
    def test_cached_fully_exited_zombie_accepted(self):
        self.zombie(True); self.assertIs(self.obs.own(100), True)
        self.assertTrue(self.obs.zombie_proofs[100]['pidfd_readable'])
    def test_uncached_zombie_foreign(self):
        self.backend.values[100].update(state='Z', cgroup='1:freezer:/\n')
        self.assertIs(self.obs.own(100), False); self.assertFalse(self.obs.known_own)
    def test_uncached_zombie_even_cgroup_contains_job_not_cached(self):
        self.backend.values[100]['state'] = 'Z'
        self.assertIs(self.obs.own(100), False); self.assertFalse(self.obs.known_own)
    def test_reaped_identity_none(self):
        self.zombie(True); del self.backend.values[100]
        self.assertIsNone(self.obs.own(100)); self.assertEqual(self.events[-1]['reason'], 'PROC_IDENTITY_DISAPPEARED')
    def test_reused_foreign_pid_start_never_exempted(self):
        self.zombie(True); self.backend.values[100].update(start=999, state='S')
        self.assertIs(self.obs.own(100), False)
    def test_reused_foreign_zombie_start_never_exempted(self):
        self.zombie(True); self.backend.values[100]['start'] = 999
        self.assertIs(self.obs.own(100), False)
    def test_cached_fd_pid_mismatch_never_historical_exemption(self):
        fd = self.zombie(True); self.backend.fds[fd]['pid'] = -1
        self.assertIs(self.obs.own(100), False)
    def test_two_read_exit_race_retries_without_losing_cache(self):
        self.zombie(True)
        z = copy.deepcopy(self.backend.values[100]); s = dict(z, state='S')
        self.backend.identity_queues[100] = [s, z]
        self.assertIs(self.obs.own(100), True)
        self.assertFalse(self.events[-1]['matched_reads'][0]['stable'])
    def test_always_unstable_is_none(self):
        live = copy.deepcopy(self.backend.values[100]); changed = dict(live, start=999)
        self.backend.identity_queues[100] = [live, changed] * 3
        self.assertIsNone(self.obs.own(100)); self.assertFalse(self.obs.known_own)
    def test_live_scheduling_state_changes_keep_same_identity(self):
        live = copy.deepcopy(self.backend.values[100])
        for before in sorted(self.obs.LIVE_STATES):
            for after in sorted(self.obs.LIVE_STATES):
                self.backend.identity_queues[100] = [dict(live, state=before), dict(live, state=after)]
                pair = self.obs._pair(100)
                self.assertTrue(pair['stable'], (before, after))
                self.assertEqual(pair['before']['state'], before)
                self.assertEqual(pair['after']['state'], after)
                self.assertEqual(pair['state_categories'], ['live', 'live'])
    def test_r_s_repeated_transition_is_owned_not_none(self):
        self.cache(); live = copy.deepcopy(self.backend.values[100])
        self.backend.identity_queues[100] = [dict(live, state='R'), dict(live, state='S')] * 3
        self.assertIs(self.obs.own(100), True)
    def test_live_scheduling_change_cannot_hide_start_change(self):
        self.cache(); live = copy.deepcopy(self.backend.values[100])
        self.backend.identity_queues[100] = [dict(live, state='R'), dict(live, state='S', start=999)] * 3
        self.assertIsNone(self.obs.own(100))
    def test_live_scheduling_change_cannot_hide_cgroup_change(self):
        self.cache(); live = copy.deepcopy(self.backend.values[100])
        self.backend.identity_queues[100] = [dict(live, state='R'), dict(live, state='S')] * 3
        self.backend.cgroup_queues[100] = ['1:freezer:/job_25/step_0\n', '1:freezer:/outside\n'] * 3
        self.assertIsNone(self.obs.own(100))
    def test_unknown_state_not_owned_or_cached(self):
        self.backend.values[100]['state'] = '?'
        self.assertIs(self.obs.own(100), False); self.assertFalse(self.obs.known_own)
    def test_live_thread_r_s_same_identity_scope_owned(self):
        self.live_threads(); live = copy.deepcopy(self.backend.values[101])
        self.backend.identity_queues[101] = [dict(live, state='R'), dict(live, state='S')] * 2
        self.assertIs(self.obs.own(100), True)
    def test_leader_z_all_live_threads_still_owned(self):
        self.live_threads(); self.assertIs(self.obs.own(100), True)
        self.assertEqual(self.events[-1]['reason'], 'OWN_CURRENT_LIVE_THREADGROUP')
        self.assertFalse(self.obs.zombie_proofs)
    def test_leader_z_live_thread_moved_foreign_rejected(self):
        self.live_threads(); self.backend.values[101]['cgroup'] = '1:freezer:/outside\n'
        self.assertIs(self.obs.own(100), False)
        self.assertEqual(self.events[-1]['reason'], 'LIVE_THREAD_OUTSIDE_OWN_JOB')
    def test_leader_z_incomplete_thread_inventory_rejected(self):
        self.live_threads(); self.backend.task_list = [100]
        self.assertIs(self.obs.own(100), False)
    def test_leader_z_wrong_tgid_rejected(self):
        self.live_threads(); self.backend.values[101]['tgid'] = 500
        self.assertIs(self.obs.own(100), False)
    def test_live_thread_start_reused_during_scope_proof_rejected(self):
        self.live_threads(); a = copy.deepcopy(self.backend.values[101]); b = dict(a, start=125)
        self.backend.identity_queues[101] = [a, a, b, b]
        self.assertIs(self.obs.own(100), False)
    def test_leader_z_unreadable_live_thread_rejected(self):
        self.live_threads(); del self.backend.values[101]
        self.assertIs(self.obs.own(100), False)
    def test_close_held_fds(self):
        fd = self.cache(); self.obs.close()
        self.assertIn(fd, self.backend.closed); self.assertFalse(self.obs.known_own)
    def test_every_false_none_preserves_decision_metadata(self):
        self.backend.values[100]['cgroup'] = '1:freezer:/outside\n'
        self.assertIs(self.obs.own(100), False)
        event = self.events[-1]
        self.assertIn('cached_identity', event); self.assertIn('matched_reads', event)
        self.assertIn('raw_stat', event['matched_reads'][0]['before'])
        self.assertIn('raw_cgroup', event['matched_reads'][0]['before'])


if __name__ == '__main__': unittest.main(verbosity=2)
