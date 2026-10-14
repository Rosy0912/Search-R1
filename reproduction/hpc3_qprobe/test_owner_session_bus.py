"""Safety boundaries for the exact stale CPU-test session cleanup."""
import copy
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe.audit_scripts import owner_session_bus as target


class OwnerScopeProofTest(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(target.time,'time',return_value=1791972400.)
        self.clock.start();self.addCleanup(self.clock.stop)
        rows = ['PID PPID SID PGID STARTED NLWP COMMAND']
        for pid in target.TEST_PIDS:
            main = pid == 2337454
            args = target.TEST_COMMAND if main else 'from multiprocessing.spawn import spawn_main;'
            rows.append('%s %s 2337454 2337454 Wed Oct 14 %s 2026 1 /env/python %s' % (pid, '1' if main else '2337454', '15:48:20' if main else '15:48:56', args))
        self.proof = dict(epoch=1791972399., session=dict(returncode=0,stdout='Id=680116\nUser=14090\nName=jxu957\nTimestamp=Wed 2026-10-14 15:48:20 CST\nService=sshd\nScope=session-680116.scope\nLeader=0\nState=closing\n'), processes=dict(returncode=0,stdout='\n'.join(rows)), scope_pids='\n'.join(map(str,target.TEST_PIDS)), **{'pids.current':'499','pids.max':'500'})
        self.inspected = dict(real_bus_uid=14090,session=dict(Timestamp=target.SESSION_TIMESTAMP))

    def test_actual_single_test_scope_allowed(self):
        target.verify_proof(self.proof,self.inspected)

    def test_unexpected_scope_process_rejected(self):
        self.proof['scope_pids'] += '\n123'
        with self.assertRaises(ValueError):target.verify_proof(self.proof,self.inspected)

    def test_reused_pid_birth_rejected(self):
        self.proof['processes']['stdout'] = self.proof['processes']['stdout'].replace('15:48:56','18:00:00')
        with self.assertRaises(ValueError):target.verify_proof(self.proof,self.inspected)

    def test_other_process_command_rejected(self):
        self.proof['processes']['stdout'] = self.proof['processes']['stdout'].replace('from multiprocessing.spawn import spawn_main;','train.py')
        with self.assertRaises(ValueError):target.verify_proof(self.proof,self.inspected)

    def test_foreign_real_bus_uid_rejected(self):
        self.inspected['real_bus_uid'] = 21930
        with self.assertRaises(ValueError):target.verify_proof(self.proof,self.inspected)

    def test_stale_or_future_proof_rejected(self):
        for epoch in (1791972300.,1791972401.):
            proof=copy.deepcopy(self.proof);proof['epoch']=epoch
            with self.assertRaises(ValueError):target.verify_proof(proof,self.inspected)

    def test_reused_session_birth_rejected(self):
        self.inspected['session']['Timestamp'] += 1
        with self.assertRaises(ValueError):target.verify_proof(self.proof,self.inspected)

    def test_failed_observer_query_rejected(self):
        self.proof['processes']['returncode'] = 1
        with self.assertRaises(ValueError):target.verify_proof(self.proof,self.inspected)


if __name__ == '__main__':unittest.main()
