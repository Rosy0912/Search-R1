import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reproduction.a800 import stop_failed_boundary as m

class FailedBoundaryTests(unittest.TestCase):
    def test_checkpoint_hash_change_prevents_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'weights').write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                m.verify_files({'checkpoint':d,'file_sha256':{'weights':'wrong'}})

    def test_cleanup_after_main_already_gone_is_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            logs=Path(d);log=logs/'train';log.write_text('step:300 - critic/rewards/mean:0.000\n')
            task=dict(pid=1,start_ticks=10,sid=3,uid=0,command='task')
            driver=dict(pid=2,start_ticks=11,sid=3,uid=0,command='driver')
            leader=dict(pid=3,start_ticks=12,sid=3,uid=0,command='leader')
            result=dict(task_identity=task,driver_identity=driver,session_leader_identity=leader,
                        owned_member_ledger=[task,driver,leader],completed_updates=[300],accepted=False)
            with patch.object(m,'matches',return_value=False),patch.object(m,'owned_session_records',return_value=[]),patch.object(m.time,'sleep'),patch.object(m,'send') as send:
                m.cleanup(result,logs,log)
                send.assert_not_called()
            self.assertEqual(result['state'],'STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT')
            self.assertFalse(json.loads((logs/'r1-stop300-status.json').read_text())['accepted'])

    def test_reused_session_member_is_not_signaled(self):
        with tempfile.TemporaryDirectory() as d:
            logs=Path(d);log=logs/'train';log.write_text('')
            r=dict(pid=3,start_ticks=12,sid=3,uid=0,command='leader')
            result=dict(task_identity=r,driver_identity=r,session_leader_identity=r,owned_member_ledger=[r])
            with patch.object(m,'matches',return_value=False),patch.object(m,'owned_session_records',return_value=[dict(r,start_ticks=99)]),patch.object(m,'send') as send:
                with self.assertRaises(AssertionError):m.cleanup(result,logs,log)
                send.assert_not_called()

    def test_live_leader_cannot_overwrite_reused_member(self):
        with tempfile.TemporaryDirectory() as d:
            logs=Path(d);log=logs/'train';log.write_text('')
            leader=dict(pid=3,start_ticks=12,sid=3,uid=0,command='leader')
            child=dict(pid=4,start_ticks=13,sid=3,uid=0,command='child')
            gone=dict(pid=1,start_ticks=10,sid=3,uid=0,command='gone')
            result=dict(task_identity=gone,driver_identity=gone,session_leader_identity=leader,owned_member_ledger=[leader,child])
            with patch.object(m,'matches',side_effect=lambda r:r['pid']==3),patch.object(m,'identity',return_value=leader),patch.object(m,'owned_session_records',return_value=[leader,dict(child,start_ticks=99)]),patch.object(m,'send') as send:
                with self.assertRaises(AssertionError):m.cleanup(result,logs,log)
                send.assert_not_called()

    def test_failed_terminal_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'logs').mkdir()
            (base/'logs/r1-stop300-status.json').write_text(json.dumps(dict(run_id='r',state='STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT')))
            with patch('sys.argv',['stop','--base',d,'--run-id','r']),patch.object(m,'send') as send:
                m.main();send.assert_not_called()

if __name__=='__main__':unittest.main()
