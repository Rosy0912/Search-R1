"""Actual small cleanup/cache metadata files and mocked Linux/GPU inventories."""
import copy
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup,birth_inventory
from reproduction.a800.owned_processes import remaining_cuda_contexts,owned_session_survivors
from tests.reproduction.test_actor_probe_recovery import documents


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fixture(root):
    base=Path(root).resolve();run=documents()[0];producer=dict(pid=101,start_ticks=1,sid=101,uid=os.getuid())
    run.update(pipeline_identity=producer,initial_model=str(base/'official'),reference_model=str(base/'official'),
        retrieval_asset_paths={'fixture':str(base/'original-asset')})
    (base/'official').mkdir();(base/'original-asset').write_text('small asset stand-in')
    leader=dict(pid=102,start_ticks=2,sid=102,uid=os.getuid())
    write(base/'logs/cleanup-status.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
        execution_backend='ssh_local',slurm_job_id=None,cleaned=True,
        scope='only recorded project PID births and private local corpus cache'))
    write(base/'logs/process-ledger.json',dict(run_id=run['run_id'],jobs=dict(trainer=dict(
        state='STOPPED',leader=leader,ledger={'102':leader}))))
    # The actual fresh claim has no source_commit key; recovery claims do.
    write(base/'logs/pipeline-launch-claim.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],pipeline_identity=producer))
    root=base/'cache/retrieval'/run['execution_id']
    cache=dict(run_id=run['run_id'],execution_id=run['execution_id'],root=str(root))
    write(base/'logs/corpus-cache-claim.json',dict(cache,uid=os.getuid(),phase='CLEANED'))
    write(base/'logs/corpus-cache-cleanup.json',dict(cache,cleaned=True,original_assets_removed=False))
    return base,run,leader


class OwnedCleanupTests(unittest.TestCase):
    def test_cleanup_cannot_drop_historical_service_births_from_mutable_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,leader=fixture(root);path=base/'logs/process-ledger.json'
            ledger=json.loads(path.read_text());member=dict(leader,pid=103,start_ticks=3)
            ledger['jobs']['trainer']['ledger']['103']=member;write(path,ledger)
            saved=birth_inventory(base,run)
            newer=copy.deepcopy(ledger);new_leader=dict(leader,pid=104,sid=104,start_ticks=4)
            newer['jobs']['final-verifier']=dict(state='STOPPED',leader=new_leader,ledger={'104':new_leader})
            newer['jobs']['trainer']['leader']['command']='a changed process title'
            write(path,newer)
            with patch('reproduction.a800.owned_processes.identity',side_effect=FileNotFoundError), \
                patch('pathlib.Path.iterdir',return_value=iter([])), \
                patch('reproduction.common8.run_pipeline.gpu_rows',return_value=[]):
                self.assertTrue(verify_cleanup(base,run,preserved_births=saved)['accepted'])
                newer['jobs']['trainer']['ledger'].pop('103');write(path,newer)
                with self.assertRaisesRegex(AssertionError,'Historical owned'):
                    verify_cleanup(base,run,preserved_births=saved)
                newer['jobs'].pop('trainer');write(path,newer)
                with self.assertRaises(KeyError):verify_cleanup(base,run,preserved_births=saved)

    def test_reused_numeric_sid_is_foreign_but_orphaned_old_session_is_owned(self):
        old=dict(pid=102,start_ticks=2,sid=102,uid=os.getuid())
        new=dict(old,start_ticks=999,state='S')
        member=dict(new,pid=103,start_ticks=1000)
        def actual(pid):return new if pid==102 else member
        with patch('reproduction.a800.owned_processes.identity',side_effect=actual), \
            patch('pathlib.Path.iterdir',return_value=iter([Path('/proc/102'),Path('/proc/103')])):
            self.assertEqual(owned_session_survivors([old]),([],[102]))
        # The leader may disappear while its old session children survive.
        def orphan(pid):
            if pid==102:raise FileNotFoundError()
            return member
        with patch('reproduction.a800.owned_processes.identity',side_effect=orphan), \
            patch('pathlib.Path.iterdir',return_value=iter([Path('/proc/103')])):
            self.assertEqual(owned_session_survivors([old]),([member],[]))
        # More than one project job may historically use the numeric SID.
        with patch('reproduction.a800.owned_processes.identity',side_effect=actual), \
            patch('pathlib.Path.iterdir',return_value=iter([Path('/proc/103')])):
            self.assertEqual(owned_session_survivors([old,new]),([member],[]))

    def test_shared_pipeline_shell_session_is_not_owned(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,leader=fixture(root)
            run['pipeline_identity']['sid']=99
            write(base/'logs/pipeline-launch-claim.json',dict(run_id=run['run_id'],
                execution_id=run['execution_id'],pipeline_identity=run['pipeline_identity']))
            shell=dict(pid=99,start_ticks=999,sid=99,uid=os.getuid(),state='S')
            def actual(pid):
                if pid==99:return shell
                raise FileNotFoundError()
            with patch('reproduction.a800.owned_processes.identity',side_effect=actual), \
                patch('pathlib.Path.iterdir',return_value=iter([Path('/proc/99')])), \
                patch('reproduction.common8.run_pipeline.gpu_rows',return_value=[]):
                self.assertTrue(verify_cleanup(base,run)['accepted'])

    def test_cuda_release_uses_exact_birth_and_preserves_unwinding_context(self):
        recorded=dict(pid=102,start_ticks=2,sid=102,uid=os.getuid())
        rows=[['owned-gpu','102'],['foreign-gpu','777']]
        with patch('reproduction.a800.owned_processes.identity',return_value=dict(recorded,state='Z')):
            self.assertEqual(remaining_cuda_contexts([recorded],rows),(rows[:1],[]))
        with patch('reproduction.a800.owned_processes.identity',side_effect=FileNotFoundError):
            self.assertEqual(remaining_cuda_contexts([recorded],rows),(rows[:1],[]))
        with patch('reproduction.a800.owned_processes.identity',return_value=dict(recorded,start_ticks=99)):
            self.assertEqual(remaining_cuda_contexts([recorded],rows),([],[102]))

    def test_marker_does_not_admit_live_owned_birth_or_unrecorded_session_member(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,leader=fixture(root)
            with patch('reproduction.a800.owned_processes.identity',return_value=dict(leader,state='S')):
                with self.assertRaises(AssertionError):verify_cleanup(base,run)
            survivor=dict(leader,pid=103,start_ticks=3,state='S')
            with patch('reproduction.a800.owned_processes.identity',side_effect=lambda pid:survivor if pid==103 else (_ for _ in ()).throw(FileNotFoundError())), \
                patch('pathlib.Path.iterdir',return_value=iter([Path('/proc/103')])):
                with self.assertRaises(AssertionError):verify_cleanup(base,run)

    def test_reused_pid_on_foreign_gpu_is_not_stopped_but_owned_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,leader=fixture(root);reused=dict(leader,start_ticks=999,state='S')
            with patch('reproduction.a800.owned_processes.identity',return_value=reused), \
                patch('pathlib.Path.iterdir',return_value=iter([Path('/proc/102')])), \
                patch('reproduction.common8.run_pipeline.gpu_rows',return_value=[['foreign-gpu4','102']]):
                proof=verify_cleanup(base,run)
                self.assertEqual(proof['reused_pids_belonging_to_other_births'],[102])
                self.assertEqual(proof['owned_cuda_contexts_remaining'],0)
            def gone(pid):raise FileNotFoundError()
            with patch('reproduction.a800.owned_processes.identity',side_effect=gone), \
                patch('pathlib.Path.iterdir',return_value=iter([])), \
                patch('reproduction.common8.run_pipeline.gpu_rows',return_value=[['assigned-gpu0','102']]):
                with self.assertRaisesRegex(AssertionError,'CUDA context'):verify_cleanup(base,run)

    def test_cache_root_must_really_be_absent_and_original_assets_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,leader=fixture(root)
            def gone(pid):raise FileNotFoundError()
            with patch('reproduction.a800.owned_processes.identity',side_effect=gone), \
                patch('pathlib.Path.iterdir',return_value=iter([])), \
                patch('reproduction.common8.run_pipeline.gpu_rows',return_value=[]):
                proof=verify_cleanup(base,run);self.assertTrue(proof['accepted'])
                cache=base/'cache/retrieval'/run['execution_id'];cache.parent.mkdir(parents=True);cache.mkdir()
                with self.assertRaises(AssertionError):verify_cleanup(base,run)
                cache.rmdir();(base/'original-asset').unlink()
                with self.assertRaises(AssertionError):verify_cleanup(base,run)


if __name__=='__main__':unittest.main()
