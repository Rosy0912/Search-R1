import copy
import hashlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.common8.actor_probe_segment_acceptance import (
    discover_chain, read_manifest, verify_refresh_isolation, verify_trainer_exit,
    verify_canonical_files,
)
from search_g1.actor_probe_contract import experiment_fields
from search_g1.reward_schedule import ACTOR_PROBE_SCHEDULE


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(run_id):
    return dict(run_id=run_id,execution_id=run_id[::-1],source_commit='c'*40,
        execution_backend='ssh_local',execution_host=socket.gethostname(),training_world_size=4,
        reward_schedule=ACTOR_PROBE_SCHEDULE,**experiment_fields(ACTOR_PROBE_SCHEDULE))


def pair(root):
    parent=Path(root).resolve()/'parent';child=Path(root).resolve()/'child';old=manifest('a'*31+'1');new=manifest('b'*31+'2')
    sha=write(parent/'logs/run-manifest.json',old)
    new['actor_probe_experiment_root_run_id']=old['run_id']
    new['actor_probe_recovery']=dict(role='actor_probe_same_experiment_full_state_recovery',
        state_semantics='full_model_optimizer_scheduler_rng_sampler_and_heads',parent_base=str(parent),
        parent_run_id=old['run_id'],parent_execution_id=old['execution_id'],parent_source_commit=old['source_commit'],
        parent_manifest_sha256=sha,experiment_root_run_id=old['run_id'],completed_outer=50,
        actual_optimizer_writes=997,attempt_index=51,aborted_attempts=1)
    write(child/'logs/run-manifest.json',new)
    return parent,child,old,new


class SegmentAcceptanceTests(unittest.TestCase):
    def test_full_state_wrong_source_identity_rejected_before_any_tensor_load(self):
        from reproduction.common8.actor_probe_checkpoint import verify_full_state
        from search_g1.actor_probe_recovery import identity
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve();run=dict(manifest('a'*32),profile='common8',slurm_job_id=None,checkpoint_dir=str(base/'checkpoints'))
            directory=Path(run['checkpoint_dir'])/'actor/global_step_200'
            for field,value in [('source_commit','d'*40),('execution_backend','slurm'),('training_world_size',8),('profile','other'),('slurm_job_id','123')]:
                with self.subTest(field=field):
                    wrong=dict(identity(run));wrong[field]=value
                    write(directory/'full-state-complete.json',dict(role='actor_probe_full_state_checkpoint',identity=wrong))
                    with patch('torch.load') as load:
                        with self.assertRaises(AssertionError):verify_full_state(base,run,200,load_optimizer=True)
                        load.assert_not_called()

    def test_finite_shape_claim_does_not_admit_replaced_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve();directory=base/'checkpoint';directory.mkdir()
            target=directory/'config.json';original=b'{"shape":[3,4],"finite":true}\n';target.write_bytes(original)
            proof=dict(checkpoint=str(directory),finite_all_saved_tensors=True,
                files_sha256={'config.json':hashlib.sha256(original).hexdigest()})
            verify_canonical_files(base,directory,proof)
            target.write_bytes(b'{"shape":[3,4],"finite":true,"changed":true}\n')
            with self.assertRaises(AssertionError):verify_canonical_files(base,directory,proof)
            target.unlink();outside=base/'outside.json';outside.write_bytes(original);target.symlink_to(outside)
            with self.assertRaises(AssertionError):verify_canonical_files(base,directory,proof)

    def test_chain_reads_actual_parent_bytes_and_retains_disjoint_intervals(self):
        with tempfile.TemporaryDirectory() as root:
            parent,child,old,new=pair(root);chain=discover_chain(child)
            self.assertEqual([(r['origin']['completed_outer'],r['last']) for r in chain],[(0,50),(50,200)])
            self.assertEqual(chain[0]['run'],old);self.assertEqual(chain[1]['run'],new)
            write(parent/'logs/run-manifest.json',dict(old,source_commit='d'*40))
            with self.assertRaises(AssertionError):discover_chain(child)

    def test_unknown_contract_identity_root_and_nonprogress_recovery_are_rejected(self):
        for field,value in [('run_id','bad'),('probe_parameter_ownership','generation_head'),
                            ('actor_probe_experiment_root_run_id','d'*32)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                parent,child,old,new=pair(root);new[field]=value;write(child/'logs/run-manifest.json',new)
                with self.assertRaises((AssertionError,ValueError)):discover_chain(child)
        with tempfile.TemporaryDirectory() as root:
            parent,child,old,new=pair(root)
            # A parent also starting at50 cannot supply its child's retained1..50.
            old.update(actor_probe_experiment_root_run_id=new['actor_probe_experiment_root_run_id'],
                actor_probe_recovery=dict(new['actor_probe_recovery'],parent_run_id='e'*32,parent_execution_id='f'*32))
            new['actor_probe_recovery']['parent_manifest_sha256']=write(parent/'logs/run-manifest.json',old)
            write(child/'logs/run-manifest.json',new)
            with self.assertRaises(AssertionError):discover_chain(child)

    def test_manifest_symlink_cannot_redirect_to_another_namespace(self):
        with tempfile.TemporaryDirectory() as root:
            parent,child,old,new=pair(root);path=child/'logs/run-manifest.json';path.unlink()
            path.symlink_to(parent/'logs/run-manifest.json')
            with self.assertRaises(AssertionError):read_manifest(child)

    def test_actual_exit_uses_birth_and_session_not_stopped_label(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve();run=manifest('a'*32)
            leader=dict(pid=123,start_ticks=456,sid=123,uid=os.getuid(),state='S')
            write(base/'logs/process-ledger.json',dict(run_id=run['run_id'],jobs=dict(trainer=dict(
                leader=leader,ledger={'123':leader},state='STOPPED',exit_code=0))))
            write(base/'logs/training-launch-claim.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
                shell_pid=123,shell_start_ticks=456))
            with patch('reproduction.a800.owned_processes.identity',return_value=leader):
                with self.assertRaises(AssertionError):verify_trainer_exit(base,run,terminal=True)
            # A reused PID is not the recorded trainer; a same-session survivor is.
            with patch('reproduction.a800.owned_processes.identity',return_value=dict(leader,start_ticks=789)),patch.object(Path,'iterdir',return_value=iter([])):
                self.assertTrue(verify_trainer_exit(base,run,terminal=True)['actual_trainer_exit_verified'])
            def current(pid):
                if pid==123:raise FileNotFoundError()
                return dict(leader,pid=987,start_ticks=100)
            with patch('reproduction.a800.owned_processes.identity',side_effect=current),patch.object(Path,'iterdir',return_value=iter([Path('/proc/987')])):
                with self.assertRaises(AssertionError):verify_trainer_exit(base,run)

    def test_refresh_binds_actor_reference_rng_and_actual_next_head(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve();run=manifest('a'*32)
            before=dict(driver_rng_sha256='a'*64)
            for role in ('actor','reference'):
                keys=['parameters_sha256','gradients_sha256','module_modes_sha256','rng_sha256']
                if role=='actor':keys+=['optimizer_sha256','scheduler_sha256','rollout_rng_sha256']
                before[role]=[dict(rank=rank,role=role,**{key:'b'*64 for key in keys}) for rank in range(4)]
            proof=dict(accepted=True,step=10,version='actual-head',before=before,after=copy.deepcopy(before))
            path=base/'logs/actor-probe-refresh-isolation-10.json';write(path,proof)
            records=[dict(step=10,probe_version='old-head'),dict(step=11,probe_version='actual-head')]
            self.assertEqual(verify_refresh_isolation(base,run,records,{})['10']['version'],'actual-head')
            for mutation in ('version','rng','actor_rng','float_rank'):
                with self.subTest(mutation=mutation):
                    changed=copy.deepcopy(proof)
                    if mutation=='version':changed['version']='another-head'
                    if mutation=='rng':changed['after']['driver_rng_sha256']='c'*64
                    if mutation=='actor_rng':changed['after']['actor'][0]['rng_sha256']='c'*64
                    if mutation=='float_rank':
                        changed['before']['actor'][0]['rank']=0.;changed['after']=copy.deepcopy(changed['before'])
                    write(path,changed)
                    with self.assertRaises(AssertionError):verify_refresh_isolation(base,run,records,{})

    def test_driver_refresh_snapshot_exposes_rng_changes_with_unchanged_workers(self):
        from reproduction.common8.actor_probe_checkpoint import refresh_isolation_snapshot
        import torch
        worker=SimpleNamespace(probe_isolation_state=lambda:[dict(rank=rank) for rank in range(4)])
        trainer=SimpleNamespace(actor_rollout_wg=worker,ref_policy_wg=worker)
        before=refresh_isolation_snapshot(trainer);torch.rand(2);after=refresh_isolation_snapshot(trainer)
        self.assertEqual(before['actor'],after['actor']);self.assertEqual(before['reference'],after['reference'])
        self.assertNotEqual(before['driver_rng_sha256'],after['driver_rng_sha256'])


if __name__=='__main__':unittest.main()
