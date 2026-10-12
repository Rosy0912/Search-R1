"""Constructed scalar/RAM admissions; no native GPU, tensor or launch claims."""
import copy
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8.actor_probe_restore_guard import (
    METHOD, ROLE, _unclaimed, build, method, verify,
)
from reproduction.common8.formal_contract import formal_mathematics
from reproduction.common8.formal_inline_safety import verify_plan
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_recovery import identity, binding_digest
from search_g1.actor_probe_namespace import namespace_origin
from tests.reproduction.test_actor_probe_head_service import fixture as heads_fixture, fitted_fixture_loader


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def fixture(root,step=50):
    base,run,manifest,service,heads=heads_fixture(Path(root)/'child',step)
    logs=base/'logs';logs.mkdir();(base/'preparation').mkdir()
    run.update(source_sha256={'scalar-fixture.py':'6'*64},started_unix=100.,
        completed_updates=0,acceptance_update=5,initial_probe_version=manifest['version'],
        checkpoint_dir=str(base/'checkpoints/search-g1-common8'),train_log=str(logs/'training.log'))
    run['input_sha256']['stage.parquet']='8'*64
    run['budget'].update(formal_mathematics(run),profile='common8',contract_kind='fresh200_warmup57',
        model_id='Qwen/Qwen2.5-3B-Instruct',model_revision=run['model_revision'],
        common_manifest_sha256=run['input_sha256']['manifest.json'])
    parent_base=Path(root).resolve()/'parent';parent=copy.deepcopy(run)
    binding=run['actor_probe_recovery']
    parent.update(run_id=binding['parent_run_id'],execution_id=binding['parent_execution_id'],
        source_commit=binding['parent_source_commit'],checkpoint_dir=str(parent_base/'checkpoints/search-g1-common8'))
    parent.pop('actor_probe_recovery');parent.pop('actor_probe_experiment_root_run_id')
    parent_path=parent_base/'logs/run-manifest.json';write(parent_path,parent)
    saved=dict(role='actor_probe_full_state_checkpoint',identity=identity(parent),completed_outer=step,
        next_logical_update=step+1,actual_optimizer_writes=step*20,
        workers=[dict(rank=i,completed_outer=step,optimizer_writes=step*20) for i in range(4)],
        probe_version=manifest['version'],probe_actor_step=manifest['actor_step'],
        probe_manifest_sha256='7'*64,probe_head_hashes=manifest['head_sha256'])
    saved_path=Path(parent['checkpoint_dir'])/'actor'/f'global_step_{step}'/'full-state-complete.json'
    write(saved_path,saved)
    binding.update(parent_base=str(parent_base),parent_manifest_sha256=digest(parent_path),
        full_state_receipt_sha256=digest(saved_path))
    candidate=logs/'actor-probe-recovery-candidate.json';write(candidate,run)
    preparation=dict(role='actor_probe_full_recovery_preparation',prepared=True,production_admission=False,
        identity=identity(run),origin=namespace_origin(run),recovery_binding_sha256=binding_digest(binding),
        candidate_manifest='logs/actor-probe-recovery-candidate.json',candidate_manifest_sha256=digest(candidate),
        parent_trainer_session_exit_verified=True,canonical_finite_tensor_count=435,
        parent_manifest_sha256=binding['parent_manifest_sha256'],parent_full_state_receipt_sha256=binding['full_state_receipt_sha256'])
    write(logs/'actor-probe-recovery-preparation.json',preparation)
    archive=base/'artifacts/restored-version-manifest.json';write(archive,manifest)
    write(base/'artifacts/live/manifest.json',manifest)
    staged=dict(role='actor_probe_restored_heads_staged',staged=True,production_admission=False,
        identity=identity(run),recovery_binding_sha256=binding_digest(binding),
        preparation_sha256=digest(logs/'actor-probe-recovery-preparation.json'),
        parent_full_state_receipt_sha256=binding['full_state_receipt_sha256'],
        parent_probe_manifest_sha256=saved['probe_manifest_sha256'],version=manifest['version'],
        actor_step=manifest['actor_step'],round=manifest['round'],head_sha256=manifest['head_sha256'],
        archive_manifest=str(archive),archive_manifest_sha256=digest(archive),live_manifest_sha256=digest(archive),
        calibration_data_sha256=manifest['calibration_data_sha256'])
    write(logs/'actor-probe-recovery-heads-staged.json',staged)
    (logs/'actor-probe-head-service-manifest.json').write_bytes(archive.read_bytes())
    with fitted_fixture_loader(heads):health=service.health();smoke=service.smoke(manifest['version'])
    write(logs/'initial-probes-accepted.json',dict(role='actual_actor_binary_head_endpoint_admission',accepted=True,
        identity=identity(run),origin=namespace_origin(run),version=manifest['version'],
        archived_manifest='logs/actor-probe-head-service-manifest.json',
        live_manifest_sha256=digest(archive),archived_manifest_sha256=digest(archive),
        actor_features_exercised=False,task_accuracy_claimed=False,
        actual_http_health=health,actual_http_readouts=smoke))
    producer=dict(pid=11,start_ticks=12,sid=11,uid=os.getuid())
    run.update(pipeline_identity=producer,probe_visible_device='6',probe_logical_ordinal=0,retrieval_placement='fixture placement')
    allocation=dict(accepted=True,run_id=run['run_id'],execution_id=run['execution_id'],execution_backend='ssh_local',
        slurm_job_id=None,training_world_size=4,pipeline_identity=producer,node=socket.gethostname(),
        training_physical_indices=[0,1,2,3],actual_allocated_indices=[0,1,2,3,5,6],
        auxiliary_physical_indices=[5,6],retriever_physical_indices=[5,6],probe_physical_index=6,
        assigned_compute_applications=[],assigned_gpu_uuids={str(i):f'fixture-gpu-{i}' for i in (0,1,2,3,5,6)},
        actual_cuda_runtime_uuids=[f'fixture-gpu-{i}' for i in (0,1,2,3)])
    write(logs/'native-allocation-admission.json',allocation)
    header=dict(run_id=run['run_id'],execution_id=run['execution_id'],execution_backend='ssh_local',slurm_job_id=None)
    docs=[['9'*64]*3 for _ in range(64)];ids=[f'fixture-stage-{i}' for i in range(64)]
    retrieval=dict(header,accepted=True,stage_sha256=run['input_sha256']['stage.parquet'],question_ids=ids,
        gpu_document_sha256=docs,host=socket.gethostname(),physical_gpu_indices=[5,6],
        actual_cuda_runtime_uuids=[f'fixture-gpu-{i}' for i in (5,6)])
    write(logs/'retrieval-admission.json',retrieval)
    write(logs/'retrieval-http-admission.json',dict(header,accepted=True,stage_sha256=retrieval['stage_sha256'],
        question_ids=ids,document_sha256=docs,retrieval_admission_sha256=digest(logs/'retrieval-admission.json')))
    from reproduction.common8.actor_probe_head_service import ROLE as HEAD_ROLE
    release_header=dict(version=manifest['version'],role=HEAD_ROLE,frozen_encoder_loaded=False)
    write(logs/'evaluator-release-admission.json',dict(header,
        released=dict(release_header,suspended=True,allocated_bytes_after_release=0),
        resumed=dict(release_header,resumed=True)))
    resource=dict(accepted=True,run_id=run['run_id'],execution_id=run['execution_id'],execution_backend='ssh_local',
        slurm_job_id=None,host=socket.gethostname(),training_world_size=4,historical_gpu_smoke_executed=False,
        occupancy_policy='monitor_only',sampled_peak_fraction_stop_limit=None,fitted_probe_manifest_sha256=digest(archive),
        allocation_sha256=digest(logs/'native-allocation-admission.json'))
    for name in ('retrieval-admission','retrieval-http-admission','evaluator-release-admission'):
        resource[name.replace('-','_')+'_sha256']=digest(logs/(name+'.json'))
    write(logs/'local-resource-admission.json',resource)
    write(base/METHOD,method(run,101.))
    write(logs/'run-manifest.json',run)
    document={'scope':'scalar resolved-config fixture; real mathematics checker separately tested'}
    return base,run,document


class RestoreGuardTests(unittest.TestCase):
    def setUp(self):
        # Only the Hydra/model mathematics gate is replaced. Real namespace,
        # saved-receipt, Head hash/metadata, resource and endpoint documents run.
        self.math=patch('reproduction.common8.formal_contract.verify_formal_config',
            side_effect=lambda document,run,base:dict(resolved_config_sha256=document_sha256(document)))
        self.math.start();self.addCleanup(self.math.stop)

    def test_saved_origin_is_truthful_and_routes_separately_from_fresh0(self):
        for step in (5,50,100,150):
            with self.subTest(step=step),tempfile.TemporaryDirectory() as root:
                base,run,doc=fixture(root,step);_unclaimed(base,run)
                artifact=build(base,run,doc,102.)
                tail,sha=verify_plan(base,artifact,run)
                self.assertEqual(artifact['role'],ROLE);self.assertEqual(tail,.01)
                self.assertEqual(sha,document_sha256(artifact))
                self.assertEqual(artifact['origin']['completed_outer'],step)
                self.assertEqual(artifact['inherited_optimizer_writes_at_freeze'],20*step)
                self.assertEqual(artifact['new_namespace_optimizer_writes_at_freeze'],0)
                self.assertFalse(artifact['production_launch_admitted'])
                self.assertFalse(artifact['actual_behavior_probability_accepted'])
                self.assertTrue(artifact['full_state_restore_required_before_rollout'])
                self.assertFalse(json.loads((base/METHOD).read_text())['fresh_states'])
                with self.assertRaises(AssertionError):verify_plan(base,dict(artifact,role='formal_inline_first_five_candidates'),run)
                fresh=dict(run);fresh.pop('actor_probe_recovery')
                with self.assertRaises(AssertionError):verify_plan(base,artifact,fresh)

    def test_self_asserted_health_restore_probability_counters_and_extra_forward_reject(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,doc=fixture(root);artifact=build(base,run,doc,102.)
            for field,value in [('healthy_checkpoint_accepted',True),('short_train_accepted',True),
                ('full_state_restore_required_before_rollout',False),('production_launch_admitted',True),
                ('inherited_optimizer_writes_at_freeze',0),('new_namespace_optimizer_writes_at_freeze',1),
                ('actual_behavior_probability_accepted',True),('hf_recomputation_required',True),
                ('negative_tail_limit',.1),('trajectory_count',32),('tolerance',.05)]:
                with self.subTest(field=field),self.assertRaises(AssertionError):
                    verify(base,dict(artifact,**{field:value}),run)

    def test_changed_child_source_data_recipe_and_parent_receipt_reject(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,doc=fixture(root);artifact=build(base,run,doc,102.)
            for field,value in [('source_sha256',{'other.py':'a'*64}),
                ('input_sha256',dict(run['input_sha256'],**{'calibration-300.parquet':'b'*64})),
                ('reward_schedule_boundary',50),('probe_refresh_interval',50)]:
                with self.subTest(field=field),self.assertRaises((AssertionError,ValueError)):
                    verify(base,artifact,dict(run,**{field:value}))
            path=Path(run['actor_probe_recovery']['parent_base'])/'logs/run-manifest.json'
            path.write_text('{"fixture":"changed parent"}')
            with self.assertRaises(AssertionError):verify(base,artifact,run)

    def test_frozen_service_head_and_resource_bytes_cannot_change(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,doc=fixture(root);artifact=build(base,run,doc,102.)
            for name in ('logs/initial-probes-accepted.json','logs/local-resource-admission.json',
                         'artifacts/commit_probe.pkl','logs/actor-probe-recovery-candidate.json'):
                path=base/name;old=path.read_bytes();path.write_text('{"fixture":"corrupted"}')
                with self.subTest(name=name),self.assertRaises(AssertionError):verify(base,artifact,run)
                path.write_bytes(old)

    def test_historical_guard_does_not_require_current_live_head_or_process_liveness(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,doc=fixture(root);artifact=build(base,run,doc,102.)
            write(base/'artifacts/live/manifest.json',dict(later_version=True))
            # Completed training adds logs; historical receipt reading remains valid.
            write(base/'logs/training-launch-claim.json',{'fixture':'claimed later'})
            self.assertEqual(verify(base,artifact,run)[0],.01)
            with self.assertRaises(AssertionError):_unclaimed(base,run)

    def test_child_attempt_and_dangling_symlink_cannot_be_frozen_again(self):
        for name in ('training-launch-claim.json','actual-attempts.jsonl','actual-updates.jsonl',
                     'actor-probe-recovery-restored.json','actor-probe-recovery-launch-admission.json',
                     'optimizer-rank-0.jsonl','scheduler-rank-3.jsonl','formal-inline-optimizer-0.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as root:
                base,run,_=fixture(root);path=base/'logs'/name;path.symlink_to(base/'missing-fixture')
                with self.assertRaises(AssertionError):_unclaimed(base,run)

    def test_method_timestamps_fresh_claim_and_missing_critical_receipt_reject(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,doc=fixture(root);artifact=build(base,run,doc,102.)
            for timestamp in (99.,103.,True,float('nan')):
                with self.subTest(timestamp=timestamp),self.assertRaises(AssertionError):
                    wrong=method(run,timestamp);write(base/METHOD,wrong)
                    candidate=build(base,run,doc,102.);verify(base,candidate,run)
            write(base/METHOD,method(run,101.))
            wrong=copy.deepcopy(artifact);wrong['receipts_sha256'].pop('logs/actor-probe-recovery-preparation.json')
            with self.assertRaises(AssertionError):verify(base,wrong,run)

    def test_failed_foreign_retrieval_and_release_cannot_be_rehashed_into_admission(self):
        for name,field,value in [('retrieval-admission','run_id','0'*32),
                ('retrieval-admission','accepted',False),('retrieval-admission','stage_sha256','0'*64),
                ('retrieval-http-admission','execution_id','0'*32),
                ('retrieval-http-admission','accepted',False),
                ('retrieval-http-admission','retrieval_admission_sha256','0'*64),
                ('evaluator-release-admission','run_id','0'*32)]:
            with self.subTest(name=name,field=field),tempfile.TemporaryDirectory() as root:
                base,run,doc=fixture(root);path=base/'logs'/(name+'.json');wrong=json.loads(path.read_text())
                wrong[field]=value;write(path,wrong)
                resource_path=base/'logs/local-resource-admission.json';resource=json.loads(resource_path.read_text())
                resource[name.replace('-','_')+'_sha256']=digest(path);write(resource_path,resource)
                with self.assertRaises(AssertionError):build(base,run,doc,102.)
        for field,value in [('probe_visible_device','4'),('probe_logical_ordinal',6),('probe_logical_ordinal',False)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                base,run,doc=fixture(root);run[field]=value
                with self.assertRaises(AssertionError):build(base,run,doc,102.)

    def test_actual_worker_threshold_consumer_accepts_only_typed_restored_origin_schema(self):
        from types import SimpleNamespace
        from search_g1.policy_safety import bind_thresholds
        with tempfile.TemporaryDirectory() as root:
            base,run,doc=fixture(root);artifact=build(base,run,doc,102.)
            target=base/'logs/policy-safety-calibration.json';write(target,artifact)
            config=SimpleNamespace(calibration_manifest=str(target),dual_clip=3.,target_old_policy_kl=.2,
                kl_log_bound=30.,negative_tail_limit=.01)
            with patch.dict(os.environ,SEARCH_G1_BASE=str(base)):
                self.assertEqual(bind_thresholds(config),(.01,document_sha256(artifact)))
                for wrong in (dict(artifact,schema_version=True),dict(artifact,role='other-schema1')):
                    write(target,wrong)
                    with self.assertRaises(RuntimeError):bind_thresholds(config)
                write(target,artifact);fresh=copy.deepcopy(run);fresh.pop('actor_probe_recovery')
                write(base/'logs/run-manifest.json',fresh)
                with self.assertRaises(RuntimeError):bind_thresholds(config)


if __name__=='__main__':unittest.main()
