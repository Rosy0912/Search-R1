"""Pure CPU admission counterexamples. Synthetic controls are never GPU proof."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from formal_probability_proofs_v1 import file_record, manifest_records, validate_manifest_closure
from formal_probability_v1 import validate_live_arms
from formal_protection_contract_v1 import admit_binding, validate_production_artifact, DATA, MODEL, MANIFEST, REVISION
from formal_first5_control_v1 import approval_payload
from formal_native_evidence_v1 import validate_native_payload

SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'


class EvidenceClosureCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        def record(name):
            path = self.root/name; path.write_text('synthetic CPU bytes '+name)
            return file_record(path)
        self.manifest = dict(observed_turns=record('observed.json'), actions=record('actions.json'),
            scored_global_batch=record('global.pt'),
            raw_files=[dict(rank=r,serial=0,**record(f'raw-{r}.pt')) for r in range(8)],
            raw_receipts=[dict(rank=r,serial=0,**record(f'raw-{r}.json')) for r in range(8)],
            reference_records=[dict(rank=r,raw=record(f'ref-{r}.pt'),receipt=record(f'ref-{r}.json')) for r in range(8)])
        self.manifest['evidence_files'] = [dict(row) for row in manifest_records(self.manifest)]

    def deny(self, change):
        change()
        with self.assertRaises(ValueError): validate_manifest_closure(self.manifest, self.root)

    def test_all_exact_declared_originals(self):
        self.assertEqual(len(validate_manifest_closure(self.manifest,self.root)),35)

    def test_same_path_different_physical_sha_cannot_overwrite_candidate(self):
        self.deny(lambda: self.manifest['evidence_files'][0].update(sha256='f'*64))

    def test_same_path_different_physical_bytes_cannot_overwrite_candidate(self):
        self.deny(lambda: self.manifest['evidence_files'][0].update(bytes=0))

    def test_duplicate_original_path_refused(self):
        self.deny(lambda: self.manifest['evidence_files'].append(copy.deepcopy(self.manifest['evidence_files'][0])))

    def test_missing_actual_receipt_refused(self):
        self.deny(lambda: self.manifest['evidence_files'].pop())

    def test_changed_actual_bytes_refused(self):
        self.deny(lambda: Path(self.manifest['raw_files'][7]['path']).write_text('different'))

    def test_last_rank_duplicate_not_hidden_by_correct_count(self):
        self.deny(lambda: self.manifest['raw_files'][7].update(rank=6))

    def test_reference_coverage_not_inferred_from_raw_gpu_count(self):
        self.deny(lambda: self.manifest['reference_records'][7].update(rank=6))

    def test_symlink_original_refused(self):
        path=Path(self.manifest['raw_files'][7]['path']); target=self.root/'alternate'; target.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(target)
        with self.assertRaises(ValueError): validate_manifest_closure(self.manifest,self.root)


class ArmIdentityCPU(unittest.TestCase):
    def rows(self):
        return [dict(rank=r,pid=1000+r,start_ticks=str(100+r),outer_step=26,manifest_sha256='a'*64,
            complete_pause_state_sha256='b'*64,arm_receipt=dict(path='/cpu-fixture'),pause_state=dict(path='/cpu-fixture')) for r in range(8)]

    def test_eight_actual_identity_declarations(self): self.assertEqual(len(validate_live_arms(self.rows(),26,'a'*64)),8)
    def test_duplicate_worker_pid_start_refused(self):
        rows=self.rows(); rows[7].update(pid=1000,start_ticks='100')
        with self.assertRaises(ValueError): validate_live_arms(rows,26,'a'*64)
    def test_wrong_outer_refused(self):
        rows=self.rows(); rows[7]['outer_step']=27
        with self.assertRaises(ValueError): validate_live_arms(rows,26,'a'*64)
    def test_missing_full_state_digest_refused(self):
        rows=self.rows(); rows[7]['complete_pause_state_sha256']=None
        with self.assertRaises(ValueError): validate_live_arms(rows,26,'a'*64)


class First5BindingCPU(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        paused=[dict(rank=r,pid=1000+r,start_ticks=str(100+r)) for r in range(8)]
        identity=dict(job_id='9999',code_commit=SOURCE,data_identity=DATA,source=MODEL,logical_step=30,
            actual_new_updates=5,config_sha256='c'*64,files=[dict(pause_identity=copy.deepcopy(r)) for r in paused])
        self.ready=dict(manifest=dict(path='/cpu-fixture',sha256='m'*64,identity=identity),driver_cursor=15360,
            checkpoint_pause_identity=paused,actual_state_roundtrip=[dict(rank=r,logical_step=30) for r in range(8)])
        self.binding=dict(status='REVIEWED_FOR_PRODUCTION',code_commit=SOURCE,production_artifact=dict(sha256='p'*64),
            native_acceptance=dict(review_sha256='n'*64))
        checks={key:True for key in ('all8_nonzero_parameter_change','finite_loss_and_gradient',
            'same_driver_and8worker_pids_start_ticks','raw_retrieval_and_fixed_reference','microbatch_global_sequence_mean',
            'raw_trajectories_and_gigpo_state','stage64_health','baseline512_health','full_hf_semantic_reload',
            'allrank_model_adam_scheduler_rng_actual_reload','driver_rng_cursor_permutation_gigpo_actual_roundtrip',
            'approved_pause_unchanged','actual_full_fit_every_outer2560_probability_and_fixed_reference',
            'actual_flash8x36_same_workers_and_originals','actual_matched_probability_collectives')}
        checks.update(adam_per_rank=50,gradient_micro_per_rank=800,saved_and_actual_restored_adam_total=550,scheduler=30,cursor=15360)
        self.review=dict(status='PASS',job_id='9999',source=MODEL,logical_step=30,actual_new_updates=5,
            manifest_sha256='m'*64,restore_manifest_sha256=MANIFEST,source_commit=SOURCE,config_sha256='c'*64,
            production_artifact_sha256='p'*64,actual_native_acceptance_sha256='n'*64,first5_actual_checks=checks)

    def invoke(self):
        manifest=self.root/'training_manifest.json'; manifest.write_text(json.dumps(self.ready['manifest']['identity']))
        self.ready['manifest'].update(path=str(manifest),sha256=file_record(manifest)['sha256'])
        self.review['manifest_sha256']=self.ready['manifest']['sha256']
        ready=self.root/'ready.json'; ready.write_text(json.dumps(self.ready)); r=file_record(ready)
        self.review['ready_sha256']=r['sha256']; review=self.root/'review.json'; review.write_text(json.dumps(self.review))
        return approval_payload(self.binding,r,file_record(review))

    def test_synthetic_good_declaration_returns_payload_only(self):
        self.assertEqual(self.invoke()['same_process_next_fetch'],31)
        self.assertFalse((self.root/'approval.json').exists())
    def test_old6000_or_20step_budget_cannot_approve_new_branch(self):
        self.review['first5_actual_checks']['saved_and_actual_restored_adam_total']=600
        with self.assertRaises(ValueError): self.invoke()
    def test_driver_cursor_rebuild_cannot_approve(self):
        self.ready['driver_cursor']=12800
        with self.assertRaises(ValueError): self.invoke()
    def test_changed_worker_start_cannot_approve(self):
        self.ready['checkpoint_pause_identity'][7]['start_ticks']='999'
        with self.assertRaises(ValueError): self.invoke()
    def test_calibration_adam10_cannot_count_as_formal_first5(self):
        self.review['first5_actual_checks']['adam_per_rank']=10
        with self.assertRaises(ValueError): self.invoke()
    def test_null_production_draft_refuses_without_reading_assets(self):
        with self.assertRaises(ValueError): admit_binding(dict(status='DRAFT_NOT_READY',code_commit=None),SOURCE)

    def test_actual_full_fit_probability_not_waived_by_metadata_static_PASS(self):
        self.review['first5_actual_checks']['actual_full_fit_every_outer2560_probability_and_fixed_reference']=False
        with self.assertRaises(ValueError):self.invoke()
    def test_actual8_FLASH_same_workers_not_waived_at_gate30(self):
        self.review['first5_actual_checks']['actual_flash8x36_same_workers_and_originals']=False
        with self.assertRaises(ValueError):self.invoke()
    def test_actual_matched_probability_collectives_not_waived_at_gate30(self):
        self.review['first5_actual_checks']['actual_matched_probability_collectives']=False
        with self.assertRaises(ValueError):self.invoke()


class ActualNativeMetadataAdapterCPU(unittest.TestCase):
    """One inherited raw metadata interface, not repeated native execution/tests."""
    @classmethod
    def setUpClass(cls):
        out=Path(__file__).resolve().parents[3]/'outputs/gigpo-e1-policy-safety'
        path=out/'policy-safety-native3267-terminal-raw-v4.json.gz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()=='628ec2fa23e037292c0dcfd509c313f76d2eb0562382a54237c866e2c2521d9c'
        cls.physical=json.loads(gzip.decompress(path.read_bytes())); cls.result=cls.physical['result']; cls.terminal=cls.physical['terminal']
        cls.acceptance=json.loads((out/'NATIVE3267-CALIBRATION-ACCEPTANCE-SPEC-v1.json').read_text())
        cls.review=json.loads((out/'NATIVE3267-INDEPENDENT-REVIEW-v2.json').read_text())
        native=next(r for r in cls.physical['originals'] if r['path']==cls.acceptance['native_binding_path'])
        cls.binding=json.loads(base64.b64decode(native['content_base64']))

    def invoke(self,result=None,terminal=None,source=SOURCE):
        return validate_native_payload(self.acceptance,result or self.result,self.review,self.binding,
            self.result['guard_identity'],terminal or self.terminal,self.physical,source,self.binding['helpers'])
    def test_actual_original3267_metadata_interface_preserved(self): self.assertIsNone(self.invoke())
    def test_same_gpu_count_wrong_restore_step_refused(self):
        result=copy.deepcopy(self.result); result['restore_step']=26
        with self.assertRaises(ValueError): self.invoke(result=result)
    def test_official_phase_Adam500_identity_not_relabelled(self):
        result=copy.deepcopy(self.result); result['phases'][0][7]['optimizer_steps_before']=250
        with self.assertRaises(ValueError): self.invoke(result=result)
    def test_different_compatible_source_requires_new_actual_native(self):
        with self.assertRaises(ValueError): self.invoke(source='f'*40)
    def test_failed_terminal_not_accepted_as_new_current_job(self):
        terminal=copy.deepcopy(self.terminal); terminal['fields']['JobState']='FAILED'
        with self.assertRaises(ValueError): self.invoke(terminal=terminal)


class ProductionArtifactSchemaCPU(unittest.TestCase):
    """Pure fixture numbers are not chosen operational tolerances or thresholds."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        runner=self.root/'runner.py'; runner.write_text('synthetic CPU declaration, not installed backend')
        self.policy=dict(dual_clip=3.,kl_log_bound=30.,target_old_policy_kl=.02,negative_tail_limit=.05)
        self.identity=dict(source_commit=SOURCE,common_manifest_sha256=DATA,restore_manifest_sha256=MANIFEST,
            node='h100-8',official_model_revision=REVISION)
        def proof(**kwargs): return dict(status='PASS',source_commit=SOURCE,actual_identity=self.identity,**kwargs)
        self.values=dict(
            probability=proof(prefix_mismatch_count=0,token_mismatch_count=0,hf_position_and_mask_identity=True,
                actual_trajectories=2560,post_sampling_live_pause_verified=True,
                probability_tolerances={key:0. for key in ('actor_recompute_abs_tolerance','actor_vllm_abs_tolerance',
                    'actor_recompute_joint_abs_tolerance','actor_vllm_joint_abs_tolerance')}),
            sampler_position_semantics=proof(scope='INSTALLED_RUNNER_DEFAULT_POSITION_CONTRACT_AND_ACTUAL_LITERAL_PREFIX',
                sampler_runner_positions_observed=False,actual_engine_literal_prefix_observed=True,
                default_causal_positions_and_attention_semantics_independently_verified=True,
                actual_runner_model_config_and_active_branch_verified=True,active_position_branch='NON_MROPE_DEFAULT_ABSOLUTE_CAUSAL',
                installed_runner_path=str(runner),installed_runner_sha256=file_record(runner)['sha256']),
            calibration=proof(actual_adam_per_rank=10,actual_gradient_micro_per_rank=160,retained_training_updates_contributed=0),
            quality=proof(before_after_fixed512_and64=True,raw_text_actions_rewards_bound=True,
                healthy_restored25_baseline=True,after_calibration_no_confirmed_quality_collapse=True,
                actual_qa_counts=dict(before512=512,before64=64,after512=512,after64=64)),
            state=proof(all8_worker_complete_state_actual_reload=True,full_hf_semantic_reload=True,
                driver_saved_state_physical_verified=True,driver26_actual_replayed=False),
            terminal=proof(slurm_completed_0_0=True,owned_cleanup_empty=True,job_and_scope_cgroup_removed=True),
            threshold_derivation=proof(target_old_policy_kl=.02,negative_tail_limit=.05,raw_monitor_statistics_bound=True))

    def invoke(self):
        files={}
        for role,value in self.values.items():
            path=self.root/(role+'.json'); path.write_text(json.dumps(value)); files[role]=file_record(path)
        artifact=dict(status='PASS',scope='GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION',rank_count=8,
            source_commit=SOURCE,restore_manifest_sha256=MANIFEST,actual_proofs=files,actual_identity=self.identity,**self.policy)
        return validate_production_artifact(artifact,SOURCE,self.policy)

    def test_full_synthetic_declaration_checks_schema_only(self): self.assertIsNone(self.invoke())
    def test_null_thresholds_still_refuse_despite_native_PASS(self):
        self.policy['target_old_policy_kl']=None
        with self.assertRaises(ValueError): self.invoke()
    def test_oldPPO_KL_cannot_be_a_probability_tolerance(self):
        self.values['probability']['probability_tolerances']=dict(target_old_policy_kl=.02)
        with self.assertRaises(ValueError): self.invoke()
    def test_structural_mismatch_cannot_be_absorbed_by_tolerance(self):
        self.values['probability']['prefix_mismatch_count']=1
        with self.assertRaises(ValueError): self.invoke()
    def test_missing_backend_position_original_role_refuses(self):
        del self.values['sampler_position_semantics']
        with self.assertRaises(ValueError): self.invoke()
    def test_sampler_runner_tensor_is_not_invented(self):
        self.values['sampler_position_semantics']['sampler_runner_positions_observed']=True
        with self.assertRaises(ValueError): self.invoke()
    def test_incompatible_active_mrope_branch_refuses(self):
        self.values['sampler_position_semantics']['active_position_branch']='MROPE'
        with self.assertRaises(ValueError): self.invoke()
    def test_full512_quality_collapse_refuses_production(self):
        self.values['quality']['after_calibration_no_confirmed_quality_collapse']=False
        with self.assertRaises(ValueError): self.invoke()
    def test_reduced_stage_count_not_accepted_as_full512(self):
        self.values['quality']['actual_qa_counts']['after512']=64
        with self.assertRaises(ValueError): self.invoke()
    def test_calibration_not_counted_as_retained_formal26(self):
        self.values['calibration']['retained_training_updates_contributed']=1
        with self.assertRaises(ValueError): self.invoke()
    def test_HF_only_replacement_not_complete_worker_reload(self):
        self.values['state']['all8_worker_complete_state_actual_reload']=False
        with self.assertRaises(ValueError): self.invoke()
    def test_driver26_replay_truth_value_must_be_explicit(self):
        del self.values['state']['driver26_actual_replayed']
        with self.assertRaises(KeyError): self.invoke()


if __name__=='__main__': unittest.main(verbosity=2)
