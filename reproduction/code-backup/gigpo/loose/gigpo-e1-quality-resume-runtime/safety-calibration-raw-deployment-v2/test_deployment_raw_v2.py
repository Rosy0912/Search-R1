"""New raw deployment security controls only; no target/model/GPU acceptance."""
import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import deployment_contract_raw_v2 as c

LOCAL=Path(__file__).resolve().parents[3]
OUT=LOCAL/'outputs/gigpo-e1-policy-safety'


def read_candidate():
    return json.loads((OUT/'CALIBRATION-RAW-CANDIDATE-2262942-v2.json').read_text())


def native_fixture():
    scope=c.NATIVE_SCOPE;job='9999';binding_sha='a'*64
    guard={k:dict(path=str(c.NATIVE_HELPERS/name),sha256=digest) for k,name,digest in [
        ('resource_guard','resource_guard_v2.py','9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
        ('resource_ownership','resource_ownership_v2.py','7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed')]}
    outer=dict(optimizer_writes=10,gradient_micro_forwards=160,before_monitor_forwards=160,after_monitor_forwards=160,
               original_local_trajectory_rows=320,mini_denominators=[32.]*10)
    ranks=[dict(status='PASS',rank=rank,world=8,code_commit=c.SOURCE,binding_sha256=binding_sha,native_scope=scope,
        native_initialization='OFFICIAL_MODEL_INITIALIZATION',native_actual_restore=False,formal_updates_contributed=0,
        official_3b=True,official_revision=c.REVISION,actual_fsdp=True,flash_attention2=True,remove_padding=True,manual_offload=True,
        prefix_4096_action500_forward_backward=True,old_ref_rank4_action16_no_grad_maximum=True,
        mapping_formula_value_and_gradient_equal=True,state_save_restore_next_outer_equal=True,native_hf_semantic_reload_equal=True,
        real_sampler_executed=False,real_fixed_reference_executed=False,hf_engine_probability_acceptance=False,
        production_thresholds_selected=False,scheduler_at_retained_boundary=2,adam_at_retained_boundary=20,
        outer_records=[copy.deepcopy(outer) for _ in range(3)],rank5_refusals={k:dict(all8_refused=True,extra_adam=0)
                                                                             for k in ('temperature','output_shape')}) for rank in range(8)]
    run=str(c.ROOT/('runs/gigpo-e1-action-prefix-native-job'+job));cg='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_'+job
    result=dict(status='PASS',validation_scope=c.ADMISSION_SCOPE,native_scope=scope,code_commit=c.SOURCE,
        job=job,node='h100-8',binding_sha256=binding_sha,native_initialization='OFFICIAL_MODEL_INITIALIZATION',
        native_actual_restore=False,formal_updates_contributed=0,fixtures_passed=True,cpu_exit=0,gpu_exit=0,
        rank_records=ranks,production_probability_acceptance=False,cleanup_residual_states={},final_gpu={'processes':[]},
        guard_identity=guard,cgroup=cg+'/step_0')
    binding=dict(code_commit=c.SOURCE,helpers=str(c.NATIVE_HELPERS),native_scope=scope,native_initialization='OFFICIAL_MODEL_INITIALIZATION',
        native_actual_restore=False,formal_updates_contributed=0,official_revision=c.REVISION,allowed_safety_modes=['native_fixture'],
        restore_step=25,restore_checkpoint=c.CHECKPOINT,restore_manifest_sha256=c.MANIFEST,**guard)
    terminal=dict(fields=dict(JobId=job,NodeList='h100-8',JobState='COMPLETED',ExitCode='0:0',Requeue='0',Restarts='0'))
    acceptance=dict(status='PASS',code_commit=c.SOURCE,native_scope=scope,job=job,binding_sha256=binding_sha,
        result_path=run+'/result.json',result_sha256='b'*64,review_path=str(c.ROOT/'logs/new-native-review-9999.json'),review_sha256='c'*64,
        native_binding_path=str(c.ROOT/'logs/new-native-binding.json'),native_binding_sha256=binding_sha,
        terminal_path=str(c.ROOT/'logs/new-native-terminal-9999.json'),terminal_sha256='d'*64,
        physical_payload_path=str(c.ROOT/'logs/new-native-physical-9999.json.gz'),physical_payload_sha256='e'*64,
        terminal_state='COMPLETED',terminal_exit_code='0:0',terminal_requeue=0,terminal_restarts=0,cgroup_removed=True,guard_identity=guard)
    originals=[]
    for key,data in [('result',result),('native_binding',binding),('terminal',terminal)]:
        payload=json.dumps(data).encode();digest=hashlib.sha256(payload).hexdigest();acceptance[key+'_sha256']=digest
        originals.append(dict(path=acceptance[key+'_path'],bytes=len(payload),sha256=digest,content_base64=base64.b64encode(payload).decode()))
    # Bind the native binding's actual bytes SHA everywhere, exactly like the runtime.
    binding_sha=acceptance['native_binding_sha256'];result['binding_sha256']=acceptance['binding_sha256']=binding_sha
    for rank in result['rank_records']:rank['binding_sha256']=binding_sha
    payload=json.dumps(result).encode();acceptance['result_sha256']=hashlib.sha256(payload).hexdigest()
    originals[0].update(bytes=len(payload),sha256=acceptance['result_sha256'],content_base64=base64.b64encode(payload).decode())
    review=dict(status='PASS',code_commit=c.SOURCE,native_scope=scope,job=job,binding_sha256=binding_sha,
        result_sha256=acceptance['result_sha256'],native_initialization='OFFICIAL_MODEL_INITIALIZATION',native_actual_restore=False,
        formal_updates_contributed=0,terminal_path=acceptance['terminal_path'],terminal_sha256=acceptance['terminal_sha256'],
        terminal_state='COMPLETED',terminal_exit_code='0:0',terminal_requeue=0,terminal_restarts=0,cgroup_removed=True,
        physical_payload_path=acceptance['physical_payload_path'],physical_payload_sha256=acceptance['physical_payload_sha256'])
    physical=dict(status='ORIGINAL_NATIVE_TERMINAL_AND_PHYSICAL_EVIDENCE_UNREVIEWED',job=job,run=run,result=result,terminal=terminal,
        own_cgroups=[],own_job_queue='',current_compute={'returncode':0,'stdout':'123 other-job'},current_node_queue='123 other-job RUNNING',
        cgroup_release=dict(job_id=job,path=cg,result_cgroup=result['cgroup'],ancestor_path='/sys/fs/cgroup/freezer/slurm_h100-8',
            ancestor_readable=True,stat_outcome='ENOENT',scope_stat_outcome='ENOENT',removed=True),originals=originals,
        physical=[{k:originals[0][k] for k in ('path','bytes','sha256')}])
    return acceptance,result,review,binding,guard,terminal,physical


class RawDeploymentTests(unittest.TestCase):
    def reject_candidate(self,mutate):
        value=read_candidate();mutate(value)
        with self.assertRaises(ValueError):c.validate_candidate(value)

    def reject_native(self,mutate):
        values=native_fixture();mutate(*values)
        with self.assertRaises(ValueError):c.validate_native_payload(*values)

    def test_actual_candidate_remains_not_ready(self):
        value=read_candidate();self.assertTrue(c.validate_candidate(value))
        with self.assertRaises(ValueError):c.validate_ready(value)
        self.assertIsNone(value['source_native_acceptance'])

    def test_exact_parent_physical_closure_retained(self):
        value=read_candidate()
        parent=json.loads((OUT/'ACTION-PREFIX-NATIVE-CANDIDATE-2262942-v1.json').read_text())
        physical=json.loads((OUT/'ACTION-PREFIX-NATIVE-PHYSICAL-2262942-v1.json').read_text())
        self.assertTrue(c.validate_inheritance(value,parent,physical))
        for key in ('metadata','evidence'):
            bad=copy.deepcopy(physical);bad[key][0]['sha256']='0'*64
            with self.assertRaises(ValueError):c.validate_inheritance(value,parent,bad)
        bad=copy.deepcopy(physical);bad['evidence'].append(copy.deepcopy(bad['evidence'][0]))
        with self.assertRaises(ValueError):c.validate_inheritance(value,parent,bad)

    def test_no_old_source_or_helper_scope(self):
        self.reject_candidate(lambda x:x.update(code_commit='52befe968c856162d1bc79cd0171bb38309f4f9e'))
        self.reject_candidate(lambda x:x.update(helpers=str(c.ROOT/'tools/policy-safety-calibration-52befe9-v9')))

    def test_no_trained_origin_or_mode_switch(self):
        self.reject_candidate(lambda x:x.update(restore_step=80))
        self.reject_candidate(lambda x:x.update(calibration_phase='one_outer'))
        self.reject_candidate(lambda x:x.update(calibration_phase='probability_probe_only'))

    def test_no_threshold_or_tolerance_and_no_formal_update(self):
        self.reject_candidate(lambda x:x.update(actor_vllm_abs_tolerance=3.))
        self.reject_candidate(lambda x:x.update(target_old_policy_kl=.1))
        self.reject_candidate(lambda x:x.update(retained_training_updates_contributed=1))

    def test_exact18_helper_bytes_and_inventory(self):
        self.reject_candidate(lambda x:x['runtime_helpers'].pop())
        self.reject_candidate(lambda x:x['runtime_helpers'][0].update(sha256='0'*64))
        self.reject_candidate(lambda x:x['files'].append(copy.deepcopy(x['files'][0])))

    def test_cannot_disable_actual_runner_observation_or_alignment(self):
        self.reject_candidate(lambda x:x.update(capture_actual_runner_inputs=False))
        self.reject_candidate(lambda x:x.update(action_token_mismatch_limit=1))

    def test_source_native_acceptance_cannot_be_borrowed_at_candidate_stage(self):
        self.reject_candidate(lambda x:x.update(source_native_acceptance={'status':'PASS','job':'3267'}))
        with self.assertRaises(ValueError):c.read_native_acceptance(None,read_candidate())

    def test_initialized_native_scope_not_restore_probability_or_formal(self):
        self.assertTrue(c.validate_native_payload(*native_fixture()))
        self.reject_native(lambda a,r,*_:r.update(native_actual_restore=True))
        self.reject_native(lambda a,r,*_:r.update(production_probability_acceptance=True))
        self.reject_native(lambda a,r,*_:r.update(formal_updates_contributed=1))

    def test_native_newsource_rank_count_replay_and_budget(self):
        self.reject_native(lambda a,r,*_:r['rank_records'][5].update(code_commit='0'*40))
        self.reject_native(lambda a,r,*_:r['rank_records'][5]['outer_records'][0].update(optimizer_writes=20))
        self.reject_native(lambda a,r,*_:r['rank_records'][5]['rank5_refusals']['temperature'].update(extra_adam=1))

    def test_exact_terminal_state_exit_node_job_requeue_restarts(self):
        for key,value in [('JobId','3267'),('NodeList','h100-11'),('JobState','FAILED'),('ExitCode','1:0'),('Requeue','1'),('Restarts','1')]:
            self.reject_native(lambda a,r,v,b,g,t,p:t['fields'].update({key:value}))

    def test_cgroup_absence_not_unknown_emptyglob_or_false_review(self):
        self.reject_native(lambda a,r,v,b,g,t,p:v.update(cgroup_removed=False))
        for key,value in [('ancestor_readable',False),('stat_outcome','EACCES'),('scope_stat_outcome','EACCES'),('removed',False)]:
            self.reject_native(lambda a,r,v,b,g,t,p:p['cgroup_release'].update({key:value}))

    def test_result_review_job_and_binding_identity(self):
        self.reject_native(lambda a,r,v,*_:v.update(job='3267'))
        self.reject_native(lambda a,r,v,*_:v.update(binding_sha256='0'*64))
        self.reject_native(lambda a,r,v,*_:v.update(result_sha256='0'*64))

    def test_native_original_bytes_and_inventory_not_interchangeable(self):
        self.reject_native(lambda a,r,v,b,g,t,p:p['originals'][0].update(content_base64='eA=='))
        self.reject_native(lambda a,r,v,b,g,t,p:p['physical'][0].update(sha256='0'*64))
        self.reject_native(lambda a,r,v,b,g,t,p:p['originals'].append(copy.deepcopy(p['originals'][0])))

    def test_old_guard_path_or_byte_refused(self):
        self.reject_candidate(lambda x:x['resource_guard'].update(path=str(c.ROOT/'tools/resource_guard_v2.py')))
        self.reject_native(lambda a,r,v,b,g,*_:r.update(guard_identity={}))

    def test_write_once_and_bound_changed_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            # macOS /var may be an alias of /private/var; use the same strict
            # canonical-path identity required from actual target originals.
            p=Path(tmp).resolve()/'new.json';c.write_once(p,{'a':1});before=p.read_bytes()
            with self.assertRaises(FileExistsError):c.write_once(p,{'a':2})
            self.assertEqual(p.read_bytes(),before)
            record=dict(path=str(p),bytes=p.stat().st_size,sha256=c.sha(p));self.assertEqual(c.read_bound(record),{'a':1})
            p.write_text('{"a":2}')
            with self.assertRaises(ValueError):c.read_bound(record)
            link=Path(tmp).resolve()/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):c.read_bound(dict(path=str(link),sha256=c.sha(p)))

    def test_existing_or_uncertain_once_claim_refused_before_sbatch(self):
        import submit_once_raw_v2 as submit
        with tempfile.TemporaryDirectory() as tmp,patch.object(submit.socket,'gethostname',return_value='h100-8'),\
                patch.object(submit,'INTENT',Path(tmp)/'intent'),patch.object(submit,'RECEIPT',Path(tmp)/'receipt'),\
                patch.object(submit.subprocess,'run') as run:
            submit.INTENT.write_text('{}')
            with self.assertRaises(ValueError):submit.main(str(c.BINDING),'a'*64,'unused','b'*64)
            run.assert_not_called()

    def test_actual_hydra_raw_preview_scope_and_environment_restore(self):
        import os
        from prepare_candidate_raw_v2 import preview_config,validate_preview
        before=dict(os.environ);config=preview_config({'quality_selection_control':c.QUALITY_CONTROL});self.assertTrue(validate_preview(config))
        self.assertEqual(before,dict(os.environ))
        config['actor_rollout_ref']['actor']['ppo_micro_batch_size']=128
        with self.assertRaises(ValueError):validate_preview(config)


if __name__=='__main__':unittest.main()
