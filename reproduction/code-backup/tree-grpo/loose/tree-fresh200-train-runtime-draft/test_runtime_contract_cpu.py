"""New production intake/gate/count contracts with tiny synthetic JSON only.

No Torch, PT, tokenizer/model, assets, remote connection or submission.
Synthetic PASS-shaped data in temporary paths test rejection/acceptance logic;
they are never actual model/threshold/independent release evidence.
"""
import ast
import copy
import hashlib
import io
import importlib.util
import sys
import types
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import production_contract as c
import train_monitor as monitor
import build_package_draft as builder
import render_dispatch
import attempt_accounting as attempts

HERE=Path(__file__).resolve().parent


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True,allow_nan=False).encode();path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def package():return dict(code_commit=c.SOURCE,common_data_identity='data',model_identity={'sha256':'model'},official_model_path='/official/base')


def observation(root):
    p=package();ready=dict(status='FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES',calibration_profile='fresh_base_zero_v1',
        initialized_global_steps=0,parent_completed_outer=0,new_completed_outer=0,optimizer_committed=0,scheduler_advanced=False,
        observed_outer=1,behavior_mu_corrected_profile=True,original_sampler_actor_probability_gate_accepted=False,
        code_commit=c.SOURCE,data_identity='data',model_identity='model',workers=[dict(rank=r,world_size=8,
            optimizer_committed=0,optimizer_state_entries=0,restored_optimizer_steps=[],scheduler_step=0,scheduler_advanced=False,
            fresh_post_monitor_state=dict(calibration_profile='fresh_base_zero_v1',restored_optimizer_steps=[],optimizer_state_entries=0,scheduler_step=0)) for r in range(8)])
    descriptors={'actual_calibration_ready':write(root/'ready.json',ready)}
    descriptors['actual_calibration_acceptance']=write(root/'calibration-review.json',dict(status='INDEPENDENT_ACTUAL_FRESH200_MU_CALIBRATION_ACCEPTED',
        fork_turns='none',calibration_ready_sha256=descriptors['actual_calibration_ready']['sha256'],actual_zero_writes_accepted=True,
        actual_behavior_mu_binding_accepted=True,fresh_initialization_accepted=True,original_sampler_actor_gate_accepted=False))
    descriptors['threshold_acceptance']=write(root/'threshold.json',dict(status='INDEPENDENT_FRESH200_MU_PRODUCTION_THRESHOLDS_ACCEPTED',fork_turns='none',
        trial_only=False,production_thresholds_accepted=True,calibration_ready_sha256=descriptors['actual_calibration_ready']['sha256'],
        calibration_acceptance_sha256=descriptors['actual_calibration_acceptance']['sha256'],code_commit=c.SOURCE,training_protocol=c.PROFILE,
        original_sampler_actor_gate_accepted=False,production_thresholds=dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),
        target_kl_authority=dict(kind='explicit_latest_human_goal_edit',value=.2,original_instruction_time='synthetic fixture only',source='synthetic fixture only')))
    descriptors['source_context_free_review']=write(root/'source-review.json',dict(status='CONTEXT_FREE_FRESH200_SOURCE_ACCEPTED',fork_turns='none',code_commit=c.SOURCE,blocking_findings=[]))
    p['production_inputs']=descriptors
    release=write(root/'release.json',dict(status='READY_FOR_FRESH200_BEHAVIOR_MU_TRAIN200',fork_turns='none',manifest_sha256='manifest',code_commit=c.SOURCE,
        production_thresholds_sha256=descriptors['threshold_acceptance']['sha256'],source_review_sha256=descriptors['source_context_free_review']['sha256'],
        actual_calibration_acceptance_sha256=descriptors['actual_calibration_acceptance']['sha256'],original_sampler_actor_gate_accepted=False))
    return p,release


def identity(run):return dict(job_id='99',run_id=run.name,code_commit=c.SOURCE,config_sha256='config',data_identity='data',model_identity='model')


OWNER_SOURCE=HERE.parent/'tree-behavior-mu-ppo/search_r1/baselines/tree_guard_continue.py'


def ledger(writes):
    # Exactly the owner's real mini enumeration, including one brake then skips.
    return [dict(index=i,rows=16,local_row_indices=list(range((i-1)*16,i*16)),actual_writes=int(i<=writes),
        status=('SUCCESS_ADAM_WRITE' if i<=writes else 'SKIPPED_SAFETY_GUARD_BEFORE_WRITE' if i==writes+1 else 'SKIPPED_BY_PRIOR_ROLLOUT_BRAKE')) for i in range(1,21)]


def owner_module(run,trainer):
    """Load actual JSON-only owner producer; fake only tensor-free IO/context dependencies."""
    name='actual_owner_source_cpu';pkg=types.ModuleType(name);pkg.__path__=[]
    partial=types.ModuleType(name+'.tree_partial_state')
    partial.partial_context=lambda t:dict(identity=dict(identity(run),training_protocol=c.PROFILE,attempted_outer=t.global_steps,
        completed_outer=t.actual_outer_updates,attempt_id=t.training_attempts))
    state=types.ModuleType(name+'.training_state')
    state.file_sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    state.atomic_save=lambda value,path,json_format:write(path,value)['sha256']
    with patch.dict(sys.modules,{name:pkg,partial.__name__:partial,state.__name__:state}):
        spec=importlib.util.spec_from_file_location(name+'.tree_guard_continue',OWNER_SOURCE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,{name:pkg,partial.__name__:partial,state.__name__:state}


def direct(run,outer=1,*,writes=20,attempt=None):
    attempt=outer if attempt is None else attempt
    existing=list((run/'checkpoints/training_state/attempt-results').glob('*.json'))
    totals={str(r):0 for r in range(8)}
    for path in existing:
        value=json.loads(path.read_text())
        if value['status']=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED':
            for rank in value['all_rank_result']['ranks']:totals[str(rank['rank'])]+=rank['optimizer_calls']
    ranks=[]
    for rank in range(8):
        record=dict(identity(run),status='DIRECT_RANK_COMPLETE_OUTER_OBSERVED',training_protocol=c.PROFILE,rank=rank,world_size=8,
            observed_outer=outer,attempt_id=attempt,actual_optimizer_calls=writes,returned_optimizer_calls=writes,
            skipped_empty_minibatches=20-writes,opportunities_processed=20,opportunities_skipped=20-writes,optimizer_opportunity_ledger=ledger(writes),
            scheduler_step_before=outer-1,scheduler_step_after=outer,scheduler_step_invoked=True,
            all_rank_before_admitted=True,all_rank_scheduler_after_valid=True,local_before_error=None,local_scheduler_error=None,
            learning_rates_before=[1e-6*min((outer-1)/57,1)],learning_rates_after=[1e-6*min(outer/57,1)])
        descriptor=write(run/'output/fresh-state'/f'outer-{outer:06d}-attempt-{attempt:06d}-rank{rank:02d}.json',record)
        ranks.append(dict(rank=rank,attempt_id=attempt,accepted_outer=outer,completed_outer_before=outer-1,
            optimizer_calls=writes,total_optimizer_calls=totals[str(rank)]+writes,opportunities_processed=20,opportunities_skipped=20-writes,
            opportunity_ledger=ledger(writes),scheduler_before=outer-1,scheduler_after=outer,
            direct_rank_record=dict(descriptor,rank=rank,actual_optimizer_calls=writes,scheduler_step_before=outer-1,scheduler_step_after=outer)))
        audit=dict(identity(run),outer_step=outer,attempt_id=attempt,rank=rank,world_size=8,local_rows=320,local_minibatch=16,local_microbatch=2,
            epochs=1,successful_optimizer_calls=writes,skipped_empty_minibatches=20-writes,finite_grad_norms=writes,
            existing_grad_norms=[1.]*writes,local_shard_sample_layout_unchanged=True)
        del audit['model_identity']
        write(run/'output/tree-rank-audit'/f'step-{outer:06d}-attempt-{attempt:06d}-rank-{rank:02d}.json',audit)
    ids=dict(identity(run),training_protocol=c.PROFILE,attempt_id=attempt,attempted_outer=outer,completed_outer=outer-1)
    result=dict(protocol=attempts.RESULT_PROTOCOL,decision='COMPLETED_OUTER',identity=ids,attempt_id=attempt,
        accepted_outer=outer,completed_outer_before=outer-1,ranks=ranks)
    trainer=NS(global_steps=outer,actual_outer_updates=outer-1,training_attempts=attempt,actor_rollout_wg=NS(world_size=8),
        config=NS(trainer=NS(default_local_dir=str(run/'checkpoints'))),completed_optimizer_calls_per_rank=totals)
    module,dependencies=owner_module(run,trainer)
    with patch.dict(sys.modules,dependencies),patch.dict(os.environ,{'SEARCH_R1_BASELINE_OUTPUT':str(run/'output')}):
        module.record_completed_result(trainer,NS(meta_info={'tree_fresh_actor_result':result}))
    return result


def allskip(run,attempt,completed):
    ids=dict(identity(run),training_protocol=c.PROFILE,attempt_id=attempt,attempted_outer=completed+1,completed_outer=completed)
    workers=[dict(rank=r,committed_optimizer_calls=0,optimizer_opportunity_ledger=ledger(0),safety_guard=True,
        model_adam_finite=True,scheduler_step_before_actual=completed,scheduler_step_after_actual=completed) for r in range(8)]
    evidence=write(run/'checkpoints/training_state'/f'partial_attempt_{attempt:06d}_outer_{completed+1:06d}'/'manifest.json',
        dict(identity=ids,worker_states_verified=True,verification_error=None,continuation_decision='CONTINUE_FRESH_ROLLOUT',
        outer_completed=False,completed_outer_updates=completed,credited_outer_candidate=False,automatic_rollback=False,
        actual_optimizer_calls_per_rank={str(r):0 for r in range(8)},worker_outcome={'worker_states':workers}))
    epoch,cursor=attempts.sampler_position(attempt)
    write(run/'checkpoints/training_state/attempt-results'/f'attempt-{attempt:06d}.json',dict(protocol='tree_partial_state_v1',
        status='ALL_OPPORTUNITIES_SKIPPED_NO_OUTER_CREDIT',identity=ids,partial_evidence=evidence,
        actual_optimizer_calls_per_rank={str(r):0 for r in range(8)},consumed_epoch=epoch,consumed_cursor=cursor))
    for rank in range(8):
        audit=dict(identity(run),outer_step=completed+1,attempt_id=attempt,rank=rank,world_size=8,local_rows=320,local_minibatch=16,
            local_microbatch=2,epochs=1,successful_optimizer_calls=0,skipped_empty_minibatches=20,finite_grad_norms=0,
            existing_grad_norms=[],local_shard_sample_layout_unchanged=True)
        del audit['model_identity'];write(run/'output/tree-rank-audit'/f'step-{completed+1:06d}-attempt-{attempt:06d}-rank-{rank:02d}.json',audit)


def checkpoint(run,outer):
    if not (run/'checkpoints/training_state/attempt-results'/f'attempt-{outer:06d}.json').exists():
        for step in range(1,outer+1):
            if not (run/'checkpoints/training_state/attempt-results'/f'attempt-{step:06d}.json').exists():direct(run,step)
    observed=attempts.attempt_results(run,package(),'config',outer)
    accounting=dict(protocol=attempts.PROTOCOL,training_attempts=observed['training_attempts'],refused_attempts=0,guard_refused_attempts=0,
        completed_outer_updates=outer,partial_optimizer_calls_per_rank={},completed_optimizer_calls_per_rank=observed['completed_optimizer_calls_per_rank'],
        total_optimizer_calls_per_rank=observed['total_optimizer_calls_per_rank'],partial_evidence=[])
    directory=run/'checkpoints/training_state'/f'global_step_{outer}'
    ids={key:value for key,value in identity(run).items() if key!='run_id'};ids.update(completed_step=outer,actual_update_limit=200,guard_continue_accounting=accounting)
    manifest=dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',identity=ids,
        worker_states=[dict(identity=dict(ids,rank=r,world_size=8,cuda_visible_devices='UUID-synthetic'),full_worker_state_finite=True,
            scheduler_last_epoch=outer,exact_same_world_reload_verified=outer==5,path=str(directory/f'rank-{r:02d}.pt')) for r in range(8)],
        driver=dict(next_step=outer+1,exact_reload_verified=outer==5,path=str(directory/'driver.pt')),
        hf_checkpoint=str(run/'checkpoints/actor'/f'global_step_{outer}'),hf_identity={'sha256':'hf','files':{}})
    write(directory/'manifest.json',manifest);return manifest


class ProductionTests(unittest.TestCase):
    def test_null_draft_receipts_cannot_start(self):
        with tempfile.TemporaryDirectory() as root:
            p=package();p['production_inputs']={key:None for key in c.INPUT_NAMES}
            with self.assertRaises(ValueError):c.production_inputs(p,manifest_sha256='manifest',dispatch_review_descriptor=None,trusted_root=root)

    def test_byte_bound_actual_shaped_inputs_and_external_package_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p,release=observation(root)
            values,pins=c.production_inputs(p,manifest_sha256='manifest',dispatch_review_descriptor=release,trusted_root=root)
            self.assertEqual(values['target_old_policy_kl'],.2);self.assertEqual(len(pins),5)
            with self.assertRaises(ValueError):c.production_inputs(p,manifest_sha256='other',dispatch_review_descriptor=release,trusted_root=root)
            Path(p['production_inputs']['actual_calibration_ready']['path']).write_text('{}')
            with self.assertRaises(ValueError):c.production_inputs(p,manifest_sha256='manifest',dispatch_review_descriptor=release,trusted_root=root)

    def test_trial_or_missing_tail_and_wrong_human_kl_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p,_=observation(root)
            values={name:c.read_bound(d['path'],d['sha256']) for name,d in p['production_inputs'].items()}
            for field,value in (('trial_only',True),('fork_turns','all'),('production_thresholds',dict(target_old_policy_kl=.2,negative_tail_limit=None,initial_actor_repeat_max_abs_delta=0)),('production_thresholds',dict(target_old_policy_kl=.01,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0))):
                bad=copy.deepcopy(values);bad['threshold_acceptance'][0][field]=value
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):c.observed_production_inputs(p,bad)

    def test_config_retains_user_kl_and_rejects_hidden_trial_tail(self):
        spec=json.loads((HERE/'CONFIGURATION_SPEC.json').read_text());values=dict(target_old_policy_kl=.2,negative_tail_limit=.03,initial_actor_repeat_max_abs_delta=0)
        result=c.configuration_overrides(spec,values)
        self.assertIn('actor_rollout_ref.actor.policy_safety.target_old_policy_kl=0.2',result)
        self.assertIn('actor_rollout_ref.actor.policy_safety.negative_tail_limit=0.03',result)
        spec['configuration_overrides'].append('actor_rollout_ref.actor.policy_safety.negative_tail_limit=0.01')
        with self.assertRaises(ValueError):c.configuration_overrides(spec,values)

    def test_first5_needs_real_external_receipt_and_copies_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';gate=run/'output/acceptance-step5';p=package()
            self.assertFalse(c.publish_external_approval(run,p,'config',{}));self.assertFalse((gate/'approved.json').exists())
            ready=write(gate/'ready.json',dict(completed_outer_updates=5,next_outer_update=6,actual_update_limit=200,full_state_manifest_sha256='state',hf_identity={'sha256':'hf'}))
            proofs=dict(full_state_manifest_sha256='state',hf_reload_report_sha256='hfcheck',state_cpu_report_sha256='statecheck',behavior_mu_cpu_report_sha256='mucheck',development={'sha256':'dev'},rank_counts={'record_sha256':{'original':'sha'}})
            review=dict({key:value for key,value in identity(run).items() if key!='run_id'},status='PASS',fork_turns='none',ready_sha256=ready['sha256'],
                full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True,fixed512_evaluation_accepted=True,behavior_mu_binding_accepted=True,
                full_state_manifest_sha256='state',hf_identity_sha256='hf',hf_reload_report_sha256='hfcheck',state_cpu_report_sha256='statecheck',behavior_mu_cpu_report_sha256='mucheck',development_report_sha256='dev',direct_rank_record_sha256={'original':'sha'})
            descriptor=write(gate/'external-review.json',review);write(gate/'independent-envelope.json',{'review':descriptor})
            result=c.publish_external_approval(run,p,'config',proofs)
            self.assertEqual((gate/'approved.json').read_bytes(),(gate/'external-review.json').read_bytes());self.assertEqual(result['review_sha256'],descriptor['sha256'])
            review['fixed512_evaluation_accepted']=False;descriptor=write(gate/'external-review.json',review);write(gate/'independent-envelope.json',{'review':descriptor})
            with self.assertRaises(ValueError):c.publish_external_approval(run,p,'config',proofs)

    def test_direct_all8_counts_and_scheduler_are_not_driver_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';direct(run)
            result=c.direct_rank_records(run,package(),'config',1);self.assertEqual(result['actual_optimizer_returns'],160)
            path=run/'output/fresh-state/outer-000001-attempt-000001-rank03.json';value=json.loads(path.read_text());value['actual_optimizer_calls']=19;write(path,value)
            with self.assertRaises(ValueError):c.direct_rank_records(run,package(),'config',1)

    def test_partial_failed_outer_keeps_writes_but_gets_no_completed_credit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';direct(run)
            write(run/'checkpoints/training_state/partial_outer_000002/manifest.json',dict(status='partial',actual_committed=1,state_finite=False))
            write(run/'checkpoints/training_state/partial_outer_000003/unmatched-worker-failure/manifest.json',dict(status='driver-only',actual_committed=None))
            value=monitor.progress_evidence(run)
            self.assertEqual(value['actual_completed_outer_from_attempt_results'],1);self.assertEqual(sum(value['completed_actual_optimizer_calls_per_rank'].values()),160)
            self.assertEqual(value['partial_attempt_evidence'][0]['value']['actual_committed'],1);self.assertFalse(value['exact_resume_supported'])
            self.assertEqual(len(value['partial_attempt_evidence']),2)

    def test_actual_saved_state_api_includes_cuda_identity_and_reload_only5(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99'
            checkpoint(run,5);checkpoint(run,200)
            c.checkpoint_manifest(run,package(),'config',5);c.checkpoint_manifest(run,package(),'config',200)
            path=run/'checkpoints/training_state/global_step_5/manifest.json';value=json.loads(path.read_text());value['worker_states'][6]['exact_same_world_reload_verified']=False;write(path,value)
            with self.assertRaises(ValueError):c.checkpoint_manifest(run,package(),'config',5)

    def test_validation_requires_coverage_counter_identity_and_ledger_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';folder=run/'output/validation-cost';folder.mkdir(parents=True)
            ledger=folder/'validation-1-1.jsonl';ledger.write_text('{"actual":"ledger"}\n')
            value=dict(identity(run),run=str(run/'output'),actual_completed_outer=5,global_step=5,phase='acceptance',status='VALIDATION_COST_COMPLETE',training_protocol=c.PROFILE,
                dataset_rows=512,evaluated_rows=512,metrics={'val/evaluated_count/synthetic':512,'val/qa_count/all':512,'val/f1/all':.4,'val/em/all':.3},ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest())
            path=folder/'validation-1-1.json';write(path,value);c.validation_reports(run,5,package=package(),config_hash='config')
            value['evaluated_rows']=511;write(path,value)
            with self.assertRaises(ValueError):c.validation_reports(run,5,package=package(),config_hash='config')
            value['evaluated_rows']=512;value['config_sha256']='different';write(path,value)
            with self.assertRaises(ValueError):c.validation_reports(run,5,package=package(),config_hash='config')

    def test_serial_cpu_checks_use_actual_checker_cli_and_cuda_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';checkpoint(run,5)
            (run/'output').mkdir(exist_ok=True)
            owned=NS(calls=[])
            def spawn(command,log,env):owned.calls.append((command,env));return NS(poll=lambda:0,returncode=0)
            owned.spawn=spawn
            queue=monitor.CpuChecks(owned,run,HERE,package(),'config',5)
            self.assertEqual(len(owned.calls),1);self.assertFalse(queue.poll());self.assertEqual(len(owned.calls),2)
            self.assertFalse(queue.poll());self.assertTrue(queue.poll())
            self.assertIn('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/check_real_checkpoint.py',owned.calls[0][0])
            self.assertTrue(all(env['CUDA_VISIBLE_DEVICES']=='' and env['TREE_FIXED_ACTOR_PROFILE']=='0' for command,env in owned.calls))
            self.assertIn('--manifest-sha',owned.calls[1][0]);self.assertIn('--config-sha',owned.calls[2][0])

    def test_complete_review_inventory_and_allowlist_exclude_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir();frozen={}
            for name in builder.WRAPPER_FILES:
                path=source/name;path.write_text(name);frozen[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            inventory={'wrapper_files':frozen,'files':frozen};inputs={'runtime_entries':{name:{'path':str(source/name),'sha256':frozen[name]} for name in builder.RUNTIME_ENTRIES}}
            prepared=builder.prepare_wrapper_inventory(inventory,inputs,source);output=root/'out';output.mkdir()
            copied=builder.copy_wrapper_inventory(output,prepared);self.assertEqual(set(copied),set(builder.WRAPPER_FILES))
            (source/'PLAN.md').write_text('changed')
            with self.assertRaises(ValueError):builder.prepare_wrapper_inventory(inventory,inputs,source)
            archive=io.BytesIO()
            with tarfile.open(fileobj=archive,mode='w') as tar:
                for name in ('search_r1/a.py','verl/version/version','e5_Flat.index','wiki-18.jsonl.gz','model.safetensors'):
                    raw=name.encode();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
            selected=builder.materialize_source_archive(archive.getvalue(),output/'repo')
            self.assertEqual(selected['selected_count'],2);self.assertEqual(selected['excluded_count'],3)

    def test_actual_owner_producer_variable_writes_and_allskip_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99'
            direct(run,1,writes=3,attempt=1);allskip(run,2,1);direct(run,2,writes=20,attempt=3)
            result=c.direct_rank_records(run,package(),'config',2)
            self.assertEqual(result['training_attempts'],3);self.assertEqual(result['direct_rank_records'],16)
            self.assertEqual(result['tree_rank_records'],24);self.assertEqual(result['opportunities_processed'],480)
            self.assertEqual(result['actual_writes_per_rank'],{str(r):23 for r in range(8)})
            self.assertEqual(result['skipped_opportunities_per_rank'],{str(r):37 for r in range(8)})
            self.assertEqual(attempts.sampler_position(3),(0,1536));self.assertEqual(attempts.sampler_position(331),(1,512))
            path=run/'checkpoints/training_state/attempt-results/attempt-000002.json'
            value=json.loads(path.read_text());value['consumed_cursor']=1024-512;write(path,value)
            with self.assertRaises(ValueError):c.direct_rank_records(run,package(),'config',2)

    def test_ledger_brake_types_incomplete_and_zero_cannot_credit(self):
        value=ledger(3);self.assertEqual(attempts.ledger_writes(value),3)
        bad=copy.deepcopy(value);bad[9]['status']='SUCCESS_ADAM_WRITE';bad[9]['actual_writes']=1
        with self.assertRaises(ValueError):attempts.ledger_writes(bad)
        bad=copy.deepcopy(value);bad[0]['index']=True
        with self.assertRaises(ValueError):attempts.ledger_writes(bad)
        with self.assertRaises(ValueError):attempts.ledger_writes(value[:-1])
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99'
            with self.assertRaises(ValueError):direct(run,writes=0)

    def test_real_owner_physical_digest_and_gathered_identity_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';direct(run,writes=3)
            path=run/'checkpoints/training_state/attempt-results/attempt-000001.json'
            value=json.loads(path.read_text());value['all_rank_result']['identity']['model_identity']='different'
            write(path,value)
            with self.assertRaises(ValueError):attempts.attempt_results(run,package(),'config',1)
            value['all_rank_result']['identity']['model_identity']='model';write(path,value)
            physical=run/'output/fresh-state/outer-000001-attempt-000001-rank03.json'
            record=json.loads(physical.read_text());record['optimizer_opportunity_ledger'][0]['gradient_preclip']=.5;write(physical,record)
            with self.assertRaises(ValueError):attempts.attempt_results(run,package(),'config',1)

    def test_once_submitting_waits_read_only_and_unknown_never_retries(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'intent.json';write(path,dict(status='SUBMITTING',manifest_sha256='m',review_sha256='r'))
            elapsed=[0.];sleeps=[]
            def sleep(value):
                sleeps.append(value);elapsed[0]+=value;write(path,dict(status='SUBMITTED',manifest_sha256='m',review_sha256='r',job_id=99))
            result=c.submitted_intent(path,'m','r','99',clock=lambda:elapsed[0],sleep=sleep)
            self.assertEqual(result['job_id'],99);self.assertEqual(sleeps,[.2])
            write(path,dict(status='UNKNOWN',manifest_sha256='m',review_sha256='r'))
            with self.assertRaises(ValueError):c.submitted_intent(path,'m','r','99',clock=lambda:elapsed[0],sleep=sleep)
            self.assertEqual(sleeps,[.2])
            write(path,dict(status='SUBMITTED',manifest_sha256='m',review_sha256='r',job_id=98))
            with self.assertRaises(ValueError):c.submitted_intent(path,'m','r','99')

    def test_frozen_actual_fifteen_interface_bytes_not_owner_or_gpu_acceptance(self):
        inputs=json.loads((HERE/'INPUTS.schema.json').read_text());source=inputs['source']
        value,sha,raw=builder.verify_source_interface(source,Path(source['path']))
        self.assertEqual(len(value['source_files']),15);self.assertEqual(sha,source['source_interface_manifest']['sha256'])
        self.assertFalse(value['actual_gpu_executed']);self.assertFalse(value['training_release'])
        changed=copy.deepcopy(source);changed['code_commit']='0'*40
        with self.assertRaises(ValueError):builder.verify_source_interface(changed,Path(source['path']))
        changed=copy.deepcopy(source);changed['source_interface_receipt_sha256']='0'*64
        with self.assertRaises(ValueError):builder.verify_source_interface(changed,Path(source['path']))

    def test_checkpoint_accounting_keeps_allskip_histories_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'tree-fresh200-train-99';allskip(run,1,0);allskip(run,2,0);direct(run,1,attempt=3)
            observed=attempts.attempt_results(run,package(),'config',1)
            entries=[]
            for attempt in (1,2):
                value=json.loads((run/'checkpoints/training_state/attempt-results'/f'attempt-{attempt:06d}.json').read_text())
                entries.append(dict(value['partial_evidence'],attempt_id=attempt,credited_outer_candidate=False,optimizer_calls_per_rank={str(r):0 for r in range(8)}))
            trainer=NS(actor_rollout_wg=NS(world_size=8),actual_outer_updates=1,training_attempts=3,refused_attempts=2,
                guard_refused_attempts=2,partial_optimizer_calls_per_rank={},completed_optimizer_calls_per_rank={str(r):20 for r in range(8)},partial_attempt_evidence=entries)
            module,_=owner_module(run,trainer);value=module.accounting(trainer)
            attempts.checkpoint_accounting(value,observed,1,run)
            value['partial_evidence'][1]=copy.deepcopy(value['partial_evidence'][0])
            with self.assertRaises(ValueError):attempts.checkpoint_accounting(value,observed,1,run)

    def test_actual_profile_finish_keeps_repeated_candidate_attempts_distinct(self):
        with tempfile.TemporaryDirectory() as root:
            tree=ast.parse((HERE/'sitecustomize.py').read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='finish')
            calls=[];namespace={'Path':Path,'os':os,'json':json,'hashlib':hashlib,'__file__':str(HERE/'sitecustomize.py'),
                'memory':lambda:{'cuda_available':False},'original_finish':lambda self,*args,**kw:calls.append(args)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(method)],type_ignores=[])),str(HERE/'sitecustomize.py'),'exec'),namespace)
            module,_=owner_module(Path(root),None)
            with patch.dict(os.environ,{'SEARCH_R1_TREE_RANK_AUDIT_DIR':root}):
                for attempt in (1,2):
                    record=NS(step=1,rank=0,world=8,attempt_id=attempt,attempt_suffix=module.evidence_suffix({'baseline_attempt_id':attempt}),_profile={})
                    namespace['finish'](record,0,20)
            paths=sorted((Path(root)/'memory-profile').glob('*.json'))
            self.assertEqual([json.loads(p.read_text())['attempt_id'] for p in paths],[1,2]);self.assertEqual(calls,[(0,20),(0,20)])

    def test_renderer_requires_explicit_train_limit_and_no_calibration_flags(self):
        path=c.ROOT/'instruct300/candidates/new-train200'
        with self.assertRaises(ValueError):render_dispatch.render(path,'a'*64,c.ROOT/'review.json','b'*64,None)
        text=render_dispatch.render(path,'a'*64,c.ROOT/'review.json','b'*64,72)
        self.assertIn('--time=72:00:00',text);self.assertIn('--nodelist=h100-9',text);self.assertNotIn('CALIBRATION_ONLY',text)


if __name__=='__main__':unittest.main()
