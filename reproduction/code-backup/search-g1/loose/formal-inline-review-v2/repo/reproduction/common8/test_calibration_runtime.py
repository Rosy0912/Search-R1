"""CPU scalar/identity fixtures only; no native execution or healthy admission."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from reproduction.common8 import calibration_runtime as runtime
from reproduction.common8 import test_policy_calibration_plan as plan_fixtures
from reproduction.common8 import test_calibration_evaluation as evaluation_fixtures
from search_g1.policy_safety import calibration_state_snapshot


def state_fixtures():
    module=torch.nn.Linear(2,1);adam=torch.optim.AdamW(module.parameters(),lr=1e-6)
    for _ in range(20):
        adam.zero_grad();module(torch.ones(1,2)).sum().backward();adam.step()
    tensors,summary=calibration_state_snapshot(module,adam)
    assert all(torch.isfinite(value).all() for value in tensors.values())
    run=dict(run_id='fixture-child')
    plan=dict(role='policy_safety_calibration_discovery',parent_production_run_id='fixture-parent',slurm_job_id='123')
    sha='1'*64
    allocation=dict(accepted=True,run_id=run['run_id'],slurm_job_id='123',
                    actual_cuda_runtime_uuids=['fixture-uuid-'+str(i) for i in range(8)])
    proofs={rank:dict(run_id=run['run_id'],execution_role=plan['role'],parent_production_run_id='fixture-parent',
                     plan_sha256=sha,rank=rank,slurm_job_id='123',pid=100+rank,start_ticks=200,
                     cuda_uuid=allocation['actual_cuda_runtime_uuids'][7-rank],
                     initial_optimizer_writes=0,initial_adam_state_entries=0,final_optimizer_writes=20,
                     **summary,all_parameters_and_adam_finite=True) for rank in range(8)}
    return proofs,run,plan,sha,allocation


class CalibrationRuntimeTests(unittest.TestCase):
    def test_freeze_reads_real_model_writer_schema_and_rejects_changed_rank_proofs(self):
        import hashlib
        from reproduction.common8.freeze_calibration_plan import _actual_prerequisites
        from reproduction.common8.test_model_smoke import ModelSmokeTests
        rows,ids,uuids=ModelSmokeTests().proofs()
        run=dict(run_id='cpu-only');job=dict(JobId='123',TRES='mem=512G')
        allocation=dict(actual_cuda_runtime_uuids=uuids)
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            for phase in ('initial','coexistence'):
                root=base/'logs'/('model-smoke-'+phase);root.mkdir(parents=True)
                proofs=[];hashes={}
                for rank in range(8):
                    row=dict(rows[rank],phase=phase,slurm_job_id='123',initial_parameters_finite=True)
                    path=root/f'rank-{rank}.json';path.write_text(json.dumps(row))
                    proofs.append(row);hashes[str(rank)]=hashlib.sha256(path.read_bytes()).hexdigest()
                (root/'complete.json').write_text(json.dumps(dict(accepted=True,run_id='cpu-only',slurm_job_id='123',
                    phase=phase,ranks=proofs,rank_file_sha256=hashes,actual_grpo_updates=0,
                    scope='CPU schema fixture only; no actual GPU execution')))
            with patch('reproduction.common8.stage_retrieval.binding',return_value=(run,job,allocation)), \
                 patch('reproduction.common8.probe_admission.verify_training_gate',return_value=run), \
                 patch('reproduction.common8.validation_identity.expected',return_value=(None,None,ids)), \
                 patch('reproduction.common8.model_smoke.coexistence'):
                # Real JSON/SHA reader, aggregate and runtime-precision schema all execute.
                _actual_prerequisites(base,run,{},'same','123')
                root=base/'logs/model-smoke-coexistence';path=root/'rank-7.json';complete=root/'complete.json'
                original=path.read_bytes();original_complete=complete.read_bytes()
                for field in ('hash','finite','config','rank','missing'):
                    row=json.loads(original);summary=json.loads(original_complete)
                    if field=='missing': path.unlink()
                    elif field=='hash': path.write_bytes(original+b' ')
                    else:
                        if field=='finite': row['initial_parameters_finite']=False
                        elif field=='config': row['resolved_config_sha256']='different'
                        else: row['rank']=True
                        path.write_text(json.dumps(row));summary['ranks'][7]=row
                        summary['rank_file_sha256']['7']=hashlib.sha256(path.read_bytes()).hexdigest()
                        complete.write_text(json.dumps(summary))
                    with self.subTest(field=field),self.assertRaises((ValueError,AssertionError,FileNotFoundError)):
                        _actual_prerequisites(base,run,{},'same','123')
                    path.write_bytes(original);complete.write_bytes(original_complete)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Actual /proc birth/environment test requires Linux')
    def test_live_worker_birth_and_bound_environment_use_real_owned_cpu_process(self):
        from reproduction.a800.owned_processes import identity
        with tempfile.TemporaryDirectory() as root:
            process=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE,
                env=dict(SEARCH_G1_BASE=root,SLURM_JOB_ID='fixture-job'))
            try:
                actual=identity(process.pid);row=dict(pid=process.pid,start_ticks=actual['start_ticks'])
                observed=runtime.verify_live_optimizer_workers([row],Path(root),'fixture-job')
                self.assertEqual(observed[0]['pid'],process.pid)
                self.assertEqual(observed[0]['start_ticks'],actual['start_ticks'])
                with self.assertRaises(AssertionError):
                    runtime.verify_live_optimizer_workers([dict(row,start_ticks=row['start_ticks']+1)],Path(root),'fixture-job')
                with self.assertRaises(AssertionError):
                    runtime.verify_live_optimizer_workers([row],Path(root),'different-job')
            finally:
                process.stdin.close();process.wait(timeout=5)

    def test_actual_dataproto_repeat_reorder_numpy_group_keys_preserve_duplicate_occurrences(self):
        try:
            from verl import DataProto
        except ModuleNotFoundError as failure:
            self.skipTest('Actual DataProto dependency unavailable locally: '+str(failure))
        import numpy as np
        inputs=torch.arange(512*3,dtype=torch.int64).reshape(512,3)
        inputs[1]=inputs[0]  # Canonical duplicate occurrences are retained.
        indices=np.arange(512,dtype=np.int64);indices[1]=indices[0]
        batch=DataProto.from_single_dict(dict(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                    position_ids=torch.arange(3).expand(512,3).clone(),index=indices,
                    data_source=np.array(['nq']*256+['hotpotqa']*256,dtype=object)))
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);(base/'logs').mkdir()
            phase=(base,dict(run_id='fixture-child',input_sha256={'train.parquet':'2'*64}),
                   dict(slurm_job_id='123'),'1'*64)
            original=batch.batch['input_ids'].clone()
            parents=runtime.training_inputs(batch,phase)
            self.assertEqual(len(parents),512);self.assertEqual(parents[0],parents[1])
            repeated=batch.repeat(5,interleave=True)
            observed=runtime.training_inputs(repeated,phase,parents)
            self.assertEqual(observed['actual_trajectory_count'],2560)
            self.assertTrue(torch.equal(batch.batch['input_ids'],original))
            repeated.non_tensor_batch['uid']=repeated.non_tensor_batch['index'].copy()
            repeated.reorder(torch.arange(2559,-1,-1))
            tensor_before=repeated.batch['input_ids'].clone()
            uid_before=repeated.non_tensor_batch['uid'].copy()
            runtime.training_groups_before_update(repeated,phase)
            self.assertTrue(torch.equal(repeated.batch['input_ids'],tensor_before))
            self.assertTrue(np.array_equal(repeated.non_tensor_batch['uid'],uid_before))
            repeated.non_tensor_batch['uid'][0]=-1
            with self.assertRaises(AssertionError): runtime.training_groups_before_update(repeated,phase)

    def test_real_cpu_adam_summaries_bind_all_eight_permuted_fixture_devices(self):
        args=state_fixtures()
        result=runtime.verify_optimizer_states(*args)
        self.assertEqual(len(result),8)
        self.assertTrue(all(row['adam_steps']==[20.,20.] for row in result))
        # This CPU fixture deliberately does not establish physical GPU identity.
        self.assertTrue(all(row['cuda_uuid'].startswith('fixture-') for row in result))

    def test_actual_state_summary_missing_rank_and_false_integer_rank_rejected(self):
        args=state_fixtures()
        for variant in ('missing','boolkey','boolrow'):
            changed=copy.deepcopy(args);proofs=changed[0]
            if variant=='missing': proofs.pop(7)
            elif variant=='boolkey': proofs[False]=proofs.pop(0)
            else: proofs[7]['rank']=True
            with self.subTest(variant=variant),self.assertRaises(AssertionError):
                runtime.verify_optimizer_states(*changed)

    def test_actual_state_summary_rejects_cross_run_role_job_plan_and_parent(self):
        args=state_fixtures()
        for field in ('run_id','execution_role','slurm_job_id','plan_sha256','parent_production_run_id'):
            changed=copy.deepcopy(args);changed[0][7][field]='different'
            with self.subTest(field=field),self.assertRaises(AssertionError):
                runtime.verify_optimizer_states(*changed)

    def test_actual_state_summary_rejects_dirty_initial_partial_steps_nonfinite_and_duplicate_worker(self):
        args=state_fixtures()
        for field,value in (('initial_optimizer_writes',1),('initial_adam_state_entries',1),
                            ('final_optimizer_writes',19),('final_optimizer_writes',True),
                            ('adam_steps',[19.,20.]),('adam_steps',[float('nan'),20.]),
                            ('adam_steps',[True,20.]),('all_parameters_and_adam_finite',False),
                            ('all_parameters_and_adam_finite',1),('adam_state_tensor_count',5),
                            ('pid',100),('cuda_uuid','fixture-uuid-7'),('start_ticks',0)):
            changed=copy.deepcopy(args);changed[0][7][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(AssertionError):
                runtime.verify_optimizer_states(*changed)

    def test_active_plan_does_not_change_production_and_rejects_unknown_child_role(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);(base/'logs').mkdir()
            path=base/'logs/run-manifest.json';path.write_text(json.dumps(dict(profile='common8',run_id='production')))
            with patch.dict(os.environ,SEARCH_G1_PROFILE='common8',SEARCH_G1_BASE=root):
                self.assertIsNone(runtime.active_plan())
                path.write_text(json.dumps(dict(profile='common8',run_id='fixture',policy_calibration_role='unknown')))
                with self.assertRaises(ValueError): runtime.active_plan()
            with patch.dict(os.environ,SEARCH_G1_PROFILE='a800'):
                self.assertIsNone(runtime.active_plan())

    def test_active_plan_uses_actual_external_job_and_original_plan_files(self):
        fixture=plan_fixtures.CalibrationPlanTests();fixture.setUp()
        with tempfile.TemporaryDirectory() as root:
            base,run=fixture.write_plan_files(root)
            with patch.dict(os.environ,SEARCH_G1_PROFILE='common8',SEARCH_G1_BASE=root,SLURM_JOB_ID='3265'):
                selected=runtime.active_plan()
                self.assertEqual(selected[1],run);self.assertEqual(selected[2],fixture.discovery)
            with patch.dict(os.environ,SEARCH_G1_PROFILE='common8',SEARCH_G1_BASE=root,SLURM_JOB_ID='other'):
                with self.assertRaises(ValueError): runtime.active_plan()

    def test_cpu_measurement_wrapper_matches_original_em_and_preserves_exclusive_evidence(self):
        rows,ids,sources=evaluation_fixtures.canonical_rows()
        class Sources:
            def tolist(self): return sources
        record=dict(sha256='2'*64);frame=dict(data_source=Sources())
        run=dict(run_id='fixture-child');plan=dict(parent_production_run_id='fixture-parent',slurm_job_id='123',
                                                 role='policy_safety_calibration_discovery')
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);(base/'logs').mkdir()
            # Only the canonical-file adapter is replaced; real scalar aggregate/write run.
            with patch('reproduction.common8.validation_identity.expected',return_value=(record,frame,ids)), \
                 patch.dict('sys.modules',{'verl.utils.reward_score':evaluation_fixtures._reward_package}):
                result=runtime.evaluation(base,run,plan,'1'*64,0,rows,[row['em'] for row in rows])
                self.assertEqual(result['question_count'],512);self.assertFalse(result['production_admission'])
                path=base/'logs/policy-calibration-evaluation-before.json';original=path.read_bytes()
                with self.assertRaises(FileExistsError):
                    runtime.evaluation(base,run,plan,'1'*64,0,rows,[row['em'] for row in rows])
                self.assertEqual(path.read_bytes(),original)
                with self.assertRaises(AssertionError): runtime.evaluation(base,run,plan,'1'*64,1,rows,[0]*512)
                self.assertFalse((base/'logs/policy-calibration-evaluation-after.json').exists())

    def test_wrong_horizon_second_outer_or_final_phase_rejects_before_save(self):
        with tempfile.TemporaryDirectory() as root:
            phase=(Path(root),{},dict(phase_budget=dict(outer_count=1,final_evaluation_in_this_phase=False)),'1'*64)
            for horizon,step in ((2,1),(301,2)):
                trainer=SimpleNamespace(total_training_steps=horizon,global_steps=step)
                with self.subTest(horizon=horizon,step=step),self.assertRaises(AssertionError):
                    runtime.finish_after_update(trainer,phase)
            bad=copy.deepcopy(phase);bad[2]['phase_budget']['final_evaluation_in_this_phase']=True
            with self.assertRaises(AssertionError):
                runtime.finish_after_update(SimpleNamespace(total_training_steps=301,global_steps=1),bad)


if __name__=='__main__': unittest.main()
