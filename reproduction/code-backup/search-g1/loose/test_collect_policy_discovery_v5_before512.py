"""PureCPU original-byte/schema fixtures; not actual native512 evidence."""
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_before512',HERE/'collect_policy_discovery_v5_before512.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);h=m.helper()


def originals():
    names=dict(launch='policy-discovery-v5-native-launch.json',birth='policy-discovery-v5-observation-1791254632444188.json',
        terminal='policy-discovery-v5-observation-1791256551399966.json')
    return {label:dict(path=str(HERE/name),sha256=h.sha((HERE/name).read_bytes())) for label,name in names.items()}


def before_fixture(base):
    source=HERE.parent/'repo/reproduction/common8/calibration_evaluation.py'
    launch=h.strict_json((HERE/'policy-discovery-v5-native-launch.json').read_bytes())
    target=base/'repo/reproduction/common8/calibration_evaluation.py';target.parent.mkdir(parents=True);target.write_bytes(source.read_bytes())
    aggregate=h.evaluation_aggregate(base,launch['source_sha256'])
    ids=[h.sha(str(i).encode()) for i in range(512)];sources=['nq']*256+['hotpotqa']*256
    rows=[dict(question_id=ids[i],data_source=sources[i],em=int(i<93),f1=float(i<93),actual_calls=0,
        valid_action=1,turns=1,answer_sha256='1'*64,answer_present=True,response_token_count=2,model_token_count=2) for i in range(512)]
    run=dict(run_id='CPU',parent_production_run_id='parent',policy_calibration_slurm_job_id='1',
        policy_calibration_plan_sha256='2'*64,policy_calibration_role='policy_safety_calibration_discovery')
    validation=dict(validation_sha256='3'*64,question_ids=ids,data_sources=sources)
    measured=aggregate(rows,ids,sources)|dict(run_id='CPU',parent_production_run_id='parent',slurm_job_id='1',
        plan_sha256='2'*64,step=0,validation_sha256='3'*64,phase_role='policy_safety_calibration_discovery')
    return aggregate,run,validation,measured


class ExistingBefore512Tests(unittest.TestCase):
    def test_actual_original_failure_binding_and_embedded_reader_source_compile(self):
        reader,launch,terminal=h.input_contract(originals())
        self.assertEqual(terminal['slurm_terminal_state'],'FAILED');self.assertEqual(launch['job_id'],'3294')
        code=m.remote_code(h,reader);ns={};exec(code.split('p=json.load(sys.stdin)',1)[0],ns)
        self.assertNotIn('torch',ns);self.assertNotIn('AutoTokenizer',ns)
        self.assertNotIn('journal_diagnosis',ns);self.assertIn('before512_diagnosis',ns)
        self.assertEqual(ns['EMPTY_SOURCE_FILES'],h.EMPTY_SOURCE_FILES)

    def test_default_reject_before_any_rpc_or_input_inspection(self):
        with patch.object(h,'input_contract',side_effect=AssertionError('must not read inputs')):
            with patch.object(m,'helper',return_value=h):
                with self.assertRaisesRegex(ValueError,'Explicit future'):
                    m.execute_once({},HERE/'must-not-create.json',rpc=lambda *args:self.fail('must not RPC'))
        result=subprocess.run([sys.executable,'-B',str(HERE/'collect_policy_discovery_v5_before512.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('--terminal',result.stderr)

    def test_bound_reader_bytes_deterministic_without_hashseed_set_repr(self):
        reader,_,_=h.input_contract(originals())
        self.assertEqual(m.remote_code(h,reader),m.remote_code(h,reader))
        self.assertEqual(m._literal(frozenset(('b','a'))),"frozenset(('a', 'b'))")
        self.assertEqual(m._literal(set(('b','a'))),"frozenset(('a', 'b'))")

    def test_actual_pure512_aggregate_returns_only_scalar_hash_not_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate,run,validation,value=before_fixture(Path(directory).resolve())
            result=h.before512_projection(value,run,validation,aggregate)
            self.assertEqual(result['weighted_em'],93/512);self.assertEqual(result['question_count'],512)
            self.assertEqual(result['source_counts'],{'nq':256,'hotpotqa':256})
            self.assertEqual(result['ordered_question_ids_sha256'],h.sha(h.canonical(validation['question_ids'])))
            self.assertFalse(result['health_evaluated']);self.assertFalse(result['accepted'])
            for key in ('rows','question_ids','data_sources','prompt','raw_generated_token_ids'):self.assertNotIn(key,result)

    def test_ordered_identity_score_arithmetic_plan_job_and_step_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate,run,validation,value=before_fixture(Path(directory).resolve())
            for key,replacement in (('weighted_em',1),('step',1),('plan_sha256','4'*64),('slurm_job_id','2')):
                with self.assertRaises(ValueError):h.before512_projection(value|{key:replacement},run,validation,aggregate)
            for field,replacement in (('question_id','different'),('data_source','hotpotqa'),('em',True),('f1',float('nan'))):
                changed=copy.deepcopy(value);changed['rows'][0][field]=replacement
                with self.assertRaises(ValueError):h.before512_projection(changed,run,validation,aggregate)

    def test_absent_initial_file_is_missing_not_score_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result=h.before512_diagnosis(Path(directory).resolve(),{}, {})
            self.assertEqual(result,dict(exists=False,health_evaluated=False,accepted=False))
            self.assertNotIn('weighted_em',result)

    def test_native_call_order_source_and_manifest_before_existing512_boundary(self):
        # Native proof boundaries are CPU fixtures; no actual source/data/SSH evidence.
        reader,launch,terminal=h.input_contract(originals());ns={}
        exec(m.remote_code(h,reader).split('p=json.load(sys.stdin)',1)[0],ns)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'logs').mkdir()
            manifest=dict(run_id=h.RUN_ID,parent_production_run_id=h.PARENT_RUN_ID,policy_calibration_slurm_job_id='3294',
                source_commit=h.SOURCE_COMMIT,source_sha256=launch['source_sha256'],
                model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',completed_updates=0)
            path=base/'logs/run-manifest.json';path.write_bytes(h.canonical(manifest))
            events=[];ns['reader']=SimpleNamespace(current_completed_terminal=lambda _:events.append('terminal') or {'fixture':True})
            ns['authentic_sources']=lambda *args:events.append('source')
            ns['before512_diagnosis']=lambda *args:events.append('existing512') or dict(exists=False,accepted=False,health_evaluated=False)
            payload=dict(launch=launch|{'base':str(base)},terminal=terminal)
            original_path=list(sys.path)
            try:
                result=ns['native_before512'](payload,ns['reader'])
                self.assertEqual(events,['terminal','source','existing512'])
                self.assertFalse(result['new_evaluation_generated']);self.assertFalse(result['model_or_adam_tensors_read'])
                self.assertFalse(result['token_journal_read']);self.assertFalse(result['optimizer_writes_authenticated'])
                events.clear();path.write_bytes(h.canonical(manifest|{'completed_updates':1}))
                with self.assertRaisesRegex(ValueError,'completed count'):ns['native_before512'](payload,ns['reader'])
                self.assertEqual(events,['terminal','source'])
            finally:sys.path[:]=original_path

    def test_simulated_rpc_requires_false_admission_before_exclusive_local_scalar_write(self):
        reader,launch,terminal=h.input_contract(originals())
        # Only output persistence/identity fixture, not a native read or score.
        result=dict(role='failed3294_existing_before512_readonly_cpu',run_id=h.RUN_ID,slurm_job_id='3294',
            source_commit=h.SOURCE_COMMIT,original_failed_terminal_sha256=h.TERMINAL_SHA256)
        for name in ('accepted','context_verified','production_admission','healthy_checkpoint_accepted','physical_execution_authenticated',
            'health_evaluated','new_evaluation_generated','optimizer_writes_authenticated','model_or_adam_tensors_read',
            'token_journal_read','raw_text_or_ids_or_lp_values_published'):result[name]=False
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory).resolve()/'scalar.json'
            bad=result|{'new_evaluation_generated':True}
            with self.assertRaisesRegex(ValueError,'boundary differs'):
                m.execute_once(originals(),output,execute=True,rpc=lambda *args:bad)
            self.assertFalse(output.exists())
            receipt=m.execute_once(originals(),output,execute=True,rpc=lambda *args:result)
            self.assertEqual(receipt['sha256'],h.sha(output.read_bytes()));self.assertEqual(output.stat().st_mode&0o777,0o600)
            with self.assertRaisesRegex(ValueError,'exclusive'):
                m.execute_once(originals(),output,execute=True,rpc=lambda *args:self.fail('must not RPC twice'))

    def test_frozen_helper_unchanged_and_no_heavy_tokenizer_model_or_eval_creation(self):
        self.assertEqual(h.sha((HERE/'diagnose_policy_discovery_v5_special_tokens_v3.py').read_bytes()),m.HELPER_SHA256)
        reader,_,_=h.input_contract(originals());code=m.remote_code(h,reader)
        for banned in ('AutoTokenizer','raw_generated_token_ids','safe_open(', 'get_tensor(', 'torch.load(',
            "['sbatch'",'rows_from_batch(', 'generate_sequences(', 'artifacts/behavior-traces'):
            self.assertNotIn(banned,code)
        self.assertIn("columns=['prompt','data_source']",code)
        self.assertIn("hidden={'rows','question_ids','data_sources'}",code)


if __name__=='__main__':unittest.main()
