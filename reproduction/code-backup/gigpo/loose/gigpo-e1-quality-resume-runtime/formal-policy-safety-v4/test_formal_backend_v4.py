"""CPU metadata/dispatch contracts only; no Ray cluster, GPU, sampler or Adam.

Read-only engine attributes and caller spies are explicit fixtures. The complete
backend validators, production binding bridge and startup ordering execute.
"""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import formal_backend_v1 as backend
from formal_protection_contract_v1 import (validate_same_calibration_flash_backend,
    validate_hook_startup_review,MANIFEST,DATA,REVISION)
from formal_probability_proofs_v1 import file_record

SOURCE='52befe968c856162d1bc79cd0171bb38309f4f9e'
BINDING='b'*64


def observations():
    return [dict(scope='ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY',rank=rank,pid=1000+rank,
        source_commit=SOURCE,binding_sha256=BINDING,requested_backend='FLASH_ATTN',
        runner_backend_class='vllm.attention.backends.flash_attn.FlashAttentionBackend',runner_backend_name='flash-attn',
        model_dtype='torch.bfloat16',cache_dtype='auto',block_size=16,model_sliding_window=None,
        attention_layers=[dict(name=f'model.layers.{n}.self_attn.attn',
            implementation_class='vllm.attention.backends.flash_attn.FlashAttentionImpl',head_size=128,
            num_heads=16,num_kv_heads=2,sliding_window=[-1,-1]) for n in range(36)],
        observation_forward_calls=0,observation_tensor_value_reads=0,
        actual_vllm_runner_positions_observed=False,probability_accepted=False) for rank in range(8)]


class ActualBackendMetadataCPU(unittest.TestCase):
    def invoke(self,rows):return backend.validate_actual_flash_backend(rows,dict(code_commit=SOURCE),BINDING)
    def test_full8_by36_declarations_only_do_not_accept_probability(self):
        value=self.invoke(observations());self.assertEqual(value['rank_count'],8);self.assertFalse(value['probability_accepted'])
    def test_fallback_on_rank7_refuses(self):
        r=observations();r[7]['runner_backend_name']='xformers'
        with self.assertRaises(ValueError):self.invoke(r)
    def test_impl_fallback_on_last_layer_refuses(self):
        r=observations();r[7]['attention_layers'][-1]['implementation_class']='XFormersImpl'
        with self.assertRaises(ValueError):self.invoke(r)
    def test_missing_layer_refuses(self):
        r=observations();r[7]['attention_layers'].pop()
        with self.assertRaises(ValueError):self.invoke(r)
    def test_duplicate_layer_name_refuses(self):
        r=observations();r[7]['attention_layers'][-1]['name']=r[7]['attention_layers'][0]['name']
        with self.assertRaises(ValueError):self.invoke(r)
    def test_source_binding_pid_rank_missing_or_relabel_refuses(self):
        for field,value in (('source_commit','other'),('binding_sha256','a'*64),('pid',1000),('rank',0)):
            with self.subTest(field=field):
                r=observations();r[7][field]=value
                with self.assertRaises(ValueError):self.invoke(r)
    def test_wrong_attention_shape_or_position_branch_refuses(self):
        for field,value in (('head_size',64),('num_heads',8),('num_kv_heads',4),('sliding_window',[100,100])):
            with self.subTest(field=field):
                r=observations();r[7]['attention_layers'][-1][field]=value
                with self.assertRaises(ValueError):self.invoke(r)
    def test_unknown_dtype_cache_or_block_refuses(self):
        for field,value in (('model_dtype','torch.float16'),('cache_dtype','fp8'),('block_size',32),('model_sliding_window',1000)):
            with self.subTest(field=field):
                r=observations();r[7][field]=value
                with self.assertRaises(ValueError):self.invoke(r)
    def test_observed_backend_does_not_invent_tensor_forward_or_probability_scope(self):
        for field,value in (('observation_forward_calls',1),('observation_tensor_value_reads',1),
            ('actual_vllm_runner_positions_observed',True),('probability_accepted',True)):
            with self.subTest(field=field):
                r=observations();r[7][field]=value
                with self.assertRaises(ValueError):self.invoke(r)
    def test_environment_explicit_missing_or_old_XFORMERS_refuses(self):
        for value in (None,'XFORMERS'):
            env={} if value is None else dict(VLLM_ATTENTION_BACKEND=value)
            with patch.dict(os.environ,env,clear=True),self.assertRaises(ValueError):backend.require_flash_environment()


class ReadOnlyActualObserverBodyCPU(unittest.TestCase):
    """Execute same exact observer attribute path with explicit CPU engine spies."""
    def engine(self):
        impl_cls=type('FlashAttentionImpl',(),{'__module__':'vllm.attention.backends.flash_attn'})
        attention_cls=type('Attention',(),{'__module__':'vllm.attention.layer'})
        modules=[]
        for n in range(36):
            impl=impl_cls();impl.head_size=128;impl.num_heads=16;impl.num_kv_heads=2;impl.sliding_window=(-1,-1)
            module=attention_cls();module.impl=impl;modules.append((f'layers.{n}.attn',module))
        cls=type('FlashAttentionBackend',(),{'__module__':'vllm.attention.backends.flash_attn','get_name':staticmethod(lambda:'flash-attn')})
        runner=types.SimpleNamespace(attn_backend=cls,model=types.SimpleNamespace(named_modules=lambda:iter(modules)),
            model_config=types.SimpleNamespace(dtype='torch.bfloat16',get_sliding_window=lambda:None),
            cache_config=types.SimpleNamespace(cache_dtype='auto',block_size=16))
        engine=types.SimpleNamespace(llm_engine=types.SimpleNamespace(model_executor=types.SimpleNamespace(worker=types.SimpleNamespace(model_runner=runner))),
            generate=lambda *a,**k:self.fail('metadata must not generate/forward'))
        return engine,modules,runner
    def test_exact_observer_reads36_impl_attributes_no_generate_or_tensor_method(self):
        engine,modules,runner=self.engine()
        before=[vars(module.impl).copy() for _,module in modules]
        with patch.dict(os.environ,dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN')):
            row=backend.observe_backend_identity(engine,7,SOURCE,BINDING)
        self.assertEqual(row['rank'],7);self.assertEqual(len(row['attention_layers']),36)
        self.assertEqual([vars(module.impl) for _,module in modules],before)
        self.assertEqual(row['binding_sha256'],BINDING);self.assertEqual(row['observation_forward_calls'],0)
        self.assertFalse(row['actual_vllm_runner_positions_observed'])
    def test_missing_actual_private_model_runner_fails_before_any_forward(self):
        engine,modules,runner=self.engine();del engine.llm_engine.model_executor.worker.model_runner
        with self.assertRaises(AttributeError):backend.observe_backend_identity(engine,0,SOURCE,BINDING)
    def test_unknown_impl_metadata_is_preserved_then_validator_rejects(self):
        engine,modules,runner=self.engine();modules[-1][1].impl.head_size=64
        with patch.dict(os.environ,dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN')):
            row=backend.observe_backend_identity(engine,7,SOURCE,BINDING)
        rows=observations();row['pid']=1007;rows[7]=row
        with self.assertRaises(ValueError):backend.validate_actual_flash_backend(rows,dict(code_commit=SOURCE),BINDING)


class StartupAndPhysicalCPU(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.run=Path(self.temp.name)
        self.calls=[];self.rows=observations();self.binding=dict(code_commit=SOURCE)
        def inspect():self.calls.append('existing_metadata_rpc_fixture');return self.rows
        self.trainer=types.SimpleNamespace(actor_rollout_wg=types.SimpleNamespace(production_backend_identity=inspect))
        self.environment=patch.dict(os.environ,dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN'));self.environment.start();self.addCleanup(self.environment.stop)
    def invoke(self):return backend.observe_before_original_fit(self.trainer,self.binding,self.run,BINDING)
    def test_valid_metadata_saved_before_original_fit_no_optimizer(self):
        self.invoke();self.assertEqual(self.calls,['existing_metadata_rpc_fixture'])
        self.assertEqual(self.trainer.production_backend_record,file_record(self.run/'actual-vllm-backend-identity.json'))
        workers={r:dict(pid=1000+r) for r in range(8)}
        result=backend.audit_backend_startup(self.run,self.binding,workers,BINDING)
        self.assertFalse(result['probability_accepted'])
    def test_fallback_preserves_all_raw_rows_then_refuses_before_fit(self):
        self.rows[7]['runner_backend_name']='xformers'
        with self.assertRaises(ValueError):self.invoke()
        raw=json.loads((self.run/'actual-vllm-backend-identity.json').read_text())
        self.assertEqual(len(raw['actual_observations']),8)
        self.assertEqual(raw['status'],'OBSERVED_UNREVIEWED')
        self.assertFalse(hasattr(self.trainer,'production_backend_record'))
    def test_startup_material_existing_never_reused_or_overwritten(self):
        p=self.run/'actual-vllm-backend-identity.json';p.write_bytes(b'original')
        with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(p.read_bytes(),b'original');self.assertEqual(self.calls,[])
    def test_wrong_worker_PID_or_original_refuses_final_physical_scope(self):
        self.invoke();workers={r:dict(pid=1000+r) for r in range(8)};workers[7]['pid']=999
        with self.assertRaises(ValueError):backend.audit_backend_startup(self.run,self.binding,workers,BINDING)
    def test_missing_FLASH_before_metadata_RPC_refuses(self):
        with patch.dict(os.environ,dict(VLLM_ATTENTION_BACKEND='XFORMERS')):
            with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(self.calls,[])


class SameCalibrationBackendBridgeCPU(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.identity=dict(source_commit=SOURCE,common_manifest_sha256=DATA,restore_manifest_sha256=MANIFEST,
            official_model_revision=REVISION,node='h100-8',binding_sha256=BINDING)
        raw=self.root/'actual_backend.json';raw.write_text(json.dumps(backend.validate_actual_flash_backend(observations(),dict(code_commit=SOURCE),BINDING)))
        self.proof=dict(status='PASS',scope='ACTUAL_CALIBRATION_EIGHT_RANK_FLASH_BACKEND_METADATA',source_commit=SOURCE,
            actual_identity=self.identity,calibration_runtime_manifest_sha256=backend.CALIBRATION_RUNTIME_SHA,
            all8_runner_and36_attention_layers=True,actual_backend_record=file_record(raw))
        self.record=self.root/'independent_backend.json'
        self.binding=dict(code_commit=SOURCE,rollout_backend=dict(requested='FLASH_ATTN',
            calibration_runtime_manifest_sha256=backend.CALIBRATION_RUNTIME_SHA,
            gate_before_original_fit_restore_baseline_adam=True))
        self.artifact=dict(actual_identity=self.identity)
    def invoke(self):
        self.record.write_text(json.dumps(self.proof));self.binding['rollout_backend']['calibration_backend_acceptance']=file_record(self.record)
        return validate_same_calibration_flash_backend(self.binding,self.artifact,SOURCE)
    def test_same_actual_identity_fixture_checks_schema_only(self):self.assertEqual(self.invoke()['status'],'PASS')
    def test_CPU_manifest_is_not_actual_same_run_acceptance(self):
        self.proof['scope']='LOCAL_CPU_ONLY'
        with self.assertRaises(ValueError):self.invoke()
    def test_new_production_cannot_reuse_old_backend_even_with_same_source(self):
        self.binding['rollout_backend']['requested']='XFORMERS'
        with self.assertRaises(ValueError):self.invoke()
    def test_different_calibration_binding_refuses(self):
        self.proof['actual_identity']=dict(self.identity,binding_sha256='a'*64)
        with self.assertRaises(ValueError):self.invoke()
    def test_unverified_rank_layer_claim_refuses(self):
        self.proof['all8_runner_and36_attention_layers']=False
        with self.assertRaises(ValueError):self.invoke()
    def test_original_backend_json_tampering_refuses(self):
        Path(self.proof['actual_backend_record']['path']).write_text('{}')
        with self.assertRaises(ValueError):self.invoke()


class StartupHookReviewScopeCPU(unittest.TestCase):
    def good(self):
        hook=dict(helper_manifest_sha256='h'*64,actual_formal_hook_acceptance_stage='SAME_PROCESS26_TO30_BEFORE31')
        artifact=dict(actual_proofs=dict(calibration=dict(sha256='c'*64)));native=dict(review_sha256='n'*64)
        review=dict(status='PASS',source_commit=SOURCE,scope='GIGPO_EXTERNAL_FORMAL_HOOK_STATIC_AND_PHYSICAL_READY',
            helper_manifest_sha256='h'*64,actual_native_acceptance_sha256='n'*64,actual_calibration_acceptance_sha256='c'*64,
            metadata_only_hooks_and_first_adam_collective_static_verified=True,expected_rollout_backend='FLASH_ATTN',
            extra_diagnostic_adam_required=False,actual_formal_GPU_hook_accepted=False)
        return hook,review,artifact,native
    def test_static_physical_scope_does_not_claim_actual_formal_GPU_or_extra_AB(self):
        h,r,a,n=self.good();validate_hook_startup_review(h,r,SOURCE,a,n)
    def test_missing_native_or_actual_calibration_refuses(self):
        for field in ('actual_native_acceptance_sha256','actual_calibration_acceptance_sha256'):
            h,r,a,n=self.good();r[field]='f'*64
            with self.assertRaises(ValueError):validate_hook_startup_review(h,r,SOURCE,a,n)
    def test_extra_AB_or_fabricated_actual_hook_PASS_refuses(self):
        for field in ('extra_diagnostic_adam_required','actual_formal_GPU_hook_accepted'):
            h,r,a,n=self.good();r[field]=True
            with self.assertRaises(ValueError):validate_hook_startup_review(h,r,SOURCE,a,n)
    def test_actual_fit_gate30_cannot_be_omitted_or_moved(self):
        h,r,a,n=self.good();h['actual_formal_hook_acceptance_stage']='OPTIONAL_AFTER300'
        with self.assertRaises(ValueError):validate_hook_startup_review(h,r,SOURCE,a,n)


class ExplicitRayPropagationCPU(unittest.TestCase):
    def class_body(self):
        text=(Path(__file__).with_name('formal_training_entry_v1.py')).read_text();tree=ast.parse(text)
        node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BoundTask')
        namespace=dict(require=backend.require)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-v4-BoundTask>','exec'),namespace)
        return namespace['BoundTask'],tree
    def test_actual_BoundTask_merges_FLASH_and_other_source_audit_env(self):
        cls,_=self.class_body();calls=[]
        task=types.SimpleNamespace(options=lambda **kwargs:calls.append(kwargs) or task,remote=lambda *a,**k:'unchangedCPUresult')
        bound=cls(task,dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN',SOURCE='52'))
        self.assertEqual(bound.options(runtime_env=dict(env_vars=dict(RANK='7'))).remote(),'unchangedCPUresult')
        self.assertEqual(calls[0]['runtime_env']['env_vars'],dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN',SOURCE='52',RANK='7'))
    def test_actual_BoundTask_rejects_conflicting_XFORMERS(self):
        cls,_=self.class_body();task=types.SimpleNamespace(options=lambda **k:self.fail('no actual task may start'))
        with self.assertRaises(ValueError):cls(task,dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN')).options(runtime_env=dict(env_vars=dict(VLLM_ATTENTION_BACKEND='XFORMERS')))
    def test_actual_main_task_backend_gate_precedes_fit_and_restore(self):
        _,tree=self.class_body();main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main_task')
        calls=[x for x in ast.walk(main) if isinstance(x,ast.Call)]
        backend_call=next(x for x in calls if isinstance(x.func,ast.Name) and x.func.id=='observe_before_original_fit')
        fit=next(x for x in calls if isinstance(x.func,ast.Attribute) and x.func.attr=='fit')
        init=next(x for x in calls if isinstance(x.func,ast.Attribute) and x.func.attr=='init_workers')
        self.assertLess(init.lineno,backend_call.lineno);self.assertLess(backend_call.lineno,fit.lineno)
        self.assertEqual(sum(isinstance(x.func,ast.Attribute) and x.func.attr=='fit' for x in calls),1)
    def test_outer_wrapper_literal_and_Ray_environment_both_FLASH(self):
        wrapper=Path(__file__).with_name('formal_policy_safety_v1.py').read_text();entry=Path(__file__).with_name('formal_training_entry_v1.py').read_text()
        self.assertIn("VLLM_ATTENTION_BACKEND='FLASH_ATTN'",wrapper);self.assertNotIn("VLLM_ATTENTION_BACKEND='XFORMERS'",wrapper)
        self.assertIn("'VLLM_ATTENTION_BACKEND')}",entry)
        self.assertIn("ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':dict(environment,",entry)
        self.assertIn("TOKENIZERS_PARALLELISM='true',NCCL_DEBUG='WARN'",entry)


if __name__=='__main__':unittest.main(verbosity=2)
