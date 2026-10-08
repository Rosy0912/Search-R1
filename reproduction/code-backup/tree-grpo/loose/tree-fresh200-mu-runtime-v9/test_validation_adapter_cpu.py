"""Actual frozen Adapter AST -> external observer integration, CPU tiny objects."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import validation_cost_hooks_instruct300 as hooks

SOURCE=Path(__file__).resolve().parents[1]/'tree-behavior-mu-ppo/search_r1/baselines/controller.py'
ROLLOUT_SOURCE=SOURCE.with_name('rollout.py')


class Scalar:
    def __init__(self,value):self.value=value
    def item(self):return self.value
    def __int__(self):return self.value


class Array:
    def __init__(self,rows,width):self.shape=(rows,width)
    def sum(self):return Scalar(self.shape[0]*self.shape[1])
    def __ne__(self,other):return self
    def long(self):return self
    def __getitem__(self,selection):return Array(1 if isinstance(selection,slice) and selection.stop==1 else self.shape[0],self.shape[1])
    def repeat(self,rows,columns):return Array(self.shape[0]*rows,self.shape[1]*columns)


class Proto:
    @staticmethod
    def from_dict(tensors,meta_info):return NS(batch=tensors,meta_info=meta_info)


class ValidationAdapterTests(unittest.TestCase):
    def test_zero_install_keeps_flags_prefix_and_passes_actual_controller(self):
        source=Path(hooks.__file__).read_text();module=ast.parse(source)
        install=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='install')
        text=ast.unparse(install)
        self.assertIn("os.environ.get('TREE_PPO_CALIBRATION_ONLY') == '1'",text)
        self.assertIn("os.environ.get('TREE_FRESH_INITIALIZATION_ONLY') == '1'",text)
        self.assertIn("'tree-fresh200-' + job",text)
        self.assertNotIn("tree-fresh200-train-",text)
        self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module=='search_r1.baselines'
                            and any(a.name=='controller' and a.asname=='controller_module' for a in n.names)
                            for n in install.body))
        call=next(n for n in ast.walk(install) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id=='wrap_trainer')
        self.assertEqual({k.arg:ast.unparse(k.value) for k in call.keywords},
                         {'final_expected_rows':'51713','controller_module':'controller_module'})

    def test_user_kl02_is_explicit_and_original_sampler_gate_is_unchanged(self):
        here=Path(hooks.__file__).parent
        configuration=json.loads((here/'CONFIGURATION_OVERRIDES.json').read_bytes())
        inputs=json.loads((here/'INPUTS.schema.json').read_bytes())
        for metadata in (configuration,inputs):
            self.assertEqual(metadata['trial_thresholds'],dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.))
            self.assertEqual(metadata['production_thresholds'],dict(target_old_policy_kl=.2,negative_tail_limit=None,initial_actor_repeat_max_abs_delta=None))
            provenance=metadata['threshold_provenance']['target_old_policy_kl']
            self.assertEqual(provenance['goal_updated_at'],1791282960)
            self.assertEqual(provenance['source'],'explicit_user_goal');self.assertIs(provenance['actual_gpu_accepted'],False)
        resolved=configuration['expected_resolved_values']
        self.assertEqual(resolved['actor_rollout_ref.actor.policy_safety.target_old_policy_kl'],.2)
        self.assertEqual(resolved['actor_rollout_ref.actor.guard_continue_protocol'],'tree_guard_continue_fresh_rollout_v1')
        self.assertEqual(resolved['actor_rollout_ref.rollout.sampling_consistency.max_abs_delta'],.01)
        self.assertIs(resolved['trainer.val_before_train'],True)
        protocol=inputs['protocol']
        self.assertNotIn('optimizer_calls_per_rank_per_complete_outer',protocol)
        self.assertEqual(protocol['optimizer_opportunities_per_rank_per_complete_outer'],20)
        self.assertEqual(protocol['minimum_successful_optimizer_writes_per_complete_outer'],1)
        self.assertIs(protocol['successful_optimizer_writes_counted_separately'],True)
        self.assertIs(protocol['zero_write_or_abandoned_batch_credits_outer'],False)
        self.assertIs(protocol['zero_write_or_abandoned_batch_advances_scheduler'],False)
        self.assertEqual(protocol['guard_continue_protocol'],'tree_guard_continue_fresh_rollout_v1')
        self.assertIs(inputs['expected_source_guard_continue_config']['actual_continue_gpu_accepted'],False)
        supervisor=(here/'supervisor.py').read_text()
        self.assertIn('safety.target_old_policy_kl==.2',supervisor)
        self.assertIn('dict(target_old_policy_kl=.2,negative_tail_limit=.01',supervisor)
        self.assertIn('sampling.max_abs_delta==.01',supervisor)

    def build(self,step,rows,final=False,fail=False):
        class WG:
            calls=0
            def generate_sequences(self,batch):
                self.calls+=1
                if fail:raise RuntimeError('actual native synthetic failure')
                return NS(batch={'responses':Array(batch.batch['input_ids'].shape[0],2)})
        wg=WG()
        class Manager:
            def __init__(self):self.actor_rollout_wg=wg;self.config=NS(num_gpus=8);self.tensor_fn=NS(create_position_ids=lambda value:value)
            def run_llm_loop(self,*args):raise AssertionError('Actual Adapter must not invoke old Manager loop')
            def _generate_with_gpu_padding(self,batch):return wg.generate_sequences(batch)
            def _batch_search(self,queries):return {'result':[[] for _ in queries]}
        manager=Manager()
        class Rollout:
            def __init__(self,value,**kwargs):
                assert value is manager and 'behavior_mu' not in kwargs
                self.manager=manager;self.config=manager.config;self.context_alignment='error';self.token_preservation={'enabled':True};self.tokenizer=NS(pad_token_id=0);self.cost={'context_mismatches':0,'prefill_tokens':0,'padded_rows':0}
            def run(self,batch):
                # Exact _advance AST prefix through nativecall. Parsing/reward suffix excluded.
                states=[NS(prompt=[1,2,3],response=[],rolling=[1,2,3]) for _ in range(batch.batch['input_ids'].shape[0])]
                with patch.dict('sys.modules',{'verl':NS(DataProto=Proto)}):out=self._advance(states,{'do_sample':False})
                manager._batch_search(['query'])
                return out,None
        actual_advance=next(n for n in ast.walk(ast.parse(ROLLOUT_SOURCE.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_advance')
        generation=next(i for i,n in enumerate(actual_advance.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='output' for t in n.targets))
        actual_advance.body=actual_advance.body[:generation+1]+[ast.Return(value=ast.Name(id='output',ctx=ast.Load()))]
        ast.fix_missing_locations(actual_advance)
        advance_scope={'padded':lambda rows,pad,left:Array(len(rows),max(map(len,rows))), 'torch':NS(cat=lambda values,dim:Array(sum(x.shape[0] for x in values),values[0].shape[1]))}
        exec(compile(ast.Module(body=[actual_advance],type_ignores=[]),str(ROLLOUT_SOURCE),'exec'),advance_scope);Rollout._advance=advance_scope['_advance']
        module=ast.parse(SOURCE.read_text());function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='token_preserving_validation_manager')
        scope={'SearchRollout':Rollout};exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),scope)
        controller_module=NS(token_preserving_validation_manager=scope['token_preserving_validation_manager'],SearchRollout=Rollout)
        original_factory=controller_module.token_preserving_validation_manager
        batch=NS(batch={'input_ids':Array(rows,3),'attention_mask':Array(rows,3)})
        controller=NS(config=NS(tree=NS(token_preservation={'enabled':True})))
        class Trainer:
            config=NS(do_search=True,baselines=NS(training_protocol='tree_fresh200_five_leaf_v1'))
            val_dataset=list(range(rows));global_steps=step;actual_outer_updates=200 if final else step
            tokenizer=NS(pad_token_id=0,eos_token_id=9);actor_rollout_wg=wg
            def _prepare_final_validation(self):return None
            def _validate(self):
                adapter=controller_module.token_preserving_validation_manager(manager,controller)
                adapter.run_llm_loop(batch,batch.batch['input_ids'])
                return {'val/evaluated_count/synthetic':rows,'val/qa_count/all':rows,'val/f1/all':.5,'val/em/all':.25}
        instance=Trainer()
        if final:instance._tree_cost_final_phase=True
        return Trainer,Manager,instance,controller_module,original_factory,wg

    def test_actual_adapter_counts_initial_acceptance_and_final_once(self):
        for step,rows,final,phase in ((0,512,False,'fresh_initial'),(5,512,False,'acceptance'),(201,51713,True,'final')):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'SLURM_JOB_ID':'99','SEARCH_R1_BASELINE_COMMIT':'source','SEARCH_R1_BASELINE_CONFIG_SHA256':'config','SEARCH_R1_BASELINE_DATA_IDENTITY':'data','SEARCH_R1_MODEL_IDENTITY':'model'}):
                trainer,manager,instance,module,original,wg=self.build(step,rows,final)
                hooks.wrap_trainer(trainer,manager,root,controller_module=module)
                result=instance._validate();report=json.loads(next(Path(root).glob('*.json')).read_text())
                self.assertEqual(report['counters']['real_sample_rows'],rows);self.assertEqual(report['evaluated_rows'],rows)
                self.assertEqual(report['metrics'],result);self.assertEqual(report['phase'],phase);self.assertEqual(wg.calls,1)
                self.assertEqual(report['counters']['active_row_events'],rows)
                self.assertEqual(report['counters']['submitted_padding_row_events'],(-rows)%8)
                self.assertEqual(report['counters']['submitted_rows'],rows+(-rows)%8)
                self.assertIs(module.token_preserving_validation_manager,original)

    def test_native_failure_persists_failed_report_and_restores_all_hooks(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,manager,instance,module,original,wg=self.build(0,512,fail=True)
            old_padding=manager._generate_with_gpu_padding;old_advance=module.SearchRollout._advance
            hooks.wrap_trainer(trainer,manager,root,controller_module=module)
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):instance._validate()
            report=json.loads(next(Path(root).glob('*.json')).read_text())
            self.assertEqual(report['status'],'VALIDATION_COST_FAILED');self.assertEqual(report['counters']['real_sample_rows'],512)
            self.assertEqual(report['counters']['generation_failed'],1);self.assertIsNone(report['evaluated_rows'])
            self.assertIs(module.token_preserving_validation_manager,original);self.assertIs(manager._generate_with_gpu_padding,old_padding)
            self.assertNotIn('generate_sequences',vars(wg));self.assertIs(module.SearchRollout._advance,old_advance)


if __name__=='__main__':unittest.main()
