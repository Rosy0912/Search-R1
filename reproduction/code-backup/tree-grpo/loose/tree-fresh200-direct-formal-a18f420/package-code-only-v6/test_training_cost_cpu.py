"""Actual owner Controller.generate AST and passive cost observer; no model/PT."""
import ast,copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from training_cost_hooks import wrap_controller
from train_monitor import training_cost_reports
from test_runtime_contract_cpu import direct,allskip,package,identity
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'tree-behavior-mu-ppo/search_r1/baselines/controller.py'


def actual_controller(fail=False):
    tree=ast.parse(SOURCE.read_text());node=next(x for x in ast.walk(tree) if isinstance(x,ast.ClassDef) and x.name=='Controller')
    method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='generate')
    code=ast.Module(body=[copy.deepcopy(method)],type_ignores=[]);namespace={'os':os,'option':lambda obj,key,default=None:obj.get(key,default)}
    class SearchRollout:
        def __init__(self,*args,**kwargs):self.audit=None;self.cost={'generated_tokens':10,'prefill_tokens':20,'retrieval_calls':2,'trajectories':2560,'padded_rows':0,'raw_emitted_tokens':11}
        def run(self,batch,tree):
            self.calls=getattr(self,'calls',0)+1
            if fail:raise RuntimeError('original native failure')
            return batch,object()
    namespace['SearchRollout']=SearchRollout;exec(compile(ast.fix_missing_locations(code),str(SOURCE),'exec'),namespace)
    class Controller:pass
    Controller.generate=namespace['generate']
    return Controller


class CostTests(unittest.TestCase):
    def test_actual_controller_preserves_return_call_and_keeps_credit_independent(self):
        with tempfile.TemporaryDirectory() as root:
            C=actual_controller();wrap_controller(C,root);obj=C();obj.method='tree_grpo';obj.tree_rng=None;obj.ranker=None
            obj.config=NS(tree={},gigpo=NS(anchor_mode='x'),seed=42,context_alignment='error')
            obj.trainer=NS(training_attempts=2,global_steps=1,actual_outer_updates=0)
            batch=NS(batch=range(512));metrics={}
            with patch.dict(os.environ,{'SEARCH_R1_TREE_AUDIT_DIR':''}):result=obj.generate(None,batch,batch,metrics)
            self.assertIs(result[0],batch);value=json.loads((Path(root)/'attempt-000002.json').read_text())
            self.assertEqual(value['costs']['raw_emitted_tokens'],11);self.assertEqual(value['question_rows'],512)
            self.assertIsNone(value['actual_optimizer_writes']);self.assertTrue(value['outer_credit_not_inferred'])
            self.assertEqual(value['completed_outer_before'],0)

    def test_real_attempt_join_separates_credited_and_all_skipped_rollout_cost(self):
        with tempfile.TemporaryDirectory() as root:
            run=Path(root)/'tree-fresh200-train-99';direct(run,1,writes=3);allskip(run,2,1);direct(run,2,writes=20,attempt=3)
            ids=identity(run);env={'SEARCH_R1_TREE_AUDIT_DIR':'','SLURM_JOB_ID':ids['job_id'],
                'SEARCH_R1_BASELINE_RUN_ID':ids['run_id'],'SEARCH_R1_BASELINE_COMMIT':ids['code_commit'],
                'SEARCH_R1_BASELINE_CONFIG_SHA256':ids['config_sha256'],'SEARCH_R1_BASELINE_DATA_IDENTITY':ids['data_identity'],
                'SEARCH_R1_MODEL_IDENTITY':ids['model_identity']}
            for attempt,candidate,completed in ((1,1,0),(2,2,1),(3,2,1)):
                C=actual_controller();wrap_controller(C,run/'output/training-rollout-cost');obj=C();obj.method='tree_grpo';obj.tree_rng=None;obj.ranker=None
                obj.config=NS(tree={},gigpo=NS(anchor_mode='x'),seed=42,context_alignment='error')
                obj.trainer=NS(training_attempts=attempt,global_steps=candidate,actual_outer_updates=completed);batch=NS(batch=range(512))
                with patch.dict(os.environ,env):obj.generate(None,batch,batch,{})
            value=training_cost_reports(run,package(),'config',2)
            self.assertEqual(value['training_attempts'],3);self.assertEqual(value['costs_by_persisted_credit']['credited']['raw_emitted_tokens'],22)
            self.assertEqual(value['costs_by_persisted_credit']['all_skipped']['raw_emitted_tokens'],11)
            self.assertEqual(len(value['record_sha256']),3)

    def test_original_generation_failure_keeps_unknown_counts_and_original_exception(self):
        with tempfile.TemporaryDirectory() as root:
            C=actual_controller(True);wrap_controller(C,root);obj=C();obj.method='tree_grpo';obj.tree_rng=None;obj.ranker=None
            obj.config=NS(tree={},gigpo=NS(anchor_mode='x'),seed=42,context_alignment='error')
            obj.trainer=NS(training_attempts=1,global_steps=1,actual_outer_updates=0);batch=NS(batch=range(512))
            with patch.dict(os.environ,{'SEARCH_R1_TREE_AUDIT_DIR':''}),self.assertRaisesRegex(RuntimeError,'original native failure'):obj.generate(None,batch,batch,{})
            value=json.loads((Path(root)/'attempt-000001.json').read_text());self.assertEqual(value['status'],'TRAINING_ROLLOUT_COST_FAILED')
            self.assertIsNone(value['costs']);self.assertIsNone(value['actual_optimizer_writes'])

if __name__=='__main__':unittest.main()
