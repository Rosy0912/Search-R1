"""Evaluation timing must not require the training-only outer annotation."""
import io,json,unittest
from contextlib import redirect_stdout,nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch
from search_r1.baselines.rollout import SearchRollout

class EvalRolloutTimingTests(unittest.TestCase):
    def engine(self):
        engine=SearchRollout.__new__(SearchRollout)
        engine.cost={'generated_tokens':0};engine.token_preservation={}
        engine.tokenizer=object();engine.tokenizer_identity={}
        return engine

    def test_eval_without_outer_keeps_return_and_records_null(self):
        engine=self.engine();result=object();engine._advance=lambda states,meta:result
        out=io.StringIO()
        with patch('search_r1.baselines.tree_raw_tokens.tokenizer_validation_scope',return_value=nullcontext()), redirect_stdout(out):
            self.assertIs(engine.advance([],{}),result)
        event=json.loads(out.getvalue().split('TREE_ACTION_TIMING=')[1])
        self.assertIsNone(event['outer']);self.assertTrue(event['successful'])

    def test_eval_timing_does_not_mask_original_exception(self):
        engine=self.engine()
        def fail(states,meta):raise ValueError('original evaluator failure')
        engine._advance=fail;out=io.StringIO()
        with patch.dict('os.environ',{'SEARCH_R1_BASELINE_OUTPUT':''}), patch('search_r1.baselines.tree_raw_tokens.tokenizer_validation_scope',return_value=nullcontext()), redirect_stdout(out):
            with self.assertRaisesRegex(ValueError,'original evaluator failure'):engine.advance([],{})
        self.assertFalse(json.loads(out.getvalue().split('TREE_ACTION_TIMING=')[1])['successful'])

if __name__=='__main__':unittest.main()
