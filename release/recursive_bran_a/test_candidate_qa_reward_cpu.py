"""Focused CPU contract tests; real saved-row equality is an independent audit."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from search_r1.baselines.candidate_qa_reward import _decoded_model_spans, candidate_qa_rewards

class CandidateQARewardTests(unittest.TestCase):
    def test_contiguous_semantic_spans(self):
        tok=NS(decode=lambda ids: ''.join(map(chr,ids)))
        self.assertEqual(_decoded_model_spans(list(map(ord,'aXbcYd')),[1,0,1,1,0,1],tok),['a','bc','d'])
        self.assertEqual(_decoded_model_spans(list(map(ord,'<answer>x</answer>')),[1]*18,tok),['<answer>x</answer>'])
        self.assertEqual(_decoded_model_spans([],[],tok),[])

    def test_unknown_reward_manager_refused(self):
        engine=NS(tokenizer=object())
        with self.assertRaises(Exception):
            candidate_qa_rewards(engine,[],[],NS(tokenizer=engine.tokenizer))

    def test_unknown_scorer_refused_before_ledger(self):
        from verl.trainer.main_ppo import RewardManager
        tok=object();engine=NS(tokenizer=tok,tokenizer_identity={},token_preservation={},config=NS(max_prompt_length=4096))
        batch=type('Batch',(),{'__len__':lambda self:1,'non_tensor_batch':{'data_source':['unknown']}})()
        # Only the tokenizer scope is mocked; the production scorer selector must reject.
        from contextlib import nullcontext
        with patch('search_r1.baselines.candidate_qa_reward.tokenizer_validation_scope',return_value=nullcontext()):
            with self.assertRaises(NotImplementedError):
                candidate_qa_rewards(engine,batch,[NS(source=0)],RewardManager(tok,0))

if __name__=='__main__':unittest.main()
