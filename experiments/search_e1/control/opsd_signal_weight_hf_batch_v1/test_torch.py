"""Run in the original Node environment, CVD empty; no model/GPU/HTTP."""
import unittest,torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from hf_backend import PaddingAwareRepetition,BatchStopTags

class ByteLevel:pass
class Tok:
    backend_tokenizer=type('Backend',(),{'decoder':ByteLevel()})()
    def decode(self,tokens,**kwargs):return ''.join(chr(int(v)) for v in tokens if int(v))

class Tests(unittest.TestCase):
    def test_padding_penalty_matches_true_single_prompts_including_real_eos(self):
        ids=torch.tensor([[0,0,10,11],[10,12,0,11]])
        logits=torch.linspace(-2,2,20).repeat(2,1)
        processor=RepetitionPenaltyLogitsProcessor(1.05)
        actual=PaddingAwareRepetition(processor,4,[2,0])(ids,logits.clone())
        expected=torch.cat([processor(ids[i:i+1,n:],logits[i:i+1].clone()) for i,n in enumerate([2,0])])
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(float(actual[0,0]),float(logits[0,0]))
        self.assertNotEqual(float(actual[1,0]),float(logits[1,0]))
        self.assertTrue(torch.equal(ids,torch.tensor([[0,0,10,11],[10,12,0,11]])))
    def test_per_row_stop_preserves_first_stop_length_and_other_live_rows(self):
        stops=BatchStopTags(Tok(),2,2)
        first=torch.tensor([[1,1]+list(map(ord,'</answer>')),[1,1]+list(map(ord,'thinking!'))])
        ended=stops(first,None);self.assertEqual(ended.tolist(),[True,False]);self.assertEqual(stops.ends,[9,None])
        second=torch.tensor([[1,1]+list(map(ord,'</answer>'))+[0]*9,[1,1]+list(map(ord,'thinking!</search>'))])
        ended=stops(second,None);self.assertEqual(ended.tolist(),[True,True]);self.assertEqual(stops.ends,[9,18])
    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
