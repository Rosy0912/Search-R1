"""CPU journal invariants only: no sampler/HF/CUDA probability admission."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace as NS

spec=importlib.util.spec_from_file_location('sampling_trace',Path(__file__).resolve().parents[1]/'search_r1/llm_agent/sampling_trace.py')
trace=importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)


def completion(request,tokens,probabilities=None):
    lp=[{t:NS(logprob=p)} for t,p in zip(tokens,probabilities or [-.1]*len(tokens))]
    return NS(prompt_token_ids=request,request_id='real-request',outputs=[NS(token_ids=tokens,logprobs=lp,finish_reason='stop',stop_reason=0)])


def turn(request,tokens,env,ordinal=0):
    record=trace.capture_requests([completion(request,tokens)], [request], ['row'], 'snapshot','call',ordinal,{},0,0)[0]
    record.update(kept_token_count=len(tokens),environment_token_ids=env)
    record['turn_sha256']=trace.sha(record)
    return record


class SamplingTraceTests(unittest.TestCase):
    def test_raw_eos_equal_pad_is_real_selected_token(self):
        result=trace.capture_requests([completion([10],[2,0])],[[10]],['row'],'snapshot','call',0,{},0,0)[0]
        self.assertEqual(result['raw_generated_token_ids'],[2,0])
        self.assertEqual(len(result['selected_logprobs']),2)
        trace.audit_row([10],[2,0],[1,1],[-.1,-.1],[turn([10],[2,0],[])],'row','snapshot')

    def test_embedded_special_preserved(self):
        tokens=[1,0,3]
        texts={1:'<answer>',0:'<special>',3:'x</answer>'}
        n,text=trace.prefix(tokens,lambda ids: ''.join(texts[i] for i in ids))
        self.assertEqual(n,3); self.assertIn('<special>',text)

    def test_closed_tag_token_straddle_rejects(self):
        with self.assertRaisesRegex(ValueError,'straddles'):
            trace.prefix([1],lambda ids: '<answer>x</answer>extra' if ids else '')

    def test_closed_tag_boundary_keeps_whitespace(self):
        self.assertEqual(trace.prefix([1,2],lambda ids: '<answer>x</answer> ' if len(ids)==1 else '<answer>x</answer> trailing')[0],1)

    def test_empty_raw_and_missing_selected_rejected(self):
        for raw in ([],[2]):
            value=completion([10],raw); value.outputs[0].logprobs=[]
            with self.assertRaises(ValueError):
                trace.capture_requests([value],[[10]],['row'],'snapshot','call',0,{},0,0)

    def test_actual_prompt_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'context'):
            trace.capture_requests([completion([11],[2])],[[10]],['row'],'snapshot','call',0,{},0,0)

    def test_model_retrieval_model_exact_reconstruction(self):
        turns=[turn([10],[2],[7,8]),turn([10,2,7,8],[3,0],[],1)]
        result=trace.audit_row([10],[2,7,8,3,0],[1,0,0,1,1],[-.1,0,0,-.1,-.1],turns,'row','snapshot')
        self.assertEqual(result['turns'],2)
        # A cropped second sampler context must fail even with valid native hashes.
        turns[1]=turn([7,8],[3,0],[],1)
        with self.assertRaisesRegex(ValueError,'context'):
            trace.audit_row([10],[2,7,8,3,0],[1,0,0,1,1],[-.1,0,0,-.1,-.1],turns,'row','snapshot')

    def test_pad_token_inside_model_span_cannot_be_filtered(self):
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            trace.audit_row([10],[2],[1],[-.1],[turn([10],[2,0],[])],'row','snapshot')

    def test_journal_mutation_and_prob_transport_rejected(self):
        record=turn([10],[2],[])
        with self.assertRaisesRegex(ValueError,'probability transport'):
            trace.audit_row([10],[2],[1],[-.2],[record],'row','snapshot')
        record['raw_generated_token_ids']=[3]
        with self.assertRaisesRegex(ValueError,'SHA'):
            trace.audit_row([10],[3],[1],[-.1],[record],'row','snapshot')

if __name__=='__main__': unittest.main()
