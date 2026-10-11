"""Real tiny HF ByteLevel CPU tests; no models, downloads or GPU calls."""
import copy
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from test_tree_native_prefix_fastpath_cpu import NativePrefixTests
from test_tree_raw_tokens_cpu import ENV,params
import search_r1.baselines.tree_raw_tokens as raw


class LinearPrefixTests(unittest.TestCase):
    setUp=NativePrefixTests.setUp
    tearDown=NativePrefixTests.tearDown

    def make_action(self,ids):
        settings=params()
        # Tiny native stopping facade, no vLLM import/model.
        class NativeSettings(NS):
            def clone(self):return copy.deepcopy(self)
            def update_from_generation_config(self,config,eos):
                self.all_stop_token_ids={eos,*self.stop_token_ids}
        settings=NativeSettings(**vars(settings))
        output=NS(token_ids=ids,logprobs=[{t:NS(logprob=-.25)} for t in ids],
            finish_reason='stop' if ids[-1]==1 else 'length',stop_reason=None)
        request=NS(request_id='linear-fixture',finished=True,outputs=[output],prompt_token_ids=[ord('Q')+2])
        stopping=raw.capture_native_stopping(settings,1,{})
        with patch.dict(os.environ,ENV):
            emission=raw.capture_native_emissions([request],settings,'linear',0,native_stopping=stopping)[0]
        full=self.tok.decode(ids,skip_special_tokens=True);text=raw.canonical_text(full)
        context=raw._TokenizerValidationContext(self.tok,self.identity)
        native=context.native_prefix_decoder
        with patch.object(context,'native_prefix_decoder',wraps=native) as calls:
            with patch.dict(os.environ,{'TREE_LINEAR_PREFIX_SCAN':'1'}):
                action,ledger=raw.sampled_action(self.tok,emission,ids+[0]*(500-len(ids)),
                    self.tok.encode(text,add_special_tokens=False),text,self.identity,
                    response_mask=[1]*len(ids)+[0]*(500-len(ids)),_context=context)
            count=calls.call_count
        return action,ledger,count

    def test_split_utf8_special_and_tag_exact_boundaries_with_fewer_native_calls(self):
        for text in [b'X\xe4\xb8\xad<answer>Paris</answer>tail',b'\xff\xe2\x82\xac',b'<search>capital</search>']:
            ids=[x+2 for x in text]+[1]
            ids.insert(1,0)  # genuine emitted special PAD is not tensor padding
            action,ledger,count=self.make_action(ids)
            self.assertEqual(count,len(set((len(ids),ledger['cut']))))
            self.assertLess(count,len(ids))
            proof=ledger['utf8_boundary_proof']
            self.assertNotIn('native_prefix_agreement_count',proof)
            self.assertEqual(proof['native_decode_call_count'],count)
            context=raw._TokenizerValidationContext(self.tok,self.identity)
            # Revalidation works even when opt-in is no longer set.
            with patch.dict(os.environ,{'TREE_LINEAR_PREFIX_SCAN':'0'}):
                raw.validate_action_utf8_proof(ledger,self.tok,context)
            self.assertEqual(action,ids[:ledger['cut']])
            mutated=copy.deepcopy(ledger);mutated['utf8_boundary_proof']['native_decode_call_count']=99
            mutated['action_binding_sha256']=raw.fingerprint({k:v for k,v in mutated.items() if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})
            with self.assertRaises(ValueError):
                raw.validate_action_utf8_proof(mutated,self.tok,raw._TokenizerValidationContext(self.tok,self.identity))

    def test_default_still_checks_every_native_prefix_and_native_disagreement_rejects(self):
        ids=[x+2 for x in b'prefix\xe4\xb8\xad']+[1]
        context=raw._TokenizerValidationContext(self.tok,self.identity)
        native=context.native_prefix_decoder
        with patch.object(context,'native_prefix_decoder',wraps=native) as calls:
            raw._scan_utf8_prefixes(self.tok,ids,self.identity,_context=context)
            self.assertEqual(calls.call_count,len(ids))
        decoded,complete,proof=raw._scan_utf8_prefixes(self.tok,ids,self.identity,False,context)
        with patch.object(context,'native_prefix_decoder',return_value='tampered'),self.assertRaises(ValueError):
            raw._linear_boundary_proof(self.tok,ids,len(ids),decoded,proof,context)

if __name__=='__main__':unittest.main()
