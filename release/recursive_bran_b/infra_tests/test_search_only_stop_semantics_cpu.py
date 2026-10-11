"""Synthetic original-ID/mu and SEARCH-first boundaries; no GPU/EM claim."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_tree_raw_tokens_cpu import Tokenizer, params, ENV
from search_r1.baselines import tree_raw_tokens as raw
from search_r1.baselines.tree_behavior_mu import _emission


class Params(NS):
    def clone(self): return copy.deepcopy(self)
    def update_from_generation_config(self, generation, eos):
        # Explicit synthetic native EOS expansion, not a real vLLM API proof.
        self.all_stop_token_ids = {eos}


class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        directory = Path(self.tmp.name); (directory/'tokenizer.json').write_text('{}')
        self.tok = Tokenizer(directory)
        self.identity = raw.tokenizer_identity(self.tok, hashlib.sha256(b'{}').hexdigest())

    def tearDown(self): self.tmp.cleanup()

    def capture(self, ids, stopped=False, returned=None):
        p = Params(**vars(params()))
        if stopped:
            p.stop = ['</search>']; p.include_stop_str_in_output = True; p.detokenize = True
        stopping = raw.capture_native_stopping(p, self.tok.eos_token_id, {})
        native = NS(token_ids=ids,
            logprobs=[{token:NS(logprob=-.25*(i+1))} for i,token in enumerate(ids)],
            finish_reason='stop',stop_reason='</search>' if stopped else None,
            text=returned if stopped else self.tok.decode(ids,skip_special_tokens=True))
        request = NS(request_id='synthetic-fixture',finished=True,outputs=[native],prompt_token_ids=[8])
        with patch.dict(os.environ, ENV):
            return raw.capture_native_emissions([request],p,'synthetic-native-call',0,native_stopping=stopping)[0]

    def action(self, emission):
        ids=emission['raw_emitted_token_ids'];text=raw.canonical_text(self.tok.decode(ids,skip_special_tokens=True))
        return raw.sampled_action(self.tok,emission,ids+[self.tok.pad_token_id]*(500-len(ids)),
            self.tok.encode(text,add_special_tokens=False),text,self.identity)

    def test_search_stop_keeps_exact_original_id_and_selected_mu(self):
        old=self.capture([7,6,self.tok.eos_token_id])
        stopped=self.capture([7],True,'<search>capital</search>')
        oldids,oldaction=self.action(old);newids,newaction=self.action(stopped)
        self.assertEqual(oldids,newids);self.assertEqual(newids,[7])
        self.assertEqual(oldaction['selected_logprobs'],newaction['selected_logprobs'])
        self.assertEqual(newaction['selected_logprobs'],[-.25])
        self.assertEqual(newaction['canonical_text'],oldaction['canonical_text'])
        self.assertEqual(newaction['indivisible_extra_text'],'extra')
        # Actual Tree mu validation sees the same raw prefix, not re-encoded IDs.
        self.assertEqual(_emission(newaction,[8],ENV)['raw_selected_logprobs'],[-.25])
        raw.validate_action_utf8_proof(newaction,self.tok)

    def test_missing_or_forged_actual_native_stopping_text_refused(self):
        emission=self.capture([7],True,'<search>capital</search>')
        for edit in (lambda e:e.pop('native_returned_text'),
                     lambda e:e.update(native_returned_text='<search>different</search>'),
                     lambda e:e.update(raw_emitted_token_ids=[10]),
                     lambda e:e.update(raw_selected_logprobs=[float('nan')])):
            changed=copy.deepcopy(emission);edit(changed)
            with self.assertRaises(ValueError):self.action(changed)

    def test_continuing_after_complete_search_stop_refused(self):
        # First token already contains a complete SEARCH stop: a second emitted
        # token would contradict a min_tokens0 native stop, even if text is cut.
        emission=self.capture([7,6],True,'<search>capital</search>')
        with self.assertRaises(ValueError):self.action(emission)

    def test_search_only_preserves_search_first_but_dual_stop_can_change_action(self):
        full='<answer>a</answer> then <search>q</search> discarded'
        search_stopped=full[:full.index('</search>')+len('</search>')]
        answer_stopped=full[:full.index('</answer>')+len('</answer>')]
        self.assertEqual(raw.canonical_text(full),raw.canonical_text(search_stopped))
        self.assertNotEqual(raw.canonical_text(full),raw.canonical_text(answer_stopped))
        answer_only='<answer>a</answer> discarded'
        self.assertNotIn('</search>',answer_only)
        self.assertEqual(raw.canonical_text(answer_only),'<answer>a</answer>')


if __name__=='__main__': unittest.main()
