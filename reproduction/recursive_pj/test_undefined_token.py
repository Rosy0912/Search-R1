"""Real Qwen tokenizer tests, including replay of the captured rejected emission."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from search_r1.baselines.tree_raw_tokens import (
    tokenizer_identity, _scan_utf8_prefixes, _validation_context, sampled_action,
    canonical_text, validate_action_utf8_proof,
)


@unittest.skipUnless(os.environ.get('TREE_TEST_TOKENIZER'), 'requires real frozen Qwen tokenizer')
class NativeUndefinedToken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        root=Path(os.environ['TREE_TEST_TOKENIZER'])
        cls.tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
        cls.identity=tokenizer_identity(cls.tok,hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest())
        cls.missing=151819
        assert cls.tok.convert_ids_to_tokens(cls.missing) is None

    def test_all_prefixes_match_native_even_inside_split_utf8(self):
        t=self.tok
        a=t.convert_tokens_to_ids('Ã');b=t.convert_tokens_to_ids('©')
        for ids in ([self.missing], [a,self.missing,b], [self.missing,a,b,self.missing],
                    t.encode('<answer>Paris',add_special_tokens=False)+[self.missing]+t.encode('</answer>',add_special_tokens=False)):
            for check in (True,False):
                decoded,_,proof=_scan_utf8_prefixes(t,ids,self.identity,check)
                self.assertEqual(decoded[-1],t.decode(ids,skip_special_tokens=True))
                self.assertEqual(len(proof['native_undefined_id_empty_projection_checks']),ids.count(self.missing))

    def test_known_token_cannot_be_disguised_as_undefined(self):
        t=self.tok;token=t.encode('Paris',add_special_tokens=False)[0]
        with patch.object(t,'convert_ids_to_tokens',return_value=None):
            with self.assertRaisesRegex(ValueError,'Undefined token requires'):
                _scan_utf8_prefixes(t,[token],self.identity)

    def test_changed_native_projection_rejected(self):
        t=self.tok;context=_validation_context(t,self.identity)
        context.native_prefix_decoder=lambda ids,**kw: 'changed' if self.missing in ids else ''
        with self.assertRaisesRegex(ValueError,'changes native decoded text'):
            _scan_utf8_prefixes(t,[self.missing],self.identity,_context=context)

    def test_captured_emission_retains_original_ids_and_probabilities(self):
        path=os.environ.get('TREE_TEST_REJECTION')
        if not path:self.skipTest('captured failure unavailable')
        r=json.loads(Path(path).read_text());context=r['failure_context']
        e=next(e for e in context['native_emissions'] if self.missing in e['raw_emitted_token_ids'])
        t=self.tok;ids=e['raw_emitted_token_ids'];n=len(ids)
        text=canonical_text(t.decode(ids,skip_special_tokens=True))
        for linear in ('0','1'):
            with patch.dict(os.environ,TREE_LINEAR_PREFIX_SCAN=linear):
                action,ledger=sampled_action(t,e,ids+[t.pad_token_id]*(500-n),
                    t.encode(text,add_special_tokens=False),text,self.identity,
                    response_mask=[1]*n+[0]*(500-n))
                self.assertEqual(action,ids[:ledger['cut']])
                self.assertIn(self.missing,action)
                self.assertEqual(ledger['selected_logprobs'],e['raw_selected_logprobs'][:ledger['cut']])
                self.assertEqual(ledger['canonical_text'],context['legacy_text'][context['native_emissions'].index(e)])
                validate_action_utf8_proof(ledger,t)
                changed=copy.deepcopy(ledger)
                changed['utf8_boundary_proof']['native_undefined_id_empty_projection_checks'][0]['token_id']+=1
                with self.assertRaises(ValueError):validate_action_utf8_proof(changed,t)

if __name__=='__main__':unittest.main()
