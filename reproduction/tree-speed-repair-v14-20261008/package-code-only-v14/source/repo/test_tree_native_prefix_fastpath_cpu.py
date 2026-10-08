"""Tiny real HF/tokenizers fixture; no downloaded assets or model/GPU calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, decoders
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

import search_r1.baselines.tree_raw_tokens as raw
from test_tree_raw_tokens_cpu import Tokenizer as CustomTokenizer


class NativePrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.vocab={'<pad>':0,'<eos>':1}
        self.vocab.update({character:byte+2 for character,byte in raw.BYTELEVEL_INVERSE.items()})
        backend=Tokenizer(models.BPE(self.vocab,[]))
        backend.decoder=decoders.ByteLevel()
        # Decoder constructor ignores those flags in tokenizers 0.19; load the
        # exact frozen native JSON state, as the production tokenizer file does.
        state=json.loads(backend.to_str());state['decoder']=raw.BYTELEVEL_DECODER_STATE
        backend=Tokenizer.from_str(json.dumps(state))
        backend.save(str(self.path/'tokenizer.json'))
        self.tok=Qwen2TokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='<eos>',
            bos_token=None,clean_up_tokenization_spaces=False)
        self.tok.name_or_path=str(self.path)
        self.identity=raw.tokenizer_identity(self.tok,hashlib.sha256((self.path/'tokenizer.json').read_bytes()).hexdigest())

    def tearDown(self):self.temp.cleanup()

    def test_bound_native_prefixes_preserve_every_public_prefix_and_complete_proof(self):
        # Split/invalid/truncated UTF8, internal genuine PAD/EOS and JSON escapes.
        ids=[byte+2 for byte in b'X\xe4']+[0]+[byte+2 for byte in b'\xb8\xad"\n\\\xff\xe2\x82\xac\xe4']+[1]
        context=raw._TokenizerValidationContext(self.tok,self.identity)
        self.assertIsNotNone(context.native_prefix_decoder)
        native=context.native_prefix_decoder
        with patch.object(context,'native_prefix_decoder',wraps=native) as calls:
            result=raw._scan_utf8_prefixes(self.tok,ids,self.identity,_context=context)
            self.assertEqual(calls.call_count,len(ids))
        context.native_prefix_decoder=None
        public=raw._scan_utf8_prefixes(self.tok,ids,self.identity,_context=context)
        self.assertEqual(result,public)
        prefix_hashes=[raw.fingerprint(self.tok.decode(ids[:stop],skip_special_tokens=True))
            for stop in range(1,len(ids)+1)]
        self.assertEqual(result[2]['native_prefix_agreement_sha256'],raw.fingerprint(prefix_hashes))
        self.assertEqual(result[2]['native_prefix_agreement_count'],len(ids))
        self.assertTrue(result[2]['invalid_original_utf8_events']);context.check_complete()

    def test_custom_replaced_private_decode_and_native_disagreement_do_not_bypass_checks(self):
        custom=CustomTokenizer(self.path)
        self.assertIsNone(raw._bound_fast_prefix_decoder(custom))
        original=self.tok._decode
        with patch.object(self.tok,'_decode',wraps=original):
            self.assertIsNone(raw._bound_fast_prefix_decoder(self.tok))
        context=raw._TokenizerValidationContext(self.tok,self.identity)
        with patch.object(self.tok,'_decode',wraps=original),self.assertRaises(ValueError):context.check_runtime()
        context=raw._TokenizerValidationContext(self.tok,self.identity)
        with patch.object(context,'native_prefix_decoder',return_value='wrong-prefix'),self.assertRaises(ValueError):
            raw._scan_utf8_prefixes(self.tok,[ord('X')+2],self.identity,_context=context)


if __name__=='__main__':unittest.main()
