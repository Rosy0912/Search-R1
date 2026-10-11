"""Dependency-light CPU scanner mechanism test, not approved HF-binding evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_tree_raw_tokens_cpu import Tokenizer
import search_r1.baselines.tree_raw_tokens as raw

class ScannerMechanismTests(unittest.TestCase):
    def test_linear_full_cut_exact_utf8_and_native_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'tokenizer.json').write_text('{}')
            tok=Tokenizer(path);identity=raw.tokenizer_identity(tok,hashlib.sha256(b'{}').hexdigest())
            for ids in [[1000+x for x in b'X\xe4\xb8\xad\xff']+[151645],
                        [1,2,3,151645],[1000+x for x in b'<answer>X</an']+[5,6,151645],
                        [1000+x for x in b'\xe4']+[151643]+[1000+x for x in b'\xb8\xad']+[151645]]:
                context=raw._TokenizerValidationContext(tok,identity)
                # Synthetic mechanism injection only; production requires the
                # approved unchanged real HF decoder, tested in the other file.
                context.native_prefix_decoder=tok.decode
                with patch.object(context,'native_prefix_decoder',wraps=tok.decode) as calls:
                    old=raw._scan_utf8_prefixes(tok,ids,identity,True,context)
                    oldcalls=calls.call_count
                linear=raw._scan_utf8_prefixes(tok,ids,identity,False,context)
                self.assertEqual(old,linear)
                decoded,complete,proof=linear;text=raw.canonical_text(decoded[-1])
                choices=[i+1 for i,p in enumerate(decoded) if complete[i] and p.startswith(text)]
                cut=choices[0] if '</answer>' in decoded[-1] or '</search>' in decoded[-1] else len(ids)
                with patch.object(context,'native_prefix_decoder',wraps=tok.decode) as calls:
                    new=raw._linear_boundary_proof(tok,ids,cut,decoded,proof,context)
                    self.assertEqual(calls.call_count,len(set((len(ids),cut))))
                    self.assertLess(calls.call_count,oldcalls)
                self.assertEqual(new['native_decode_call_count'],len(set((len(ids),cut))))
                self.assertNotIn('native_prefix_agreement_count',new)
                self.assertEqual(tok.decode(ids[:cut],skip_special_tokens=True),decoded[cut-1])
                if ids==[1,2,3,151645]:self.assertEqual(decoded[cut-1][len(text):],'tail')
                with patch.object(context,'native_prefix_decoder',return_value='changed'),self.assertRaises(ValueError):
                    raw._linear_boundary_proof(tok,ids,cut,decoded,proof,context)

if __name__=='__main__':unittest.main()
