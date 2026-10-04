import json
import tempfile
import unittest
from pathlib import Path
from tokenizers import Tokenizer,models
from reproduction.a800.tokenizer_identity import validate_tokenizer_identity


class TokenizerIdentity(unittest.TestCase):
    def test_serialization_change_allowed_actual_semantic_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'original';b=Path(tmp)/'saved';a.mkdir();b.mkdir()
            backend=Tokenizer(models.BPE(vocab={'a':0,'b':1,'ab':2},merges=[('a','b')]))
            saved=json.loads(backend.to_str());old=json.loads(backend.to_str())
            old['model']['merges']=['a b'];old['model'].pop('ignore_merges',None)
            for p,value in [(a,old),(b,saved)]:
                (p/'tokenizer.json').write_text(json.dumps(value))
                (p/'tokenizer_config.json').write_text(json.dumps({'chat_template':'test','tokenizer_class':'PreTrainedTokenizerFast'}))
            self.assertTrue(validate_tokenizer_identity(a,b)['verified'])
            saved['model']['merges']=[];(b/'tokenizer.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(AssertionError,'backend semantics'):validate_tokenizer_identity(a,b)

    def test_chat_template_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'original';b=Path(tmp)/'saved';a.mkdir();b.mkdir()
            for p in (a,b):
                Tokenizer(models.BPE(vocab={'a':0},merges=[])).save(str(p/'tokenizer.json'))
            (a/'tokenizer_config.json').write_text(json.dumps({'chat_template':'original'}))
            (b/'tokenizer_config.json').write_text(json.dumps({'chat_template':'changed'}))
            with self.assertRaisesRegex(AssertionError,'config/chat template'):validate_tokenizer_identity(a,b)


if __name__=='__main__':unittest.main()
