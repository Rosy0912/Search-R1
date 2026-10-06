"""Ordered EM/F1 scalar fixtures; no native evaluation or model result is claimed."""
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8 import validation_identity as v


class Column:
    def __init__(self,values):self.values=values
    def tolist(self):return self.values


class FormalValidationTests(unittest.TestCase):
    def fixture(self,root):
        base=Path(root);(base/'logs').mkdir()
        (base/'logs/run-manifest.json').write_text(json.dumps(dict(run_id='CPU-fixture-only')))
        record=dict(sha256='CPU product hash fixture',source_counts=dict(nq=1,triviaqa=1))
        frame={'data_source':Column(['nq','triviaqa'])}
        return base,(record,frame,['first-original-id','second-original-id'])

    def test_actual_alias_overlap_f1_is_separate_from_exact_match(self):
        with tempfile.TemporaryDirectory() as root:
            base,expected=self.fixture(root)
            with patch.object(v,'_ACTIVE_SPLIT','stage'),patch.object(v,'COUNTS',{'stage':2}),patch.object(v,'expected',return_value=expected):
                result=v.record_validation(base,5,expected[2],['nq','triviaqa'],[1.,0.],f1=[1.,.5])
            self.assertEqual(result['weighted_em'],.5)
            self.assertEqual(result['weighted_f1'],.75)
            self.assertEqual(result['f1'],[1.,.5])

    def test_final_and_stage_identity_and_ledger_are_durable_before_return(self):
        for final in (False,True):
            for fail in (False,True):
                with self.subTest(final=final,fail=fail),tempfile.TemporaryDirectory() as root:
                    base,expected=self.fixture(root);split='final' if final else 'stage';seen=[];real=os.fsync
                    def sync(fd):
                        directory=stat.S_ISDIR(os.fstat(fd).st_mode);seen.append('directory' if directory else 'file')
                        if directory and fail:raise OSError('constructed validation parent durability failure')
                        return real(fd)
                    with patch.object(v,'_ACTIVE_SPLIT',split),patch.object(v,'COUNTS',{split:2}), \
                         patch.object(v,'expected',return_value=expected),patch.object(os,'fsync',side_effect=sync):
                        if fail:
                            with self.assertRaisesRegex(OSError,'parent durability failure'):
                                v.record_validation(base,200 if final else 5,expected[2],['nq','triviaqa'],[1.,0.],final=final,f1=[1.,.5])
                        else:v.record_validation(base,200 if final else 5,expected[2],['nq','triviaqa'],[1.,0.],final=final,f1=[1.,.5])
                    self.assertEqual(seen,['file','directory'] if fail else ['file','directory','file','directory'])
                    self.assertEqual((base/'logs/validation-identities.jsonl').exists(),not fail)

    def test_missing_nonfinite_invalid_f1_or_changed_identity_rejected(self):
        variants=[dict(f1=None),dict(f1=[1.,float('nan')]),dict(f1=[1.,1.1]),
                  dict(f1=[0.,.5]),dict(f1=[1.]),dict(ids=['second-original-id','first-original-id'],f1=[1.,.5])]
        for variant in variants:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                base,expected=self.fixture(root);args=dict(ids=expected[2],f1=[1.,.5]);args.update(variant)
                with patch.object(v,'_ACTIVE_SPLIT','stage'),patch.object(v,'COUNTS',{'stage':2}),patch.object(v,'expected',return_value=expected),self.assertRaises(AssertionError):
                    v.record_validation(base,5,args['ids'],['nq','triviaqa'],[1.,0.],f1=args['f1'])
                self.assertFalse((base/'logs/latest-stage-identity.json').exists())

    def test_all_source_scalar_reader_keeps_calibration_source_default_strict(self):
        from reproduction.common8.test_calibration_evaluation import tensor_batch,CharacterTokenizer,rows_from_batch
        batch=tensor_batch([dict(segments=[('<answer>Alpha</answer>',True)],gold=['Alpha'],source='triviaqa')])
        tokenizer=CharacterTokenizer()
        with self.assertRaises(ValueError):rows_from_batch(batch,tokenizer)
        rows=rows_from_batch(batch,tokenizer,allowed_sources={'triviaqa'})
        self.assertEqual(rows[0]['data_source'],'triviaqa')
        self.assertEqual((rows[0]['em'],rows[0]['f1']),(1,1.))


if __name__=='__main__':unittest.main()
