"""Small CPU mathematical/reader tests; these fixtures are not GPU evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',HERE/'analyze_outer.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
REPO=HERE.parents[1]/'opsd-unclipped-multitraj200-teacherrecompute'


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules,_=a.source_modules(REPO)

    def score(self,spans,gold=None):
        return a.answer_metrics(spans,gold or ['Paris'],self.modules)

    def test_terminal_answer_eos(self):
        row=self.score(['<answer>The Paris.</answer><|im_end|>'])
        self.assertEqual(row['em'],1);self.assertEqual(row['f1'],1)

    def test_previous_answer_not_reused(self):
        self.assertEqual(self.score(['<answer>Paris</answer>','<search>query</search>'])['em'],0)

    def test_split_answer_not_joined_over_observation(self):
        self.assertFalse(self.score(['<answer>Pa','ris</answer>'])['completed_terminal_answer'])

    def test_nested_or_extra_answer_rejected(self):
        for value in ('<answer><search>Paris</search></answer>','<answer>Paris</answer><answer>Rome</answer>'):
            self.assertEqual(self.score([value])['em'],0)

    def test_partial_token_f1(self):
        self.assertAlmostEqual(self.score(['<answer>Paris France</answer>'])['f1'],2/3)

    def test_reasoning_compatibility_and_strict_format(self):
        row=self.score(['I believe <answer>Paris</answer>'])
        self.assertEqual(row['em'],1);self.assertFalse(row['legal_complete_trajectory'])

    def test_strict_complete_search_then_answer(self):
        self.assertTrue(self.score(['<search>q</search>','<answer>Paris</answer>'])['legal_complete_trajectory'])

    def test_decode_authorship_excludes_gold_observation(self):
        class Tok:
            def decode(self,ids):return ''.join({1:'<search>q</search>',2:'<answer>Paris</answer>',3:'<answer>Rome</answer>'}[i] for i in ids)
        p=self.modules['protocol']
        self.assertEqual(self.score(p.decode_model_spans(Tok(),[1,2,3],[1,0,1]))['em'],0)

    def test_empty_pair_is_not_agreement(self):
        q={'question_uid':'fixture','question_position':0,'data_source':'fixture','dataset_row_position':1}
        students=[dict(**self.score([]),response_index=i) for i in range(5)]
        out=a.question_summary(q,students,self.score([]))
        self.assertEqual(out['complete_pair_count'],0);self.assertEqual(out['same_completed_pair_count'],0)
        self.assertEqual(out['student_teacher_agreement_count'],0);self.assertFalse(out['all_five_complete_and_same'])

    def test_any_correct_consistency_separate(self):
        q={'question_uid':'fixture','question_position':0,'data_source':'fixture','dataset_row_position':1}
        students=[dict(**self.score([f'<answer>{v}</answer>']),response_index=i)
                  for i,v in enumerate(('Paris','Rome','Rome','Rome','Rome'))]
        out=a.question_summary(q,students,self.score(['<answer>Paris</answer>']))
        self.assertTrue(out['student_any_correct']);self.assertFalse(out['student_majority_correct'])
        self.assertEqual(out['same_completed_pair_count'],6);self.assertEqual(out['consensus_max_count'],4)
        self.assertEqual(out['student_teacher_agreement_count'],1)

    def test_exact_five_group(self):
        with self.assertRaises(ValueError):a.question_summary({},[],{})


class MaskAndReaderTests(unittest.TestCase):
    def tensors(self):
        return {'responses':[2,3,0],'prompts':[1],'input_ids':[1,2,3,0],
                'attention_mask':[1,1,1,0],'info_mask':[1,1,0,0],'loss_mask':[1,0,0]}

    def test_mask_observation_and_padding(self):
        self.assertEqual(a.authored_mask(self.tensors()),[1,0,0])

    def test_mask_bool_rejected(self):
        t=self.tensors();t['info_mask'][1]=True
        with self.assertRaises(ValueError):a.authored_mask(t)

    def test_mask_padding_supervision_rejected(self):
        t=self.tensors();t['info_mask'][-1]=1;t['loss_mask'][-1]=1
        with self.assertRaises(ValueError):a.authored_mask(t)

    def test_mask_loss_drift_rejected(self):
        t=self.tensors();t['loss_mask'][1]=1
        with self.assertRaises(ValueError):a.authored_mask(t)

    def test_duplicate_nonfinite_json_rejected(self):
        for value in (b'{"x":1,"x":2}',b'{"x":NaN}'):
            with self.assertRaises(ValueError):a.parse(value)

    def test_saved_merge_header_exact_body_only(self):
        raw=b'a b\nx y\n';pin=a.sha(raw)
        self.assertEqual(a.official_merges_equivalence(raw,pin),'BYTE_EXACT_OFFICIAL')
        self.assertEqual(a.official_merges_equivalence(b'#version: 0.2\n'+raw,pin),
                         'EXACT_OFFICIAL_BODY_AFTER_SINGLE_HF_BPE_VERSION_HEADER')
        for changed in (b'#version: 0.3\n'+raw,b'#version: 0.2\n'+raw+b'bad',b'#version: 0.2\n'*2+raw):
            with self.assertRaises(ValueError):a.official_merges_equivalence(changed,pin)

    def test_read_exact_hash_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'actual';p.write_bytes(b'raw original')
            raw,record=a.read(p,a.sha(b'raw original'));self.assertTrue(record['stable_read'])
            with self.assertRaises(ValueError):a.read(p,'0'*64)
            link=Path(tmp)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):a.read(link)

    def test_exclusive_output_keeps_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report';a.publish(p,b'first')
            with self.assertRaises(FileExistsError):a.publish(p,b'second')
            self.assertEqual(p.read_bytes(),b'first')


if __name__=='__main__':unittest.main()
