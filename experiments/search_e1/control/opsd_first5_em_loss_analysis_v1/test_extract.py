"""Focused CPU checks for mask, target interpretation and native reductions."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name('extract.py')
spec = importlib.util.spec_from_file_location('first5_extract', SOURCE)
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def fixture(root, group='M', absent_answer=False, corrupt=None):
    root = Path(root).resolve()
    text = '<think>x</think><search>q</search>' + ('' if absent_answer else '<answer>A</answer>')
    offsets = [[text.index('x'), text.index('x')+1], [text.index('q'), text.index('q')+1]]
    if not absent_answer:
        a = text.index('A'); offsets.append([a,a+1])
    pair = dict(question_id='fixture:train:1',question='private fixture',answers=['A'],
                reference='successful original fixture',student=text,reference_reward=1,student_reward=0,
                student_policy_spans=[[0,len(text)]],pair_weight=1.)
    old_ref = dict(path='/fixture/old.json',sha256='a'*64,bytes=100)
    config = dict(variant=group,source_model_manifest=dict(sha256='b'*64))
    contract = dict(variant=group,direction='student_to_target_reverse',mixture_alpha=.1 if group=='M' else 0.)
    receipt = dict(run_id='r',outer_index=0,optimizer_updates=1,contract=contract,retries=0,
                   pre_update_adapter_sha256='c'*64,outer_old_adapter=old_ref,global_actual_pairs=7)
    identity = dict(question_id=pair['question_id'],pair_row_sha256=e.object_sha(pair),student_reward=0,reference_reward=1,
        student_response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        reference_response_sha256=hashlib.sha256(pair['reference'].encode()).hexdigest())
    header = dict(schema='opsd-online-training-token-header-v1',run_id='r',outer_index=0,
        optimizer_update=1,rank=0,pair_index=0,encoded_sha256='d'*64,identity=identity,
        contract=contract,component_clip=10,retry=0,pair_weight=1.,student_pre_update_adapter_sha256='c'*64,
        outer_old_adapter=old_ref,frozen_teacher_identity='b'*64,loss_scope='actual_training_forward_before_optimizer',
        token_ids=list(range(len(offsets))),token_offsets=offsets,policy_mask=[1]*len(offsets),sampling_model_identity={'test':'fixed'})
    rows = []
    raw = .02 if group=='M' else .2
    for index,offset in enumerate(offsets):
        target_logp = math.log(.1*math.exp(-2.)+.9*math.exp(-1.3)) if group=='M' else -2.
        row = dict(token_index=index,token_id=index,token_offsets=offset,policy_mask=True,
            valid_trajectory_tokens=len(offsets),global_actual_pairs=7,pair_weight=1.,
            direction='student_to_fixed_target_reverse',target_kind='mixture' if group=='M' else 'teacher',
            loss_scope='actual_training_forward_before_optimizer',teacher_forced_fixed_student_prefix=True,
            actual_generation_scores=False,outer_old_identity={'manifest':old_ref},
            pre_clip_full_vocabulary_kl=raw,post_clip_full_vocabulary_loss=raw,
            student_to_teacher_raw_kl=.2,student_to_outer_old_raw_kl=.01,student_to_mixture_raw_kl=raw,
            clipped_vocabulary_count=0,vocabulary_size=151936,clipped_vocabulary_fraction=0.,
            clipped_student_probability_mass=0.,clipped_teacher_probability_mass=0.,clipped_mixture_probability_mass=0.,
            trajectory_contribution=raw/len(offsets),global_batch_contribution=raw/len(offsets)/7,
            student_observed_token_logp=-1.,teacher_observed_token_logp=-2.,outer_old_observed_token_logp=-1.3,
            mixture_observed_token_logp=target_logp,student_entropy=1.,teacher_entropy=1.,outer_old_entropy=1.,mixture_entropy=1.,
            token_text='single-token decode deliberately differs from response slice')
        rows.append(row)
    metric = dict(pair_index=0,valid_tokens=len(offsets),pair_weight=1.,raw_loss=raw,post_loss=raw,
        student_to_teacher_raw_kl=.2,student_to_outer_old_raw_kl=.01,student_to_mixture_raw_kl=raw,
        weighted_post_loss=raw,clipped_vocabulary_count=0)
    if corrupt:
        corrupt(header,rows)
    path = Path(root)/'committed.jsonl'; rawfile=b''.join(e.canonical(v)+b'\n' for v in [header]+rows)
    path.write_bytes(rawfile)
    ref=dict(path=str(path),sha256=hashlib.sha256(rawfile).hexdigest(),bytes=len(rawfile))
    return ref,pair,metric,receipt,config


class ExtractionTests(unittest.TestCase):
    def test_actual_tail_region_sum_and_geometric_probability(self):
        with tempfile.TemporaryDirectory() as root:
            ref,pair,metric,receipt,config=fixture(root)
            regions,proof=e.token_features(e.Reader(),ref,pair,metric,receipt,0,'d'*64,config)
            self.assertEqual(regions['all_policy']['valid_tokens'],3)
            self.assertAlmostEqual(sum(v['global_batch_contributions']['target_post_loss'] for k,v in regions.items() if k!='all_policy'),.02/7)
            self.assertAlmostEqual(regions['final_answer']['probability_geometric_means']['student'],math.exp(-1.))
            self.assertAlmostEqual(regions['final_answer']['means']['student_teacher_observed_logp_gap'],1.)
            self.assertEqual(proof['token_log']['sha256'],ref['sha256'])

    def test_empty_answer_null_probability_zero_contribution(self):
        with tempfile.TemporaryDirectory() as root:
            ref,pair,metric,receipt,config=fixture(root,absent_answer=True)
            regions,_=e.token_features(e.Reader(),ref,pair,metric,receipt,0,'d'*64,config)
            answer=regions['final_answer']
            self.assertEqual(answer['valid_tokens'],0)
            self.assertIsNone(answer['means']['student_observed_logp'])
            self.assertIsNone(answer['probability_geometric_means']['student'])
            self.assertEqual(answer['trajectory_contributions']['target_post_loss'],0.)

    def test_wrong_arithmetic_mixture_logp_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            args=fixture(root,corrupt=lambda h,r:r[0].update(mixture_observed_token_logp=.1*(-2)+.9*(-1.3)))
            ref,pair,metric,receipt,config=args
            with self.assertRaisesRegex(ValueError,'arithmetic mixture'):
                e.token_features(e.Reader(),ref,pair,metric,receipt,0,'d'*64,config)

    def test_invalid_mask_outside_authorship_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            ref,pair,metric,receipt,config=fixture(root)
            pair['student_policy_spans']=[[0,7]]
            # Identity check is updated so the test reaches the frozen mask check.
            rows=[json.loads(x) for x in Path(ref['path']).read_bytes().splitlines()]
            rows[0]['identity']['pair_row_sha256']=e.object_sha(pair)
            raw=b''.join(e.canonical(v)+b'\n' for v in rows);Path(ref['path']).write_bytes(raw)
            ref.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError,'mask differs'):
                e.token_features(e.Reader(),ref,pair,metric,receipt,0,'d'*64,config)

    def test_native_mean_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            ref,pair,metric,receipt,config=fixture(root)
            metric['post_loss']+=.1
            with self.assertRaisesRegex(ValueError,'reconstruction'):
                e.token_features(e.Reader(),ref,pair,metric,receipt,0,'d'*64,config)

    def test_boundary_and_final_payload_do_not_score_nested_answer(self):
        text='<think>x<answer>A</answer>'
        regions=e.character_regions(text,[[0,len(text)]])
        self.assertEqual(e.category([6,8],regions),'boundary')
        self.assertEqual(e.category([text.index('A'),text.index('A')+1],regions),'final_answer')
        # Category is deliberately textual; frozen native QA separately rejects this action.

    def test_sha_mismatch_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root).resolve()/'input.json';path.write_text('{}')
            reader=e.Reader()
            with self.assertRaisesRegex(ValueError,'SHA'):
                reader.json(dict(path=str(path),sha256='0'*64))
            alias=Path(root).resolve()/'alias.json';alias.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'nonsymlink'):
                reader.json(dict(path=str(alias),sha256=hashlib.sha256(b'{}').hexdigest()))


if __name__=='__main__':
    unittest.main()
