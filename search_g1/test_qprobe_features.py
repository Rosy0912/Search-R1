"""Immutable cache and question-only input/provenance failures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from search_g1.qprobe_features import (COMMON_MANIFEST_SHA, OFFICIAL_ENCODER, OFFICIAL_REVISION,
    QUESTION_EXTRACTION_RULE, FeatureCache, FeatureCacheWriter, extract_question_features,
    last_nonpadding, original_question_and_id, question_text_sha, write_feature_cache)
from search_g1.qprobe_prepare import bootstrap_labels
from search_g1.mixed_selection import question_identity


def spec():
    return dict(encoder_id=OFFICIAL_ENCODER,encoder_revision=OFFICIAL_REVISION,
        encoder_files_sha256='a'*64,tokenizer_sha256='b'*64,prompt_template_sha256='c'*64,
        feature_layer=27,pooling='last_nonpadding_prompt_token',feature_dtype='float16',
        feature_dim=4,max_prompt_tokens=4096,question_extraction=QUESTION_EXTRACTION_RULE,
        train_manifest_sha256=COMMON_MANIFEST_SHA,split='train',encoder_compute_dtype='float32')


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layers=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
        self.calls=0
    def forward(self,input_ids,attention_mask,**kwargs):
        self.calls+=1
        hidden=input_ids.float().unsqueeze(-1).repeat(1,1,4)
        for layer in self.layers: hidden=layer(hidden)
        return dict(last_hidden_state=hidden)


class FakeTokenizer:
    def __call__(self,questions,**kwargs):
        assert kwargs['truncation'] is False
        return dict(input_ids=torch.tensor([[0,10,11],[20,21,0]]),
                    attention_mask=torch.tensor([[0,1,1],[1,1,0]]))


class QProbeFeatureTests(unittest.TestCase):
    def test_last_token_pooling_handles_both_padding_sides(self):
        hidden=torch.arange(24).reshape(2,3,4).float()
        pooled=last_nonpadding(hidden,torch.tensor([[0,1,1],[1,1,0]]))
        self.assertTrue(torch.equal(pooled,torch.stack([hidden[0,2],hidden[1,1]])))
        with self.assertRaisesRegex(ValueError,'ALL_PADDING'):
            last_nonpadding(hidden,torch.zeros(2,3))

    def test_extraction_has_no_answer_input_and_no_lm_head(self):
        model=FakeBase(); results=extract_question_features(model,FakeTokenizer(),['q1','q2'],feature_layer=0)
        self.assertEqual(model.calls,1)
        self.assertTrue(torch.equal(results,torch.tensor([[11]*4,[21]*4]).float()))
        self.assertEqual(len(model.layers[0]._forward_hooks),0)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError,'NO_SILENT_TRUNCATION'):
            extract_question_features(model,FakeTokenizer(),['q1','q2'],feature_layer=0,max_prompt_tokens=1)

    def test_original_question_and_upstream_identity_ignore_gold(self):
        row=dict(prompt=[{'role':'user','content':'Question: Which city?'}],data_source='nq',
                 extra_info={'index':17},reward_model={'ground_truth':{'target':['DO NOT ENCODE']}})
        question,qid=original_question_and_id(row)
        self.assertEqual(question,'Question: Which city?');self.assertEqual(qid,json.dumps(['nq','17']))
        row['reward_model']['ground_truth']['target']=['Different gold']
        self.assertEqual(original_question_and_id(row),(question,qid))

    def test_cache_exact_text_binding_gets_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features'
            features=torch.arange(12).reshape(3,4).float()
            write_feature_cache(path,spec(),['q1','q2','q3'],['Question one','Question two','Question three'],features)
            cache=FeatureCache(path); cache.verify_all()
            result=cache.get_features(['q3','q1'],['Question three','Question one'])
            self.assertTrue(torch.equal(result,features[[2,0]]))
            self.assertEqual(cache.feature_dim,4);self.assertEqual(cache.spec_sha,cache.feature_spec_sha)
            with self.assertRaisesRegex(ValueError,'TEXT_MISMATCH'):
                cache.get_features(['q1'],['Question  one'])
            with self.assertRaisesRegex(KeyError,'NOT_PRECOMPUTED'):
                cache.get_features(['missing'],['new question'])
            with self.assertRaises(FileExistsError): write_feature_cache(path,spec(),['q'],['q'],torch.ones(1,4))

    def test_missing_partial_and_wrong_spec_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache'
            with self.assertRaisesRegex(FileNotFoundError,'PREPARE_FIRST'):FeatureCache(path)
            writer=FeatureCacheWriter(path,spec(),2);writer.append(['q'],['Q?'],torch.ones(1,4))
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):writer.finish()
            writer.abort();self.assertFalse(writer.partial.exists());self.assertFalse(path.exists())
            write_feature_cache(path,spec(),['q'],['Q?'],torch.ones(1,4))
            with self.assertRaisesRegex(ValueError,'SPEC_MISMATCH'):FeatureCache(path,expected_spec_sha='wrong')

    def test_duplicate_ids_nonfinite_overflow_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache'
            with self.assertRaisesRegex(ValueError,'DUPLICATE'):
                write_feature_cache(path,spec(),['q','q'],['q1','q2'],torch.ones(2,4))
            with self.assertRaisesRegex(ValueError,'NONFINITE'):
                write_feature_cache(path,spec(),['q'],['q'],torch.full((1,4),float('inf')))
            write_feature_cache(path,spec(),['q'],['Question?'],torch.ones(1,4))
            cache=FeatureCache(path)
            altered=np.load(path/'features.npy',mmap_mode='r+',allow_pickle=False);altered[0,0]=2;altered.flush()
            with self.assertRaisesRegex(ValueError,'CORRUPT'):cache.get_features(['q'],['Question?'])
            with self.assertRaisesRegex(ValueError,'FEATURES_SHA'):cache.verify_all()

    def test_bootstrap_uses_all_classes_with_exact_audit_sha_and_policy_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempt1.json';question_by_id={f'q{i}':f'Question {i}?' for i in range(6)}
            audit=dict(source_commit='commit',model_revision=OFFICIAL_REVISION,data_manifest_sha256=COMMON_MANIFEST_SHA,
                sampling_contract_sha='sampling',strict_scoring_contract_sha='strict',retriever_version='retriever',
                step=4,policy_snapshot_id='actor-before-4',responses_per_group=5,attempted_candidates=6,
                raw_generated_trajectories=30,groups=[dict(ordinal=i,question_id=f'q{i}',
                    content_sha256=question_identity(question_by_id[f'q{i}']),trajectory_ids=[f'{i}-{j}' for j in range(5)],
                    strict_correctness=[1]*i+[0]*(5-i),category='all_wrong' if i==0 else 'all_correct' if i==5 else 'mixed',
                    selected=(i>0)) for i in range(6)])
            path.write_text(json.dumps(audit));sha=hashlib.sha256(path.read_bytes()).hexdigest()
            proof=dict(actor_outer=4,actor_snapshot_id='actor-after-4',run_id='old',source_commit='commit',
                sampling_contract_sha='sampling',strict_scoring_contract_sha='strict',retriever_version='retriever',
                source_policies=[dict(source_outer=4,policy_snapshot_id='actor-before-4',audit_sha256=sha)])
            values,qids,questions=bootstrap_labels([path],proof,question_by_id)
            self.assertEqual([x['correct_count'] for x in values],list(range(6)))
            self.assertEqual(values[0]['selection_source'],'bootstrap_all_candidate')
            self.assertEqual(values[0]['question_hash'],question_text_sha(questions[0]))
            modified=json.loads(json.dumps(proof));modified['source_policies'][0]['policy_snapshot_id']='later-policy'
            with self.assertRaisesRegex(ValueError,'ACTOR_MISMATCH'):bootstrap_labels([path],modified,question_by_id)
            no_contract=json.loads(json.dumps(audit));no_contract.pop('sampling_contract_sha')
            path.write_text(json.dumps(no_contract))
            updated=json.loads(json.dumps(proof));updated['source_policies'][0]['audit_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'AUDITED_CONTRACT_MISSING'):bootstrap_labels([path],updated,question_by_id)
            audit['groups']=audit['groups'][1:];path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,'NOT_IN_ACTOR_LINEAGE'):bootstrap_labels([path],proof,question_by_id)


if __name__=='__main__':unittest.main()
