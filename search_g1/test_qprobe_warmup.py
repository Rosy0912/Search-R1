"""CPU evidence for full teacher labels, cache tamper rejection and head budget.

Cache tests run on explicitly supplied remote synthetic-fixture storage.
They cannot establish actual CUDA frozen-teacher or restore acceptance.
"""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from search_g1.qprobe_warmup import TeacherCache, moments, verify_frozen, QUESTIONS, CHUNK_SIZE, STATUS
from search_g1.qprobe_features import sha_file


class MomentTests(unittest.TestCase):
    def test_every_binary_count_and_sample_std(self):
        for k in range(6):
            vector=[1]*k+[0]*(5-k)
            count,mean,std=moments(vector)
            self.assertEqual(count,k);self.assertEqual(mean,k/5)
            self.assertAlmostEqual(std,math.sqrt(sum((x-mean)**2 for x in vector)/4))

    def test_incomplete_or_nonbinary_labels_never_become_wrong_answers(self):
        for vector in [[],[0]*4,[0]*6,[0,0,0,0,2],[False]*5,[float('nan')]*5]:
            with self.assertRaises(ValueError):moments(vector)

    def test_frozen_readback_rejects_missing_rank_parameter_changes_and_writes(self):
        before=[dict(rank=i,parameter_sha256='model',optimizer_sha256='adam',scheduler_sha256='lr',
                     actual_optimizer_writes=0,completed_outer=0) for i in range(8)]
        self.assertTrue(verify_frozen(before,copy.deepcopy(before))['actual_eight_rank_frozen_readback'])
        for key,value in [('parameter_sha256','changed'),('optimizer_sha256','changed'),
                          ('scheduler_sha256','changed'),('actual_optimizer_writes',1),('completed_outer',1)]:
            after=copy.deepcopy(before);after[3][key]=value
            with self.assertRaises(ValueError):verify_frozen(before,after)
        with self.assertRaises(ValueError):verify_frozen(before,before[:-1])

    def test_full_5120_head_fit_keeps_zero_formal_counters_and_all_classes(self):
        import torch
        from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            counts=torch.arange(QUESTIONS)%6
            features=torch.nn.functional.one_hot(counts,6).float()
            head=OnlineQProbe(6,QProbeConfig(bootstrap_max_groups_per_outer=QUESTIONS,seed=20261013),
                              feature_spec_sha='a'*64)
            labels=[dict(qid=str(i),question_hash='b'*64,strict_em=[1]*(i%6)+[0]*(5-i%6),
                complete=True,selection_source='bootstrap_all_candidate',source_outer=0,
                policy_snapshot_id='synthetic-frozen-teacher:0',run_id='synthetic-run',attempt_id='warmup',
                sampling_contract_sha='c'*64,strict_scoring_contract_sha='d'*64,retriever_version='e'*64)
                for i in range(QUESTIONS)]
            cold=torch.nn.functional.cross_entropy(head.predict(features)['logits'],counts).item()
            actor_rng=torch.get_rng_state().clone()
            fit=head.bootstrap(labels,features,actor_outer=0,actor_snapshot_id='synthetic-frozen-teacher:0')
            self.assertEqual(fit['bootstrap_groups'],QUESTIONS)
            self.assertEqual(len(head.replay_labels),QUESTIONS)
            self.assertEqual({x['correct_count'] for x in head.replay_labels},set(range(6)))
            self.assertEqual((head.completed_outer,head.update_count),(0,0))
            self.assertTrue(head.optimizer.state)
            self.assertTrue(torch.equal(actor_rng,torch.get_rng_state()))
            self.assertLess(torch.nn.functional.cross_entropy(head.predict(features)['logits'],counts).item(),cold)
        finally:torch.set_num_threads(threads)


@unittest.skipUnless(os.environ.get('QPROBE_TEST_REMOTE_ROOT'),'remote synthetic cache storage required')
class CacheTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ['QPROBE_TEST_REMOTE_ROOT']);parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.root=Path(self.temp.name)
        self.identity=dict(questions=[dict(qid=str(i),question_hash='a'*64) for i in range(QUESTIONS)],teacher='synthetic')
        self.cache=TeacherCache(self.root,self.identity)
        self.contracts=dict(sampling_contract_sha='b'*64,strict_scoring_contract_sha='c'*64,retriever_version='d'*64)

    def tearDown(self):self.cache.close();self.temp.cleanup()

    def make(self,index):
        raw=self.root/('rollout-%02d.pt'%index);raw.write_bytes(b'clearly-marked-synthetic-fixture')
        groups=[]
        for i in range(index*CHUNK_SIZE,(index+1)*CHUNK_SIZE):
            vector=[1]*(i%6)+[0]*(5-i%6);k,m,s=moments(vector)
            groups.append(dict(**self.identity['questions'][i],strict_em=vector,
                correct_count=k,mean=m,std=s,trajectory_ids=['%d-%d'%(i,j) for j in range(5)]))
        return groups,dict(relative_path=raw.name,bytes=raw.stat().st_size,sha256=sha_file(raw))

    def test_complete_5120_all_classes_and_reuse(self):
        for i in range(10):
            groups,raw=self.make(i);self.cache.commit_chunk(i,groups,raw,self.contracts)
        proof,groups,_=self.cache.verify_complete()
        self.assertEqual(proof['status'],STATUS);self.assertEqual(len(groups),5120)
        self.assertEqual(proof['trajectories'],25600)
        self.assertEqual({x['correct_count'] for x in groups},set(range(6)))
        before=(self.root/'complete.json').read_bytes()
        self.assertEqual(self.cache.verify_complete()[0],proof)
        self.assertEqual((self.root/'complete.json').read_bytes(),before)

    def test_tamper_and_partial_pool_hardfail(self):
        groups,raw=self.make(0);self.cache.commit_chunk(0,groups,raw,self.contracts)
        with self.assertRaises(ValueError):self.cache.verify_complete()
        path=self.root/'chunk-00.json';value=json.loads(path.read_text());value['groups'][0]['std']=0.5
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.cache.read_chunk(0)

    def test_raw_bytes_and_duplicate_trajectory_hardfail(self):
        groups,raw=self.make(0)
        bad=copy.deepcopy(groups);bad[1]['trajectory_ids']=bad[0]['trajectory_ids']
        with self.assertRaises(ValueError):self.cache.commit_chunk(0,bad,raw,self.contracts)
        self.cache.commit_chunk(0,groups,raw,self.contracts)
        (self.root/raw['relative_path']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):self.cache.read_chunk(0)


if __name__=='__main__':unittest.main()
