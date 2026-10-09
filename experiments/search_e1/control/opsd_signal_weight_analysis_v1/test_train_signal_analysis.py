"""CPU synthetic fixtures; no remote, Torch, model or real data."""
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
import train_signal_analysis as a
class Writer:
 def __init__(self,path):self.path=Path(path);self.n=0
 def put(self,value,lines=False):
  self.n+=1;p=self.path/('x%03d.json%s'%(self.n,'l' if lines else ''));raw=b''.join(a.canonical(r)+b'\n' for r in value) if lines else a.canonical(value)+b'\n';p.write_bytes(raw)
  return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def summary():
 qs=[dict(question_id='q%03d'%i,dataset='nq' if i<256 else 'hotpotqa',success_count=int(i==0),s_over_5=.2 if i==0 else 0.,rewards=[1,0,0,0,0] if i==0 else [0]*5,row_sha256=[hashlib.sha256(('%d-%d'%(i,j)).encode()).hexdigest() for j in range(5)]) for i in range(512)]
 return dict(schema='opsd-signal-weight-train-em-v1',accepted=True,variant='G_POS',outer_index=0,run_id='r',post_training_em=False,questions=512,unique_slots=2560,overall=dict(denominator=2560,strict_correct=1,em=1/2560),all5_wrong=511,per_question=qs,sampling_preupdate_raw_model={'sha256':'d'*64})
def fixture(w,K=1):
 s=summary();q=s['per_question'][0];sib=[dict(sibling=j,original_row_sha256=q['row_sha256'][j],answer_gap=float(j-2),student_entropy=float(j+1)) for j in range(5)]
 score=dict(scoring_scope='no_grad_current_preupdate_hf_replay_not_generation_scores',original_training_objective=False,reference_response_sha256='c'*64,schema='opsd-current-preupdate-five-sibling-scores-v1',accepted=True,variant='G_POS',question_id=q['question_id'],preupdate_identity=dict(run_id='r',optimizer_update=1,adapter_tensor_sha256='b'*64,encoded_sha256='a'*64,source_model_identity='e'*64),state_restore=dict(state_restored=True,rng_consumed=False),siblings=sib,signal='answer_gap',normalization=dict(amplitude=.9,numerical_floor=1e-6,direction=1,neutral_reason=None,scores=[-2.,-1.,0.,1.,2.],weights=[.1,.55,1.,1.45,1.9]),selected_sibling=0,selected_pair_weight=.1)
 header=dict(loss_scope='actual_training_forward_before_optimizer',frozen_teacher_identity='e'*64,schema='opsd-online-training-token-header-v1',optimizer_update=1,run_id='r',student_pre_update_adapter_sha256='b'*64,identity=dict(question_id=q['question_id'],reference_response_sha256='c'*64),encoded_sha256='a'*64,pair_weight=.1,policy_mask=[1,1],token_ids=[10,11],token_offsets=[[0,1],[1,2]],token_regions=['think_text','final_answer'],mining=dict(reference_sibling=0))
 tokens=[dict(token_index=i,policy_mask=True,token_id=10+i,token_offsets=[i,i+1],region=header['token_regions'][i],pre_clip_full_vocabulary_kl=2.+2*i,post_clip_full_vocabulary_loss=1.+2*i,student_to_teacher_raw_kl=2.+2*i,global_batch_contribution=.1*(1.+2*i)/2,student_entropy=1.,clipped_vocabulary_count=1.,clipped_student_probability_mass=.1,student_observed_token_logp=-2. if i==0 else -3.,teacher_observed_token_logp=-1.) for i in range(2)]
 metric=dict(target_kind='teacher',pair_index=0,signal_score=w.put(score),token_log=w.put([header]+tokens,True),pair_weight=.1,valid_tokens=2,raw_loss=3.,post_loss=2.,weighted_post_loss=.2,signal_scalar=dict(question_id=q['question_id'],sibling=0,strict_reward=1,success_fraction=.2))
 u=dict(schema='opsd-online-real-update-v1',accepted=True,run_id='r',outer_index=0,optimizer_updates=1,pre_update_adapter_sha256='b'*64,global_actual_pairs=1,all_rank_actual_token_metrics=[dict(rank=j,pairs=[metric] if j==0 else []) for j in range(4)],scaler=dict(grad_norm=2.))
 cp=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id='r',optimizer_updates=K,cursor=dict(outer_index=0,next_pair_cursor=K,usable_pair_count=K),outer_update_receipts=[w.put(u)] if K else [])
 done=dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id='r',outer_index=0,P=1,K=K,optimizer_updates_delta=K,optimizer_updates=K,checkpoint=w.put(cp))
 e=dict(outer_index=0,sampling_summary=w.put(s),training_completion=w.put(done))
 return dict(variant='G_POS',run_id='r'),e,s,score,u,metric,header,tokens
class Tests(unittest.TestCase):
 def test_full_population_hard_noB_and_K0_NA(self):
  with tempfile.TemporaryDirectory() as d:
   cfg,e,*_=fixture(Writer(d),0);info,rows,b=a.collect_outer(a.Reader(),cfg,e);self.assertEqual(len(rows),512);self.assertEqual(info['train_denominator'],2560);self.assertEqual(sum(not r['teacher_available'] for r in rows),511);self.assertEqual(rows[0]['missing_reason'],'selected_D_encoding_skip');self.assertFalse(rows[0]['trained']);self.assertEqual(b,[])
   result=a.aggregate_features(rows,b,0)['groups']['G_POS'];self.assertEqual(result['gap_measured_questions'],0);self.assertIsNone(result['selected_weight']['mean'])
 def test_actual_commit_token_region_weights_and_self_pair(self):
  with tempfile.TemporaryDirectory() as d:
   cfg,e,*_=fixture(Writer(d));reader=a.Reader();info,rows,b=a.collect_outer(reader,cfg,e);reader.stable_end();r=rows[0];self.assertTrue(r['self_pair']);self.assertEqual(r['answer_gap'],-2.);self.assertEqual(r['post_loss'],2.);self.assertAlmostEqual(r['weighted_post_loss'],.2);self.assertAlmostEqual(sum(v['weighted_post_contribution'] for v in r['regions'].values()),.2)
   result=a.aggregate_features(rows,b,10)['groups']['G_POS'];self.assertEqual(result['population_sampled_trajectories'],2560);self.assertEqual(result['CC_CW_exposure'],{'CC':1});self.assertEqual(result['clipped_batches'],1);self.assertEqual(result['selected_D_relationships']['answer_gap']['selected_reward']['missing_questions'],511)
 def test_partial_filtered_sampling_refused(self):
  for k,v in [('unique_slots',2559),('questions',1),('all5_wrong',0)]:
   s=summary();s[k]=v
   with self.assertRaises(ValueError):a.sampling_questions(s,'G_POS',0)
 def test_SHA_and_changed_input_refused(self):
  with tempfile.TemporaryDirectory() as d:
   w=Writer(d);ref=w.put({'a':1});Path(ref['path']).write_text('{"a":2}\n')
   with self.assertRaises(ValueError):a.Reader().json(ref)
   ref=w.put([{'a':1}],True);r=a.Reader();list(r.lines(ref));Path(ref['path']).write_text('{"a":3}\n')
   with self.assertRaises(ValueError):r.stable_end()
 def test_model_row_weight_tampering_refused(self):
  with tempfile.TemporaryDirectory() as d:
   cfg,e,s,sc,u,m,*_=fixture(Writer(d));q=s['per_question'][0]
   for mutate in (lambda x:x['preupdate_identity'].update(adapter_tensor_sha256='x'*64),lambda x:x['siblings'][2].update(original_row_sha256='x'*64),lambda x:x['normalization']['weights'].__setitem__(0,.5)):
    bad=copy.deepcopy(sc);mutate(bad)
    with self.assertRaises(ValueError):a.validate_score(bad,q,'G_POS',u,m,'r')
 def test_token_duplicate_missing_wrongdenominator_refused(self):
  with tempfile.TemporaryDirectory() as d:
   _,_,_,sc,u,m,h,t=fixture(Writer(d))
   for records in ([h]+t[:1],[h]+t+[t[0]]):
    with self.assertRaises(ValueError):a.token_features(records,m,sc,u)
   bad=copy.deepcopy(t);bad[0]['global_batch_contribution']*=4
   with self.assertRaises(ValueError):a.token_features([h]+bad,m,sc,u)
 def test_answer_NA_and_signed_region_loss_not_filledzero(self):
  with tempfile.TemporaryDirectory() as d:
   _,_,_,sc,u,m,h,t=fixture(Writer(d));h['token_regions']=['think_text']*2
   for r in t:r['region']='think_text'
   measured=a.token_features([h]+t,m,sc,u);self.assertIsNone(measured['answer_gap']);self.assertIsNone(measured['regions']['final_answer']['mean_post']);self.assertEqual(measured['regions']['final_answer']['post_contribution'],0)
   t[0]['post_clip_full_vocabulary_loss']=-1.;t[0]['global_batch_contribution']=-.05;m['post_loss']=1.;m['weighted_post_loss']=.1
   measured=a.token_features([h]+t,m,sc,u);self.assertAlmostEqual(sum(v['weighted_post_contribution'] for v in measured['regions'].values()),.1)
 def test_math_midrank_auc_pearson(self):
  self.assertEqual(a.midranks([2,1,1,3]),[3.,1.5,1.5,4.]);self.assertEqual(a.auc([1,2,3,4],[0,0,1,1]),1.);self.assertEqual(a.auc([2,2],[0,1]),.5);self.assertIsNone(a.auc([1,2],[1,1]));self.assertIsNone(a.pearson([1,1,1],[1,2,3]));self.assertAlmostEqual(a.pearson([1,2,3],[3,2,1]),-1.)
 def test_qblock_bootstrap_NA_cluster_rng(self):
  rows=[dict(question_id=str(i),x=float(i),y=float(2*i)) for i in range(8)]+[dict(question_id='missing',x=None,y=0)];before=random.getstate();r=a.bootstrap_pearson(rows,'x','y',40);self.assertEqual(r['blocks'],9);self.assertEqual(r['interval95'],[1.,1.]);self.assertEqual(before,random.getstate())
  duplicated=[dict(question_id='same',x=float(i),y=float(i)) for i in range(10)];self.assertEqual(a.bootstrap_pearson(duplicated,'x','y',10)['blocks'],1);self.assertIsNone(a.bootstrap_pearson(duplicated,'x','y',10)['interval95'])
 def test_within_vs_between_question(self):
  rows=[]
  for q in range(6):
   rows.append(dict(question_id=str(q),teacher_available=True,sampling_rewards=[1,1,0,0,0],score_siblings=[dict(sibling=i,answer_gap=float(q*100+i),student_entropy=float(q*100+i)) for i in range(5)]))
  r=a.within_question_associations(rows,20)['answer_gap'];self.assertLess(r['equal_question_centered_pearson'],0);self.assertEqual(r['questions_with_at_least_two_valid_scores'],6)
  rows.append(dict(question_id='hard',teacher_available=False,sampling_rewards=[0]*5));r=a.within_question_associations(rows,0)['answer_gap'];self.assertEqual(r['population_questions'],7);self.assertEqual(r['successful_B_questions'],6)
if __name__=='__main__':unittest.main()
