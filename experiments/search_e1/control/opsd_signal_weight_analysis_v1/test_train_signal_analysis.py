"""CPU synthetic fixtures; no remote, Torch, model or real data."""
import copy
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
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
 def test_final_SHA_recheck_same_size_preserved_mtime_coarse_metadata(self):
  # JSON and streamed JSONL must both fail even when a coarse remote stat
  # reports identical metadata after a same-size content rewrite.
  for as_lines in (False,True):
   with self.subTest(lines=as_lines),tempfile.TemporaryDirectory() as d:
    w=Writer(d);ref=w.put([{'a':1}] if as_lines else {'a':1},as_lines)
    path=Path(ref['path']);original=path.stat();metadata=a.Reader.metadata(path);r=a.Reader()
    with patch.object(r,'metadata',return_value=metadata):
     list(r.lines(ref)) if as_lines else r.json(ref)
     r.stable_end() # Unchanged content remains accepted under the same stat.
     replacement=path.read_bytes().replace(b'1',b'2');self.assertEqual(len(replacement),original.st_size)
     path.write_bytes(replacement);os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
     self.assertEqual(path.stat().st_size,original.st_size)
     self.assertEqual(path.stat().st_mtime_ns,original.st_mtime_ns)
     with self.assertRaisesRegex(ValueError,'content/SHA changed'):r.stable_end()
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
  self.assertEqual(a.midranks([2,1,1,3]),[3.,1.5,1.5,4.]);self.assertEqual(a.auc([1,2,3,4],[0,0,1,1]),1.);self.assertIsNone(a.auc([2,2],[0,1]));self.assertIsNone(a.auc([1,2],[1,1]));self.assertIsNone(a.pearson([1,1,1],[1,2,3]));self.assertAlmostEqual(a.pearson([1,2,3],[3,2,1]),-1.)
 def exposure_rows(self,values):
  rows=[]
  for i,value in enumerate(values):
   hit=int(i%2==0)
   rows.append(dict(question_id=str(i),trained=True,teacher_available=True,missing_reason=None,
    answer_gap=value,student_entropy=value,selected_reward=hit,pair_class='CC' if hit else 'CW',
    selected_weight=1.,self_pair=bool(hit and i%3==0),sampling_s_over_5=.4 if hit else .2,
    raw_teacher_kl=1.,post_loss=.5,weighted_post_loss=.5,selected_sibling=2,
    score_normalization=dict(scores=[-2.,-1.,0.,1.,2.],weights=[.1,.55,1.,1.45,1.9],
      centered_scores=[-2.,-1.,0.,1.,2.],raw_maxabs=2.,available_siblings=[0,1,2,3,4],neutral_reason=None)))
  return rows
 def test_quantile_ties_are_never_split_or_forced_to_quarter_count(self):
  rows=self.exposure_rows([0.]*4+[1.]*2+[2.]*2+[3.]*4)
  value=a.quantile_exposure(rows,'answer_gap');b=value['buckets']
  self.assertEqual([b[x]['actual_pairs'] for x in ('LOW','MIDDLE','HIGH')],[4,4,4])
  self.assertEqual((b['LOW']['score_min'],b['LOW']['score_max']),(0.,0.))
  self.assertEqual((b['HIGH']['score_min'],b['HIGH']['score_max']),(3.,3.))
  self.assertAlmostEqual(b['LOW']['fraction_of_measured_actual_pairs'],1/3)
  self.assertTrue(value['definition']['nominal_bucket_counts_not_forced'])
 def test_constant_quantile_all_middle_and_full_population_NA_kept(self):
  rows=self.exposure_rows([.1]*4)
  rows.append(dict(question_id='hard',trained=False,teacher_available=False,missing_reason='all5_wrong_no_successful_B'))
  rows.append(dict(question_id='skip',trained=False,teacher_available=True,missing_reason='selected_D_encoding_skip'))
  value=a.quantile_exposure(rows,'student_entropy')
  self.assertEqual((value['full_population_questions'],value['actual_trained_questions'],value['measured_actual_selected_D_questions']),(6,4,4))
  self.assertEqual(value['ordering_NA_reason'],'zero_feature_variance')
  self.assertEqual(value['buckets']['MIDDLE']['actual_pairs'],4)
  self.assertEqual(value['buckets']['LOW']['actual_pairs'],0);self.assertEqual(value['buckets']['HIGH']['actual_pairs'],0)
  self.assertEqual(value['missing_reason_counts'],{'all5_wrong_no_successful_B':1,'selected_D_encoding_skip':1})
 def test_gap_NA_not_zero_or_fake_weight_exposure(self):
  rows=self.exposure_rows([None,1.,2.,3.]);value=a.quantile_exposure(rows,'answer_gap')
  self.assertEqual(value['measured_actual_selected_D_questions'],3)
  self.assertEqual(value['missing_reason_counts'],{'no_effective_final_answer_payload_tokens':1})
  empty=a.quantile_exposure([dict(trained=False,missing_reason='all5_wrong_no_successful_B')],'answer_gap')
  self.assertEqual(empty['actual_trained_questions'],0)
  self.assertEqual(empty['buckets']['LOW']['weight_sum'],0.);self.assertIsNone(empty['buckets']['LOW']['weight_mean'])
 def test_neutral_axes_distinguish_constant_NA_mean_and_recorded_fallback(self):
  rows=self.exposure_rows([.1,None,0.])
  rows[0]['score_normalization'].update(scores=[.1]*5,weights=[1.]*5,centered_scores=[0.]*5,raw_maxabs=0.)
  rows[1]['score_normalization'].update(scores=[-2.,-1.,None,1.,2.],available_siblings=[0,1,3,4])
  rows[2]['score_normalization'].update(scores=[None]*5,weights=[1.]*5,centered_scores=[0.]*5,
       raw_maxabs=None,available_siblings=[],neutral_reason='sibling_encoding_failure')
  rows.append(dict(trained=False,teacher_available=False,missing_reason='all5_wrong_no_successful_B'))
  value=a.neutral_and_pair_exposure(rows);axes=value['derived_neutral_axes']
  self.assertEqual(value['actual_trained_questions'],3)
  self.assertEqual(value['recorded_group_neutral_reason_counts'],{'none_recorded':2,'sibling_encoding_failure':1})
  self.assertEqual(axes['zero_score_dispersion']['count'],1);self.assertEqual(axes['selected_signal_NA']['count'],2)
  self.assertEqual(axes['selected_weight_approximately_one']['count'],3)
  self.assertEqual(axes['all_five_weights_approximately_one']['count'],2)
  self.assertAlmostEqual(axes['recorded_group_neutral_reason']['rate_over_actual_K'],1/3)
  self.assertAlmostEqual(axes['recorded_group_neutral_reason']['rate_over_full_question_population'],1/4)
  self.assertTrue(value['axes_overlap_and_are_not_recorded_receipt_reasons'])
 def test_actual_self_nonself_exposure_uses_weights_not_fake_population(self):
  rows=self.exposure_rows([0.,1.]);rows[0].update(selected_weight=.1,weighted_post_loss=.05)
  rows[1].update(selected_weight=1.9,weighted_post_loss=.95)
  rows.append(dict(trained=False,teacher_available=False,missing_reason='all5_wrong_no_successful_B'))
  value=a.neutral_and_pair_exposure(rows)['self_vs_nonself']
  self.assertEqual(value['self']['actual_pairs'],1);self.assertAlmostEqual(value['self']['weight_sum'],.1)
  self.assertEqual(value['nonself']['actual_pairs'],1);self.assertAlmostEqual(value['nonself']['weight_sum'],1.9)
  self.assertEqual(value['nonself']['CC_CW_counts'],{'CC':0,'CW':1})
  self.assertTrue(value['self']['weighted_exposure_is_scalar_objective_scaling_not_gradient_mass'])
 def test_weighting_normalization_NA_not_actual_training_forward_NA(self):
  rows=self.exposure_rows([.1]);rows[0]['variant']='G_POS'
  rows[0]['score_normalization'].update(scores=[None]*5,weights=[1.]*5,
    raw_maxabs=None,available_siblings=[],neutral_reason='sibling_encoding_failure')
  axes=a.neutral_and_pair_exposure(rows)['derived_neutral_axes']
  self.assertEqual(axes['normalization_selected_signal_NA']['count'],1)
  self.assertEqual(axes['selected_signal_NA']['count'],1)
  self.assertEqual(axes['actual_forward_selected_signal_NA']['count'],0)
  self.assertEqual(axes['actual_forward_answer_gap_NA']['count'],0)
  self.assertEqual(axes['actual_forward_entropy_NA']['count'],0)
  rows[0].update(variant='H_POS',answer_gap=None)
  axes=a.neutral_and_pair_exposure(rows)['derived_neutral_axes']
  self.assertEqual(axes['actual_forward_selected_signal_NA']['count'],0)
  self.assertEqual(axes['actual_forward_answer_gap_NA']['count'],1)
 def test_small_positive_dispersion_is_floor_attenuated_not_zero_or_fallback(self):
  rows=self.exposure_rows([0.]);norm=rows[0]['score_normalization']
  norm.update(scores=[-2e-8,-1e-8,0.,1e-8,2e-8],centered_scores=[-2e-8,-1e-8,0.,1e-8,2e-8],
    raw_maxabs=2e-8,numerical_floor=1e-6,weights=[.982,.991,1.,1.009,1.018])
  value=a.neutral_and_pair_exposure(rows);axes=value['derived_neutral_axes']
  self.assertEqual(axes['zero_score_dispersion']['count'],0)
  self.assertEqual(axes['recorded_group_neutral_reason']['count'],0)
  self.assertEqual(axes['floor_attenuated_nonzero_dispersion']['count'],1)
  self.assertEqual(axes['floor_attenuated_nonzero_dispersion']['data_available_denominator'],1)
  self.assertEqual(axes['all_five_weights_exactly_one']['count'],0)
  self.assertEqual(axes['incidental_selected_weight_one_in_nonconstant_group']['count'],1)
  self.assertTrue(value['floor_positive_dispersion_is_attenuated_not_neutral'])
 def test_floating_constant_feature_pearson_auc_and_bootstrap_all_NA(self):
  rows=[dict(question_id=str(i),x=.1,y=i%2) for i in range(8)]
  self.assertIsNone(a.pearson([.1]*8,[0,1]*4));self.assertIsNone(a.auc([.1]*8,[0,1]*4))
  value=a.bootstrap_pearson(rows,'x','y',20)
  self.assertEqual(value['NA_reason'],'zero_feature_variance');self.assertIsNone(value['interval95']);self.assertEqual(value['valid_replicates'],0)
 def test_incidental_near_one_keeps_exact_and_approx_separate(self):
  rows=self.exposure_rows([0.]);rows[0]['selected_weight']=1.+5e-13
  rows[0]['score_normalization']['weights'][2]=rows[0]['selected_weight']
  axes=a.neutral_and_pair_exposure(rows)['derived_neutral_axes']
  self.assertEqual(axes['incidental_selected_weight_one_in_nonconstant_group']['count'],0)
  self.assertEqual(axes['incidental_selected_weight_approximately_one_in_nonconstant_group']['count'],1)
  self.assertEqual(axes['selected_weight_exactly_one']['count'],0)
  self.assertEqual(axes['selected_weight_approximately_one']['count'],1)
 def test_within_question_floating_constant_is_zero_dispersion_NA(self):
  rows=[dict(question_id=str(i),teacher_available=True,sampling_rewards=[1,1,0,0,0],
    score_siblings=[dict(sibling=j,answer_gap=.1,student_entropy=.1) for j in range(5)]) for i in range(4)]
  value=a.within_question_associations(rows,20)['answer_gap']
  self.assertEqual(value['equal_question_centered_covariance'],0.)
  self.assertIsNone(value['equal_question_centered_pearson']);self.assertIsNone(value['interval95'])
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
