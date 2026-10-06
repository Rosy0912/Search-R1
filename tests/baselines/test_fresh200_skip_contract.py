import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from search_r1.baselines.fresh200_contract import (validate_rank_updates,scheduler_may_step,
    verify_fixed_policy_safety_artifact,outer_contract)

class SkipContractTests(unittest.TestCase):
 def events(self,status='COMPLETE_WITH_SKIPS',writes=17,outer=49):
  total=outer_contract(outer)['expected_optimizer_steps']
  return [dict(rank=r,actual_outer=outer,attempt_id=11,status=status,update_complete=True,
   attempt_complete=True,postabort=False,optimizer_steps=writes,expected_optimizer_steps=total,
   optimizer_opportunities_total=total,processed_opportunities=total,
   skipped_opportunities=total-writes,unprocessed_opportunities=0) for r in range(8)]
 def test_all_opportunities_processed_with_real_skips_completes(self):
  e=self.events();self.assertTrue(validate_rank_updates(e,49));self.assertTrue(all(scheduler_may_step(x) for x in e))
 def test_whole_skip_and_retained_partial_cannot_tick(self):
  for status,writes,processed in [('SKIPPED',0,0),('SKIPPED',0,20),('ABORTED',7,7),('ABORTED',20,20)]:
   e=self.events(status,writes)
   for x in e:x.update(postabort=status=='ABORTED',update_complete=False,attempt_complete=status!='ABORTED' and processed==20,processed_opportunities=processed,skipped_opportunities=processed-writes,unprocessed_opportunities=20-processed)
   self.assertFalse(validate_rank_updates(e,49));self.assertTrue(all(not scheduler_may_step(x) for x in e))
 def test_fake_zero_write_completion_and_fake_skip_count_rejected(self):
  for field,value in [('optimizer_steps',0),('skipped_opportunities',0),('unprocessed_opportunities',1),('processed_opportunities',19),('attempt_complete',False),('postabort',True),('attempt_id',True)]:
   e=self.events();e[3][field]=value
   with self.assertRaises(ValueError):validate_rank_updates(e,49)
 def test_rank_bool_mixed_writes_and_incomplete_legacy_reject(self):
  for variant in ('rank_bool','mixed_writes','legacy_complete'):
   e=self.events()
   if variant=='rank_bool':e[0]['rank']=False
   elif variant=='mixed_writes':e[7].update(optimizer_steps=18,skipped_opportunities=2)
   else:e[0]=dict(rank=0,actual_outer=49,status='COMPLETE',update_complete=True,optimizer_steps=20,expected_optimizer_steps=20)
   with self.assertRaises(ValueError):validate_rank_updates(e,49)
 def test_fiftieth_uses_ten_opportunities(self):
  e=self.events(writes=8,outer=50);self.assertTrue(validate_rank_updates(e,50))
  self.assertEqual(e[0]['skipped_opportunities'],2)

class FixedThresholdTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name).resolve()/'fixed.json';self.source='1'*40
  self.doc=dict(protocol='BAPO_FRESH200_USER_FIXED_SAFETY_V1',scope='USER_FIXED_THRESHOLDS_NOT_NATIVE_ACCEPTANCE_NOT_EMPIRICAL_CALIBRATION',source_commit=self.source,source_sha256={'test':'2'*64},thresholds=dict(dual_clip=3,sampled_old_policy_kl=.2,negative_advantage_tail_fraction=.2,negative_tail_ratio_boundary=3,kl_log_bound=30.,on_guard='skip_and_continue'),monitor_scopes=['full_frozen_rollout','full_current_minibatch'],model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',data_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',human_goal_sha256='3'*64)
  self.safety=dict(enabled=True,dual_clip=3,target_old_policy_kl=.2,negative_tail_limit=.2,threshold_origin='user_fixed',skip_on_policy_guard=True,kl_log_bound=30.,source_commit=self.source,fixed_threshold_artifact=str(self.path))
 def tearDown(self):self.tmp.cleanup()
 def save(self):
  self.path.write_text(json.dumps(self.doc));self.safety['fixed_threshold_artifact_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest()
 def check(self,dirty=False):
  with patch('subprocess.check_output',side_effect=[self.source+'\n','modified' if dirty else '']),patch('search_r1.baselines.fresh200_contract.critical_source_hashes',return_value={'test':'2'*64}):
   return verify_fixed_policy_safety_artifact(self.safety)
 def test_fixed_identity_returns_threshold_without_fabricating_calibration(self):
  self.save();self.assertEqual(self.check(),(.2,self.safety['fixed_threshold_artifact_sha256']))
 def test_wrong_math_scope_source_and_hash_reject(self):
  for key,value in [('thresholds',{}),('scope','PASS_NATIVE_GPU'),('source_commit','4'*40),('source_sha256',{}),('human_goal_sha256',''),('model_revision','wrong'),('monitor_scopes',['full_current_minibatch'])]:
   with self.subTest(key=key):
    old=copy.deepcopy(self.doc);self.doc[key]=value;self.save()
    with self.assertRaises(ValueError):self.check()
    self.doc=old
 def test_dirty_source_old_threshold_and_fail_exit_reject(self):
  self.save()
  with self.assertRaises(ValueError):self.check(dirty=True)
  for key,value in [('target_old_policy_kl',.01),('negative_tail_limit',None),('skip_on_policy_guard',False),('threshold_origin','calibrated')]:
   old=self.safety[key];self.safety[key]=value
   with self.assertRaises(ValueError):self.check()
   self.safety[key]=old
 def test_artifact_changed_after_binding_reject(self):
  self.save();self.path.write_text('{}')
  with self.assertRaises(ValueError):self.check()

if __name__=='__main__':unittest.main()
