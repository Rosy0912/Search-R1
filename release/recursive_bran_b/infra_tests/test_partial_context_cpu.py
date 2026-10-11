import os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from search_r1.baselines.tree_partial_state import partial_context
from search_r1.baselines.partial_worker_state import _validated_context

class Context(unittest.TestCase):
 def test_actual_driver_context_accepted_by_actual_worker_and_wrong_run_refused(self):
  for enabled in (False,True):
   with tempfile.TemporaryDirectory() as root:
    run='f'*32
    env=dict(SLURM_JOB_ID='hkust-direct-'+run,SEARCH_R1_BASELINE_RUN_ID=run,SEARCH_R1_BASELINE_COMMIT='c'*40,SEARCH_R1_BASELINE_CONFIG_SHA256='a'*64,SEARCH_R1_BASELINE_DATA_IDENTITY='d'*64,SEARCH_R1_MODEL_IDENTITY='official',SEARCH_R1_BASELINE_OUTPUT=root+'/output')
    if enabled:env['RECURSIVE_B_CONTINUATION_BINDING']=root+'/binding.json'
    actor=dict(fresh200_protocol='tree_fresh200_five_leaf_v1',guard_continue_protocol='tree_guard_continue_fresh_rollout_v1')
    trainer=NS(global_steps=4,actual_outer_updates=3,training_attempts=4,config=NS(baselines=NS(training_protocol='tree_fresh200_five_leaf_v1'),actor_rollout_ref=NS(actor=actor),trainer=NS(default_local_dir=root+'/checkpoints')))
    with patch.dict(os.environ,env,clear=True):
     c=partial_context(trainer);data=NS(meta_info=dict(tree_partial_context=c,baseline_outer_step=4,baseline_attempt_id=4));worker=NS(config=NS(actor=actor))
     self.assertEqual(_validated_context(worker,data),c)
     data.meta_info['tree_partial_context']['directory'] += '_wrong'
     with self.assertRaisesRegex(ValueError,'Partial state directory'):_validated_context(worker,data)
if __name__=='__main__':unittest.main()
