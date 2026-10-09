"""Only three necessary tiny CPU contract fixtures; no Torch/model/HTTP."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import controller,completion,prepare,summary,utils


class NecessaryContracts(unittest.TestCase):
    def test_main_not_complete_spawns_zero(self):
        value=dict(schema='opsd-legacy-mandatory-closure-inputs-v1',state='NOT_READY',study_id='s',groups={})
        c=object.__new__(controller.Controller);c.config=dict(run_id='l',mandatory_completion={});c.ref={}
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'','E1_RUN_ID':'l'}), \
             patch.object(prepare,'validate_config'),patch.object(controller.gates,'publication'), \
             patch.object(completion,'checked_json',return_value=(value,{})),patch.object(controller.subprocess,'Popen') as spawn:
            with self.assertRaises(utils.Hold):c.run()
            spawn.assert_not_called()

    def test_exact_qualified99_and_wrong_identity_rejected(self):
        cfg=dict(qualification=dict(path='/qualification',sha256=utils.QUALIFICATION_SHA),
            checkpoint99=dict(path=utils.TARGET_PATH,sha256=utils.TARGET_SHA))
        source=dict(sha256=utils.SOURCE_MANIFEST_SHA);runtime=dict(sha256=utils.RUNTIME_SHA);initial=dict(sha256='initial')
        train=dict(source_model_manifest=source,runtime_manifest=runtime,common_initial_adapter_manifest=initial)
        value=dict(schema='optional-L-adapter-only-qualification-v1',accepted=True,historical_run_id=utils.HISTORICAL_RUN,
            label='L_historical_forward_outer10_Adam99',target=cfg['checkpoint99'],actual_tensor_sha256=utils.L_TENSOR_SHA,
            adapter_tensors=504,adapter_finite_fp32=True,common_actual_tensor_sha256=utils.COMMON_TENSOR_SHA,
            cuda_initialized=False,full_Adam_or_RNG_read=False,source_model_manifest=source,runtime_manifest=runtime,common_initial=initial)
        with patch.object(prepare,'checked_json',return_value=(value,{})),patch.object(prepare,'original_config',return_value=(train,{})):
            self.assertIs(prepare.qualification(cfg)[0],value)
            wrong=copy.deepcopy(cfg);wrong['checkpoint99']['path']=utils.TARGET_PATH.replace('00000099','00000374')
            with self.assertRaises(utils.Hold):prepare.qualification(wrong)
            wrong=copy.deepcopy(value);wrong['actual_tensor_sha256']='0'*64
            with patch.object(prepare,'checked_json',return_value=(wrong,{})),self.assertRaises(utils.Hold):prepare.qualification(cfg)

    def test_unique512_extra_exclusion_and_paired(self):
        data=[dict(question_id=str(i),question='question '+str(i),answers=['answer'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
        cfg=dict(run_id='l',study_id='s',variant='L',evaluation_role='historical_forward_appendix',scientific=dict(versions=list(utils.VERSIONS)))
        rows=[dict(primary=True,status='complete',run_id='l',version=utils.VERSIONS[0],validation_index=i,rank=i%8,slot=i//8,
            question_id=str(i),dataset=d['data_source'],row_semantic_sha256=utils.object_sha(d),protocol_em=1,
            protocol_valid_final_action=True,failure_class=None,search_calls=1,termination='answer',
            hook_control=dict(accepted=True,excluded_from_primary_denominator=True)) for i,d in enumerate(data)]
        result=summary.aggregate(rows,data,cfg,verify_traces=False)
        self.assertEqual(result['primary_slots'],512);self.assertEqual(result['models'][utils.VERSIONS[0]]['overall']['strict_hits'],512)
        self.assertEqual(result['excluded_extra_actual_generations']['hook_null_effect_controls'],8)
        self.assertEqual(summary.paired(rows,rows,1000)['net_hits'],0)
        with self.assertRaises(utils.Hold):summary.aggregate(rows+[copy.deepcopy(rows[0])],data,cfg,verify_traces=False)
        bad=copy.deepcopy(rows);bad[-1]=copy.deepcopy(bad[0])
        with self.assertRaises(utils.Hold):summary.aggregate(bad,data,cfg,verify_traces=False)


if __name__=='__main__':unittest.main()
