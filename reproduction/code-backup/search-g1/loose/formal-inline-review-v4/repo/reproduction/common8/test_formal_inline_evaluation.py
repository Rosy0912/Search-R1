"""Actual evaluator aggregation/dispatch tests; constructed rows are not GPU evidence."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.common8 import formal_inline_evaluation as evaluation
from reproduction.common8.test_configuration import ConfigurationTests
from reproduction.common8.test_calibration_evaluation import canonical_rows


class FormalInlineEvaluationTests(unittest.TestCase):
    def fixture(self,root):
        base,run=ConfigurationTests().fixture(root)
        (base/'logs/policy-safety-calibration.json').write_text('{}')
        rows,ids,sources=canonical_rows()
        frame={'data_source':SimpleNamespace(tolist=lambda:list(sources))}
        expected=(dict(sha256='d'*64,source_counts=dict(nq=256,hotpotqa=256)),frame,ids)
        return base,run,rows,expected

    def test_real_paired_rows_report_accuracy_decline_without_invented_no_drop_gate(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,rows,expected=self.fixture(root)
            with patch.object(evaluation,'active_run',return_value=(base,run)), \
                 patch('reproduction.common8.validation_identity.expected',return_value=expected), \
                 patch('reproduction.common8.formal_inline_safety.verify_plan',return_value=(.01,'a'*64)):
                evaluation.record_development(0,rows,[row['em'] for row in rows])
                after=copy.deepcopy(rows)
                for row in after[256:]:row.update(em=0,f1=.5)
                evaluation.record_development(5,after,[row['em'] for row in after])
                proof=evaluation.verify_pair(base,run)
            self.assertFalse(proof['accuracy_health_gate'])
            self.assertEqual(proof['changes']['weighted_em'],-.5)
            self.assertEqual(proof['changes']['weighted_f1'],-.25)
            self.assertEqual(proof['after']['correct_count'],256)
            self.assertNotIn('healthy_checkpoint_accepted',proof)
            self.assertEqual(set(proof['file_sha256']),{'0','5'})

    def test_forged_scalar_reordered_ids_or_wrong_binding_rejected_by_independent_pair_reader(self):
        for mutation in ('scalar','row_order','run','plan','split'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as root:
                base,run,rows,expected=self.fixture(root)
                with patch.object(evaluation,'active_run',return_value=(base,run)), \
                     patch('reproduction.common8.validation_identity.expected',return_value=expected), \
                     patch('reproduction.common8.formal_inline_safety.verify_plan',return_value=(.01,'a'*64)):
                    for step in (0,5):evaluation.record_development(step,rows,[row['em'] for row in rows])
                    target=base/evaluation.PATHS[5];data=json.loads(target.read_text())
                    if mutation=='scalar':data['weighted_em']=.25
                    elif mutation=='row_order':data['rows'][0],data['rows'][1]=data['rows'][1],data['rows'][0]
                    elif mutation=='run':data['run_id']='other'
                    elif mutation=='plan':data['plan_sha256']='wrong'
                    else:data['validation_sha256']='wrong'
                    target.write_text(json.dumps(data))
                    with self.assertRaises((AssertionError,ValueError)):evaluation.verify_pair(base,run)

    def test_actual_em_disagreement_cannot_publish_development_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,rows,expected=self.fixture(root)
            with patch.object(evaluation,'active_run',return_value=(base,run)), \
                 patch('reproduction.common8.validation_identity.expected',return_value=expected):
                with self.assertRaisesRegex(AssertionError,'actual EM reward'):
                    evaluation.record_development(0,rows,[0]*512)
            self.assertFalse((base/evaluation.PATHS[0]).exists())

    def test_same_trainer_evaluation_restores_stage_loader_even_when_generation_fails(self):
        from reproduction.common8 import validation_identity as validation
        loader=object();trainer=SimpleNamespace(global_steps=5,val_dataloader=loader)
        original_split=validation._ACTIVE_SPLIT;original_sources=validation._ACTIVE_SOURCE_COUNTS
        def failed(**kwargs):
            self.assertEqual(kwargs,dict(inline_development=True))
            trainer.val_dataloader=object();validation._ACTIVE_SPLIT='validation'
            validation._ACTIVE_SOURCE_COUNTS={'fixture':'temporary'}
            raise RuntimeError('real generation failure')
        trainer._validate=failed
        with patch.object(evaluation,'active_run',return_value=(None,None)),self.assertRaisesRegex(RuntimeError,'generation failure'):
            evaluation.evaluate_development(trainer)
        self.assertIs(trainer.val_dataloader,loader)
        self.assertEqual(validation._ACTIVE_SPLIT,original_split)
        self.assertEqual(validation._ACTIVE_SOURCE_COUNTS,original_sources)
        self.assertEqual(trainer.global_steps,5)

    def test_old_diagnostic_scope_never_enters_inline_development(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,*_=self.fixture(root)
            run.pop('policy_safety_mode');(base/'logs/run-manifest.json').write_text(json.dumps(run))
            with patch.dict(os.environ,SEARCH_G1_PROFILE='common8',SEARCH_G1_BASE=str(base)):
                self.assertIsNone(evaluation.active_run())


if __name__=='__main__':unittest.main()
