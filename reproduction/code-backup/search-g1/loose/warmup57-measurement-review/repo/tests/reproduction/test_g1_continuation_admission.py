"""Exercise rejection of stale evaluator identity on a new recovery segment."""
import copy
import tempfile
import unittest
from pathlib import Path
from reproduction.a800.prepare_run import digest
from reproduction.a800.probe_admission import calibrated_version


class ContinuationAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);model=root/'base';checkpoint=root/'checkpoint200'
        model.mkdir();checkpoint.mkdir()
        names=['config.json','model.safetensors.index.json','weights.safetensors','tokenizer.json','tokenizer_config.json']
        for n in names:
            (model/n).write_text('base '+n);(checkpoint/n).write_text('checkpoint '+n)
        for n in ('commit_probe.pkl','boundary_probe.pkl'):(root/n).write_text(n)
        self.run=dict(run_id='new',completed_updates=200,acceptance_update=205,
            actor_checkpoint=str(checkpoint),actor_checkpoint_sha256={n:digest(checkpoint/n) for n in names},
            initial_model=str(model),initial_model_sha256={n:digest(model/n) for n in names})
        proof=dict(question_count=300,group_size=5,closed_book_label='exact_match_no_retrieval',
            reliance_label='normalized_answer_change_after_evidence_deletion',model_authored_answer_spans=True,
            boundary_all_reserved_questions=True,question_ids=list(range(300)),searched_valid_trajectory_count=13)
        report=dict(unique_question_count=300,question_grouped_cv=True,commit_class_counts=[3,10],
            boundary_class_counts=[265,35],commit_cv_auc=.75,boundary_cv_auc=.878)
        self.live=dict(run_id='new',round=200,version='new:200:uuid',encoder=str(checkpoint),
            encoder_sha256=self.run['actor_checkpoint_sha256'].copy(),label_proof=proof,calibration_report=report,
            commit=str(root/'commit_probe.pkl'),boundary=str(root/'boundary_probe.pkl'),
            head_sha256={n:digest(root/n) for n in ('commit_probe.pkl','boundary_probe.pkl')})

    def test_bound200_accepted(self):
        self.assertEqual(calibrated_version(self.run,self.live),'new:200:uuid')

    def test_old_identity_encoder_and_missing_hash_rejected(self):
        for field,value in [('run_id','old'),('round',0),('version','new:150:uuid'),('encoder',self.run['initial_model'])]:
            live=copy.deepcopy(self.live);live[field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError):calibrated_version(self.run,live)
        live=copy.deepcopy(self.live);del live['encoder_sha256']['weights.safetensors']
        with self.assertRaises(AssertionError):calibrated_version(self.run,live)

    def test_changed_checkpoint_rejected(self):
        (Path(self.run['actor_checkpoint'])/'weights.safetensors').write_text('modified')
        with self.assertRaises(AssertionError):calibrated_version(self.run,self.live)


if __name__=='__main__':unittest.main()
