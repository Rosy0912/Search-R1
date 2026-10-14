import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .study_evaluations import collect, collect_run


class StudyEvaluationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def fixture(self, group, step=5):
        run = self.base/'runs'/group
        receipts = run/'receipts';receipts.mkdir(parents=True)
        config = receipts/'resolved.yaml';config.write_text('frozen configuration')
        identity = dict(run_id=group, code_commit='source', data_identity='data',
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
        (receipts/'configuration.json').write_text(json.dumps(dict(
            group=group, source_commit='source', config=str(config),
            **{k:v for k,v in identity.items() if k != 'code_commit'})))
        out = run/'output'/'attempt-test';folder=out/'evaluations';folder.mkdir(parents=True)
        (out/'fresh-initialization.json').write_text(json.dumps(dict(job_id='1', **identity)))
        rows = [dict(question_id=str(i),source='nq',em=float(i%2)) for i in range(512)]
        raw = ''.join(json.dumps(row)+'\n' for row in rows).encode()
        path = folder/f'dev512-outer-{step:06d}.jsonl';path.write_bytes(raw)
        path.with_suffix('.json').write_text(json.dumps(dict(**identity,job='1',
            split='dev512',completed_outer=step,count=512,em=.5,
            rows_sha256=hashlib.sha256(raw).hexdigest(),
            ordered_ids_sha256=hashlib.sha256(json.dumps([r['question_id'] for r in rows],ensure_ascii=False).encode()).hexdigest())))
        return run,path

    def test_missing_milestones_are_null_and_reused_controls_are_not_new_runs(self):
        self.fixture('original');self.fixture('C1-root-only')
        result = collect(self.base)
        self.assertIsNone(result['runs']['original']['dev_curve']['150'])
        self.assertIsNone(result['paired_against_original']['C1-root-only']['150'])
        self.assertEqual(result['paired_against_original']['C1-root-only']['5']['overall']['both_correct'],256)
        self.assertNotIn('budget-10',result['runs'])
        self.assertFalse(result['goal_completion_assessed'])
        self.assertFalse(result['runs']['original']['all_planned_evaluations_present'])

    def test_wrong_run_cannot_be_added_to_curve(self):
        run,path=self.fixture('original')
        p=path.with_suffix('.json');meta=json.loads(p.read_text());meta['run_id']='foreign'
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'frozen run'):collect_run(run,'original')

    def test_wrong_physical_job_is_rejected(self):
        run,path=self.fixture('original')
        p=path.with_suffix('.json');meta=json.loads(p.read_text());meta['job']='2'
        p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'physical attempt'):collect_run(run,'original')

    def test_incomplete_native_write_remains_pending(self):
        run,path=self.fixture('original');path.with_suffix('.json').unlink()
        result=collect_run(run,'original')
        self.assertIsNone(result['dev_curve']['5'])
        self.assertEqual(result['pending_evaluation_files'],[str(path.resolve())])

    def test_duplicate_physical_attempt_does_not_silently_replace_result(self):
        run,path=self.fixture('original')
        shutil.copytree(path.parent.parent,run/'output/attempt-duplicate')
        with self.assertRaisesRegex(ValueError,'duplicate'):collect_run(run,'original')

    def test_extra_evaluation_is_not_a_planned_milestone(self):
        run,_=self.fixture('original',step=10)
        with self.assertRaisesRegex(ValueError,'schedule'):collect_run(run,'original')


if __name__ == '__main__':unittest.main()
