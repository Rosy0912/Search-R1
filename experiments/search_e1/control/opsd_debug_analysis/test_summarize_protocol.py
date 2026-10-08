"""CPU fixtures use actual frozen extraction/reward; never import a model."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('protocol_analysis',HERE/'summarize_protocol.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
# Locate committed frozen source from the repository itself, not test reimplementations.
REPO=HERE.parents[3]
CORE=REPO/'experiments/search_e1/reproduction/runtime/search_e1/core.py'
# Actual frozen source bytes live in retained local analysis snapshots in this
# workspace; fixtures accept an explicit path for repository portability.
import os
WORKSPACE=HERE.parents[5]
CORE=Path(os.environ.get('E1_FIXTURE_FROZEN_CORE',str(WORKSPACE/'analysis-output/prep3-pair-mechanism-review-20261008/core.py')))
QA=Path(os.environ.get('E1_FIXTURE_FROZEN_QA',str(WORKSPACE/'analysis-output/opsd-validation-review-20261007/native-qa-protocol.py')))


def generation(response,spans=None,em=0,calls=0):
 return dict(kind='actual_autoregressive_generation',teacher_forced=False,response=response,
  policy_spans=spans if spans is not None else [[0,len(response)]],protocol_em=em,
  search_calls=calls,termination='answer',turns=[dict(generated_token_ids=[1,2],**({'query':'q'} if i<calls else {})) for i in range(max(1,calls))])


class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.runtime=Path(cls.temp.name)/'runtime'
  p=cls.runtime/'search_e1/core.py';p.parent.mkdir(parents=True);shutil.copyfile(CORE,p)
  p=cls.runtime/'upstream/Search-R1/search_r1/qa_protocol.py';p.parent.mkdir(parents=True);shutil.copyfile(QA,p)
  cls.core,cls.qa,_=a.load_science(cls.runtime)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def check(self,g):return a.analyze_generation(g,['Paris'],self.core,self.qa)
 def test_valid_answer_and_closed_think(self):
  r=self.check(generation('<think>check</think><answer>Paris</answer>',em=1))
  self.assertEqual(r['strict_em'],1);self.assertTrue(r['final_answer_protocol_valid']);self.assertFalse(r['invalid_tag_candidate_matches_gold'])
 def test_unclosed_think_literal_is_never_em(self):
  r=self.check(generation('<think>check <answer>Paris</answer>'))
  self.assertEqual(r['strict_em'],0);self.assertFalse(r['final_answer_protocol_valid']);self.assertTrue(r['invalid_tag_candidate_matches_gold']);self.assertTrue(r['invalid_tag_candidate_unclosed_think'])
 def test_earlier_answer_is_not_final(self):
  text='<answer>Paris</answer>outside<search>q</search>';first=len('<answer>Paris</answer>');last=text.index('<search>')
  r=self.check(generation(text,[[0,first],[last,len(text)]],calls=1))
  self.assertFalse(r['invalid_tag_candidate_matches_gold']);self.assertEqual(r['last_action_kind'],'search');self.assertEqual(r['strict_em'],0)
 def test_retrieval_answer_excluded(self):
  text='<information><answer>Paris</answer></information><think>unknown</think>'
  start=text.index('<think>');r=self.check(generation(text,[[start,len(text)]]))
  self.assertFalse(r['invalid_tag_candidate_matches_gold']);self.assertTrue(r['answer_tag_ranges'][0]['information_overlap']);self.assertTrue(r['answer_tag_ranges'][0]['excluded_from_terminal_candidate'])
 def test_info_inside_authored_span_still_excluded(self):
  r=self.check(generation('<information><answer>Paris</answer></information>'))
  self.assertEqual(r['strict_em'],0);self.assertFalse(r['invalid_tag_candidate_matches_gold'])
 def test_multiple_answers_invalid_candidate_only(self):
  r=self.check(generation('<answer>London</answer><answer>Paris</answer>'))
  self.assertEqual(r['strict_em'],0);self.assertTrue(r['invalid_tag_candidate_matches_gold'])
 def test_invalid_provenance_rejected(self):
  with self.assertRaises(ValueError):self.check(generation('<answer>Paris</answer>',[[5,100]],em=1))
 def test_wrong_recorded_em_rejected(self):
  with self.assertRaises(ValueError):self.check(generation('<think><answer>Paris</answer>',em=1))
 def test_no_teacher_is_skip_not_wrong(self):
  r=self.check({'status':'not_applicable_no_successful_reference'})
  self.assertEqual(r['status'],'skipped');self.assertIsNone(r['strict_em'])
 def test_appendable_results_prefix_ignores_partial_line(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'rows';p.write_bytes(b'{"status":"complete"}\n{"status":')
   rows,proof=a.completed_prefix(p);self.assertEqual(len(rows),1);self.assertTrue(proof['trailing_partial_line_skipped']);self.assertFalse(proof['whole_final_file_hash_claimed'])
 def test_complete_case_summary_keeps_invalid_literal_out_of_em_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);casesdir=root/'cases';casesdir.mkdir();run=root/'run';run.mkdir()
   case={'case_id':'toy-case','question_id':'toy-qid','dataset':'nq','group':'correct_wrong','source_train_row':{'row':{'answers':['Paris']}}}
   case['case_semantic_sha256']=a.sha_bytes(a.canonical(case))
   cases=(a.canonical(case)+b'\n');(casesdir/'cases.jsonl').write_bytes(cases)
   (casesdir/'manifest.json').write_text(json.dumps({'cases':{'sha256':a.sha_bytes(cases)}}))
   (run/'config.json').write_text(json.dumps({'run_id':'toy-run'}));worker=run/'worker-0';worker.mkdir();cd=worker/'toy-case';cd.mkdir()
   result={'status':'complete','case_id':'toy-case','source_case_sha256':case['case_semantic_sha256'],'dataset':'nq','group':'correct_wrong','versions':{}}
   for version in a.VERSIONS:
    bad=version=='epoch1';g=generation('<think><answer>Paris</answer>' if bad else '<answer>Paris</answer>',em=0 if bad else 1)
    g.update(active_adapter='disabled' if version=='teacher' else version,privileged_reference=version=='teacher')
    name='teacher-generation.json' if version=='teacher' else 'student-generation-'+version+'.json'
    (cd/name).write_text(json.dumps(g));result['versions'][version]={'generation_em':g['protocol_em']}
   (worker/'case-results.jsonl').write_bytes(a.canonical(result)+b'\n')
   out=root/'out';summary=a.summarize(self.runtime,casesdir,run,out)
   self.assertEqual(summary['completed_cases'],1)
   self.assertEqual(summary['by_version']['epoch1']['strict_correct'],0)
   self.assertEqual(summary['by_version']['epoch1']['invalid_tag_candidate_matches_gold'],1)
   saved=(out/'summary.json').read_bytes()
   with self.assertRaises(FileExistsError):a.summarize(self.runtime,casesdir,run,out)
   self.assertEqual((out/'summary.json').read_bytes(),saved)
 def test_science_source_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);shutil.copytree(self.runtime,root/'runtime');(root/'runtime/search_e1/core.py').write_text('# wrong')
   with self.assertRaises(ValueError):a.load_science(root/'runtime')

if __name__=='__main__':unittest.main()
