import importlib.util
from pathlib import Path
import unittest
from . import protocol as p

class ProtocolTests(unittest.TestCase):
 def test_retriever_root_and_endpoint(self):
  from .prepare import retriever_endpoint
  for url in ['http://node1:29147','http://node1:29147/','http://node1:29147/retrieve']:
   self.assertEqual(retriever_endpoint(url),'http://node1:29147/retrieve')
 def test_retriever_rejects_unregistered_destination(self):
  from .prepare import retriever_endpoint
  for url in ['http://node2:29147','http://node1:29147/wrong','http://node1:29147/retrieve?topk=1']:
   with self.assertRaises(AssertionError):retriever_endpoint(url)
 def test_prompt(self):
  self.assertEqual(p.prompt(' who? '),p.PREFIX+'who?\n')
  self.assertEqual(p.prompt(' who '),p.PREFIX+'who?\n')
 def test_question(self):
  self.assertEqual(p.question_from_old_prompt([dict(role='user',content='old\nQuestion: who')]),'who')
  with self.assertRaises(ValueError):p.question_from_old_prompt([dict(role='user',content='unknown')])
 def test_official_parser_semantics(self):
  self.assertEqual(p.answer(['unclosed</information><answer>Paris</answer>']),'Paris')
  self.assertEqual(p.answer(['<answer>first</answer>','<answer>second</answer> trailing']),'second')
  self.assertIsNone(p.answer(['<search>q</search>']))
 def test_em(self):
  self.assertEqual(p.em(' The Paris. ',['Paris']),1)
  self.assertEqual(p.em(None,['Paris']),0)
 def test_action(self):
  self.assertEqual(p.action('<think>x</think><search> q </search>'),('search','q'))
  self.assertEqual(p.postprocess('<search>q</search> junk <answer>a</answer>'),'<search>q</search>')
 def test_passages(self):
  self.assertEqual(p.passages([{'document':{'contents':'Title\nBody'}}]),'Doc 1(Title: Title) Body\n')
 def test_fixed_contract(self):
  self.assertEqual(sum(p.COUNTS.values()),51713)
  self.assertEqual(p.SCIENCE['normal_interactions'],4)
  self.assertTrue(p.SCIENCE['final_generation_for_active'])
  self.assertEqual(p.SCIENCE['temperature'],0)
 def test_final_answer_after_four_searches(self):
  spans=['<search>q</search>']*4+['<answer>Paris</answer>']
  self.assertEqual(p.em(p.answer(spans),['Paris']),1)
 def test_no_retrieved_answers_in_scoring_input(self):
  self.assertIsNone(p.answer(['<search>q</search>']))

if __name__=='__main__':unittest.main()
