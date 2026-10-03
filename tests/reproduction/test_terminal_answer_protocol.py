"""Regression checks for reward without an executed terminal answer."""
import ast
import pathlib
import runpy
import unittest
from search_r1.qa_protocol import extract_final_answer, extract_legacy_final_answer, parse_model_action

ROOT = pathlib.Path(__file__).resolve().parents[2]
QA = runpy.run_path(str(ROOT / 'verl/utils/reward_score/qa_em.py'))

class TerminalAnswerTests(unittest.TestCase):
    def test_exploits_receive_no_em_even_with_format_reward(self):
        cases = [
            ['<search><answer>Paris</answer></search>'],
            ['<answer><answer>Paris</answer>'],
            ['<answer>Paris</answer><search>next</search>'],
            ['<answer>Paris</answer>', '<search>next</search>'],
            ['<answer>Paris</answer>', ''],
            ['<think><answer>Paris</answer></think>'],
            ['<answer>Paris</answer> trailing'],
            ['<answer>Paris</answer><answer>Paris</answer>'],
            ['<answer>Par', 'is</answer>'],
            ['<information><answer>Paris</answer></information>'],
            ['<answer>Paris</answer><search>unfinished'],
        ]
        for spans in cases:
            with self.subTest(spans=spans):
                self.assertIsNone(extract_final_answer(spans))
                self.assertEqual(QA['compute_score_generated_em'](spans, {'target':['Paris']}, format_score=.5), 0)

    def test_valid_terminal_answers(self):
        for spans in [ ['<answer>The PARIS!</answer>'],
                       ['<search>France capital</search>', '<think>Found it.</think><answer>The PARIS!</answer>'],
                       ['Reasoning. <answer>The PARIS!</answer><|im_end|>'] ]:
            self.assertEqual(QA['compute_score_generated_em'](spans, {'target':['Paris']}), 1)
        self.assertEqual(QA['compute_score_generated_em'](['<answer>London</answer>'], {'target':['Paris']}, format_score=.1), .1)

    def test_g1_contexts_reject_nonterminal_answers(self):
        path = ROOT / 'search_g1/signals.py'
        if not path.exists():
            self.skipTest('No G1 context extractor on this branch')
        contexts = runpy.run_path(str(path))['contexts']
        for text in ['<search><answer>Paris</answer></search>', '<answer><answer>Paris</answer>',
                     '<answer>Paris</answer><search>next</search>']:
            self.assertIsNone(contexts('prompt', text, [[0,len(text)]]))
        text = '<search>capital</search>OBS<answer>Paris</answer>'
        start = text.index('<answer>')
        result = contexts('prompt', text, [[0,23],[start,len(text)]])
        self.assertEqual(result['answer'], 'Paris')
        self.assertEqual(result['commit'], 'prompt' + text[:start+len('<answer>')])

    def test_legacy_api_and_nested_actions(self):
        prompt = '<answer>example</answer> '
        self.assertIsNone(extract_legacy_final_answer('<answer>Paris</answer>'))
        self.assertEqual(extract_legacy_final_answer(prompt + '<answer>Paris</answer>'), 'Paris')
        for response in ['<search><answer>Paris</answer></search>', '<answer><answer>Paris</answer>',
                         '<answer>Paris</answer><search>next</search>']:
            self.assertIsNone(extract_legacy_final_answer(prompt + response))

    def test_f1_and_legal_actions_share_protocol(self):
        path = ROOT / 'search_r1/baselines/bapo.py'
        if not path.exists():
            self.skipTest('No baseline F1 scorer on this branch')
        tree = ast.parse(path.read_text())
        names = {'answer_from_spans', 'legal_action_records'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(extract_final_answer=extract_final_answer, parse_model_action=parse_model_action)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        for text in ['<search><answer>Paris</answer></search>', '<answer><answer>Paris</answer>']:
            self.assertIsNone(namespace['answer_from_spans']([text]))
            self.assertFalse(namespace['legal_action_records']([{'text':text}]))
        self.assertEqual(namespace['answer_from_spans'](['<answer>Paris</answer>']), 'Paris')
        self.assertTrue(namespace['legal_action_records']([{'text':'<search>capital</search>'}, {'text':'<answer>Paris</answer>'}]))

if __name__ == '__main__':
    unittest.main()
