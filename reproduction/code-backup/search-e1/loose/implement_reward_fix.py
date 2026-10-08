from pathlib import Path
import re
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/answer-reward-fix')
(root/'search_r1/qa_protocol.py').write_text('''"""Dependency-free extraction of terminal, non-nested QA actions."""
import re

_RESERVED = re.compile(r"</?(?:think|search|answer|information)\\b")
_ACTION = re.compile(r"<(search|answer)>(.*?)</\\1>", re.DOTALL)
_EOS = re.compile(r"(?:\\s*(?:<\\|im_end\\|>|<\\|endoftext\\|>|</s>))+$")


def parse_model_action(text, strict_prefix=False):
    """Return (kind, payload) only for a complete standalone protocol action.

    Free-text reasoning prefixes remain compatible with the executor. Structured
    reasoning must be one closed think block, with no nested protocol tags.
    """
    text = _EOS.sub('', text.strip()).strip()
    actions = list(_ACTION.finditer(text))
    if len(actions) != 1:
        return None
    action = actions[0]
    payload = action.group(2).strip()
    if not payload or _RESERVED.search(payload) or text[action.end():].strip():
        return None
    prefix = text[:action.start()].strip()
    if _RESERVED.search(prefix) or (strict_prefix and prefix):
        thought = re.fullmatch(r'<think>(.*?)</think>', prefix, re.DOTALL)
        if thought is None or _RESERVED.search(thought.group(1)):
            return None
    return action.group(1), payload


def extract_final_answer(model_spans):
    """Never join authored spans across observations or reuse an earlier answer."""
    spans = list(model_spans)
    if not spans:
        return None
    action = parse_model_action(spans[-1])
    return action[1] if action is not None and action[0] == 'answer' else None


def extract_legacy_final_answer(text):
    """Preserve the legacy prompt-example convention while rejecting nesting.

    Full-transcript callers cannot establish authorship; masked model spans are
    preferred. The last answer must nevertheless be top-level and terminal.
    """
    answers = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if len(answers) <= 1:
        return None
    candidate = answers[-1]
    stack = []
    for tag in re.finditer(r'<(/?)(think|search|answer|information)>', text[:candidate.start()]):
        closing, kind = tag.groups()
        if closing:
            if not stack or stack[-1] != kind:
                return None
            stack.pop()
        else:
            stack.append(kind)
    if stack:
        return None
    return extract_final_answer([text[candidate.start():]])
''')
for name in ['qa_em.py','qa_em_format.py']:
 p=root/'verl/utils/reward_score'/name
 s=p.read_text()
 s=s.replace('import random','import random\nfrom search_r1.qa_protocol import extract_final_answer, extract_legacy_final_answer',1)
 start=s.index('def extract_solution(');end=s.index('\ndef ',start+5)
 s=s[:start]+'''def extract_solution(solution_str):
    return extract_legacy_final_answer(solution_str)

'''+s[end:]
 if name=='qa_em.py':
  start=s.index('def compute_score_generated_em(');end=s.index('\ndef ',start+5)
  s=s[:start]+'''def compute_score_generated_em(model_spans, ground_truth, format_score=0., score=1.):
    """Score only a legal terminal answer in the final model-authored span."""
    answer = extract_final_answer(model_spans)
    if answer is None:
        return 0.
    return score if em_check(answer, ground_truth['target']) else format_score

'''+s[end:]
 p.write_text(s)
p=root/'tests/reproduction/test_terminal_answer_protocol.py';p.parent.mkdir(parents=True,exist_ok=True)
p.write_text('''"""Regression checks for reward without an executed terminal answer."""
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
''')
