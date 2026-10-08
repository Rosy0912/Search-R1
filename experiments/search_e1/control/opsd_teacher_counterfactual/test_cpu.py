"""Synthetic CPU fixtures compare actual unmodified frozen SearchAgent code.

No real model, torch, tokenizer package, GPU, HTTP or original845k data is loaded.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import state as s
import references as refs
import prepare_cpu
from prepare_cpu import science

RUNTIME = Path(os.environ.get('M3_TEST_FROZEN_RUNTIME', str(Path(__file__).resolve().parents[6] / 'work/search-e1-analysis-output/opsd-preparation3-xformers-backup-20261008/snapshot/runtime/opsd-preparation3-xformers')))
ROLLOUT_SHA = 'a88506797ea7fe24e61e1ad53ba4487aef87bdd69ec395aef26c94973b3d629a'


class Tensor:
    def __init__(self, rows): self.rows = rows; self.shape = (len(rows), len(rows[0]))
    def to(self, device): return self
    def __getitem__(self, key):
        if isinstance(key, tuple): return self.rows[key[0]][key[1]]
        return self.rows[key]


class Tokenizer:
    eos_token_id = 0
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return 'CHAT:' + messages[0]['content'] + ':ASSISTANT:'
    def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
    def decode(self, ids, skip_special_tokens=False): return ''.join(chr(i) for i in ids if not skip_special_tokens or i != 0)
    def __call__(self, text, **kwargs): return SimpleNamespace(input_ids=Tensor([self.encode(text)]))


class Model:
    def __init__(self, tokenizer, segments): self.tokenizer = tokenizer; self.segments = list(segments); self.inputs = []; self.options = []
    def eval(self): return self
    def parameters(self): return iter([SimpleNamespace(device='cpu')])
    def generate(self, ids, **kwargs):
        self.inputs.append(self.tokenizer.decode(ids.rows[0])); self.options.append(kwargs)
        if not self.segments: raise AssertionError('Extra generation opportunity')
        return Tensor([ids.rows[0] + self.tokenizer.encode(self.segments.pop(0))])


class Requests:
    def __init__(self): self.calls = []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        passages = [dict(document=dict(contents='fixture observation ' + str(i))) for i in range(3)]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(result=[passages]))


class CPUFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUNTIME.exists(): raise RuntimeError('Actual frozen runtime required; no silently skipped comparator')
        cls.core, cls.qa, _ = science(RUNTIME)
        raw = (RUNTIME / 'search_e1/rollout.py').read_bytes()
        if hashlib.sha256(raw).hexdigest() != ROLLOUT_SHA: raise AssertionError('Wrong frozen SearchAgent source')
        tree = ast.parse(raw)
        definitions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('StopTags', 'SearchAgent')], type_ignores=[])
        cls.globals = dict(re=re, torch=SimpleNamespace(no_grad=lambda: lambda function: function),
                           StoppingCriteria=object, StoppingCriteriaList=list, prompt=cls.core.prompt)
        exec(compile(definitions, str(RUNTIME / 'search_e1/rollout.py'), 'exec'), cls.globals)
        cls.Agent = cls.globals['SearchAgent']

    def native(self, segments, max_context=8192):
        tokenizer = Tokenizer(); model = Model(tokenizer, segments); requests = Requests()
        self.globals['requests'] = requests
        result = self.Agent(model, tokenizer, 'fixture://no-http', max_context=max_context).run('fixture question', temperature=0)
        result['reward'] = self.core.reward(result['response'], ['x'], result['policy_spans'])
        return result, model, requests, tokenizer

    def compare_transition(self, segments):
        row, model, requests, tokenizer = self.native(segments)
        state = s.initial(); prefix = tokenizer.apply_chat_template(self.core.prompt('fixture question'))
        for i, (a, b) in enumerate(row['policy_spans']):
            self.assertEqual(s.context(state, prefix), model.inputs[i])
            end = row['policy_spans'][i + 1][0] if i + 1 < len(row['policy_spans']) else len(row['response'])
            observation = row['response'][b:end]
            if i + 1 == 4: observation = observation[:-len(s.FORCED)]
            segment = row['response'][a:b]
            has_search = '</answer>' not in segment and re.search(r'<search>(.*?)</search>', segment, re.S) and state['calls_completed'] < 4
            state = s.advance(state, segment, observation if has_search else None)
        self.assertEqual(state['response_before_turn'], row['response'])
        self.assertEqual(state['policy_spans'], [list(x) for x in row['policy_spans']])
        self.assertEqual(state['calls_completed'], row['search_calls'])
        self.assertEqual(state['terminal'], row['termination'])
        return row, model, requests

    def test_direct_answer_native_exact_and_empty_history_label(self):
        row, _, _ = self.compare_transition(['<think>a</think><answer>x</answer>'])
        result = s.reconstruct_original(row, 'hf')
        self.assertEqual(len(result['executable_states']), 1)
        self.assertEqual(result['skipped_boundaries'][0]['reason'], 'terminal_boundary_skip:answer')
        self.assertEqual(result['executable_states'][0]['history_label'], 'empty_no_failure_history')

    def test_search_observation_native_exact_and_alias_dedup(self):
        row, _, requests = self.compare_transition(['<think>a</think><search>q</search>', '<answer>x</answer>'])
        result = s.reconstruct_original(row, 'hf')
        state = result['executable_states'][0]
        self.assertEqual(len(result['executable_states']), 1); self.assertEqual(len(state['aliases']), 2)
        self.assertEqual((state['next_turn'], state['calls_completed'], state['remaining_turns']), (1, 1, 4))
        self.assertEqual(len(requests.calls), 1)

    def test_four_searches_then_forced_final_native_exact(self):
        row, model, _ = self.compare_transition(['<search>q</search>'] * 4 + ['<answer>x</answer>'])
        result = s.reconstruct_original(row, 'hf')
        last = next(x for x in result['executable_states'] if 'before_last_policy_generation' in x['aliases'])
        self.assertTrue(last['forced_pending']); self.assertEqual(last['remaining_turns'], 1)
        self.assertTrue(model.inputs[4].endswith(s.FORCED))
        self.assertFalse(last['response_before_turn'].endswith(s.FORCED))

    def test_invalid_first_is_terminal_never_advanced(self):
        row, _, requests = self.compare_transition(['no action'])
        result = s.reconstruct_original(row, 'hf')
        self.assertEqual(result['skipped_boundaries'][0]['reason'], 'terminal_boundary_skip:invalid_or_budget')
        terminal = s.advance(s.initial(), 'no action')
        with self.assertRaises(ValueError): s.advance(terminal, '<answer>x</answer>')
        self.assertFalse(requests.calls)

    def test_bad_think_search_executes_but_bad_final_strict_fails(self):
        row, _, requests = self.compare_transition(['<think>a<search>q</search>', '<think>a<answer>x</answer>'])
        self.assertEqual(row['termination'], 'answer'); self.assertEqual(row['reward'], 0)
        self.assertEqual(len(requests.calls), 1)
        self.assertTrue(s.reconstruct_original(row, 'hf')['executable_states'][0]['failure_history'])

    def test_search_budget_four_is_terminal_no_extra_http(self):
        row, _, requests = self.compare_transition(['<search>q</search>'] * 5)
        self.assertEqual(row['termination'], 'invalid_or_budget'); self.assertEqual(len(requests.calls), 4)
        self.assertEqual(row['search_calls'], 4)

    def test_context_exact_threshold_and_one_over_native(self):
        prefix = Tokenizer().apply_chat_template(self.core.prompt('fixture question'))
        good, model, _ , _= self.native(['<answer>x</answer>'], len(prefix) + 500)
        self.assertEqual(good['termination'], 'answer'); self.assertEqual(len(model.inputs), 1)
        bad, model, _, _ = self.native([], len(prefix) + 499)
        self.assertEqual(bad['termination'], 'context_limit'); self.assertFalse(model.inputs)
        self.assertFalse(s.reconstruct_original(bad, 'hf')['executable_states'])

    def test_cut_keeps_actual_ids_and_posthoc_distinction(self):
        row, model, _, _ = self.native(['<answer>x</answer>EXTRA\x00'])
        self.assertEqual(row['response'], '<answer>x</answer>')
        self.assertEqual(s.cut('<answer>x</answer>EXTRA\x00'), row['response'])
        self.assertNotEqual(len(Tokenizer().encode('<answer>x</answer>EXTRA\x00')), len(Tokenizer().encode(row['response'])))

    def test_missing_observation_and_bad_spans_rejected(self):
        with self.assertRaisesRegex(ValueError, 'observation'): s.advance(s.initial(), '<search>q</search>')
        row, _, _, _ = self.native(['<search>q</search>', '<answer>x</answer>'])
        altered = copy.deepcopy(row); altered['policy_spans'] = [list(x) for x in altered['policy_spans']]; altered['policy_spans'][1][0] -= 1
        with self.assertRaises(ValueError): s.reconstruct_original(altered, 'hf')
        altered = copy.deepcopy(row); altered['search_calls'] += 1
        with self.assertRaises(ValueError): s.reconstruct_original(altered, 'hf')

    def test_observation_taglike_content_not_naive_regex_truncated(self):
        obs = s.OBS_START + 'quoted </information> and <answer>fixture</answer>' + s.OBS_END
        state = s.advance(s.initial(), '<search>q</search>', obs)
        final = s.advance(state, '<answer>x</answer>')
        row = dict(response=final['response_before_turn'], policy_spans=final['policy_spans'], search_calls=1, termination='answer', reward=1)
        rebuilt = s.reconstruct_original(row)
        self.assertEqual(rebuilt['executable_states'][0]['observations'][0]['inserted_information'], obs)

    def test_vllm_forced_context_tail_distinguished_from_hf(self):
        state = s.initial('vllm')
        for _ in range(4): state = s.advance(state, '<search>q</search>', s.OBS_START + 'd' + s.OBS_END)
        row = dict(response=state['response_before_turn'] + s.FORCED, policy_spans=state['policy_spans'], search_calls=4, termination='context_limit', reward=0)
        rebuilt = s.reconstruct_original(row)
        self.assertTrue(any(x['next_turn'] == 1 for x in rebuilt['executable_states']))
        self.assertTrue(any('context_limit' in x['reason'] for x in rebuilt['skipped_boundaries']))
        with self.assertRaises(ValueError): s.reconstruct_original(row, 'hf')

    def trace(self):
        row, model, _, tok = self.native(['<search>q</search>', '<answer>x</answer>'])
        events = []
        for i, (a, b) in enumerate(row['policy_spans']):
            event = dict(turn=i, forced_information='', context_text=model.inputs[i], context_tokens=len(tok.encode(model.inputs[i])),
                         policy_span=[a, b], policy_segment=row['response'][a:b],
                         generated_token_ids=tok.encode(row['response'][a:b]), decoded_segment_before_structural_cut=row['response'][a:b])
            if i == 0:
                event.update(query='q', inserted_information=row['response'][b:row['policy_spans'][1][0]],
                             retrieval_payload=dict(result=[[{}, {}, {}]]), observation_content_budget=480, inserted_content_token_ids=[1])
            events.append(event)
        return dict(kind='actual_autoregressive_generation', teacher_forced=False, response=row['response'], policy_spans=row['policy_spans'],
                    prompt_text=tok.apply_chat_template(self.core.prompt('fixture question')), search_calls=1, termination='answer', protocol_em=1, turns=events)

    def test_actual_hf_trace_context_observation_ids_and_aliases(self):
        trace = self.trace(); result = s.reconstruct_hf(trace)
        self.assertEqual(len(result['executable_states'][0]['aliases']), 2)
        altered = copy.deepcopy(trace); altered['turns'][1]['context_text'] += 'wrong'
        with self.assertRaises(ValueError): s.reconstruct_hf(altered)
        altered = copy.deepcopy(trace); del altered['turns'][0]['generated_token_ids']
        with self.assertRaisesRegex(ValueError, 'IDs missing'): s.reconstruct_hf(altered)

    def test_reference_final_only_deletion_keeps_information_answers(self):
        prefix = s.OBS_START + '<answer>fixture info</answer>' + s.OBS_END
        policy = '<think>a</think><answer>x</answer>'
        row = dict(response=prefix + policy, policy_spans=[[len(prefix), len(prefix) + len(policy)]])
        deleted, proof = refs.delete_final(row, self.qa)
        self.assertIn('<answer>fixture info</answer>', deleted); self.assertNotIn('<answer>x</answer>', deleted)
        self.assertEqual(deleted, prefix + '<think>a</think>')

    def cases(self, same_answer=False):
        cases = []
        for dataset, group in refs.CELLS:
            for i in range(32):
                qid = f'{dataset}-{group}-{i}'; answer = 'x' if same_answer else 'fixture' + str(i)
                response = '<answer>' + answer + '</answer>'
                rows = []
                for j in range(5):
                    text = response if j == 0 else '<think>' + str(j) + '</think>' + response if group == 'correct_correct' else '<answer>wrong</answer>'
                    rows.append(dict(row=dict(sibling=j, response=text, reward=int(j == 0 or group == 'correct_correct'),
                                      search_calls=0, token_count=10 + j, policy_spans=[[0, len(text)]], answers=[answer],
                                      question_id=qid, question='fixture question ' + qid, data_source=dataset, termination='answer'),
                             provenance=dict(line=j + 1)))
                case = dict(case_id=qid, question_id=qid, dataset=dataset, group=group,
                            selection_sha256=hashlib.sha256(f'42|{dataset}|{group}|{qid}'.encode()).hexdigest(),
                            original_siblings=rows, original_pair=dict(row=dict(reference=response, student=rows[1]['row']['response'],
                              student_policy_spans=rows[1]['row']['policy_spans'], student_reward=rows[1]['row']['reward'])),
                            source_train_row=dict(row=dict(answers=[answer], question='fixture question ' + qid)))
                case['case_semantic_sha256'] = s.sha(case); cases.append(case)
        return cases

    def test_seed64_pilot16_order_independent_donor_and_none(self):
        cases = self.cases(); selected = refs.select(cases)
        self.assertEqual(len(selected), 64); self.assertEqual(sum(x['pilot'] for x in selected), 16)
        self.assertEqual([x['case']['case_id'] for x in selected], [x['case']['case_id'] for x in refs.select(list(reversed(cases)))])
        conditions = refs.conditions(selected, self.core, self.qa)
        for key, value in conditions.items():
            self.assertIsNone(value['conditions'][1]['reference']); self.assertEqual(value['conditions'][3]['status'], 'ready')
            self.assertNotEqual(value['conditions'][3]['donor_case_id'], key)
        same = refs.conditions(refs.select(self.cases(True)), self.core, self.qa)
        self.assertTrue(all(x['conditions'][3]['status'] == 'skipped' for x in same.values()))

    def test_native_reference_source_order_ties_not_sibling_sort(self):
        case = self.cases()[0]; rows = case['original_siblings']
        rows[1]['row'].update(response='<answer>y</answer>', reward=1, token_count=10, policy_spans=[[0, len('<answer>y</answer>')]], answers=['y'])
        rows[1]['provenance']['line'] = 0; case['original_pair']['row']['reference'] = '<answer>y</answer>'
        self.assertEqual(refs.reference_row(case, self.core)['sibling'], 1)

    def test_no_real_gpu_or_http_dependency_imported(self):
        for path in (Path(s.__file__), Path(refs.__file__), Path(__file__).with_name('prepare_cpu.py')):
            names = {n.name for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Import) for n in node.names}
            self.assertFalse(names & {'torch', 'transformers', 'requests', 'subprocess', 'peft'})

    def test_cpu_prepare_private_output_frozen_identity_and_nooverwrite(self):
        cases = self.cases()
        for dataset in ('nq', 'hotpotqa'):
            for i in range(32):
                c = copy.deepcopy(next(x for x in cases if x['dataset'] == dataset))
                c.update(case_id=f'{dataset}-hard-{i}', question_id=f'{dataset}-hard-{i}', group='all5_wrong', original_pair=None)
                c['case_semantic_sha256'] = s.sha({k: v for k, v in c.items() if k != 'case_semantic_sha256'}); cases.append(c)
        data = b''.join(json.dumps(c, sort_keys=True).encode() + b'\n' for c in cases)
        wanted = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'cases').mkdir(); (root / 'cases/cases.jsonl').write_bytes(data)
            (root / 'cases/manifest.json').write_text(json.dumps(dict(cases=dict(sha256=wanted), selected_cases=192, per_cell=32, seed=42, strict_reward_checked=True)))
            with patch.object(prepare_cpu, 'CASES_SHA', wanted):
                summary = prepare_cpu.prepare(root / 'cases', RUNTIME, root / 'out')
                self.assertEqual((summary['selected_cases'], summary['pilot_cases']), (64, 16))
                self.assertFalse(summary['gpu_admitted']); self.assertFalse(summary['teacher_effect_evaluated'])
                records = [json.loads(x) for x in (root / 'out/states.private.jsonl').read_bytes().splitlines()]
                self.assertEqual(len(records), 64)
                with self.assertRaises(FileExistsError): prepare_cpu.prepare(root / 'cases', RUNTIME, root / 'out')
            (root / 'cases/cases.jsonl').write_bytes(data + b'{}\n')
            with patch.object(prepare_cpu, 'CASES_SHA', wanted):
                with self.assertRaisesRegex(ValueError, 'hash differs'): prepare_cpu.prepare(root / 'cases', RUNTIME, root / 'bad')

    def test_exhausted_loop_state_is_terminal_and_not_resumable(self):
        terminal = dict(s.initial(), next_turn=5, calls_completed=4, terminal='budget')
        with self.assertRaisesRegex(ValueError, 'terminal'): s.context(terminal, 'prefix')

    def test_later_context_limit_does_not_retroactively_terminalize_search_boundary(self):
        st = s.advance(s.initial('vllm'), '<search>q</search>', s.OBS_START + 'd' + s.OBS_END)
        row = dict(response=st['response_before_turn'], policy_spans=st['policy_spans'], search_calls=1, termination='context_limit', reward=0)
        result = s.reconstruct_original(row)
        self.assertEqual(len(result['executable_states']), 1)
        self.assertEqual((result['executable_states'][0]['next_turn'], result['executable_states'][0]['calls_completed']), (1, 1))
        self.assertEqual(result['skipped_boundaries'][0]['reason'], 'terminal_boundary_skip:context_limit')
        trace = self.trace(); first = trace['turns'][0]
        trace.update(response=st['response_before_turn'], policy_spans=st['policy_spans'], termination='context_limit', protocol_em=0)
        # The observation differs from the synthetic full-run trace; bind the
        # first real event to this exact completed response before skipped turn1.
        first['inserted_information'] = s.OBS_START + 'd' + s.OBS_END
        trace['turns'] = [first, dict(turn=1, forced_information='', context_text=trace['prompt_text'] + st['response_before_turn'],
                                    context_tokens=8190, skipped='context_limit_without_independent_truncation')]
        result = s.reconstruct_hf(trace)
        self.assertEqual(len(result['executable_states']), 1)
        self.assertEqual(result['executable_states'][0]['next_turn'], 1)
        self.assertEqual(result['skipped_boundaries'][0]['reason'], 'terminal_boundary_skip:context_limit')


if __name__ == '__main__': unittest.main()
