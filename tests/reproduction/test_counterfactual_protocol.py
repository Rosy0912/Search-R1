import ast
from pathlib import Path
import unittest
from search_r1.qa_protocol import parse_model_action
from search_g1.calibration_evidence import invalid_action_feedback, validate_provenance, evidence_deleted_prefix

ROOT=Path(__file__).resolve().parents[2]
tree=ast.parse((ROOT/'ts_probe_dump.py').read_text());fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('counterfactual_answer','counterfactual_labels','rollouts_batched')]
ns=dict(parse_model_action=parse_model_action,normalize=lambda x:x.lower().strip(),
        invalid_action_feedback=invalid_action_feedback,validate_provenance=validate_provenance)
exec(compile(ast.Module(body=fns,type_ignores=[]),str(ROOT/'ts_probe_dump.py'),'exec'),ns)
class CounterfactualProtocol(unittest.TestCase):
    def test_prefilled_answer_payload_and_source_forced_close_preserved(self):
        for text in ('Paris</answer>','Paris','The PARIS!</answer>','Paris</answer><|im_end|>'):
            self.assertEqual(ns['counterfactual_answer'](text),text.split('</answer>')[0])
    def test_nested_or_search_answer_is_not_a_usable_counterfactual_answer(self):
        for text in ('<answer>Paris</answer>','<search><answer>Paris</answer></search>','Paris</answer> trailing','<information>Paris</information>'):
            self.assertEqual(ns['counterfactual_answer'](text),'')
    def test_actual_labels_preserve_source_failure_as_flip_and_report_validity(self):
        labels, valid = ns['counterfactual_labels'](['Paris</answer>', 'Rome</answer>', '<search>Paris</search>'], ['Paris']*3)
        self.assertEqual(labels, [0,1,1])
        self.assertEqual(valid, [True,True,False])
        with self.assertRaises(ValueError): ns['counterfactual_labels'](['Paris'], [])
    def test_malformed_search_never_retrieves_or_continues(self):
        for text in ('Paris</search>', '<search><answer>Paris</answer></search>', '<search>Paris</search> trailing'):
            calls=[]
            ns['gen_batch']=lambda *a, **k: [text]
            ns['retrieve_batch']=lambda q: calls.append(q) or ['evidence']
            gen,ranges,count=ns['rollouts_batched'](None,['prompt'],False,return_counts=True)
            self.assertEqual(gen,[text]); self.assertEqual(count,[0]); self.assertEqual(calls,[])
    def test_valid_search_executes_then_final_answer(self):
        rounds=iter([['<search>Paris</search>'], ['<answer>Paris</answer>']]); calls=[]
        ns['gen_batch']=lambda *a, **k: next(rounds)
        ns['retrieve_batch']=lambda q: calls.append(q) or ['evidence']
        gen,ranges,count=ns['rollouts_batched'](None,['prompt'],False,return_counts=True)
        self.assertEqual(calls,[['Paris']]); self.assertEqual(count,[1]); self.assertEqual(len(ranges[0]),2)
    def test_optin_correction_recovers_search_and_keeps_typed_feedback(self):
        rounds=iter([['plain reasoning'],['<search>Paris</search>'],['<answer>Paris</answer>']]); calls=[]
        ns['gen_batch']=lambda *a, **k: next(rounds)
        ns['retrieve_batch']=lambda q: calls.append(q) or ['actual evidence']
        texts,models,counts,evidence,corrections=ns['rollouts_batched'](
            None,['prompt'],False,max_new=500,correction_feedback=True,return_provenance=True)
        self.assertEqual(calls,[['Paris']]);self.assertEqual(counts,[1]);self.assertEqual(len(corrections[0]),1)
        end=models[0][-1][0]+len('<answer>')
        deleted=evidence_deleted_prefix(texts[0],end,models[0],evidence[0],corrections[0])
        self.assertIn(invalid_action_feedback(),deleted);self.assertNotIn('actual evidence',deleted)
        self.assertTrue(deleted.endswith('<answer>'))
    def test_correction_cannot_extend_final_turn(self):
        budgets=[];calls=[]
        ns['gen_batch']=lambda *a, **k: budgets.append(a[2]) or ['invalid']
        ns['retrieve_batch']=lambda q: calls.append(q) or ['evidence']
        text,models,counts,evidence,corrections=ns['rollouts_batched'](
            None,['prompt'],False,max_new=500,correction_feedback=True,return_provenance=True)
        self.assertEqual(budgets,[500]*5);self.assertEqual(len(corrections[0]),4)
        self.assertEqual(counts,[0]);self.assertEqual(calls,[])
    def test_search_in_final_turn_never_calls_retriever(self):
        round_number=[0];calls=[]
        def generate(*a,**k):
            round_number[0]+=1
            return ['<search>Paris</search>' if round_number[0]==5 else 'invalid']
        ns['gen_batch']=generate;ns['retrieve_batch']=lambda q: calls.append(q) or ['evidence']
        text,models,counts,evidence,corrections=ns['rollouts_batched'](
            None,['prompt'],False,correction_feedback=True,return_provenance=True)
        self.assertEqual(round_number[0],5);self.assertEqual(calls,[]);self.assertEqual(counts,[0])
if __name__=='__main__':unittest.main()
