"""Captured native output-kind transition and strict metadata CPU checks only."""
import ast
import copy
from enum import Enum
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import unittest

ROOT = Path(__file__).parent
PINS = {'entrypoints/llm.py': '1393441b1b5cf4750b2e3a72ff30272a7e0db2220fa23ffa459f135265686582',
        'sampling_params.py': 'f379042fe14ca7adb2a74eeb1b746de4583481c2a10416f7fe1b05aba81952bc'}


def require(value, message):
    if not value: raise ValueError(message)


def production_transition():
    module = ast.parse((ROOT/'window_runtime.py').read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == 'sampling_params_transition')
    namespace = dict(require=require)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'exact-production-check.py', 'exec'), namespace)
    return namespace['sampling_params_transition']


def captured_native():
    trees = {}
    for key, name in [('entrypoints/llm.py', 'vllm-063-llm-output-source.txt'),
                      ('sampling_params.py', 'vllm-063-sampling-params-source.txt')]:
        source = (ROOT/'fixtures'/name).read_bytes()
        require(hashlib.sha256(source).hexdigest() == PINS[key], 'Captured source SHA changed')
        trees[key] = ast.parse(source.decode())
    enum = next(n for n in trees['sampling_params.py'].body
                if isinstance(n, ast.ClassDef) and n.name == 'RequestOutputKind')
    namespace = dict(Enum=Enum, __name__='vllm.sampling_params')
    exec(compile(ast.Module(body=[enum], type_ignores=[]), 'captured-enum.py', 'exec'), namespace)
    kind = namespace['RequestOutputKind']
    cls = next(n for n in trees['entrypoints/llm.py'].body
               if isinstance(n, ast.ClassDef) and n.name == 'LLM')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_validate_and_add_requests')
    # Preserve the full exact native method; heavyweight annotations are postponed.
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ast.fix_missing_locations(future)
    class Params:
        def __init__(self):
            self.output_kind=kind.CUMULATIVE
            self.temperature=1.; self.n=1; self.max_tokens=500
    namespace.update(SamplingParams=Params, Sequence=Sequence)
    exec(compile(ast.Module(body=[future, method], type_ignores=[]), 'captured-native-entry.py', 'exec'), namespace)
    return Params, kind, namespace['_validate_and_add_requests'], trees


def metadata(kind):
    return dict(temperature=1., n=1, max_tokens=500, top_p=1., top_k=-1,
                output_kind=dict(enum_class='vllm.sampling_params.RequestOutputKind', value=kind))


class SamplingOutputTransitionTests(unittest.TestCase):
    def setUp(self):
        self.check = production_transition()
        self.sources = {key:dict(sha256=value) for key,value in PINS.items()}

    def test_actual_native_method_changes_only_output_delivery_in_plain_fixture(self):
        Params, kind, method, trees = captured_native()
        params = Params(); before=copy.deepcopy(params.__dict__); requests=[]
        host=SimpleNamespace(_add_guided_params=lambda *args:None,
                             _add_request=lambda *args,**kwargs:requests.append((args,kwargs)))
        method(host, prompts=[{'prompt_token_ids':[17,18]}], params=params,
               lora_request=None, prompt_adapter_request=None,
               guided_options=None, priority=None)
        self.assertEqual(params.output_kind, kind.FINAL_ONLY)
        self.assertEqual({k:v for k,v in params.__dict__.items() if k!='output_kind'},
                         {k:v for k,v in before.items() if k!='output_kind'})
        self.assertEqual(len(requests),1)
        sampling=next(n for n in trees['sampling_params.py'].body
                      if isinstance(n,ast.ClassDef) and n.name=='SamplingParams')
        default=next(n.value for n in sampling.body if isinstance(n,ast.AnnAssign)
                     and isinstance(n.target,ast.Name) and n.target.id=='output_kind')
        self.assertEqual(ast.unparse(default),'RequestOutputKind.CUMULATIVE')

    def test_exact_transition_and_already_final_pass_with_complete_evidence(self):
        for initial in [0,2]:
            before,after=metadata(initial),metadata(2)
            record=self.check(before,after,self.sources)
            self.assertTrue(record['allowed']); self.assertEqual(record['before'],before)
            self.assertEqual(record['after'],after)
            self.assertEqual(record['changed_fields'],['output_kind'] if initial==0 else [])
            self.assertFalse(record['observer_changed_sampling_parameters'])

    def test_distribution_length_stop_and_unknown_field_mutation_all_refuse(self):
        for key,value in [('temperature',.7),('top_p',.8),('n',2),('max_tokens',501),
                          ('stop_token_ids',[17]),('unexpected_field',None)]:
            with self.subTest(key=key):
                before,after=metadata(0),metadata(2);after[key]=value
                record=self.check(before,after,self.sources)
                self.assertFalse(record['allowed']);self.assertIn(key,record['changed_fields'])
                self.assertEqual(record['after'],after)

    def test_wrong_enum_or_source_is_not_an_output_delivery_exception(self):
        for initial,final in [(1,2),(0,0),(0,1)]:
            self.assertFalse(self.check(metadata(initial),metadata(final),self.sources)['allowed'])
        before,after=metadata(0),metadata(2);after['output_kind']['enum_class']='other.RequestOutputKind'
        self.assertFalse(self.check(before,after,self.sources)['allowed'])
        invalid=copy.deepcopy(self.sources);invalid['entrypoints/llm.py']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'exact installed source'):
            self.check(metadata(0),metadata(2),invalid)

    def test_runtime_import_and_evidence_publication_precede_admission(self):
        tree=ast.parse((ROOT/'single_window_runner.py').read_text())
        imported={a.name for n in tree.body if isinstance(n,ast.ImportFrom)
                  and n.module=='window_runtime' for a in n.names}
        self.assertIn('sampling_params_transition',imported)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_window')
        calls=sorted((n.lineno,ast.unparse(n)) for n in ast.walk(fn) if isinstance(n,ast.Call)
                     and isinstance(n.func,ast.Attribute) and n.func.attr=='local_stage')
        labels=[ast.literal_eval(ast.parse(expr,mode='eval').body.args[0]) for _,expr in calls]
        self.assertLess(labels.index('actual sampling transition evidence before admission'),
                        labels.index('original actual sampling distribution unchanged'))


if __name__=='__main__': unittest.main()
