"""Execute actual validator AST branches; synthetic schema, never GPU evidence."""
import ast
import collections
import copy
import hashlib
import itertools
import math
from pathlib import Path
import tempfile
import unittest


class ActualValidatorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tool = Path('work/e1-300/tools/verify_hardware_receipt.py')
        module = ast.parse(self.tool.read_text())
        branch = next(n for n in ast.walk(module) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "kind == 'native-actor'")
        self.code = compile(ast.Module(body=branch.body, type_ignores=[]), str(self.tool), 'exec')
        self.names = {'verl/workers/actor/dp_actor.py', 'search_r1/baselines/objectives.py',
                      'search_r1/baselines/gradient_norm.py', 'verl/trainer/ppo/core_algos.py',
                      'verl/utils/fsdp_utils.py', 'verl/workers/fsdp_workers.py', 'verl/protocol.py',
                      'verl/utils/torch_functional.py', 'search_r1/baselines/distillation.py'}
        for name in self.names | {'../tools/native-actor-fixture.py'}:
            p = self.root / 'repo' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('synthetic_schema_binding')
        self.value = dict(status='PASS_NATIVE_ACTOR_FSDP8', world=8,
                          source_commit='6a7bcec66d6d7a5da527d5e703bec83819dab183',
                          repo=str(self.root / 'repo'), actor_file=str(self.root / 'repo/verl/workers/actor/dp_actor.py'),
                          fixture_sha256=self.sha(self.root / 'tools/native-actor-fixture.py'),
                          source_files_sha256={name: self.sha(self.root / 'repo' / name) for name in self.names})
        records = [self.record('actual_actor_micro_invariance', off, empty, pg, kl, micro)
                   for off, empty, (pg, kl), micro in itertools.product(
                       (False, True), (False, True), ((True, False), (False, True), (True, True)), (1, 2, 4, 16))]
        for off in (False, True):
            records += [self.record('actual_actor_global_empty_step', off, micro=2),
                        self.record('actual_actor_2560_rows_20_updates', off, micro=2, rows=2560)]
            records += [self.record('actual_actor_production_bf16_fp32_micro_invariance', off, micro=m, production=True)
                        for m in (1, 2, 4, 16)]
            records += [self.record('actual_actor_production_bf16_20_updates', off, micro=2, rows=2560, production=True)]
        self.value['records'] = records

    @staticmethod
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    @staticmethod
    def record(case, off=False, empty=False, pg=True, kl=True, micro=1, rows=128, production=False):
        steps = 20 if rows == 2560 else 1
        return dict(case=case, manual_outer_offload=off, empty_rank=empty, pg_on=pg, kl_on=kl,
                    micro_per_rank=micro, native_cpu_offload=False, mini_per_rank=16,
                    optimizer_steps=steps, global_rows=rows, gradients=[[0.] * 16 for _ in range(steps)],
                    max_gradient_error=0., production_bf16_fp32=production,
                    gradient_norm_min=0., gradient_norm_max=0.)

    def evaluate(self, value):
        exec(self.code, dict(v=value, ROOT=self.root, source=self.value['source_commit'],
                             sha=self.sha, pathlib=__import__('pathlib'), collections=collections,
                             itertools=itertools, math=math))

    def reject(self, mutate):
        value = copy.deepcopy(self.value)
        mutate(value)
        with self.assertRaises((AssertionError, KeyError)):
            self.evaluate(value)

    def test_complete_synthetic_schema_is_accepted_not_native_proof(self):
        self.evaluate(self.value)

    def test_duplicate_bf16_micro_matrix_refused(self):
        def mutate(v):
            cases = [x for x in v['records'] if x['case'] == 'actual_actor_production_bf16_fp32_micro_invariance']
            for x in cases:
                x['manual_outer_offload'] = False
                x['micro_per_rank'] = 1
        self.reject(mutate)

    def test_short_gradient_width_refused(self):
        self.reject(lambda v: v['records'][0].update(gradients=[[0.]]))

    def test_20_steps_with_one_gradient_refused(self):
        self.reject(lambda v: next(x for x in v['records'] if x['global_rows'] == 2560).update(gradients=[[0.] * 16]))

    def test_missing_source_binding_refused(self):
        self.reject(lambda v: v['source_files_sha256'].pop('verl/protocol.py'))

    def test_wrong_source_binding_refused(self):
        self.reject(lambda v: v['source_files_sha256'].update({'verl/protocol.py': '0' * 64}))

    def test_negative_error_refused(self):
        self.reject(lambda v: v['records'][0].update(max_gradient_error=-1.))

    def test_wrong_precision_refused(self):
        self.reject(lambda v: next(x for x in v['records'] if x['production_bf16_fp32']).update(production_bf16_fp32=False))

    def test_reversed_norm_range_refused(self):
        self.reject(lambda v: v['records'][0].update(gradient_norm_min=2., gradient_norm_max=1.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
