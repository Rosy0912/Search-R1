"""Exercise actual patched update/validation AST with four CPU rank replicas."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'control'))
from opsd_execution import patch_opsd, _function_text

SOURCE = (ROOT / 'archive/search_e1/opsd.py').read_text()


class Collective:
    """Blocking rank collective: mismatched call order fails rather than hangs."""
    def __init__(self, world):
        self.world = world
        self.condition = threading.Condition()
        self.rounds = {}
        self.steps = [0] * world

    def rank_dist(self, rank):
        def reduce(tensor, op='sum'):
            with self.condition:
                step = self.steps[rank]
                self.steps[rank] += 1
                state = self.rounds.setdefault(step, {'op': op, 'values': {}, 'result': None})
                if state['op'] != op:
                    raise AssertionError('Collective order mismatch')
                state['values'][rank] = tensor.detach().clone()
                if len(state['values']) == self.world:
                    values = torch.stack(list(state['values'].values()))
                    state['result'] = values.amax(0) if op == 'max' else values.sum(0)
                    self.condition.notify_all()
                if not self.condition.wait_for(lambda: state['result'] is not None, timeout=15):
                    raise AssertionError('Collective missing rank')
                tensor.copy_(state['result'])
        return SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MAX='max', SUM='sum'))


def compiled_functions(source, namespace):
    tree = ast.parse(source)
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('_opsd_')]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'start')
    wrapper = ast.parse('def update(model, examples, a, world, rank, optimizer):\n'
                        '    epoch = 0\n    total = 0.\n'
                        '    trainable = [p for p in model.parameters() if p.requires_grad]\n'
                        '    global_batch = a.accumulation * world\n').body[0]
    wrapper.body.append(loop)
    wrapper.body.append(ast.Return(value=ast.Name(id='total', ctx=ast.Load())))
    module = ast.fix_missing_locations(ast.Module(body=helpers + [wrapper], type_ignores=[]))
    exec(compile(module, '<patched-opsd>', 'exec'), namespace)
    return namespace


class OPSDExecutionTests(unittest.TestCase):
    def namespace(self, collective, rank, pair_loss=None, evaluate=None):
        def tensor(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return torch.tensor(*args, **kwargs)
        proxy = SimpleNamespace(tensor=tensor, int64=torch.int64, isfinite=torch.isfinite,
                                zeros_like=torch.zeros_like, nn=torch.nn)
        return compiled_functions(patch_opsd(SOURCE), dict(torch=proxy, dist=collective.rank_dist(rank),
                                  pair_loss=pair_loss, evaluate=evaluate, print=lambda *a, **k: None))

    def run_ranks(self, examples, bad_gradient=False):
        collective = Collective(4)
        models, optimizers, functions = [], [], []
        for rank in range(4):
            model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
            with torch.no_grad(): model.weight.fill_(0.2)
            if bad_gradient and rank == 1:
                model.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            def loss(model, example, clip, chunk):
                return (model.weight.sum() - example) ** 2
            functions.append(self.namespace(collective, rank, pair_loss=loss)['update'])
            models.append(model); optimizers.append(optimizer)
        def run(rank):
            try:
                functions[rank](models[rank], examples, SimpleNamespace(accumulation=2, clip=10., chunk=32),
                                4, rank, optimizers[rank])
            except FloatingPointError as error:
                return str(error)
            return None
        with ThreadPoolExecutor(4) as pool:
            errors = list(pool.map(run, range(4)))
        return models, optimizers, errors, collective

    def test_pair_loss_byte_exact_and_overlay_refuses_unexpected_source(self):
        patched = patch_opsd(SOURCE)
        self.assertEqual(_function_text(SOURCE, 'pair_loss'), _function_text(patched, 'pair_loss'))
        self.assertIn('(loss / len(batch)).backward()', patched)
        self.assertNotIn('loss / world', patched)
        with self.assertRaises(ValueError): patch_opsd(patched)
        with self.assertRaises(ValueError): patch_opsd(SOURCE.replace('for example in local:', 'for item in local:'))

    def test_existing_local_finite_guards_are_replaced(self):
        guarded = SOURCE.replace('                (loss / len(batch)).backward()',
                  "                if not torch.isfinite(loss).all():\n"
                  "                    raise FloatingPointError('Nonfinite OPSD loss')\n"
                  '                (loss / len(batch)).backward()')
        guarded = guarded.replace('            torch.nn.utils.clip_grad_norm_(trainable, 1.)',
                  '            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)\n'
                  '            if not torch.isfinite(grad_norm).all():\n'
                  "                raise FloatingPointError('Nonfinite OPSD gradient; optimizer update refused')")
        self.assertEqual(patch_opsd(guarded), patch_opsd(SOURCE))

    def test_four_rank_full_and_unequal_tail_updates_equal_serial_global_mean(self):
        examples = [0.03 * i for i in range(10)]  # Last batch leaves ranks 2/3 empty.
        models, optimizers, errors, collective = self.run_ranks(examples)
        self.assertEqual(errors, [None] * 4)
        baseline = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad(): baseline.weight.fill_(0.2)
        optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
        for start in range(0, len(examples), 8):
            batch = examples[start:start + 8]
            optimizer.zero_grad(set_to_none=True)
            for example in batch: (((baseline.weight.sum() - example) ** 2) / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
            optimizer.step()
        for model, opt in zip(models, optimizers):
            torch.testing.assert_close(model.weight, baseline.weight, atol=1e-14, rtol=1e-14)
            self.assertEqual(opt.state[model.weight]['step'].item(), 2)
        self.assertEqual(collective.steps, [6] * 4)  # reject, gradient SUM, reject, per batch.

    def test_one_rank_nan_loss_rejects_every_rank_without_update_or_missing_collective(self):
        models, optimizers, errors, collective = self.run_ranks([0., float('nan'), 0., 0.])
        self.assertTrue(all(error and 'loss' in error for error in errors))
        self.assertEqual(collective.steps, [1] * 4)
        for model, optimizer in zip(models, optimizers):
            self.assertEqual(model.weight.item(), 0.2)
            self.assertEqual(len(optimizer.state), 0)

    def test_finite_loss_nan_gradient_rejects_every_rank_without_update(self):
        models, optimizers, errors, collective = self.run_ranks([0.] * 4, bad_gradient=True)
        self.assertTrue(all(error and 'gradient' in error for error in errors))
        self.assertEqual(collective.steps, [3] * 4)
        for model, optimizer in zip(models, optimizers):
            self.assertEqual(model.weight.item(), 0.2)
            self.assertEqual(len(optimizer.state), 0)

    def test_validation_reduces_integer_counts_with_unequal_and_empty_ranks(self):
        for rows in ([1, 0, 1, 1, 0, 0, 1], [1, 0]):
            collective = Collective(4)
            seen = [None] * 4
            def run(rank):
                def evaluate(agent, local):
                    seen[rank] = local
                    return sum(local) / len(local)
                function = self.namespace(collective, rank, evaluate=evaluate)['_opsd_global_validation']
                return function(None, rows, rank, 4)
            with ThreadPoolExecutor(4) as pool: values = list(pool.map(run, range(4)))
            self.assertEqual(values, [sum(rows) / len(rows)] * 4)
            self.assertEqual(seen, [rows[rank::4] or None for rank in range(4)])
            for state in collective.rounds.values():
                self.assertTrue(all(value.dtype == torch.int64 for value in state['values'].values()))

    def test_both_validation_sites_and_rank_zero_selection_remain(self):
        patched = patch_opsd(SOURCE)
        self.assertEqual(patched.count('_opsd_global_validation(agent, validation, rank, world)'), 3)
        self.assertIn("if em > best:\n                best = em\n                model.save_pretrained(out/'best_adapter')", patched)
        self.assertEqual(_function_text(SOURCE, 'encoded_pair'), _function_text(patched, 'encoded_pair'))


if __name__ == '__main__': unittest.main()
