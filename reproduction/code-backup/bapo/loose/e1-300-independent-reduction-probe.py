"""Independent actual DataProto/core function probe; not an actor/GPU acceptance."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path('work/bapo-e1-300').resolve()))
sys.path.insert(0, '/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-audit/search-e1-reproduction-20261003-233328')
import torch
from verl import DataProto
from verl.trainer.ppo import core_algos
from search_e1.grpo_batching import trajectory_policy_loss, trajectory_mean
from search_r1.baselines.rollout import select_rows

torch.manual_seed(27)
mask = torch.zeros(16, 7, dtype=torch.float64)
for i in range(16):
    mask[i, :i % 8] = 1
old = torch.full_like(mask, -3)
ref = old + torch.linspace(-.8, .8, mask.numel()).reshape_as(mask)
adv = torch.linspace(-2, 2, mask.numel()).reshape_as(mask)
features = torch.linspace(-.7, .9, mask.numel()).reshape_as(mask)
proto = DataProto.from_dict({'old': old, 'ref': ref, 'adv': adv, 'mask': mask, 'features': features})

def evaluate(rank_rows, micros, correct):
    parameter = torch.tensor(.31, dtype=torch.float64, requires_grad=True)
    values = []
    world = len(rank_rows)
    for indices in rank_rows:
        part = select_rows(proto, indices)
        for b in part.batch.split(micros):
            cur = b['old'] + parameter * b['features']
            pg = trajectory_policy_loss(b['old'], cur, b['adv'], b['mask'], .2)[0]
            kl = trajectory_mean(core_algos.kl_penalty(cur, b['ref'], 'low_var_kl'), b['mask'])
            scale = len(b) / (len(proto) / world) if correct else 1 / max(1, (len(part) // micros))
            values.append((pg + .001 * kl) * scale / world)
    total = sum(values)
    total.backward()
    return [total.item(), parameter.grad.item()]

reference = evaluate([list(range(16))], 16, True)
cases = {}
for world in (1, 2, 4, 8):
    rows = [list(range(rank, 16, world)) for rank in range(world)]
    for micro in (1, 2, 3, 7, 16):
        result = evaluate(rows, micro, True)
        assert torch.allclose(torch.tensor(result), torch.tensor(reference), atol=2e-7, rtol=2e-7), (world, micro, result, reference)
        cases[f'world{world}_micro{micro}'] = result
bad = evaluate([list(range(16))], 3, False)
assert abs(bad[1] - reference[1]) > 1e-4
empty = trajectory_mean(torch.ones(2, 3), torch.tensor([[1., 0., 0.], [0., 0., 0.]])).item()
assert empty == .5
print(json.dumps({'status': 'PASS_ACTUAL_DATAPROTO_CORE_CPU_PROBE_NOT_ACTOR_ACCEPTANCE', 'cases': len(cases), 'reference_loss_gradient': reference, 'ragged_fixed_accumulation_counterexample': bad, 'empty_row_denominator_value': empty}, indent=2))
