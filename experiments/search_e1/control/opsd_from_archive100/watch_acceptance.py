from pathlib import Path
import json
import os
import time

import torch
from safetensors import safe_open

b = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
out = b / 'runs/opsd'


def read_json_ready(path, timeout=120):
    """Retry missing/partially written observations without accepting them."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if deadline is not None and time.monotonic() >= deadline:
                raise
            time.sleep(2)


def write_receipt(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def assert_identical(expected, actual, where):
    """Require exact CPU tensor values and optimizer configuration equality."""
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), f'{where}: expected tensor'
        assert expected.dtype == actual.dtype, f'{where}: tensor dtype differs'
        assert expected.shape == actual.shape, f'{where}: tensor shape differs'
        assert torch.equal(expected, actual), f'{where}: tensor values differ'
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), f'{where}: keys differ'
        for key in expected:
            assert_identical(expected[key], actual[key], f'{where}.{key}')
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual) and len(expected) == len(actual), f'{where}: sequence differs'
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_identical(left, right, f'{where}[{index}]')
    else:
        assert type(expected) is type(actual) and expected == actual, f'{where}: value differs'


def inspect_optimizer(saved, rank):
    assert saved['rank'] == rank and saved['updates'] == 5, f'Rank {rank}: wrong saved update identity'
    optimizer = saved['optimizer']
    states = optimizer['state']
    parameter_ids = [key for group in optimizer['param_groups'] for key in group['params']]
    assert parameter_ids and len(set(parameter_ids)) == len(parameter_ids), f'Rank {rank}: invalid parameter IDs'
    assert set(states) == set(parameter_ids), f'Rank {rank}: missing optimizer states'
    for parameter_id, state in states.items():
        assert {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys(), f'Rank {rank}, parameter {parameter_id}: incomplete Adam state'
        assert torch.is_tensor(state['exp_avg']) and torch.is_tensor(state['exp_avg_sq']), f'Rank {rank}, parameter {parameter_id}: Adam moments are not tensors'
        for key, value in state.items():
            if torch.is_tensor(value):
                assert bool(torch.isfinite(value).all()), f'Rank {rank}, parameter {parameter_id}, {key}: nonfinite'
        assert float(state['step']) == 5., f'Rank {rank}, parameter {parameter_id}: Adam step is not 5'
    return optimizer


def main():
    # This is an observation wait; it cannot turn initialization into an update.
    acceptance_path = out / 'five-update-acceptance.json'
    a = read_json_ready(acceptance_path, timeout=None)
    assert a['world_size'] == 8 and a['opsd_updates'] == 5 and a['source_grpo_step'] == 100, 'Expected archive GRPO 100 plus five actual OPSD updates on eight ranks'
    assert sorted(r['rank'] for r in a['ranks']) == list(range(8)), 'Update evidence does not cover ranks 0 through 7 exactly once'
    assert all(r['updates'] == 5 and r['source_grpo_step'] == 100 and r['finite']
               and r['changed_from_initial'] and r['adam_steps'] == [5.] for r in a['ranks']), 'A rank lacks finite changed weights or an actual Adam update count of five'
    config = read_json_ready(out / 'config.json')
    assert config['world_size'] == 8 and config['accumulation'] == 4, 'Global trajectory batch is not 8 x 4'
    assert config['model'] == a['source_model'], 'Saved model identity differs from live evidence'

    files = [out / f'adapter_update_{s}/adapter_model.safetensors' for s in (1, 5)]
    with safe_open(str(files[0]), framework='pt', device='cpu') as one, safe_open(str(files[1]), framework='pt', device='cpu') as five:
        assert set(one.keys()) == set(five.keys()) and list(five.keys()), 'Adapter tensor keys are missing or differ'
        changed = []
        for key in five.keys():
            first, last = one.get_tensor(key), five.get_tensor(key)
            assert bool(torch.isfinite(first).all()), f'Update 1 adapter tensor {key}: nonfinite'
            assert bool(torch.isfinite(last).all()), f'Update 5 adapter tensor {key}: nonfinite'
            assert first.shape == last.shape and first.dtype == last.dtype, f'Adapter tensor {key}: shape/dtype changed'
            if not torch.equal(first, last):
                changed.append(dict(key=key, max_abs_delta=float((last.float() - first.float()).abs().max())))
        assert changed, 'No adapter weights changed between actual updates 1 and 5'

    optimizer_evidence = []
    reference_path = out / 'update_5_rank_0_state.pt'
    reference_saved = torch.load(reference_path, map_location='cpu', weights_only=False)
    reference = inspect_optimizer(reference_saved, 0)
    for rank in range(8):
        path = out / f'update_5_rank_{rank}_state.pt'
        if rank == 0:
            optimizer = reference
        else:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            optimizer = inspect_optimizer(saved, rank)
        assert_identical(reference, optimizer, f'Rank {rank} optimizer')
        optimizer_evidence.append(dict(rank=rank, steps=[5.], finite=True,
            identical_to_rank_0=True, compared_fields=['state', 'param_groups'],
            parameter_states=len(optimizer['state']), path=str(path), bytes=path.stat().st_size))
        if rank != 0:
            del optimizer, saved
    del reference, reference_saved

    # A healthy generation-stage sample cannot prove OPSD memory headroom.
    deadline = time.monotonic() + 120
    while True:
        memory = read_json_ready(b / 'receipts/memory-continuous-v2.json')
        if memory['sample_end_unix'] >= acceptance_path.stat().st_mtime and 0 <= time.time() - memory['updated_at_unix'] <= 60:
            break
        assert time.monotonic() < deadline, 'No fresh GPU memory observation after actual update 5'
        time.sleep(2)
    assert sorted(r['gpu'] for r in memory['current']) == list(range(8)), 'Memory observer does not cover all eight GPUs'
    assert memory['schema_version'] == 2 and memory['free_measurement'] == 'nvidia-smi memory.free', 'Require directly sampled actual free GPU memory'
    assert memory['sample_start_unix'] <= (out / 'config.json').stat().st_mtime, 'Actual free memory sampling did not cover OPSD from initialization'
    assert memory['sample_end_unix'] <= memory['updated_at_unix'] <= time.time() + 2, 'Invalid actual memory sampling timestamps'
    assert memory['minimum_actual_free_mib'] >= 1024, 'Less than 1 GiB actual sampled free memory; do not certify GPU headroom'
    report = dict(status='ACCEPTED', time=time.time(), source_grpo_step=100,
        actual_opsd_updates=5, display_step=105, continuous_run=True,
        source_model=a['source_model'], global_trajectory_batch=32, world=8,
        adapter_tensors_all_finite=True, adapter_updates_checked=[1, 5],
        changed_adapter_tensors_1_to_5=changed, actual_saved_optimizer_ranks=optimizer_evidence,
        optimizer_states_identical_all_ranks=True, memory=memory, live_update_evidence=a,
        scope='Five actual OPSD updates and numerical/memory acceptance, not completed effect reproduction')
    write_receipt(b / 'receipts/five-real-opsd-updates-accepted.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        failure = dict(status='FAILED_ACCEPTANCE', time=time.time(), error_type=type(error).__name__,
            reason=str(error), source_grpo_step=100, required_actual_opsd_updates=5,
            scope='External numerical/memory verification failed; no acceptance was written by this attempt')
        write_receipt(b / 'receipts/five-real-opsd-updates-failed.json', failure)
        print(json.dumps(failure), flush=True)
        raise
