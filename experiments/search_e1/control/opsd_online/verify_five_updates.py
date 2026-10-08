"""Independent CPU reads of protected checkpoints. Five steps are not effect replication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

try:
    from .checkpoint import checked_json, check_checkpoint, exclusive_json, require, fileproof
    from .acceptance import process, same_process
except ImportError:
    from checkpoint import checked_json, check_checkpoint, exclusive_json, require, fileproof
    from acceptance import process, same_process


def fresh_memory_snapshot(binding_ref, config, output, reader=None):
    """Read a current atomic sample after CPU checkpoint inspection, not an old content SHA."""
    binding, binding_proof = checked_json(binding_ref)
    require(binding.get('schema') == 'opsd-online-memory-observer-binding-v1'
            and binding.get('accepted') is True, 'Missing bound live observer identity')
    require(binding['experiment_run_id'] == config['run_id']
            and binding['config_sha256'] == config['_config_sha256'], 'Observer binding belongs to another config')
    receipts = Path(config['paths']['receipts']).resolve()
    live_path = Path(binding['live_report_path']).resolve()
    require(live_path.is_relative_to(receipts), 'Live observer path outside bound receipts')
    expected = binding['observer_identity']
    require(expected['uid'] == config['owner']['uid'] and expected.get('run_id')
            and expected['run_id'] != config['run_id'], 'Observer must be an independent owned CPU process')
    manifest, _ = checked_json(config['helper_source_manifest'])
    source = manifest['files']['memory_observer.py']
    source = source['sha256'] if isinstance(source, dict) else source
    require(expected['source_sha256'] == source, 'Observer binding has wrong published source')
    reader = reader or process
    def alive():
        actual = reader(expected['pid'])
        require(same_process(actual, expected) and actual.get('run_id') == expected['run_id'],
                'Observer PID/UID/starttick/run identity is no longer live')
    alive()
    last_error = None
    for attempt in range(5):
        try:
            memory, actual = checked_json(fileproof(live_path))
            break
        except (ValueError, FileNotFoundError) as exc:
            last_error = exc
            if attempt < 4: time.sleep(.05)
    else:
        raise ValueError('Could not bind a stable atomic memory sample') from last_error
    require(memory['schema'] == 'opsd-online-direct-memory-v1' and memory['run_id'] == config['run_id']
            and memory['config_sha256'] == config['_config_sha256'], 'Memory samples belong to another run/config')
    require(memory['observer_identity'] == expected, 'Live memory report differs from bound observer identity')
    require(memory['samples'] > 0 and memory['samples_below_limit'] == 0, 'Actual memory limit failed in observed history')
    require(0 <= time.time() - memory['updated_unix'] <= 30, 'Direct-memory observer is stale')
    require(set(map(str, memory['minimum_actual_free_mib'])) == {str(index) for index in range(8)} and
            all(free >= config['resources']['training_minimum_free_mib'] for free in memory['minimum_actual_free_mib'].values()),
            'Missing/failed actual memory.free history')
    require([item['uuid'] for item in memory['current']['devices']] == config['owner']['gpu_uuids'],
            'Direct memory observer is monitoring another physical allocation')
    alive()
    snapshot = exclusive_json(Path(str(output) + '.direct-memory.json'), memory)
    return snapshot, dict(binding=binding_proof, observed_current_report=actual)


def equal_tree(left, right, torch):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b, torch) for a, b in zip(left, right))
    return left == right


def adapter_digest(tensors, torch):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def verify_rank_states(checkpoint, config, torch, actual_adapter_digest, actual_adapter_count):
    first, reports = None, []
    for entry in sorted(checkpoint['ranks'], key=lambda value: value['rank']):
        value = torch.load(entry['file']['path'], map_location='cpu', weights_only=False)
        require(value.get('schema') == 'opsd-online-rank-state-v1', 'Unknown actual rank state schema')
        require(value['rank'] == entry['rank'] and value['run_id'] == checkpoint['run_id'], 'Rank state identity differs')
        require(value['config_sha256'] == config['_config_sha256'], 'Rank state config differs')
        for key in ('optimizer_updates', 'cumulative_pair_exposure', 'cursor', 'training_input',
                    'source_model_identity', 'runtime_identity'):
            require(value[key] == checkpoint[key], 'Actual saved rank state differs from manifest: ' + key)
        require(set(value['rng']) == {'python', 'numpy', 'torch_cpu', 'cuda_all'}, 'Missing actual RNG fields')
        require(isinstance(value['rng']['torch_cpu'], torch.Tensor) and value['rng']['cuda_all']
                and all(isinstance(item, torch.Tensor) for item in value['rng']['cuda_all']), 'Incomplete actual Torch RNG')
        optimizer = value['optimizer']
        parameter_ids = [index for group in optimizer['param_groups'] for index in group['params']]
        require(len(parameter_ids) == len(set(parameter_ids)) == actual_adapter_count,
                'Actual optimizer does not cover every adapter parameter exactly once')
        require(value['adapter_tensor_sha256'] == actual_adapter_digest == checkpoint['all_rank_adapter_sha256'][entry['rank']],
                'Actual adapter bytes differ from saved rank/manifest tensor identity')
        if checkpoint['optimizer_updates']:
            require(set(optimizer['state']) == set(parameter_ids), 'An actual Adam parameter state is missing')
        else:
            require(not optimizer['state'], 'Initial checkpoint contains optimizer history')
        steps = sorted({float(state['step']) for state in optimizer['state'].values() if 'step' in state})
        expected = [float(checkpoint['optimizer_updates'])] if checkpoint['optimizer_updates'] else []
        require(steps == expected, 'Actual serialized Adam step differs')
        finite_count = 0
        for state in optimizer['state'].values():
            require(all(name in state for name in ('step', 'exp_avg', 'exp_avg_sq')), 'Incomplete actual Adam moments')
            for tensor in state.values():
                if isinstance(tensor, torch.Tensor):
                    require(torch.isfinite(tensor).all().item(), 'Nonfinite actual optimizer tensor')
                    finite_count += 1
        for group in optimizer['param_groups']:
            require(group['lr'] == config['scientific']['learning_rate'], 'Actual optimizer learning rate changed')
        if first is None:
            first = optimizer
        else:
            require(equal_tree(first, optimizer, torch), 'Eight serialized Adam tensors/groups are not identical')
        reports.append(dict(rank=entry['rank'], actual_adam_steps=steps, finite_optimizer_tensor_count=finite_count,
                            complete_rng=True, optimizer_equal_to_rank0=True))
    return reports


def verify(config_ref, initial_ref, fifth_ref, memory_ref, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from safetensors.torch import load_file
    require(not torch.cuda.is_initialized(), 'Independent acceptance must not initialize CUDA')
    config, config_proof = checked_json(config_ref)
    config['_config_sha256'] = config_proof['sha256']
    initial, initial_proof = check_checkpoint(initial_ref, config, config['run_id'], expected_updates=0)
    fifth, fifth_proof = check_checkpoint(fifth_ref, config, config['run_id'], expected_updates=5)
    old = load_file(str(Path(initial['adapter']['path']) / 'adapter_model.safetensors'), device='cpu')
    new = load_file(str(Path(fifth['adapter']['path']) / 'adapter_model.safetensors'), device='cpu')
    require(old.keys() == new.keys() and old, 'Protected adapters have different tensor inventory')
    initial_states = verify_rank_states(initial, config, torch, adapter_digest(old, torch), len(old))
    fifth_states = verify_rank_states(fifth, config, torch, adapter_digest(new, torch), len(new))
    changed, maximum_delta = [], 0.
    for name in sorted(old):
        require(old[name].shape == new[name].shape and old[name].dtype == new[name].dtype, 'Adapter tensor shape/dtype differs')
        require(torch.isfinite(old[name]).all().item() and torch.isfinite(new[name]).all().item(), 'Nonfinite adapter tensor')
        if not torch.equal(old[name], new[name]):
            changed.append(name)
            maximum_delta = max(maximum_delta, float((old[name].float() - new[name].float()).abs().max()))
    require(changed and maximum_delta > 0, 'Fifth update has not actually changed source adapter')
    memory_proof, memory_binding = fresh_memory_snapshot(memory_ref, config, output)
    result = dict(schema='opsd-online-five-real-updates-independent-v1', accepted=True,
                  verified_unix=time.time(), run_id=config['run_id'], config=config_proof,
                  source_grpo_step=100, actual_opsd_updates=5, display_step=105,
                  initial_checkpoint=initial_proof, fifth_checkpoint=fifth_proof,
                  initial_actual_ranks=initial_states, fifth_actual_ranks=fifth_states,
                  finite_adapter_tensors=len(new), changed_adapter_tensors=len(changed), maximum_absolute_delta=maximum_delta,
                  direct_memory=memory_proof, direct_memory_observer=memory_binding, cuda_initialized=False,
                  scope='Five synchronized real OPSD updates only; continue this same run. No GRPO Adam recovery or effect claim.')
    require(not torch.cuda.is_initialized(), 'CPU acceptance unexpectedly initialized CUDA')
    return exclusive_json(output, result)


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'initial-checkpoint', 'fifth-checkpoint', 'memory'):
        parser.add_argument('--' + name, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.description = '--memory binds the static observer-binding receipt; a fresh report is captured after CPU state checks.'
    args = parser.parse_args()
    refs = [dict(path=getattr(args, name), sha256=getattr(args, name+'_sha256'))
            for name in ('config', 'initial_checkpoint', 'fifth_checkpoint', 'memory')]
    print(json.dumps(verify(*refs, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
