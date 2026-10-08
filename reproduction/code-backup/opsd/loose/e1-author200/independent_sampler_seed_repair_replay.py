"""Local independent CPU boundaries for minimal default-seed repair."""
from pathlib import Path
import copy
import hashlib
import json
import types
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NATIVE = 'f7d8482e54c727c878a3114d4d748889e4bcd38cc7dac573000f6ba2cda1864a'

def load(path, name, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module

def rejection(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Mutation accepted')

def main():
    state = load(ROOT / 'work/opsd-e1-300/search_r1/baselines/full_state.py', 'exact_frozen_sampler', 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6')
    n = load(HERE / 'final-acceptance-candidate/inspect_author200_checkpoint_cpu.py', 'repaired_checker', NATIVE)
    config_path = HERE / 'actual-snapshots/3261-20261005T203236002279Z/original-files/full_state/global_step_5/resolved-config.json'
    raw = config_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
    config = json.loads(raw)
    settings = n.sampler_settings(state, config)
    assert settings == {'seed': None, 'shuffle': True}
    sampler = state.StatefulSampler(169028, shuffle=settings['shuffle'])
    rows = iter(sampler)
    for _ in range(2560): next(rows)
    sampler.commit_consumed(2560)
    first = sampler.state_dict()
    for _ in range(10240): next(rows)
    sampler.commit_consumed(10240)
    current = sampler.state_dict()
    passed = n.validate_sampler(current, first, 25, torch, settings)
    assert passed['seed'] is None and passed['sampler_cursor'] == 12800
    cases = ['real_frozen_default_ctor5_to25_positive']
    for target in ('current', 'accepted5'):
        for bad in (42, 0, False, 'None'):
            a = copy.deepcopy(current); b = copy.deepcopy(first)
            (a if target == 'current' else b)['seed'] = bad
            rejection(lambda: n.validate_sampler(a, b, 25, torch, settings))
            cases.append(target + '_seed_' + repr(bad))
    for field, bad in [('shuffle', 1), ('cursor', 12799), ('consumed_cursor', 12799), ('epoch', False)]:
        a = copy.deepcopy(current); a[field] = bad
        rejection(lambda: n.validate_sampler(a, first, 25, torch, settings))
        cases.append('current_' + field + '_' + repr(bad))
    a = copy.deepcopy(current); a['order'][0], a['order'][1] = a['order'][1], a['order'][0]
    rejection(lambda: n.validate_sampler(a, first, 25, torch, settings))
    cases.append('same_set_order_changed')
    a = copy.deepcopy(current); a['generator'] = torch.Generator().manual_seed(73428).get_state()
    rejection(lambda: n.validate_sampler(a, first, 25, torch, settings))
    cases.append('valid_generator_changed')
    changed_config = copy.deepcopy(config); changed_config['data']['shuffle_train_dataloader'] = 1
    rejection(lambda: n.sampler_settings(state, changed_config))
    cases.append('config_shuffle_int_rejected')
    class WrongDefault:
        def __init__(self, data_source, seed=42, shuffle=True): pass
    rejection(lambda: n.sampler_settings(types.SimpleNamespace(StatefulSampler=WrongDefault), config))
    cases.append('constructor_default_changed_rejected')
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'PASS_INDEPENDENT_LOCAL_CPU_SAMPLER_REPAIR_NOT_ACTUAL25_INSPECTION', 'native_sha256': NATIVE, 'config_sha256': hashlib.sha256(raw).hexdigest(), 'torch_version': torch.__version__, 'cuda_is_initialized': False, 'case_count': len(cases), 'cases': cases, 'result': passed, 'limitations': ['Local fixtures use actual frozen constructor and actual signed config, not actual saved PT bytes', 'Original actual25 native/srun/controller1 remains FAILED; no repair execution is performed here']}))

if __name__ == '__main__':
    main()
