"""Independent local CPU reconstruction from exact frozen StatefulSampler bytes."""
from pathlib import Path
import hashlib
import inspect
import json
import types
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def load(path, name, expected):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == expected, (path, digest)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module

def main():
    source = load(ROOT / 'work/opsd-e1-300/search_r1/baselines/full_state.py', 'frozen_sampler', 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6')
    old_checker = load(HERE / 'final-acceptance-8441e10fa456/inspect_author200_checkpoint_cpu.py', 'old_checker', 'ca09cbca5084b1c1c8a1bf57906aca61e9805dd1176d6cfc954da5fb9981b471')
    assert inspect.signature(source.StatefulSampler.__init__).parameters['seed'].default is None
    sampler = source.StatefulSampler(169028, shuffle=True)
    iterator = iter(sampler)
    for _ in range(512 * 5):
        next(iterator)
    sampler.commit_consumed(512 * 5)
    accepted5_fixture = sampler.state_dict()
    for _ in range(512 * 20):
        next(iterator)
    sampler.commit_consumed(512 * 20)
    current25_fixture = sampler.state_dict()
    assert accepted5_fixture['seed'] is current25_fixture['seed'] is None
    assert accepted5_fixture['order'] == current25_fixture['order']
    assert torch.equal(accepted5_fixture['generator'], current25_fixture['generator'])
    try:
        old_checker.validate_sampler(current25_fixture, accepted5_fixture, 25, torch)
    except ValueError as error:
        assert str(error) == 'Sampler position/config differs: seed'
        failure = str(error)
    else:
        raise AssertionError('Old checker must reject actual frozen default constructor')
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'COUNTEREXAMPLE_FROM_FROZEN_DEFAULT_CONSTRUCTOR_NOT_ACTUAL_SAVED_PT', 'frozen_sampler_sha256': 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6', 'old_checker_sha256': 'ca09cbca5084b1c1c8a1bf57906aca61e9805dd1176d6cfc954da5fb9981b471', 'constructor_default_seed': None, 'fixture5_seed': accepted5_fixture['seed'], 'fixture25_seed': current25_fixture['seed'], 'old_checker_error': failure, 'torch_version': torch.__version__, 'cuda_is_initialized': False, 'limitations': ['Local CPU construction only; not actual3261 driver PT inspection', 'Original FAILED native/srun/controller 1 is not converted to PASS']}))

if __name__ == '__main__':
    main()
