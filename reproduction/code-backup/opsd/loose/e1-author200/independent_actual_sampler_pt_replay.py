"""Inspect copied actual small driver states; no GPU, no remote or rank PT read."""
from pathlib import Path
import hashlib
import json
import types
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SMALL = HERE / 'actual3261-sampler-driver-small-originals'
EXPECTED = {'accepted5': '5a8c2f6141a09cceee068e08ac8e8eed3033cca58fc05abf3139beaf487782c1', 'current25': 'dfbf01906077afbc6249ef2d4d57ecf665dcfa3f4f63c675d4d11a96608f9e22'}

def load_code(path, name, digest):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module

def main():
    state = load_code(ROOT / 'work/opsd-e1-300/search_r1/baselines/full_state.py', 'frozen_full_state', 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6')
    n = load_code(HERE / 'final-acceptance-candidate/inspect_author200_checkpoint_cpu.py', 'repaired_native', 'f7d8482e54c727c878a3114d4d748889e4bcd38cc7dac573000f6ba2cda1864a')
    metadata = json.loads((SMALL / 'metadata-original.json').read_bytes())
    first_root = HERE / 'actual-snapshots/3261-20261005T203236002279Z/original-files'
    accepted_native_path = first_root / 'evidence/first5-cpu-inspection-once-v1/native.json'
    assert hashlib.sha256(accepted_native_path.read_bytes()).hexdigest() == 'b50c6ed9995f6148ac250e75f20721bafc95d952f4d841c24542dce31f4adcb2'
    old_native = json.loads(accepted_native_path.read_bytes())
    original_observation = json.loads((HERE / 'checkpoint25-cpu-observed-20261005T221801Z-original.json').read_bytes())['files']
    current_report = original_observation['observation/observed-checkpoint-inventory.json']['content']['driver_report']
    expected_digests = {'accepted5': old_native['driver']['state_digest'], 'current25': current_report['digest']}
    loaded = {}
    records = {}
    for role, digest in EXPECTED.items():
        p = SMALL / (role + '-driver.pt'); raw = p.read_bytes()
        assert len(raw) == 734488 and hashlib.sha256(raw).hexdigest() == digest
        assert metadata['files'][role]['sha256'] == digest and metadata['files'][role]['bytes'] == len(raw)
        saved = torch.load(p, map_location='cpu', weights_only=False)
        step = 5 if role == 'accepted5' else 25
        expected = dict(version=1, method='pure_opsd', global_step=step, next_global_step=step+1, actual_outer_updates=step, epoch=0, pid=1533312, uid=0, job_id='3261', source_commit='188f4e5e7b63cf479290d0888a4a075499886750', config_sha256='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60')
        for key, value in expected.items():
            assert type(saved[key]) is type(value) and saved[key] == value, (role, key)
        assert state.state_digest(saved) == expected_digests[role]
        sampler = saved['sampler']
        assert sampler['seed'] is None and sampler['shuffle'] is True
        assert type(sampler['epoch']) is int and sampler['epoch'] == 0
        assert sampler['cursor'] == sampler['consumed_cursor'] == 512 * step
        assert len(sampler['order']) == 169028
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        records[role] = {'file_sha256': digest, 'full_state_digest': state.state_digest(saved), 'schema_keys': sorted(saved), 'seed': sampler['seed'], 'seed_type': type(sampler['seed']).__name__, 'shuffle': sampler['shuffle'], 'epoch': sampler['epoch'], 'cursor': sampler['cursor'], 'consumed_cursor': sampler['consumed_cursor'], 'permutation_state_digest': state.state_digest(sampler['order']), 'generator_state_digest': state.state_digest(sampler['generator'])}
        loaded[role] = saved
    config_bytes = (first_root / 'full_state/global_step_5/resolved-config.json').read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
    result = n.validate_sampler(loaded['current25']['sampler'], loaded['accepted5']['sampler'], 25, torch, n.sampler_settings(state, json.loads(config_bytes)))
    assert result['seed'] is None and result['sampler_cursor'] == 12800
    assert records['accepted5']['permutation_state_digest'] == records['current25']['permutation_state_digest']
    assert records['accepted5']['generator_state_digest'] == records['current25']['generator_state_digest']
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'PASS_ACTUAL_SMALL_DRIVER_STATE_AND_SEED_JOIN_NOT_FULL_NATIVE25_ACCEPTANCE', 'records': records, 'actual25_raw_driver_status': current_report['status'], 'repaired_sampler_result': result, 'torch_version': torch.__version__, 'cuda_is_initialized': False, 'limits': ['Actual copied5/25 driver PT only, no current25 rank/Adam/teacher/HF tensor reread', 'Prior failed8441 execution remains failure; local valid sampler does not substitute complete native2.4 repair execution', 'No same-process or new-process restore performed; stored saved identities only']}))

if __name__ == '__main__':
    main()
