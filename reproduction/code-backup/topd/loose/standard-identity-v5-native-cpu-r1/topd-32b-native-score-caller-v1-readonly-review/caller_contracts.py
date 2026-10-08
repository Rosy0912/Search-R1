"""CPU-only caller policy; no model, CUDA, process creation or authorization."""
import json
from pathlib import Path
from score_gate import require, digest, PARAMETERS


def same_cpu_identity(snapshot, provenance):
    expected = json.loads(Path(provenance['cpu_identity']['path']).read_text())['snapshot']
    require(snapshot['parameter_count'] == PARAMETERS
            and snapshot['parameter_bytes'] == PARAMETERS * 4
            and snapshot['parameter_sha256'] == expected['parameter_sha256']
            and snapshot['buffer_sha256'] == expected['buffer_sha256']
            and snapshot['parameter_devices'] == snapshot['buffer_devices'] == ['cpu']
            and snapshot['training_modules'] == snapshot['requires_grad_parameters']
            == snapshot['grad_parameters'] == 0, 'Actual CPU frozen identity differs')


def context_receipt(receipt, identity, uuids):
    require(receipt['pid'] == identity['pid'], 'Context PID differs')
    values = receipt['allocators']
    require(len(values) == 8 and {v['device'] for v in values} == set(range(8))
            and all(v['uuid'] == uuids[v['device']]
                    and v['allocated_bytes'] == v['reserved_bytes'] == 0 for v in values),
            'Actual eight context allocators/UUID order differ')
    require({(v['pid'], v['uuid']) for v in receipt['gpu_pid_uuid']}
            == {(identity['pid'], uuid) for uuid in uuids}, 'Context actual PID/UUID differs')


def authorization(receipt, phase, identity, provenance_sha, job):
    require(receipt['phase'] == phase and receipt['pid'] == identity['pid']
            and receipt['starttime_ticks'] == identity['starttime_ticks']
            and receipt['job'] == str(job)
            and receipt['provenance_sha256'] == provenance_sha,
            'Parent authorization phase/process/source differs')


def verify_bundle(directory):
    """Caller deployed source is frozen separately from the original pipeline."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'tool-files.json').read_text())
    required = {'score_gate.py', 'caller_contracts.py', 'worker.py', 'supervisor.py',
                'native-score.sbatch', 'test_score_gate.py', 'test_caller_contracts.py'}
    require(required <= set(manifest), 'Incomplete caller source manifest')
    for name, expected in manifest.items():
        path = (directory / name).resolve()
        require(directory in path.parents and path.is_file() and digest(path) == expected,
                'Caller source bundle drift: ' + name)
    return dict(manifest_sha256=digest(directory / 'tool-files.json'), files=manifest)
