"""CPU-only validation for the unique two-node NCCL connectivity probe."""
import hashlib
import json
from pathlib import Path

JOB = 'zhangjunming-s16-net-v1'
ROOT = '/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-Scaleup-20261019-v1/probes/nccl16-v1'
RUNTIME = '/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/native-env/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(plan):
    assert plan['job'] == JOB and plan['root'] == ROOT
    assert plan['runtime'] == RUNTIME
    assert plan['nodes'] == 2 and plan['gpus_per_node'] == 8
    assert plan['purpose'] == 'INFRASTRUCTURE_ONLY_NOT_SCIENTIFIC_REPEAT'
    assert plan['timeout_seconds'] == 900 and plan['no_auto_retry'] is True
    assert plan['existing_trainings_unchanged'] is True
    assert plan['production_checkpoint_read'] is False
    assert plan['production_speedup_claim'] is False
    assert set(plan['files']) == {'__init__.py', 'contract.py', 'node.py', 'worker.py', 'gate.py'}
    for h in plan['files'].values():
        assert isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
    return plan


def node_coordinates(env):
    # SSP injects one process per node; torchrun later supplies per-GPU ranks.
    node = int(env['NODE_RANK'])
    assert int(env['NODE_COUNT']) == 2 and node in (0, 1)
    assert int(env['WORLD_SIZE']) == 2 and int(env['RANK']) == node
    host = env['MASTER_ADDR']
    assert host and len(host) < 254 and all(c.isalnum() or c in '.-_' for c in host)
    port = int(env['MASTER_PORT'])
    assert 1024 <= port <= 65535
    return node, host, port


def validate_params(params):
    assert params['job_name'] == JOB
    for k, v in dict(num_nodes=2, tasks_per_node=1, gpus_per_task=8,
                     cpus_per_task=16, memory_per_task=64, shm_size_gib=16,
                     priority=1, retry=0, no_env=True, enable_privileged=False).items():
        assert params[k] == v, k
    assert params['cluster'] == 'cluster-p'
    assert params['queue'] == 'queue-p-reserved-llmarchitecture'
    assert params['machine_type'] == 'n2ls.ie.i60b'
    assert params['rdma_name'] == 'ib-cluster-01'
    assert not params.get('include') and not params.get('exclude')


def validate_gate(gate, submission, plan_sha):
    assert gate['job'] == submission['job'] == JOB
    assert gate['uid'] == submission['uid'] and gate['state'] == 'RUNNING'
    assert gate['plan_sha256'] == submission['plan_sha256'] == plan_sha
    assert gate['nodes'] == 2 and gate['total_gpus'] == 16


def validate_actual_tasks(tasks):
    assert len(tasks) == 2
    assert {t['role'] for t in tasks} == {'PYTORCH_MASTER', 'PYTORCH_WORKER'}
    assert all(t['replicas'] == 1 for t in tasks)
    for t in tasks:
        r = t['resource_spec']
        assert r['accelerate_device_count'] == 8 and r['cpu_count'] == 16
        assert r['memory_gib'] == 64 and r['rdma_name'] == 'ib-cluster-01'
        assert r['machine_types'] == ['n2ls.ie.i60b']
        assert 'kubernetes.io/hostname' not in json.dumps(t.get('scheduling', {}))


def read_plan(path):
    path = Path(path)
    plan = validate(json.loads(path.read_text()))
    package = Path(plan['package'])
    assert package == Path(ROOT) / 'source/reproduction/scaleup_probe'
    for n, h in plan['files'].items():
        assert sha(package / n) == h, n
    return plan
