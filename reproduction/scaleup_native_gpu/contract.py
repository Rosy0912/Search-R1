"""CPU-only contract for one tiny CUDA native8→16 cold-restoration probe."""
import hashlib
import json
from pathlib import Path

JOB='zhangjunming-s16-native-v1'
ROOT='/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-Scaleup-20261019-v1/probes/cgrts-native-gpu-v1'
RUNTIME='/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/native-env/bin/python'
NATIVE_SHA='042894a5fcae3363b81af1da240129b3206c353e8e03731f98092319ac24e18e'
CORE_SHA='898916fa99399ed4c01b1b3a5dd72ae05da096122545dcf4d07fa85a7cd7daff'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node_coordinates(env):
    rank, world=int(env['RANK']),int(env['WORLD_SIZE'])
    assert rank in (0,1) and world==2
    for k,v in [('NODE_RANK',rank),('NODE_COUNT',2)]:
        if env.get(k) not in (None,''): assert int(env[k])==v
    addr,port=env['MASTER_ADDR'],int(env['MASTER_PORT'])
    assert addr and len(addr)<254 and all(x.isalnum() or x in '.-_' for x in addr)
    assert 1024<=port<=65535
    return rank,addr,port


def read_plan(path):
    p=json.loads(Path(path).read_text())
    assert p['job']==JOB and p['root']==ROOT and p['runtime']==RUNTIME
    assert p['purpose']=='TINY_CUDA_NATIVE8TO16_ONLY_NOT_PRODUCTION'
    assert p['nodes']==2 and p['gpus_per_node']==8 and p['source_world']==8 and p['target_world']==16
    assert p['source_outer']==50 and p['optimizer_calls_per_outer']==20 and p['next_outer']==51
    assert p['source_optimizer_steps']==1000 and p['target_optimizer_steps']==1020
    assert p['no_auto_retry'] and not p['production_checkpoint_read'] and not p['scientific_repeat']
    assert p['cuda_rng_policy']=='tiny-native-cuda8to16-parent-digest-domain-separated-v1'
    package=Path(ROOT)/'source/reproduction/scaleup_native_gpu'
    assert set(p['files'])=={'__init__.py','transition.py','contract.py','node.py','worker.py','gate.py'}
    for n,h in p['files'].items(): assert sha(package/n)==h,n
    assert p['assets']=={'training-state-frozen.py':NATIVE_SHA,'flat-core-frozen.py':CORE_SHA}
    for n,h in p['assets'].items():assert sha(Path(ROOT)/'assets'/n)==h,n
    return p


def validate_tasks(tasks):
    assert len(tasks)==2 and {t['role'] for t in tasks}=={'PYTORCH_MASTER','PYTORCH_WORKER'}
    for t in tasks:
        assert t['replicas']==1
        r=t['resource_spec'];assert r['accelerate_device_count']==8 and r['cpu_count']==16 and r['memory_gib']==64
        assert r['machine_types']==['n2ls.ie.i60b'] and r['rdma_name']=='ib-cluster-01'
        assert 'kubernetes.io/hostname' not in json.dumps(t.get('scheduling',{}))


def validate_gate(g,s,plan_sha):
    assert g['job']==s['job']==JOB and g['uid']==s['uid'] and g['state']=='RUNNING'
    assert g['plan_sha256']==s['plan_sha256']==plan_sha and g['nodes']==2 and g['total_gpus']==16
