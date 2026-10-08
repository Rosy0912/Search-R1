"""CPU-only real Ray environment propagation, not a TOP-D training iteration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import ray
from search_r1.baselines.topd_runtime_audit import runtime_env_vars

root = Path(os.environ['AGENT_ROOT'])
job = os.environ['SLURM_JOB_ID']
identity = {'SEARCH_R1_TOPD_AUDIT_DIR':str(root/f'logs/ray-env-probe-{job}'),
            'SEARCH_R1_TOPD_RUN_ID':f'cpu-infrastructure-probe-{job}',
            'SEARCH_R1_TOPD_COMMIT':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'SEARCH_R1_TOPD_CONFIG_SHA256':hashlib.sha256(b'CPU infrastructure probe; no training config').hexdigest(),
            'SEARCH_R1_TOPD_DATA_IDENTITY':'fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'}
os.environ.update(identity)

@ray.remote(num_cpus=1)
class Worker:
    def inspect(self):
        return {'environment':runtime_env_vars(),'pid':os.getpid(),
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}

@ray.remote(num_cpus=0)
def driver():
    environment = runtime_env_vars()
    actors = [Worker.options(runtime_env={'env_vars':dict(environment, RANK=str(rank))}).remote()
              for rank in range(8)]
    results = ray.get([actor.inspect.remote() for actor in actors])
    return {'driver':{'environment':environment,'pid':os.getpid()},'workers':results}

try:
    ray.init(address='local', num_cpus=8, num_gpus=0, include_dashboard=False,
             object_store_memory=256*1024**2, _memory=8*1024**3,
             _temp_dir=f'/tmp/topd-ray-probe-{job}',
             runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN'}})
    result = ray.get(driver.options(runtime_env={'env_vars':runtime_env_vars()}).remote())
    assert result['driver']['environment'] == identity
    assert len(result['workers']) == 8
    assert len({worker['pid'] for worker in result['workers']}) == 8
    assert all(worker['environment'] == identity for worker in result['workers'])
    (root/'logs/ray-environment-probe.json').write_text(json.dumps(result,indent=2))
    print('REAL_CPU_RAY_DRIVER_AND_EIGHT_WORKER_IDENTITY_PROPAGATION_PASS',flush=True)
finally:
    ray.shutdown()
