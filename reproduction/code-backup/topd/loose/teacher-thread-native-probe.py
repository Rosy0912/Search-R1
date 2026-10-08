"""Real native CPU/Ray resource propagation probe; no teacher/model/GPU training."""
import json
import os
from pathlib import Path
import ray

root = Path(os.environ['AGENT_ROOT'])
job = os.environ['SLURM_JOB_ID']
variables = {
    'SEARCH_R1_TOPD_AUDIT_DIR': str(root / 'logs' / ('thread-probe-' + job)),
    'SEARCH_R1_TOPD_RUN_ID': 'cpu-resource-probe-' + job,
    'SEARCH_R1_TOPD_COMMIT': os.environ['PROBE_COMMIT'],
    'SEARCH_R1_TOPD_CONFIG_SHA256': 'infrastructure-probe-no-training-config',
    'SEARCH_R1_TOPD_DATA_IDENTITY': 'infrastructure-probe-no-training-data',
    'SEARCH_R1_TOPD_TEACHER_CPU_THREADS': '16',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
}
ray.init(num_cpus=10, num_gpus=0, include_dashboard=False,
         _temp_dir='/tmp/topd-th-' + job)
try:
    @ray.remote(num_cpus=0, runtime_env={'env_vars':variables})
    class Driver:
        def probe(self):
            import torch
            from search_r1.baselines.topd_runtime_audit import configure_teacher_cpu_threads, runtime_env_vars
            before = torch.get_num_threads()
            configure_teacher_cpu_threads()
            return dict(pid=os.getpid(), before=before, after=torch.get_num_threads(),
                        marker=runtime_env_vars()['SEARCH_R1_TOPD_TEACHER_CPU_THREADS'])
    @ray.remote(num_cpus=1, runtime_env={'env_vars':variables})
    class Actor:
        def probe(self):
            import torch
            from search_r1.baselines.topd_runtime_audit import runtime_env_vars
            return dict(pid=os.getpid(), threads=torch.get_num_threads(),
                        marker=runtime_env_vars()['SEARCH_R1_TOPD_TEACHER_CPU_THREADS'],
                        cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))
    driver = Driver.remote()
    actors = [Actor.remote() for _ in range(8)]
    result = dict(scope='CPU resource infrastructure only; no model or training iterations',
                  job_id=job, commit=os.environ['PROBE_COMMIT'],
                  driver=ray.get(driver.probe.remote()), actors=ray.get([a.probe.remote() for a in actors]))
    assert result['driver']['before'] == 1 and result['driver']['after'] == 16
    assert all(a['threads'] == 1 and a['marker'] == '16' and a['cuda'] == '' for a in result['actors'])
    assert len({result['driver']['pid']} | {a['pid'] for a in result['actors']}) == 9
    (root / 'logs/teacher-thread-native-probe.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_RAY_DRIVER16_EIGHT_ACTORS1_PASS_NO_TRAINING', flush=True)
finally:
    ray.shutdown()
