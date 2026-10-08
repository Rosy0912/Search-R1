import os, torch
from native_job import sanitize_runtime_env
from verl.single_controller.base.worker import Worker
from verl.utils.ray_utils import ray_noset_visible_devices
assert not torch.cuda.is_initialized()
saved=dict(os.environ)
try:
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    os.environ['ROCR_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    try:
        Worker.__new__(Worker)._setup_env_cuda_visible_devices()
    except ValueError as e:
        assert 'ROCR_VISIBLE_DEVICES' in str(e)
    else:
        raise AssertionError('Original runtime did not reproduce conflict')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','HABANA_VISIBLE_MODULES','NEURON_RT_VISIBLE_CORES','TPU_VISIBLE_CHIPS','ONEAPI_DEVICE_SELECTOR'):
        os.environ['RAY_EXPERIMENTAL_NOSET_'+name]='1'
    os.environ['HIP_VISIBLE_DEVICES']='0'; os.environ['GPU_DEVICE_ORDINAL']='0'
    sanitize_runtime_env(os.environ)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
    assert not ray_noset_visible_devices()
    assert not any(k in os.environ for k in ('ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'))
    Worker.__new__(Worker)._setup_env_cuda_visible_devices()
    assert not torch.cuda.is_initialized()
    print('ACTUAL_AUTHOR_RUNTIME_ENV_CPU_PASS synthetic=0RL CUDA_uninitialized=True original_conflict_reproduced=True')
finally:
    os.environ.clear();os.environ.update(saved)
