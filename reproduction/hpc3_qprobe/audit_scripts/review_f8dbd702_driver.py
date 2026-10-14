import os,subprocess,json,time,sys
from pathlib import Path
d=Path("/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/preparation/review-regression-f8dbd702-durable")
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SEARCH_G1_NATIVE4_SMOKE_FIXTURE='/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717765-20261014/failed-job-717827/logs/model-smoke-initial/complete.json',QPROBE_SOURCE='/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/7ddfc0ffec29c3d74fa460c2096e87a5574065ad')
cmd=['/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717827-20261014/env/bin/python','-m','unittest','-v','reproduction.hpc3_qprobe.test_launch_review_fixes','reproduction.common8.test_production_safety_admission','reproduction.common8.test_behavior_runtime','reproduction.common8.test_model_smoke']
start=time.time()
with (d/'tests.log').open('x') as f:
 r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
with (d/'complete.json').open('x') as f:
 json.dump(dict(exit=r.returncode,uid=os.getuid(),start=start,end=time.time(),actual_gpu_allocation=False,training_updates=0),f);f.flush();os.fsync(f.fileno())
