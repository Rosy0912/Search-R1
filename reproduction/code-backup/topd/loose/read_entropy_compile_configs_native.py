import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
src=r'''
import subprocess,json,pathlib
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
code="""
import os,pathlib,json,importlib.util,inspect,hashlib,torch
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools/entropy-device-proof-v1-r1-final/worker.py');s=importlib.util.spec_from_file_location('owned_entropy_worker_config',p);w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
e,o,c,identity=w.source_functions();keys=['triton.autotune_pointwise','max_autotune','max_autotune_pointwise','coordinate_descent_tuning','dynamic_scale_rblock']
import torch._dynamo as d
import torch._inductor.runtime.runtime_utils as u
out={'scope':'NATIVE_CPU_ONLY_COMPILE_CONFIG_AND_RESET_SOURCE_NO_EXECUTION','torch':torch.__version__,'original':{k:o.get_compiler_config()[k] for k in keys},'candidate':{k:c.get_compiler_config()[k] for k in keys},'original_and_candidate_share_original_callable':o._torchdynamo_orig_callable is c._torchdynamo_orig_callable,'reset_source':inspect.getsource(d.reset),'cache_dir_source':inspect.getsource(u.cache_dir),'cuda_initialized':torch.cuda.is_initialized(),'source_identity':identity}
print(json.dumps(out))
"""
r=subprocess.run([str(R/'env/bin/python'),'-B','-c',code],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'},text=True,capture_output=True,timeout=30);print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
'''
r=send(src,attempts=3,retry_safe=True,timeout=60);r.check_returncode();pathlib.Path('outputs/entropy3212-native-compile-config-and-reset-source.json').write_bytes(r.stdout);print(r.stdout.decode()[:9000])
