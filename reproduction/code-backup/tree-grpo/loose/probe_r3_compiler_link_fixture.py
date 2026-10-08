import json,os,socket,subprocess,tempfile,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
include=str(root/'python-dev/usr/include/python3.10')+':'+str(root/'python-dev/usr/include')
env=dict(os.environ);env['C_INCLUDE_PATH']=include
with tempfile.TemporaryDirectory(prefix='tree-own-header-link-',dir=root/'instruct300') as temporary:
 d=Path(temporary);source=d/'main.c';source.write_text('#include <Python.h>\nint tree_header_link_probe(void) {return PY_MAJOR_VERSION;}\n')
 target=d/'tree_header_link_probe.so'
 args=['/usr/bin/gcc',str(source),'-O3','-shared','-fPIC','-o',str(target),'-lcuda','-L'+str(root/'env/lib/python3.10/site-packages/triton/backends/nvidia/lib'),'-L/lib/x86_64-linux-gnu','-I'+str(root/'env/lib/python3.10/site-packages/triton/backends/nvidia/include'),'-I'+str(d),'-I/usr/include/python3.10']
 p=subprocess.run(args,capture_output=True,text=True,env=env,timeout=20)
 result={'host':socket.gethostname(),'epoch':time.time(),'C_INCLUDE_PATH':include,'argv':args,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'output_created':target.is_file(),'shared_library_loaded':False,'cuda_initialized':False,'model_or_tensor_assets_loaded':False,'scope':'Own tiny CPU compiler/link fixture using failed3280 GCC header/CUDA link directories. Not original Triton generated launcher or native GPU validation.'}
print(json.dumps(result,sort_keys=True))
