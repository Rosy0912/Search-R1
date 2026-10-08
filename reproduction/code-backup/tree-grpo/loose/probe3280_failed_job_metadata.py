import hashlib,json,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
out={'host':socket.gethostname(),'epoch':time.time(),'commands':{},'headers':{},'gpu_or_tensor_loaded_by_reader':False}
for args in [['sacct','-j','3280','--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES','--parsable2'],['scontrol','show','job','3280']]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);out['commands'][' '.join(args)]={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for rel in ['python-dev/usr/include/python3.10/Python.h','python-dev/usr/include/python3.10/pyconfig.h','python-dev/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h','logs/python-headers-acceptance.json']:
 p=root/rel;out['headers'][rel]={'exists':p.is_file(),'is_symlink':p.is_symlink(),'resolved':str(p.resolve())}
 if p.is_file():
  b=p.read_bytes();out['headers'][rel].update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  if p.name.endswith('.json'):out['headers'][rel]['text']=b.decode()
include=str(root/'python-dev/usr/include/python3.10')+':'+str(root/'python-dev/usr/include')
import os
env=dict(os.environ);env['C_INCLUDE_PATH']=include
p=subprocess.run(['/usr/bin/gcc','-x','c','-fsyntax-only','-'],input='#include <Python.h>\nint tree_headers_probe(void) { return PY_MAJOR_VERSION; }\n',capture_output=True,text=True,env=env,timeout=20)
out['actual_same_node_own_cpu_only_compiler_fixture']={'C_INCLUDE_PATH':include,'command':['/usr/bin/gcc','-x','c','-fsyntax-only','-'],'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'code':'#include <Python.h>\nint tree_headers_probe(void) { return PY_MAJOR_VERSION; }\n','no_output_file_or_gpu_or_training':True}
print(json.dumps(out,sort_keys=True))
