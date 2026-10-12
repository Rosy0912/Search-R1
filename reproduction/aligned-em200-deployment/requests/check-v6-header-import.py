import os,json,subprocess,hashlib
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009');h=Path('/root/miniconda3/pkgs/python-3.10.21-h17756b0_0/include/python3.10');assert (h/'pyconfig.h').is_file()
c=r/'inputs/header_import_smoke.c';c.write_text('#include <Python.h>\nstatic struct PyModuleDef m={PyModuleDef_HEAD_INIT,"_header_import_smoke",NULL,-1,NULL};\nPyMODINIT_FUNC PyInit__header_import_smoke(void){return PyModule_Create(&m);}\n')
o=r/'inputs/_header_import_smoke.so';p=subprocess.run(['/usr/bin/gcc','-shared','-fPIC','-I'+str(h),str(c),'-o',str(o)],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
e=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',CPATH=str(h));e.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(r/'env/bin/python'),'-B','-c','import sys,sysconfig;sys.path.insert(0,'+repr(str(r/'inputs'))+');import _header_import_smoke;print(sys.version);print(sysconfig.get_config_var("Py_DEBUG"));print(sysconfig.get_config_var("SIZEOF_VOID_P"))'],env=e,capture_output=True,text=True,timeout=20)
print(json.dumps({'compile_exit':0,'actual_python_import_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'headers_sha256':{n:hashlib.sha256((h/n).read_bytes()).hexdigest()for n in ['Python.h','pyconfig.h']},'scope':'CPU_EXTENSION_ABI_IMPORT_ONLY_NOT_TRAINING_ACCEPTANCE'}));assert p.returncode==0
