python3 - <<'PY'
import pathlib,subprocess,json,hashlib,os,sysconfig,time
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/assets/compiler-headers/python3.10-3.10.12-22.04.18-amd64');overlay=base/'overlay';overlay.mkdir(exist_ok=False)
shas={'libpython3.10-dev':'c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9','python3.10-dev':'7caa163f4d83ae871f985ec4339fb4c96ae7ef8cb3568b8aec9138f86aacbeb7'};packages={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in shas.items():
 p=base/'packages'/(name+'_3.10.12-1~22.04.18_amd64.deb');assert sha(p)==h
 metadata=subprocess.check_output(['dpkg-deb','--field',str(p)],text=True);assert 'Version: 3.10.12-1~22.04.18' in metadata and 'Architecture: amd64' in metadata
 subprocess.run(['dpkg-deb','--extract',str(p),str(overlay)],check=True);packages[p.name]={'sha256':h,'size':p.stat().st_size,'url':'https://security.ubuntu.com/ubuntu/pool/main/p/python3.10/'+p.name,'metadata':metadata}
headers={};links={}
for p in sorted(overlay.rglob('*')):
 n=str(p.relative_to(overlay))
 if p.is_symlink():links[n]=os.readlink(p)
 elif p.is_file():headers[n]={'size':p.stat().st_size,'sha256':sha(p)}
assert (overlay/'usr/include/python3.10/Python.h').is_file() and (overlay/'usr/include/x86_64-linux-gnu/python3.10/pyconfig.h').is_file()
record={'scope':'OFFICIAL_STATIC_HEADERS_NO_GLOBAL_INSTALL_CPU_CAPI_ONLY','actual_node':subprocess.check_output(['hostname'],text=True).strip(),'packages':packages,'overlay_directory':str(overlay),'cpath':[str(overlay/'usr/include/python3.10'),str(overlay/'usr/include')],'files':headers,'symlinks':links,'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],'actual_python_deb_version':subprocess.check_output(['dpkg-query','-W','-f=${Version}','python3.10'],text=True),'epoch':time.time()}
manifest=base/'inventory.json'
with manifest.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
smoke=base/'cpu-capi-smoke';smoke.mkdir(exist_ok=False)
source=smoke/'header_smoke.c';source.write_text('#include <Python.h>\nstatic PyObject *ver(PyObject *s,PyObject *a){return Py_BuildValue("(ss)",PY_VERSION,Py_GetVersion());}\nstatic PyMethodDef methods[]={{"versions",ver,METH_NOARGS,"versions"},{NULL,NULL,0,NULL}};\nstatic struct PyModuleDef mod={PyModuleDef_HEAD_INIT,"header_smoke",NULL,-1,methods};\nPyMODINIT_FUNC PyInit_header_smoke(void){return PyModule_Create(&mod);}\n')
python='/dataset/shaoshitong/search-r1-rl/topd/env/bin/python'
script='import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))';suffix=subprocess.check_output([python,'-c',script],text=True).strip();binary=smoke/('header_smoke'+suffix)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',CPATH=':'.join(record['cpath']))
cmd=['gcc','-shared','-fPIC','-O2',str(source),'-o',str(binary)];compile_result=subprocess.run(cmd,env=env,capture_output=True,text=True);assert compile_result.returncode==0,compile_result.stderr
check='import sys,json,header_smoke;v=header_smoke.versions();assert v[0]=="3.10.12" and v[1].startswith("3.10.12");print(json.dumps({"compiled_header_version":v[0],"loaded_runtime_version":v[1],"actual_executable":sys.executable,"actual_module":header_smoke.__file__,"cuda_modules_loaded":any(n.startswith("torch.cuda") for n in sys.modules)}))'
result=subprocess.run([python,'-c',check],cwd=smoke,env=env,capture_output=True,text=True);assert result.returncode==0,result.stderr
receipt={'status':'CPU_CAPI_COMPILE_IMPORT_PASS_NO_GPU_PROOF','inventory_sha256':sha(manifest),'compile_command':cmd,'compile_stderr':compile_result.stderr,'source_sha256':sha(source),'binary_sha256':sha(binary),'actual_import':json.loads(result.stdout),'CPATH':env['CPATH'],'epoch':time.time()}
with (base/'cpu-capi-smoke.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps({'inventory':record,'inventory_sha256':sha(manifest),'receipt':receipt,'receipt_sha256':sha(base/'cpu-capi-smoke.json')}))
PY
