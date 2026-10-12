"""One explicit zero-update recovery of the observed missing Python build headers."""
import pathlib,json,time,requests,subprocess,os
b=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert not pathlib.Path('/proc/3612156').exists()
assert pathlib.Path('/root/miniconda3/pkgs/python-3.10.21-h17756b0_0/include/python3.10/Python.h').is_file()
assert pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/lib/driver/libcuda.so').is_file()
assert 'ACTUAL_TRAIN_ENTRY_EXIT=1 LAUNCH='+str(b/'control/launch-1791789844.sh') in (b/'logs/slurm-3387.log').read_text()
t=(b/'logs/training.log').read_text();assert 'Python.h: No such file or directory' in t and 'epoch 0, step 2' not in t
assert not list((b/'checkpoints').rglob('manifest.json'))
initial=json.loads((b/'output/fresh-initialization.json').read_text());assert initial['driver_completed_outer']==0 and initial['checkpoint_loaded'] is False
h=requests.get('http://127.0.0.1:18086/health',timeout=5).json();i=json.loads((b/'retriever/receipts/source-identity.json').read_text());assert h['pid']==i['pid'] and h['source_identity_sha256']==i['source_identity_sha256']
token=(b/'retriever/receipts/private-probe-token').read_text().strip();r=requests.post('http://127.0.0.1:18086/release-gpu',headers={'x-e1-probe-token':token},timeout=60);r.raise_for_status();release=r.json();assert release['released'] and release['gpu_worker_pid'] is None
headers=pathlib.Path('/root/miniconda3/pkgs/python-3.10.21-h17756b0_0/include/python3.10')
smoke=b/'control/python-header-entry-smoke';smoke.mkdir(exist_ok=False)
c=smoke/'header.c';c.write_text('#include <Python.h>\nstatic struct PyModuleDef m={PyModuleDef_HEAD_INIT,"_rba_header",NULL,-1,NULL};\nPyMODINIT_FUNC PyInit__rba_header(void){return PyModule_Create(&m);}\n')
compile_result=subprocess.run(['/usr/bin/gcc','-shared','-fPIC','-I'+str(headers),str(c),'-o',str(smoke/'_rba_header.so')],capture_output=True,text=True,timeout=20)
assert compile_result.returncode==0,compile_result.stderr
abi=subprocess.run(['/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/env/bin/python','-B','-c','import sys;sys.path.insert(0,'+repr(str(smoke))+');import _rba_header;print("CPU_HEADER_ABI_IMPORT_OK")'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CPATH=str(headers)),capture_output=True,text=True,timeout=20)
assert abi.returncode==0,abi.stderr
(b/'receipts/python-header-entry-smoke.json').write_text(json.dumps({'compile_rc':compile_result.returncode,'abi_import_rc':abi.returncode,'stdout':abi.stdout,'scope':'CPU_ONLY_HEADER_ABI_NOT_TRAINING'}))
preserved=b/'preserved/python-entry-headers-zero-update-3612156';preserved.mkdir(parents=True,exist_ok=False)
(b/'output').rename(preserved/'output');(b/'logs/training.log').rename(preserved/'training.log')
script=(b/'control/launch-1791789844.sh').read_text().replace('source-db40459','source-78eecc1')
assert not (b/'control/launch-ready.sh').exists();(b/'control/launch-ready.sh').write_text(script)
proof={'time':time.time(),'previous_owner':3612156,'committed_updates':0,'initial_state':initial,'release':release,'preserved':str(preserved),'new_runtime_source':'78eecc1','config_unchanged':True}
(b/'receipts/python-entry-headers-zero-update-recovery.json').write_text(json.dumps(proof));print(json.dumps(proof))
