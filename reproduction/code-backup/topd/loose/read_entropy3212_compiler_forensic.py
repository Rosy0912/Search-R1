import sys,pathlib,json,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
src=r'''
import pathlib,json,hashlib,base64,subprocess
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');folder=R/'logs/entropy-device-proof-3212';out={'job':3212,'scope':'READ_ONLY_POST_JOB_SOURCE_CACHE_NOT_HISTORICAL_WORKER_ENV','files':[]}
for p in (folder/'original-compiled-cache').rglob('*.py'):
 b=p.read_bytes();out['files'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size':len(b),'base64':base64.b64encode(b).decode()})
spec="""
import os,json,torch,inspect
import torch._dynamo.config as d
import torch._inductor.config as i
names=['TORCH_COMPILE_DISABLE','TORCHDYNAMO_DISABLE','TORCHINDUCTOR_COMPILE_THREADS','TORCHDYNAMO_SUPPRESS_ERRORS']
print(json.dumps({'scope':'POST_JOB_NEW_CPU_INTERPRETER_NOT_WORKER_ENV','env':{k:os.environ.get(k) for k in names},'torch_version':torch.__version__,'actual_defaults':{'compile_threads':i.compile_threads,'suppress_errors':d.suppress_errors,'disable':d.disable},'backend_source':inspect.getsource(torch._TorchCompileInductorWrapper),'cuda_initialized':torch.cuda.is_initialized()}))
"""
r=subprocess.run([str(R/'env/bin/python'),'-B','-c',spec],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=30);out['native_metadata']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out))
'''
r=send(src,attempts=3,retry_safe=True,timeout=60);r.check_returncode();d=json.loads(r.stdout);folder=pathlib.Path('outputs/remote-evidence/entropy-device3212/compiler-forensic');folder.mkdir(exist_ok=True)
for f in d['files']:
 b=base64.b64decode(f.pop('base64'));assert hashlib.sha256(b).hexdigest()==f['sha256'];p=folder/(pathlib.Path(f['path']).name+'.'+f['sha256'][:12]);p.write_bytes(b);f['local_path']=str(p.resolve())
p=folder/'readonly-source-cache-native-metadata.json';p.write_text(json.dumps(d,indent=2)+'\n');print(d['native_metadata']['stdout']);print('cachefiles',len(d['files']))
