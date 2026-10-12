import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');a=json.loads((root/'tools/gpu-faiss-wheel.json').read_text());p=root/'wheelhouse'/a['filename'];part=p.with_name(p.name+'.partial')
def valid(x):return x.is_file() and x.stat().st_size==a['size'] and hashlib.sha256(x.read_bytes()).hexdigest()==a['sha256']
if not valid(p):
 subprocess.run(['curl','--proxy','http://10.33.0.16:3128','--http1.1','-fLsS','--connect-timeout','10','--retry','2','-C','-','-o',str(part),a['url']],check=True)
 assert valid(part);part.replace(p)
print('GPU_FAISS_SOURCE_SHA_VERIFIED',a['sha256'],flush=True)
while not (root/'logs/environment.ready').exists():
 exitfile=root/'logs/runtime-prepare-20261003.exit'
 if exitfile.exists():raise RuntimeError('Runtime preparation ended before environment acceptance')
 time.sleep(30)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['UV_CACHE_DIR']=str(root/'cache/uv')
subprocess.run([str(root/'tools/uv'),'pip','install','--python',str(root/'env/bin/python'),'--target',str(root/'retriever-gpu-lib'),'--no-deps','--offline',str(p)],check=True)
(root/'logs/gpu-faiss-overlay.installed').write_text(a['sha256']+'\n')
print('GPU_FAISS_OVERLAY_INSTALLED_NOT_GPU_VERIFIED',flush=True)
