import pathlib,hashlib,json,socket,os,time,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');tools=root/'instruct300/candidates/recovery-502e9fa-f91c4f17'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
start=time.monotonic();cpu=time.process_time();raw=(tools/'manifest.json').read_bytes();pin=hashlib.sha256(raw).hexdigest()
assert pin=='f91c4f17a3a3454bb16a81e9ca400f4db07a56a0a27cf6cfac2c1814459c3087';m=json.loads(raw)
assert len(m['files'])==694 and m['code_commit']=='502e9fafa8e3aba722dbfcb493abc2b9d3ece326'
for path,value in m['files'].items():assert sha(tools/path)==value,path
for path,value in m['external_files'].items():assert sha(path)==value,path
assert sha(m['parent_manifest']['path'])==m['parent_manifest']['sha256']
model=root/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
# Model manifest entries are content-bound, source assets are read only.
identity=m['model_identity'];assert len(identity['files'])==13
for relative,value in identity['files'].items():assert sha(model/relative)==value,relative
assert sha(root/'instruct300/common-data/manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
print(json.dumps(dict(status='INDEPENDENT_ACTUAL_H1009_RECOVERY_TARGET_BYTE_PINS_PASS',hostname=socket.gethostname(),code_commit=m['code_commit'],manifest_sha256=pin,payload_files=694,external_files=len(m['external_files']),official_model_files=13,model_identity=identity['sha256'],data_manifest_sha256=sha(root/'instruct300/common-data/manifest.json'),parent_manifest_sha256=m['parent_manifest']['sha256'],wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],scope='Read-only target actual byte verification, no GPU/native/Slurm or training execution')),flush=True)
