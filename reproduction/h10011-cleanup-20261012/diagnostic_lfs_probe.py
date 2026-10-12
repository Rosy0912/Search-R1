import pathlib,subprocess,shlex
code='''import sys,pathlib,json,subprocess,shlex,time,hashlib,re,traceback
sys.path.insert(0,'/home/lichen/h100-cleanup-20261012-bapo')
import uploader_parallel4x4_gated as mod
from huggingface_hub import HfApi,CommitOperationAdd
from huggingface_hub.lfs import UploadInfo
token=sys.stdin.readline().strip();api=HfApi(token=token)
mod.ROOT=pathlib.Path('/home/lichen/h100-cleanup-20261012-bapo/lfs-diagnostic');mod.ROOT.mkdir(exist_ok=True)
source='/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359/checkpoints/actor/global_step_150/model-00001-of-00003.safetensors'
dest='/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/cleanup-lfs80m-diagnostic.bin'
src='import pathlib,hashlib,json;source=pathlib.Path('+repr(source)+');dest=pathlib.Path('+repr(dest)+');f=source.open("rb");o=dest.open("wb");h=hashlib.sha256();left=80000000\\nwhile left:\\n b=f.read(min(left,8388608));o.write(b);h.update(b);left-=len(b)\\no.close();print(json.dumps(dict(path=str(dest),size=dest.stat().st_size,sha256=h.hexdigest())))'
p=subprocess.run(mod.source_args('python3 -c '+shlex.quote(src)),capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr[-1000:]
j=json.loads(p.stdout);rec=dict(j,archive=j['path'],archive_bytes=j['size'],relative='diagnostic-lfs80m')
r=mod.ParallelRemoteFile(rec);t=time.time()
try:
 sample=r.read(512);r.seek(0);op=CommitOperationAdd(path_in_repo='h100-cleanup-20261012/bapo/_diagnostic-lfs80m.bin',path_or_fileobj=sample);op.path_or_fileobj=r;op.upload_info=UploadInfo(size=j['size'],sha256=bytes.fromhex(j['sha256']),sample=sample)
 c=api.create_commit(repo_id='Sutoonq/TMP_MODEL',operations=[op],commit_message='Diagnose gated H100 gateway multipart upload')
 info=api.get_paths_info(repo_id='Sutoonq/TMP_MODEL',paths=[op.path_in_repo],revision=c.oid)[0];assert info.size==j['size'] and info.lfs.sha256==j['sha256']
 result=dict(status='DIAGNOSTIC_MULTIPART_HF_SHA_PASS',commit=c.oid,bytes=j['size'],elapsed=time.time()-t,source_opens=r.source_opens)
except Exception as e:
 chain=[];seen=set()
 while e is not None and id(e) not in seen and len(chain)<7:
  seen.add(id(e));msg=str(e);msg=re.sub(r'https?://[^\\s\\\"\\\']+','[redacted-url]',msg);msg=re.sub(r'\\?[^\\s\\\"\\\']+','?[redacted-query]',msg);msg=re.sub(r'hf_[A-Za-z0-9]+','[redacted-token]',msg)
  chain.append(dict(type=type(e).__name__,message=msg[:1500],frames=[dict(file=pathlib.Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(e.__traceback__)[-10:]]))
  e=e.__cause__ or e.__context__
 stderr=[]
 for p in r.processes:
  if p.poll() is not None:
   err=p.stderr.read(2048).decode(errors='replace');stderr.append(dict(pid=p.pid,returncode=p.returncode,stderr=err))
 result=dict(status='DIAGNOSTIC_MULTIPART_FAILED',elapsed=time.time()-t,source_position=r.position,source_opens=r.source_opens,chain=chain,source_errors=stderr)
finally:r.close()
(mod.ROOT/'result.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host','python3 -c '+shlex.quote(code)],input=pathlib.Path('/Users/shaoshitong/.cache/huggingface/token').read_text().strip()+'\n',capture_output=True,text=True,timeout=220)
pathlib.Path('outputs/cleanup-h10011/lfs-diagnostic.json').write_text(p.stdout)
print(p.returncode,p.stdout,p.stderr[-1000:])
