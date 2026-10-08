"""Download existing failed-job archive in bounded chunks; never remote mutation."""
import argparse,hashlib,json,re,subprocess,sys,shlex,importlib.util
from pathlib import Path
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
v=json.loads(a.inventory.read_text());assert v['status']=='FAILED_JOB_EXISTING_RAW_ARCHIVE_REQUIRED_FILES_VERIFIED_REQUIRES_INDEPENDENT_REVIEW'
assert v['root']==ROOT and v['job_id']==3156 and v['source_commit']=='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9'
meta=v['archive'];archive=meta['path'];assert re.fullmatch(re.escape(ROOT)+r'/logs/hardware-3156-raw-capture-[A-Za-z0-9T_-]+\.tar\.gz',archive)
assert type(meta['bytes'])is int and meta['bytes']>0 and re.fullmatch('[0-9a-f]{64}',meta['sha256'])
assert not a.output.exists();a.output.mkdir(parents=True);parts=a.output/'parts';parts.mkdir();transport=Path(__file__).parents[1]/'h100_exec.py'
size=512*1024
for offset in range(0,meta['bytes'],size):
 count=min(size,meta['bytes']-offset);part=parts/str(offset);code='import pathlib,sys;p=pathlib.Path('+repr(archive)+');assert p.is_file()and not p.is_symlink();f=p.open("rb");f.seek('+str(offset)+');sys.stdout.buffer.write(f.read('+str(count)+'))'
 with part.open('xb')as out,(parts/(str(offset)+'.stderr')).open('xb')as err:
  subprocess.run([sys.executable,str(transport),'--socks','--node','h100-5','--timeout','60','python3 -c '+shlex.quote(code)],stdout=out,stderr=err,check=True)
 assert part.stat().st_size==count,'Partial chunk retained; never marked accepted'
local=a.output/'raw-evidence.tar.gz'
with local.open('xb')as out:
 for offset in range(0,meta['bytes'],size):
  with (parts/str(offset)).open('rb')as src:
   for b in iter(lambda:src.read(1048576),b''):out.write(b)
assert local.stat().st_size==meta['bytes']and sha(local)==meta['sha256'],'Partial/full archive retained; not accepted'
spec=importlib.util.spec_from_file_location('collector',Path(__file__).with_name('capture-hardware.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.safe_extract(local,a.output/'raw',v)
with (a.output/'download-proof.json').open('x')as f:json.dump(dict(status='FAILED_JOB_RAW_DOWNLOAD_FULL_SHA_SAFE_EXTRACT_ONLY_NOT_ACCEPTANCE',job_id=3156,archive_sha256=sha(local),inventory_sha256=sha(a.inventory),files=len(v['files']),actual_hardware_accepted=False),f,indent=2)
print('FAILED_JOB_RAW_BYTES_VERIFIED_NOT_ACCEPTANCE',flush=True)
