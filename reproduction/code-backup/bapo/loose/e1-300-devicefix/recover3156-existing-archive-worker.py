"""Recover a complete pre-existing owned archive; never rearchive or signal."""
from pathlib import Path
import os,json,hashlib,tarfile,time,re,subprocess
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004');JOB=3156
SOURCE='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9';logs=ROOT/'logs';pid=os.getpid()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def writers():
 found=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit()or int(proc.name)==pid:continue
  try:
   if 'python'not in (proc/'comm').read_text():continue
   for fd in (proc/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if target.startswith(str(logs)+'/hardware-3156-raw-capture-'):
      stat=(proc/'stat').read_text().rsplit(')',1)[1].split();found.append(dict(pid=int(proc.name),start_ticks=int(stat[19]),target=target))
    except OSError:pass
  except OSError:pass
 return found
output=logs/'hardware-3156-existing-archive-recovery-result.json'
try:
 assert not output.exists(),'Never replace a previous recovery result'
 active=writers();assert not active,repr(active)
 archives=sorted(logs.glob('hardware-3156-raw-capture-*.tar.gz'));assert len(archives)==1,'Ambiguous/missing original archive; do not recreate'
 archive=archives[0];assert archive.is_file()and not archive.is_symlink();before=archive.stat()
 records=[];seen=set()
 with tarfile.open(archive,'r:gz')as tf:
  for member in tf:
   path=Path(member.name);assert member.isfile()and not path.is_absolute()and '..'not in path.parts and member.name not in seen;seen.add(member.name)
   current=ROOT/path;assert current.resolve()==current and current.is_file()and not current.is_symlink()
   h=hashlib.sha256();count=0
   with tf.extractfile(member)as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b);count+=len(b)
   assert count==member.size==current.stat().st_size and h.hexdigest()==sha(current)
   records.append(dict(path=str(current),archive_path=member.name,bytes=count,sha256=h.hexdigest()))
 assert not writers(),'Original writer appeared; no recovery accepted'
 after=archive.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 manifest=json.loads((ROOT/'logs/tools-manifest.json').read_text());assert len(manifest['files'])==21
 actual=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip();assert actual==SOURCE
 assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
 result=dict(status='FAILED_JOB_EXISTING_RAW_ARCHIVE_CURRENT_MEMBER_HASH_VERIFIED_NOT_ACCEPTANCE',root=str(ROOT),job_id=JOB,source_commit=SOURCE,files=records,archive=dict(path=str(archive),bytes=after.st_size,sha256=sha(archive)),collector_failure='Original collector stale20tool assertion occurs after tar completion; recovered existing immutable tar without new archive',no_signals=True,no_new_archive=True,actual_hardware_accepted=False,time=time.time())
except Exception as e:
 result=dict(status='RECOVERY_REFUSED_NO_ARCHIVE_OR_PROCESS_CHANGED',job_id=JOB,error=type(e).__name__+': '+str(e),actual_hardware_accepted=False,time=time.time())
with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
