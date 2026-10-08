"""Read original node hosts and derive a local review candidate; no remote writes."""
import base64,datetime,hashlib,json,os,socket,stat
from pathlib import Path
PID=755855
LINE=b"10.33.0.8 h100-16 # BAPO_STOPPED3315_RESOLUTION_REPAIR_20261009_V1\n"
def birth():
 p=Path('/proc')/str(PID)
 f=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert int(f[19])==1433986891 and int(f[1])==1 and p.stat().st_uid==0
 assert (p/'cmdline').read_bytes().replace(b'\0',b' ').strip()==b'slurmstepd: [3315.0]'
 assert os.readlink(p/'exe')=='/usr/sbin/slurmstepd-wlm'
 assert hashlib.sha256((p/'exe').read_bytes()).hexdigest()=='be7770948fcfd24b074745d8add09005f77c1ae0ddb380b86abceb71d2bfdd6a'
 return dict(pid=PID,start_ticks=int(f[19]),state=f[0])
def identity(p):
 s=p.stat()
 return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode),nlink=s.st_nlink)
assert socket.gethostname()=='h100-5'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='9471cbc1-9ae1-4be5-8edb-1e84e850e2cc'
before=birth();p=Path('/etc/hosts');a=identity(p)
assert p.is_file() and not p.is_symlink() and a['nlink']==1 and a['size']<1048576
assert os.path.samefile(p,Path('/proc')/str(PID)/'root/etc/hosts')
raw=p.read_bytes();assert raw.endswith(b'\n')
assert not any('h100-16' in line.split('#',1)[0].split()[1:] for line in raw.decode().splitlines())
assert b'BAPO_STOPPED3315_RESOLUTION_REPAIR' not in raw
attrs={key:base64.b64encode(os.getxattr(p,key)).decode() for key in os.listxattr(p)}
mounts=[line for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4] in ('/','/etc','/etc/hosts')]
assert identity(p)==a and p.read_bytes()==raw
print(json.dumps(dict(scope='READONLY_FULL_BACKUP_AND_SINGLE_LINE_CANDIDATE_NO_NODE_MUTATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),birth_before=before,birth_after=birth(),metadata=a,xattrs_b64=attrs,selected_mountinfo=mounts,original_b64=base64.b64encode(raw).decode(),candidate_b64=base64.b64encode(raw+LINE).decode(),original_sha256=hashlib.sha256(raw).hexdigest(),candidate_sha256=hashlib.sha256(raw+LINE).hexdigest(),added_line=LINE.decode(),action_taken='NONE')))
