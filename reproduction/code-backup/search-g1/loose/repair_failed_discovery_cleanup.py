"""One journalled, content-bound metadata repair for failed native3274 only.

The immutable original runtime and logs are retained. No GPU work, service or
training launch. A repeat/unknown result must be inspected through the claim.
"""
import json
import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--purged-job',action='store_true');args=parser.parse_args()
    launch = json.loads((PREP/'policy-discovery-native-launch.json').read_text())
    assert launch['job_id']=='3274'
    intent=dict(base=launch['base'],job_id='3274',source_sha256=launch['source_sha256'],
                created_unix=time.time(),scope='Only verified original TAR-extracted cache inode metadata; then original strict owned cleanup')
    if args.purged_job:
        preceding=json.loads((PREP/'policy-discovery-purged-job-cleanup-observation.json').read_text())
        assert preceding['repair_intent_exists'] is False and preceding['repair_complete_exists'] is False
        assert preceding['original_cache_inode']['uid']==1012
        historical=json.loads((PREP/'policy-discovery-native-observation.json').read_text())
        assert historical['job_id']=='3274' and 'JobState=FAILED ' in historical['job_stdout']
        assert historical['base']==launch['base'] and not historical['actual_updates'] and historical['source_hashes_match']
        intent['historical_failed_job']=dict(job_stdout=historical['job_stdout'],checked_unix=historical['checked_unix'],
             observation_sha256=hashlib.sha256((PREP/'policy-discovery-native-observation.json').read_bytes()).hexdigest(),
             accounting_unavailable='Slurm accounting storage is disabled',current_job_purged=True)
    path=PREP/('policy-discovery-owned-cleanup-repair-purged-intent.json' if args.purged_job else 'policy-discovery-owned-cleanup-repair-intent.json')
    with path.open('x') as f:
        json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
    descriptor=os.open(PREP,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    code=r'''
import hashlib,json,os,stat,sys,tarfile,time,subprocess
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);assert p['job_id']=='3274'
assert not b.is_symlink() and b.resolve()==b
for name,sha in p['source_sha256'].items():assert hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==sha
current_job=subprocess.run(['scontrol','show','job','3274','-o'],capture_output=True,text=True)
if 'historical_failed_job' not in p:
 assert current_job.returncode==0 and 'JobId=3274 ' in current_job.stdout and 'JobState=FAILED ' in current_job.stdout
else:
 historical=p['historical_failed_job']
 assert 'JobId=3274 ' in historical['job_stdout'] and 'JobState=FAILED ' in historical['job_stdout']
 assert str(b/'repo/reproduction/common8/launch_calibration_native.sh') in historical['job_stdout']
 assert current_job.returncode!=0 and not current_job.stdout and 'Invalid job id specified' in current_job.stderr
 queue=subprocess.run(['squeue','-j','3274','--noheader','--format=%i|%T'],capture_output=True,text=True)
 assert not queue.stdout and queue.returncode!=0 and 'Invalid job id specified' in queue.stderr
run=json.loads((b/'logs/run-manifest.json').read_text());assert run['run_id']=='cc365db0dcbf4526a9a0e79f3fc333b9'
ledger=json.loads((b/'logs/process-ledger.json').read_text());assert ledger['run_id']==run['run_id']
assert all(j['state']=='STOPPED' for j in ledger['jobs'].values())
marker=('SEARCH_G1_BASE='+str(b)).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if marker in (proc/'environ').read_bytes().split(b'\0'):assert (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
sys.path.insert(0,str(b/'repo'))
from reproduction.common8.stage_retrieval import assert_quiescent,ASSETS,cleanup as cleanup_stage
from reproduction.common8.temporary_cache import cleanup as cleanup_cache
stage=json.loads((b/'logs/retrieval-staging-claim.json').read_text());claim=json.loads((b/'logs/corpus-cache-claim.json').read_text())
assert stage['run_id']==claim['run_id']==run['run_id'] and stage['slurm_job_id']==claim['slurm_job_id']=='3274'
assert_quiescent(b,stage)
parent=Path(stage['root']);root=Path(claim['root']);assert root==parent/'datasets-cache'
assert parent==Path('/dev/shm/search-g1-common8-'+run['run_id']+'-3274')
for path,device,inode in ((parent,claim['parent_device'],claim['parent_inode']),(root,claim['root_device'],claim['root_inode'])):
 s=path.lstat();assert stat.S_ISDIR(s.st_mode) and s.st_uid==claim['uid']==os.getuid() and (s.st_dev,s.st_ino)==(device,inode)
abnormal=[]
for path in root.rglob('*'):
 s=path.lstat();assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)
 if s.st_uid!=claim['uid']:abnormal.append(path)
assert len(abnormal)==1;target=abnormal[0]
member='data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl'
relative=target.relative_to(root);parts=relative.parts
assert parts[:2]==('downloads','extracted') and len(parts[2])==64 and all(c in '0123456789abcdef' for c in parts[2])
assert Path(*parts[3:]).as_posix()==member
fds=[]
def hashed(stream):
 h=hashlib.sha256();size=0
 while True:
  data=stream.read(2*1024*1024)
  if not data:break
  h.update(data);size+=len(data)
 return h.hexdigest(),size
try:
 directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);fds.append(directory)
 assert (os.fstat(directory).st_dev,os.fstat(directory).st_ino)==(claim['root_device'],claim['root_inode'])
 for part in parts[:-1]:
  directory=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory);fds.append(directory)
  assert os.fstat(directory).st_uid==claim['uid']
 target_fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory);fds.append(target_fd)
 before=os.fstat(target_fd);assert stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode)==0o664 and before.st_uid==before.st_gid==1012 and before.st_size==14393573105 and before.st_nlink==1
 assert (target.lstat().st_dev,target.lstat().st_ino)==(before.st_dev,before.st_ino)
 source=parent/'wiki-18.jsonl';source_fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW);fds.append(source_fd)
 src=os.fstat(source_fd);assert stat.S_ISREG(src.st_mode) and src.st_uid==claim['uid'] and src.st_size==ASSETS['wiki-18.jsonl']['size']
 with os.fdopen(os.dup(source_fd),'rb') as stream:
  source_sha,source_size=hashed(stream)
  assert source_sha==ASSETS['wiki-18.jsonl']['sha256'] and source_size==src.st_size
  stream.seek(0)
  with tarfile.open(fileobj=stream,mode='r:') as archive:
   members=archive.getmembers();assert len(members)==1;m=members[0]
   assert m.name==member and m.isfile() and not m.linkname and m.uid==m.gid==1012 and m.size==before.st_size and m.mode==0o664
   with archive.extractfile(m) as content:member_sha,member_size=hashed(content)
 with os.fdopen(os.dup(target_fd),'rb') as stream:target_sha,target_size=hashed(stream)
 assert target_sha==member_sha and target_size==member_size==m.size
 after=os.fstat(target_fd)
 assert (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_uid,after.st_gid,after.st_nlink)==(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_uid,before.st_gid,1)
 proof=dict(run_id=run['run_id'],slurm_job_id='3274',original_source_sha256=source_sha,original_member=member,member_size=member_size,member_sha256=member_sha,target_sha256=target_sha,target_relative=str(relative),target_device=before.st_dev,target_inode=before.st_ino,target_nlink=before.st_nlink,uid_before=before.st_uid,gid_before=before.st_gid,uid_after=claim['uid'],gid_after=os.getgid(),checked_unix=time.time(),scope='Only this exact original-member content and fd-bound extracted cache inode; no model/data content or frozen source edits')
 with (b/'preparation/owned-cache-metadata-repair-intent.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
 journal_fd=os.open(b/'preparation',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(journal_fd)
 finally:os.close(journal_fd)
 os.fchown(target_fd,claim['uid'],os.getgid())
 os.fsync(target_fd)
 assert os.fstat(target_fd).st_uid==claim['uid'] and os.fstat(target_fd).st_gid==os.getgid() and os.fstat(target_fd).st_nlink==1
 with (b/'preparation/owned-cache-metadata-repair-complete.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
 journal_fd=os.open(b/'preparation',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(journal_fd)
 finally:os.close(journal_fd)
finally:
 for fd in reversed(fds):os.close(fd)
cache=cleanup_cache(b);stage_cleanup=cleanup_stage(b)
assert not parent.exists() and not parent.is_symlink()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if marker in (proc/'environ').read_bytes().split(b'\0'):assert (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
result=dict(base=str(b),job_id='3274',run_id=run['run_id'],metadata_repair=proof,corpus_cache_cleanup=cache,retrieval_staging_cleanup=stage_cleanup,owned_temporary_cleanup_complete=True,owned_namespace_empty=True,owned_shm_root_absent=True,cleaned=True,failed_training_run_remains_failed=True,actual_completed_updates=0,completed_outer_updates=0,production_admission=False,checked_unix=time.time(),historical_failed_job=p.get('historical_failed_job'),current_slurm_record_purged='historical_failed_job' in p)
with (b/'preparation/manual-owned-cleanup-complete.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''
    compile(code,'<scoped-old-run-cleanup>','exec')
    result=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),
                          input=json.dumps(intent),capture_output=True,text=True,timeout=55)
    assert result.returncode==0,'Inspect same saved intent and remote repair claim before any repeat: '+result.stderr[-3000:]
    proof=json.loads(result.stdout)
    with (PREP/'policy-discovery-owned-cleanup-repair-complete.json').open('x') as f:
        json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in proof.items() if k not in ('corpus_cache_cleanup','retrieval_staging_cleanup')},indent=2))


if __name__=='__main__':main()
