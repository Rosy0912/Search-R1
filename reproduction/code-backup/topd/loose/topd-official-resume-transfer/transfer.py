"""Candidate only: official pinned Mac→ownh11 shard relay; never downloads/deletes local."""
import concurrent.futures
import hashlib
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[2]
STAGING=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-qwen25-32b-20261003')
REMOTE='/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/qwen2.5-32b'
REVISION='1818d35814b8319459f4bd55ed1ac8709630f003'
OFFICIAL_METADATA_SHA='e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'
CHUNK=64*1024**2
ADOPT_LEGACY=False
SSH_COMPRESS=False


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest_file(path,size=None):
    h=hashlib.sha256();left=size
    with Path(path).open('rb') as stream:
        while left is None or left:
            raw=stream.read(8*1024**2 if left is None else min(left,8*1024**2))
            if not raw:break
            h.update(raw)
            if left is not None:left-=len(raw)
    require(left in (None,0),'Source prefix shorter than remote')
    return h.hexdigest()


def legacy_guard(name):
    path=BASE/'outputs/qwen25-32b-download-state.json'
    if path.exists():
        state=json.loads(path.read_text())
        if not state.get('complete') and state.get('active_file')==name:
            try:os.kill(int(state['pid']),0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Official legacy active file; preserve running transfer: '+name)


def route(command):
    worker='env -u LD_LIBRARY_PATH ssh -T -S /dev/shm/topd-worker-connect-v2.sock -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-11 '+shlex.quote(command)
    head='env -u LD_LIBRARY_PATH ssh -T -S none -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-16 '+shlex.quote(worker)
    return ['ssh']+(['-C'] if SSH_COMPRESS else [])+['-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-S','none','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head]


# Pure receiver callable is also tested without SSH. Arbitrary paths exist only
# in tests; CLI-generated remote invocation pins root, host and metadata entry.
RECEIVER=r'''
import fcntl,hashlib,json,os,socket,sys
from pathlib import Path

def require(ok,message):
 if not ok:raise RuntimeError(message)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()

def execute(root,name,size,expected,operation,offset,length,blocksha,source,hostname=None,adopt_legacy=False,prefix_sha=None,exclusive=False):
 root=Path(root)
 if hostname is not None:require(socket.gethostname().split('.')[0]==hostname,'Foreign target host')
 require(root.is_dir() and not root.is_symlink(),'Missing canonical own directory')
 require('/' not in name and (name.endswith('.safetensors') or name in {'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json'}),'Only pinned official file basename')
 target=root/name;partial=root/(name+'.resume-v1.transfer');lock=root/(name+'.resume-v1.lock')
 for p in (target,partial,lock):require(not p.is_symlink(),'Symlink target refused')
 if lock.exists():require(lock.is_file() and lock.stat().st_uid==os.getuid(),'Foreign own lock type/owner')
 fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'r+b') as owner:
  fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for p in (target,partial):
   if p.exists():require(p.is_file() and p.stat().st_uid==os.getuid(),'Foreign target owner/type')
  if target.exists():
   require(target.stat().st_size==size and sha(target)==expected,'Existing final official file mismatch')
   return {'status':'VERIFIED_FINAL','size':size,'sha256':expected}
  legacy=root/(name+'.transfer')
  if operation=='adopt':
   require(adopt_legacy and exclusive,'Explicit stopped-tree exclusive handoff required')
   require(not partial.exists() and legacy.is_file() and not legacy.is_symlink() and legacy.stat().st_uid==os.getuid(),'Own legacy partial only; no replace resume file')
   require(legacy.stat().st_size==offset<=size and prefix_sha and sha(legacy)==prefix_sha,'Legacy full prefix differs')
   # Linux /proc scan finds open own writers/readers; fail closed if uncertain.
   writers=[]
   if hostname is not None:
    for proc in Path('/proc').iterdir():
     if not proc.name.isdigit():continue
     try:
      if proc.stat().st_uid!=os.getuid():continue
      for fd in (proc/'fd').iterdir():
       try:
        if fd.resolve()==legacy.resolve():writers.append(int(proc.name))
       except FileNotFoundError:pass
     except FileNotFoundError:pass
    require(not writers,'Legacy partial still open; preserve active transfer')
   fd=os.open(legacy,os.O_RDONLY|os.O_NOFOLLOW)
   try:os.fsync(fd)
   finally:os.close(fd)
   os.replace(legacy,partial)
   fd=os.open(root,os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
   return {'status':'ADOPTED_EXCLUSIVE_PREFIX','size':offset,'sha256':prefix_sha,'open_legacy_pids':writers}
  if operation=='probe' and adopt_legacy and not partial.exists() and legacy.exists():
   require(not legacy.is_symlink() and legacy.is_file() and legacy.stat().st_uid==os.getuid(),'Foreign legacy partial')
   require(legacy.stat().st_size<=size,'Oversized legacy prefix')
   return {'status':'LEGACY_PREFIX_REQUIRES_EXCLUSIVE_ADOPT','size':legacy.stat().st_size,'sha256':sha(legacy)}
  actual=partial.stat().st_size if partial.exists() else 0
  require(actual<=size,'Oversized own partial')
  if operation=='probe':return {'status':'PARTIAL','size':actual,'sha256':sha(partial) if actual else hashlib.sha256(b'').hexdigest()}
  require(actual==offset,'Offset changed; retry probe without overwrite')
  if operation=='append':
   require(0<length<=64*1024**2 and offset+length<=size,'Bounded append length')
   fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
   with os.fdopen(fd,'r+b') as out:
    out.seek(offset);left=length;h=hashlib.sha256()
    while left:
     raw=source.read(min(left,1024**2))
     if not raw:
      out.flush();os.fsync(out.fileno())
      raise RuntimeError('Interrupted append; exact partial retained for verified resume')
     out.write(raw);h.update(raw);left-=len(raw)
    if h.hexdigest()!=blocksha:
     out.truncate(offset);out.flush();os.fsync(out.fileno())
     raise RuntimeError('Chunk SHA mismatch; rolled back owned chunk')
    out.flush();os.fsync(out.fileno())
   return {'status':'APPENDED','size':offset+length,'block_sha256':blocksha}
  require(operation=='finalize' and actual==size and sha(partial)==expected,'Full official size/SHA gate failed')
  os.replace(partial,target)
  fd=os.open(root,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return {'status':'VERIFIED_FINAL','size':size,'sha256':expected}
'''


def remote_call(name,item,operation,offset=0,raw=b'',prefix_sha=None):
    expected=item['lfs']['sha256']
    invocation=f"print(json.dumps(execute({REMOTE!r},{name!r},{item['size']!r},{expected!r},{operation!r},{offset!r},{len(raw)!r},{hashlib.sha256(raw).hexdigest()!r},sys.stdin.buffer,hostname='h100-11',adopt_legacy={ADOPT_LEGACY!r},prefix_sha={prefix_sha!r},exclusive=True)))"
    command='/dataset/shaoshitong/search-r1-rl/topd/env/bin/python -c '+shlex.quote(RECEIVER+'\n'+invocation)
    result=subprocess.run(route(command),input=raw,capture_output=True,timeout=600)
    require(result.returncode==0,'Receiver failed; own partial retained, reprobe required')
    return json.loads(result.stdout)


def relay(name,item):
    source=STAGING/name;require(source.is_file() and not source.is_symlink(),'Complete official staging source required')
    require(source.stat().st_size==item['size'] and digest_file(source)==item['lfs']['sha256'],'Complete source official size/SHA differs')
    started=time.time();sent=0
    for attempt in range(32):
        legacy_guard(name)
        try:
            probe=remote_call(name,item,'probe')
            if probe['status']=='VERIFIED_FINAL':return dict(name=name,status=probe['status'],sha256=probe['sha256'],sent_bytes=sent,wall_seconds=time.time()-started)
            offset=probe['size'];require(digest_file(source,offset)==probe['sha256'],'Remote prefix mismatch; preserve both copies, do not truncate')
            if probe['status']=='LEGACY_PREFIX_REQUIRES_EXCLUSIVE_ADOPT':
                adopted=remote_call(name,item,'adopt',offset,prefix_sha=probe['sha256'])
                require(adopted['status']=='ADOPTED_EXCLUSIVE_PREFIX' and adopted['size']==offset and adopted['sha256']==probe['sha256'],'Adoption evidence differs')
                print(json.dumps(dict(status=adopted['status'],name=name,offset=offset,prefix_sha256=probe['sha256'],legacy_open_pids=adopted['open_legacy_pids'])),flush=True)
            with source.open('rb') as stream:
                stream.seek(offset)
                while offset<item['size']:
                    legacy_guard(name)
                    raw=stream.read(min(CHUNK,item['size']-offset));require(raw,'Official source disappeared/shortened')
                    ack=remote_call(name,item,'append',offset,raw)
                    require(ack['status']=='APPENDED' and ack['size']==offset+len(raw) and ack['block_sha256']==hashlib.sha256(raw).hexdigest(),'Append ACK differs')
                    offset+=len(raw);sent+=len(raw)
                    print(json.dumps(dict(status='OWN_OFFICIAL_CHUNK_ACK',name=name,verified_remote_bytes=offset,sent_bytes=sent,wall_seconds=time.time()-started)),flush=True)
            legacy_guard(name);final=remote_call(name,item,'finalize',offset)
            require(final['status']=='VERIFIED_FINAL' and final['sha256']==item['lfs']['sha256'],'Final SHA differs')
            return dict(name=name,status=final['status'],sha256=final['sha256'],sent_bytes=sent,wall_seconds=time.time()-started)
        except (subprocess.TimeoutExpired,subprocess.SubprocessError,RuntimeError) as error:
            # Bad prefix is not a recoverable network retry; never overwrite it.
            if 'prefix mismatch' in str(error) or 'legacy active' in str(error):raise
            print(json.dumps(dict(status='OWN_RESUME_REPROBE_REQUIRED',name=name,attempt=attempt)),flush=True);time.sleep(3)
    raise RuntimeError('Retries exhausted; preserve official local and own remote partial')


def official_metadata():
    path=BASE/'outputs/qwen2.5-32b-official-metadata.json'
    require(digest_file(path)==OFFICIAL_METADATA_SHA,'Official complete metadata bytes SHA differs')
    metadata=json.loads(path.read_text());require(metadata['sha']==REVISION,'Wrong official revision')
    return metadata


def exclusive_handoff(path):
    proof=json.loads(Path(path).read_text())
    require(proof['scope']=='LEGACY_TRANSFER_TREE_STOPPED_EXCLUSIVE_HANDOFF','Wrong actual handoff scope')
    require(type(proof.get('legacy_local_identities')) is list and type(proof.get('legacy_remote_identities')) is list,'Full old PPID/starttime identity inventories required')
    identities=proof['legacy_local_identities']
    require(all(type(p.get('pid')) is int and p['pid']>0 and type(p.get('ppid')) is int and p['ppid']>0 and isinstance(p.get('starttime'),(str,int)) and bool(p['starttime']) for p in identities),'Missing actual old process PID/PPID/starttime')
    ids=[p['pid'] for p in identities]
    require(len(ids)==len(set(ids)) and {23765,53280,53396}<=set(ids),'Incomplete known legacy local tree')
    for pid in ids:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Old transfer PID still exists; no candidate startup')
    remote=proof['legacy_remote_identities']
    require(all(p.get('host')=='h100-11' and type(p.get('pid')) is int and p['pid']>0 and type(p.get('ppid')) is int and p['ppid']>0 and bool(p.get('starttime')) for p in remote),'Foreign/missing remote process identity')
    require(len(remote)==len({p['pid'] for p in remote}),'Duplicate old remote process identities')
    if remote:
        script="import json,socket;from pathlib import Path;assert socket.gethostname().split('.')[0]=='h100-11';ids=json.loads("+repr(json.dumps([p['pid'] for p in remote]))+");remaining=[p for p in ids if (Path('/proc')/str(p)).exists()];assert not remaining,'Old target process remains';print('ALL_OLD_TARGET_PIDS_ABSENT')"
        result=subprocess.run(route('/dataset/shaoshitong/search-r1-rl/topd/env/bin/python -c '+shlex.quote(script)),capture_output=True,text=True,timeout=180)
        require(result.returncode==0 and result.stdout.strip()=='ALL_OLD_TARGET_PIDS_ABSENT','Old remote process absence unverified; no startup')
    return proof


def main():
    global ADOPT_LEGACY,SSH_COMPRESS
    parser=argparse.ArgumentParser();parser.add_argument('--exclusive-handoff',required=True,type=Path);parser.add_argument('--adopt-legacy',action='store_true');parser.add_argument('--ssh-compress',action='store_true');parser.add_argument('names',nargs='+');args=parser.parse_args()
    names=args.names;require(0<len(names)<=3 and len(names)==len(set(names)),'Explicit maximum3 complete official shards required')
    proof=exclusive_handoff(args.exclusive_handoff)
    ADOPT_LEGACY=args.adopt_legacy;SSH_COMPRESS=args.ssh_compress
    # The existing official orchestrator owns this same advisory lock. Holding
    # it excludes any restarted official v5 and is retained through all relays.
    with (STAGING/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metadata=official_metadata()
        entries={x['rfilename']:x for x in metadata['siblings'] if x['rfilename'].endswith('.safetensors')}
        require(all(name in entries and entries[name].get('lfs',{}).get('sha256') for name in names),'Only pinned official shards allowed')
        state=json.loads((BASE/'outputs/qwen25-32b-download-state.json').read_text())
        require(not state.get('complete') and set(names)<=set(state['inflight_files']),'Do not expand existing3-file staging window')
        for name in names:legacy_guard(name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,len(names))) as pool:
            futures=[pool.submit(relay,name,entries[name]) for name in names]
            for future in concurrent.futures.as_completed(futures):print(json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
