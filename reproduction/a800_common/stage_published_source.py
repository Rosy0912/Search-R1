"""Stage an exact published source checkout on A800 without starting services.

The new directory contains code and small receipts only. It is neither a run
manifest nor a deployment/training admission. The existing physical SSH route
and an offline Git bundle are used; no credentials, datasets or weights move.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

from reproduction.a800_common.deploy import PROJECT, published_source, ssh_command, write_once

ROLE='a800_published_source_staging_only'
BASELINE='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab'
BASELINE_COMMIT='34e63b7b12f4056503803905c1b3dd1be95f0adc'
MAX_BUNDLE=32*1024**2


REMOTE=r'''
import hashlib,io,json,os,re,socket,stat,subprocess,sys,tarfile,time
from pathlib import Path
ROLE='a800_published_source_staging_only'
BASELINE='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab'
BASELINE_COMMIT='34e63b7b12f4056503803905c1b3dd1be95f0adc'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def once(path,value):
 b=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def git(*args):
 result=subprocess.run(['git',*args],capture_output=True,text=True,timeout=90,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
 assert result.returncode==0, 'Offline Git staging failed; inspect the same source namespace'
 return result.stdout.strip()
archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|')
first=archive.next();assert first and first.isfile() and first.name=='source-staging-intent.json' and first.size<=1024**2
intent=json.load(archive.extractfile(first))
assert intent['role']==ROLE and intent['host']=='baai-a800'
assert re.fullmatch('[0-9a-f]{32}',intent['staging_id']) and re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
base=Path(intent['base']);assert str(base)=='/data/shaoshitong/Search-G1-actor-probes-source-'+intent['staging_id']
assert base.parent==base.parent.resolve() and base.parent.is_dir()
assert intent['baseline_base']==BASELINE and intent['baseline_commit']==BASELINE_COMMIT
assert intent['production_admission'] is False and intent['launch_services'] is False
baseline=Path(BASELINE);assert baseline==baseline.resolve() and baseline.stat().st_uid==os.getuid()
manifest=baseline/'logs/run-manifest.json'
assert manifest==manifest.resolve() and manifest.is_file() and manifest.stat().st_uid==os.getuid()
old_run=json.loads(manifest.read_bytes())
assert old_run['execution_backend']=='ssh_local' and old_run['execution_host']==socket.gethostname()
assert old_run['run_id']=='1db662001a0f4d998928bcf6f5f33c9d' and old_run['source_commit']==BASELINE_COMMIT
old=baseline/'repo';assert old==old.resolve() and old.stat().st_uid==os.getuid()
assert git('-C',str(old),'rev-parse','HEAD')==BASELINE_COMMIT
base.mkdir(mode=0o700,exist_ok=False)
for name in ('preparation','logs'):(base/name).mkdir()
once(base/'preparation/source-staging-intent.json',intent)
item=archive.next();assert item and item.isfile() and item.name=='published-source.bundle'
assert item.size==intent['bundle_bytes'] and 0<item.size<=32*1024**2
bundle=base/'preparation/published-source.bundle';h=hashlib.sha256();count=0
with bundle.open('xb') as f:
 src=archive.extractfile(item)
 for part in iter(lambda:src.read(1024**2),b''):count+=len(part);h.update(part);f.write(part)
 f.flush();os.fsync(f.fileno())
assert count==item.size and h.hexdigest()==intent['bundle_sha256'] and archive.next() is None
repo=base/'repo'
git('clone','--local','--no-hardlinks','--no-checkout',str(old),str(repo))
git('-C',str(repo),'bundle','verify',str(bundle))
assert git('bundle','list-heads',str(bundle)).splitlines()==[intent['source_commit']+' refs/heads/actor-probes-ten']
git('-C',str(repo),'fetch',str(bundle),'refs/heads/actor-probes-ten')
assert git('-C',str(repo),'rev-parse','FETCH_HEAD')==intent['source_commit']
git('-C',str(repo),'checkout','--detach',intent['source_commit'])
assert git('-C',str(repo),'rev-parse','HEAD')==intent['source_commit']
assert not git('-C',str(repo),'status','--porcelain')
sys.path.insert(0,str(repo))
from reproduction.common8.source_files import runtime_files
actual={}
for p in runtime_files(repo):
 assert p==p.resolve() and p.is_file() and p.stat().st_uid==os.getuid()
 actual[str(p.relative_to(repo))]=sha(p)
assert actual==intent['source_sha256'], 'Staged runtime bytes differ from published blobs'
assert git('-C',str(old),'rev-parse','HEAD')==BASELINE_COMMIT
assert not any((base/name).exists() for name in ('logs/run-manifest.json','logs/process-ledger.json',
 'logs/pipeline-launch-claim.json','logs/training-launch-claim.json','data','checkpoints','env'))
receipt=dict(role=ROLE,staging_id=intent['staging_id'],base=str(base),source_commit=intent['source_commit'],
 source_sha256=actual,runtime_file_count=len(actual),bundle_sha256=sha(bundle),
 exact_published_source_verified=True,git_tree_clean=True,baseline_source_unchanged=True,
 services_started=False,trainer_started=False,production_admission=False,
 transport_scope='published Git objects only; no untracked filesystem datasets/models/environments included',
 checked_unix=time.time())
once(base/'logs/source-staging-complete.json',receipt)
print('SOURCE_STAGING_JSON:'+json.dumps(receipt),flush=True)
'''


def create_source_bundle(repo,commit,bundle,*,baseline_commit=BASELINE_COMMIT):
    """The named bundle ref itself must identify the pinned published commit."""
    tip=subprocess.check_output(['git','rev-parse','refs/heads/actor-probes-ten'],cwd=repo,text=True).strip()
    assert tip==commit, 'Local actor-probes-ten ref differs from the pinned published commit'
    assert not bundle.exists() and not bundle.is_symlink()
    subprocess.run(['git','bundle','create',str(bundle),'refs/heads/actor-probes-ten','^'+baseline_commit],cwd=repo,check=True,capture_output=True)
    heads=subprocess.check_output(['git','bundle','list-heads',str(bundle)],cwd=repo,text=True).splitlines()
    assert heads==[commit+' refs/heads/actor-probes-ten']
    raw=bundle.read_bytes();assert 0<len(raw)<=MAX_BUNDLE
    return raw


def stage(commit, handle):
    repo=Path(__file__).resolve().parents[2]
    assert re.fullmatch('[0-9a-f]{40}',commit)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/actor-probes-ten'],cwd=repo,text=True).split()[0]
    assert remote==commit, 'Stage only the actually pushed actor-probes-ten source'
    source=published_source(repo,commit)
    identifier=uuid.uuid4().hex;handle=Path(handle).resolve()
    bundle=handle.with_name(handle.stem+'.source.bundle')
    raw=create_source_bundle(repo,commit,bundle)
    intent=dict(role=ROLE,host='baai-a800',staging_id=identifier,
        base='/data/shaoshitong/Search-G1-actor-probes-source-'+identifier,
        source_commit=commit,source_sha256=source,baseline_base=BASELINE,baseline_commit=BASELINE_COMMIT,
        bundle_bytes=len(raw),bundle_sha256=hashlib.sha256(raw).hexdigest(),
        production_admission=False,launch_services=False,created_unix=time.time())
    write_once(handle,intent)
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as archive:
        for name,content in [('source-staging-intent.json',json.dumps(intent,allow_nan=False).encode()),('published-source.bundle',raw)]:
            member=tarfile.TarInfo(name);member.size=len(content);archive.addfile(member,io.BytesIO(content))
    error_path=handle.with_name(handle.stem+'.stderr')
    with error_path.open('xb') as errors:
        result=subprocess.run(ssh_command(REMOTE),input=data.getvalue(),stdout=subprocess.PIPE,stderr=errors,timeout=120)
    assert result.returncode==0, 'Source staging failed/uncertain; inspect the same intent, never launch a trainer'
    rows=[json.loads(line.removeprefix('SOURCE_STAGING_JSON:')) for line in result.stdout.decode().splitlines()
          if line.startswith('SOURCE_STAGING_JSON:')]
    assert len(rows)==1;receipt=rows[0]
    assert receipt['source_commit']==commit and receipt['source_sha256']==source
    assert receipt['staging_id']==identifier and receipt['base']==intent['base']
    assert receipt['exact_published_source_verified'] is True and receipt['git_tree_clean'] is True
    assert receipt['services_started'] is False and receipt['trainer_started'] is False
    proof=handle.with_name(handle.stem+'.complete.json');write_once(proof,receipt)
    bundle.unlink()  # Remote Git checkout and bundle hashes were actually verified.
    return dict(handle=str(handle),receipt=str(proof),base=intent['base'],source_commit=commit,
                runtime_file_count=len(source),source_only=True,new_training_started=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True)
    parser.add_argument('--handle',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(stage(args.source_commit,args.handle)))


if __name__=='__main__':main()
