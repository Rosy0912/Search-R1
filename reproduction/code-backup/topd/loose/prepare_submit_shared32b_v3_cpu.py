#!/usr/bin/env python3
"""Default is local verification only. ROOT explicitly invokes --submit once."""
import argparse, hashlib, io, json, pathlib, subprocess, sys, tarfile
NAME = 'topd-shared32b-supervisor-v3'
ARCHIVE_SHA = '817dd72ecc25d8a99149cfb7c5478e98baef38cd24c3426ffc7561a5b2d12a2a'
MANIFEST_SHA = '0919cac556c6d6fbed5e9b49fdc8b03c4b54059b9bbeb4c97db662122e7c1635'
COMMIT = 'd23c56c78fdb09f40b7859e02b14eeddbe83a863'
ROOT = '/dataset/shaoshitong/search-r1-rl/topd'

def frozen_payload(blob):
    assert hashlib.sha256(blob).hexdigest() == ARCHIVE_SHA, 'archive SHA mismatch'
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as t:
        members = t.getmembers()
        assert len(members) == 15 and all(m.isfile() for m in members)
        data = {}
        for m in members:
            parts = pathlib.PurePosixPath(m.name).parts
            assert len(parts) == 2 and parts[0] == NAME and parts[1] not in ('.', '..')
            assert parts[1] not in data, 'duplicate archive member'
            data[parts[1]] = t.extractfile(m).read()
    assert hashlib.sha256(data['files.json']).hexdigest() == MANIFEST_SHA
    manifest = json.loads(data['files.json'])
    assert manifest['cpu_expected'] == 45 and manifest['source_commit'] == COMMIT
    assert len(manifest['files']) == 14 and set(data) == set(manifest['files']) | {'files.json'}
    for name, rec in manifest['files'].items():
        assert len(data[name]) == rec['size'] and hashlib.sha256(data[name]).hexdigest() == rec['sha256']
    entry = data['native-cpu.sbatch'].decode()
    assert '#SBATCH --nodelist=h100-11' in entry and '#SBATCH --partition=gpu' in entry
    assert not any('--gres' in line or '--gpus' in line for line in entry.splitlines() if line.startswith('#SBATCH'))
    assert "CUDA_VISIBLE_DEVICES=''" in entry and '"$TASK_ROOT/env/bin/python"' in entry
    assert '$TASK_ROOT/test-tools' in entry
    return data

REMOTE = r'''
import fcntl, os, time, socket
r=pathlib.Path(ROOT); tools=r/'tools'; logs=r/'logs'
assert r.resolve()==r and tools.is_dir() and logs.is_dir()
assert socket.gethostname().split('.')[0] in ('h10011','h100-11')
record=logs/'topd-shared32b-supervisor-v3-native-cpu-submission.json'
def persist(d):
    tmp=record.with_suffix('.tmp')
    assert not record.is_symlink() and not tmp.is_symlink()
    with tmp.open('w') as f:
        json.dump(d,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,record)
    fd=os.open(str(logs),os.O_RDONLY); os.fsync(fd); os.close(fd)
lockpath=record.with_suffix('.lock')
assert not lockpath.is_symlink()
with lockpath.open('a') as lock:
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    data=frozen_payload((tools/(NAME+'-tools.tar.gz')).read_bytes())
    target=tools/NAME
    assert not target.is_symlink()
    if target.exists():
        assert target.is_dir() and all(p.is_file() and not p.is_symlink() for p in target.iterdir())
        assert {p.name for p in target.iterdir()}==set(data), 'existing tree has extra/missing files'
        for n,b in data.items(): assert (target/n).read_bytes()==b, 'existing bytes differ'
    else:
        staging=tools/(NAME+'.deploy-staging')
        assert not staging.exists() and not staging.is_symlink(), 'stale deployment requires inspection'
        staging.mkdir(mode=0o700)
        for n,b in data.items():
            with (staging/n).open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
        fd=os.open(str(staging),os.O_RDONLY); os.fsync(fd); os.close(fd)
        os.rename(staging,target)
        fd=os.open(str(tools),os.O_RDONLY); os.fsync(fd); os.close(fd)
    repo=r/'repo-shared32b'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(), 'source checkout not clean'
    assert (r/'env/bin/python').exists() and (r/'test-tools/pytest').is_dir()
    subprocess.run(['bash','-n',str(target/'native-cpu.sbatch')],check=True)
    if record.exists():
        d=json.loads(record.read_text())
        assert d['archive_sha256']==ARCHIVE_SHA and d['status']=='SUBMITTED', 'UNKNOWN record: inspect Slurm; never blindly retry'
    else:
        q=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True)
        assert not any(x.split('|',1)[-1]=='topd32b-supervisor-cpu' for x in q.splitlines()), 'same job name already queued'
        d=dict(status='UNKNOWN_BEFORE_SUBMISSION',scope='NATIVE_CPU45_NO_GPU_NO_TRAINING',epoch=time.time(),archive_sha256=ARCHIVE_SHA,manifest_sha256=MANIFEST_SHA,entry_sha256=hashlib.sha256(data['native-cpu.sbatch']).hexdigest(),source_commit=COMMIT,entry=str(target/'native-cpu.sbatch'))
        persist(d)
        p=subprocess.run(['sbatch','--parsable','--chdir='+str(r),str(target/'native-cpu.sbatch')],text=True,capture_output=True)
        d.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr); persist(d)
        assert p.returncode==0, d
        jid=p.stdout.strip().split(';')[0]; assert jid.isdigit(), 'ambiguous sbatch output: no retry'
        d.update(status='SUBMITTED',job_id=int(jid)); persist(d)
        ctl=subprocess.run(['scontrol','show','job','-dd',jid],text=True,capture_output=True)
        raw=logs/('topd-shared32b-supervisor-v3-native-cpu-'+jid+'-scontrol-initial.txt')
        assert not raw.exists()
        with raw.open('x') as f: f.write(ctl.stdout+ctl.stderr); f.flush(); os.fsync(f.fileno())
        d.update(initial_scontrol_path=str(raw),initial_scontrol_returncode=ctl.returncode,initial_scontrol_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()); persist(d)
print(json.dumps(d))
'''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--submit',action='store_true'); args=ap.parse_args()
    cwd=pathlib.Path(__file__).resolve().parent.parent
    archive=cwd/'work'/(NAME+'-tools.tar.gz')
    data=frozen_payload(archive.read_bytes())
    for n,b in data.items(): assert (cwd/'work'/NAME/n).read_bytes()==b
    if not args.submit:
        print(json.dumps(dict(status='LOCAL_FROZEN_BYTES_VERIFIED_NOT_SUBMITTED',members=15,payloads=14,cpu_expected=45,archive_sha256=ARCHIVE_SHA,manifest_sha256=MANIFEST_SHA))); return
    import fcntl
    with (cwd/'outputs/topd-shared32b-supervisor-v3-submit-local.lock').open('a') as local_lock:
        fcntl.flock(local_lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(cwd/'work/transfer_file.py'),str(archive),ROOT+'/tools/'+archive.name],check=True,cwd=str(cwd))
        source=pathlib.Path(__file__).read_text().split('REMOTE =',1)[0]
        command="python3 - <<'TOPD_SUPERVISOR_CPU_SUBMIT'\n"+source+REMOTE+'\nTOPD_SUPERVISOR_CPU_SUBMIT\n'
        p=subprocess.run([sys.executable,str(cwd/'work/remote_topd.py')],input=command,text=True,capture_output=True,cwd=str(cwd))
        print(p.stdout,end=''); print(p.stderr,end='',file=sys.stderr)
        output=cwd/'outputs/topd-shared32b-supervisor-v3-native-cpu-submit-invocation.log'
        with output.open('a') as f: f.write(p.stdout+p.stderr); f.flush(); __import__('os').fsync(f.fileno())
        if p.returncode: raise SystemExit(p.returncode)
if __name__=='__main__': main()
