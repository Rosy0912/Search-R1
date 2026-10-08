"""Root-only small independent deployment; default performs local checks only."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess

HERE = Path(__file__).absolute().parent
COMMIT = '554c52c845561bcab00ec56d469c945878689890'
MANIFEST = '4f9ee158ec1c326498898d26ee5b6a1d3212e94d5ecbbd24c8ef7864fa079b69'
GOAL = 'c65bea46dbb4ef6451ed56b2c4e26adafde5fd247e6b67fb5bea593b8689c94f'
PUBLICATION = '782fa0285167b9449e1c2db999bc3822817591bc7b807106503fac674377e9ac'
N = '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008'
HELPER = N + '/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2'
RECEIPTS = N + '/receipts/outer-backup-v2-554c52c-20261009'


def stamp(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)


def read(path, digest=None, size=None):
    p = Path(path)
    current = Path(p.anchor)
    for part in p.parts[1:]:
        current /= part
        assert not current.is_symlink(), 'Symlink source refused'
    before = p.stat()
    assert stat.S_ISREG(before.st_mode)
    with p.open('rb') as handle:
        assert stamp(os.fstat(handle.fileno())) == stamp(before)
        raw = handle.read()
        assert stamp(os.fstat(handle.fileno())) == stamp(before)
    assert stamp(p.stat()) == stamp(before), 'Source changed during read'
    actual = hashlib.sha256(raw).hexdigest()
    assert digest is None or actual == digest, 'Local SHA mismatch'
    assert size is None or len(raw) == size, 'Local size mismatch'
    return raw


def payload():
    packet = json.loads(read(HERE / 'deployment-packet.json'))
    assert packet['source_commit'] == COMMIT and packet['source_manifest_sha256'] == MANIFEST
    assert packet['helper_root'] == HELPER and packet['receipts_root'] == RECEIPTS
    assert packet['goal_sha256'] == GOAL and packet['publication_sha256'] == PUBLICATION
    goal = json.loads(read(HERE / 'goal.json', GOAL))
    pub = json.loads(read(HERE / 'publication.json', PUBLICATION))
    assert pub['accepted'] is True and pub['source_commit'] == pub['remote_readback_sha'] == COMMIT
    assert pub['source_manifest_sha256'] == MANIFEST
    proof = pub['verification_receipt']
    evidence = json.loads(read(proof['path'], proof['sha256'], proof['bytes']))
    assert evidence['accepted'] is True and evidence['source_commit'] == evidence['remote_readback_sha'] == COMMIT
    assert pub['files'] == evidence['files']
    assert pub['run_ids'] == {g['variant']: g['run_id'] for g in goal['groups']}
    expected = {HELPER + '/' + Path(r['path']).name: r for r in pub['files']}
    assert len(expected) == 5 and all(r['equal'] is True for r in expected.values())
    expected[RECEIPTS + '/goal.json'] = dict(sha256=GOAL, bytes=len(read(HERE / 'goal.json', GOAL)))
    expected[RECEIPTS + '/publication.json'] = dict(sha256=PUBLICATION, bytes=len(read(HERE / 'publication.json', PUBLICATION)))
    assert {r['node_path'] for r in packet['files']} == set(expected) and len(packet['files']) == 7
    rows = []
    for row in packet['files']:
        declaration = expected[row['node_path']]
        assert row['sha256'] == declaration['sha256'] and row['bytes'] == declaration['bytes']
        raw = read(row['local_path'], row['sha256'], row['bytes'])
        rows.append(dict(row, content_base64=base64.b64encode(raw).decode()))
    return dict(helper_root=HELPER, receipts_root=RECEIPTS, files=rows, goal=goal)


REMOTE = r'''
import base64,hashlib,json,os,stat,sys
from pathlib import Path
assert os.getuid()==os.geteuid()==1005
x=json.loads(sys.stdin.buffer.read())
helper=Path(x['helper_root']); receipts=Path(x['receipts_root'])
assert str(helper)=='/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2'
assert str(receipts)=='/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/receipts/outer-backup-v2-554c52c-20261009'
def components(p):
 c=Path(p.anchor)
 for part in p.parts[1:]:
  c/=part
  if c.exists() or c.is_symlink(): assert not c.is_symlink(), 'Symlink Node path refused'
def stamp(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
def checked(p,sha,size):
 components(p); before=p.stat(); assert stat.S_ISREG(before.st_mode)
 with p.open('rb') as h:
  assert stamp(os.fstat(h.fileno()))==stamp(before); raw=h.read(); assert stamp(os.fstat(h.fileno()))==stamp(before)
 assert stamp(p.stat())==stamp(before) and len(raw)==size and hashlib.sha256(raw).hexdigest()==sha
 return dict(path=str(p),sha256=sha,bytes=size,device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns)
# All seven original config pins must still match before any independent writes.
for g in x['goal']['groups']:
 p=Path(g['config']['path']); components(p); s=p.stat(); checked(p,g['config']['sha256'],s.st_size)
expected={r['node_path'] for r in x['files']}; assert len(expected)==7
for p in (helper,receipts):
 components(p)
 if p.exists(): assert {str(q) for q in p.iterdir()} <= expected, 'Existing independent directory has extra files'
 else: p.mkdir(parents=True,mode=0o700)
proofs=[]
for r in x['files']:
 p=Path(r['node_path']); assert p.parent in (helper,receipts)
 raw=base64.b64decode(r['content_base64'],validate=True)
 assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 if not p.exists():
  with p.open('xb') as h: h.write(raw); h.flush(); os.fsync(h.fileno())
  p.chmod(0o600)
 proofs.append(checked(p,r['sha256'],r['bytes']))
for p in (helper,receipts):
 assert {str(q) for q in p.iterdir()}=={n for n in expected if Path(n).parent==p}
 fd=os.open(str(p),os.O_RDONLY); os.fsync(fd); os.close(fd)
print(json.dumps(dict(schema='opsd-ablation-outer-backup-v2-node-deployment-readback-v1',accepted=True,files=proofs,no_training_changes=True,no_gpu=True)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Root explicitly deploys only the seven independent files')
    args = parser.parse_args()
    data = payload()
    if not args.apply:
        print(json.dumps(dict(state='LOCAL_CHECKS_PASS_NOT_DEPLOYED',files=len(data['files']),source_commit=COMMIT)))
        return
    remote = ['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','E1_RUN_ID=backup-bootstrap-v2-554c52c','python3','-B','-c',REMOTE]
    command = ['/usr/bin/ssh','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oControlMaster=no','-S','none','-oConnectTimeout=15','hkust1',shlex.join(remote)]
    result = subprocess.run(command,input=json.dumps(data).encode(),capture_output=True,check=False)
    os.write(2,result.stderr)
    assert result.returncode == 0, 'Deployment SSH did not return actual rc0; preserve stderr/partial'
    proof = json.loads(result.stdout)
    assert proof['accepted'] is True
    assert {(r['path'],r['sha256'],r['bytes']) for r in proof['files']} == {(r['node_path'],r['sha256'],r['bytes']) for r in data['files']}
    print(json.dumps(proof,indent=2))


if __name__ == '__main__':
    main()
