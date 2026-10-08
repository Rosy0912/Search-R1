import pathlib,hashlib,tarfile,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-20261004');d=root/'scheduler-head';a=root/'frozen-head-deploy.tar.gz'
assert not d.exists();assert hashlib.sha256(a.read_bytes()).hexdigest()=='3c1f9da798755f67cc0e5445478aa201f0d587ea9cc1c33e5120547380a34a2a'
d.mkdir();expected={'submit.sh': 'e8f87a0d28f4f9bc6d55c5559c8ca9aa1b8c8b251c658f286e9204ab393bd099', 'hardware.sbatch': '4f33d2bd56781300f4dc9b52c4ed9e0dd34acd177fa05c8cdb223659b2d329d8', 'formal300.sbatch': '415aee548316935a1898a77553c296b1f9b238e9fc0d6368af58ad480547293e', 'watch-terminal.py': '8a659e9952f170213eade864b937383cdf3ea0dcc5c366b83441c587ab6eda30', 'approved-launch-hashes.json': '92e06cfd8178706369ede9575ffcef5f013fd709d8ee685725e4b537fe0a0228'}
with tarfile.open(a,'r:gz')as t:
 members=t.getmembers();assert len(members)==5 and{m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile()and '/'not in m.name and m.name not in ('.','..');b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name]
  with (d/m.name).open('xb')as f:f.write(b)
  (d/m.name).chmod(m.mode&0o777)
print(json.dumps({'status':'NEW_HEAD_DURABLE_TOOLS_PREPARED_NO_SUBMISSION','path':str(d),'files':expected}))
