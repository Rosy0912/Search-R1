import pathlib,hashlib,tarfile,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004');d=root/'scheduler-head';a=root/'frozen-head-deploy.tar.gz'
assert not d.exists();assert hashlib.sha256(a.read_bytes()).hexdigest()=='81025749c3f03093ddc2b7c2dbaed4984dd553fc35e110734717545b138d571f'
d.mkdir();expected={'submit.sh': '6077c725ff32a8c2e5c22c94a3d18c9c928997c4d6f5aa49559916e4c0a21d2e', 'hardware.sbatch': '425f78cfa0dac40da5298736a926115eaad0c5c28a685ed1d4451a91d165d722', 'formal300.sbatch': 'e4107c69c5d1bb5afa357eb5b1a181435adf081996546803c3e50e856bf3b1d3', 'watch-terminal.py': '78ffa2e4fd7c485de71be5de57335c25b4be6f1d25f1528de6cfcd3869ab5b26', 'approved-launch-hashes.json': '27986dbac6811394bc70e81ad73c0bbe865ba097988adf8b8e0a2590c6ffb8b4'}
with tarfile.open(a,'r:gz')as t:
 members=t.getmembers();assert len(members)==5 and{m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile()and '/'not in m.name and m.name not in ('.','..');b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name]
  with (d/m.name).open('xb')as f:f.write(b)
  (d/m.name).chmod(m.mode&0o777)
print(json.dumps({'status':'NEW_HEAD_DURABLE_TOOLS_PREPARED_NO_SUBMISSION','path':str(d),'files':expected}))
