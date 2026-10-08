import pathlib,hashlib,tarfile,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-r2-20261004');d=root/'scheduler-head';a=root/'frozen-head-deploy.tar.gz'
assert not d.exists();assert hashlib.sha256(a.read_bytes()).hexdigest()=='433488f86ec8b45a068815edc13bbbe1c530025373ccf7c185af1207602943a2'
d.mkdir();expected={'submit.sh': '4d83d5f37ad436982fbea5e3b962618790a35d2546dca3ac2794ef9cb22dea60', 'hardware.sbatch': '43413730d646592c33aef4d3e5c80718cc4bf332c5925ecb3d48782af79abb70', 'formal300.sbatch': 'cea7e4a39886f52610ad5d19ab8e3d33f3181e9d3d583d810669471f4c8a6222', 'watch-terminal.py': 'bca6967466c4c5e709df4b51b72f141d77b324c10a957691c399a687abdb0af7', 'approved-launch-hashes.json': 'b6ab609e03973ee1ff21f09754cac6d1ea7b43ce83e755c3fda1a0de84f0d11f'}
with tarfile.open(a,'r:gz')as t:
 members=t.getmembers();assert len(members)==5 and{m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile()and '/'not in m.name and m.name not in ('.','..');b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name]
  with (d/m.name).open('xb')as f:f.write(b)
  (d/m.name).chmod(m.mode&0o777)
print(json.dumps({'status':'NEW_HEAD_DURABLE_TOOLS_PREPARED_NO_SUBMISSION','path':str(d),'files':expected}))
