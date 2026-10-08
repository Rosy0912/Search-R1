import pathlib,hashlib,tarfile,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-devicefix-r2-20261004');d=root/'scheduler-head';a=root/'frozen-head-deploy.tar.gz'
assert not d.exists();assert hashlib.sha256(a.read_bytes()).hexdigest()=='c343b27dfbe6449387f3ab9a828927d8322d1fd0aa9657ff5b63cae876c6e49a'
d.mkdir();expected={'submit.sh': 'a156e25c14e29d73c4da12443585fd4eef2caaa80777517850d069b03ac0a082', 'hardware.sbatch': '36fcf1280237e90d173ef5dc88c920a00a9eec5bbb6aaa5b6e68dc9e20c554f4', 'formal300.sbatch': '4f26a78716654f564ff647c42c03032e04ebee0bbbcec9c16bd1bc1045583ee5', 'watch-terminal.py': 'fbb7bc0567ff3b5fcbd7723620b7ab10ecf8f3fc847797eccead1d074783747c', 'approved-launch-hashes.json': 'aa02a74423a2b690656646d5b3b4898c2ce3b2c280c55cc700ca535d62d9f639'}
with tarfile.open(a,'r:gz')as t:
 members=t.getmembers();assert len(members)==5 and{m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile()and '/'not in m.name and m.name not in ('.','..');b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name]
  with (d/m.name).open('xb')as f:f.write(b)
  (d/m.name).chmod(m.mode&0o777)
print(json.dumps({'status':'NEW_HEAD_DURABLE_TOOLS_PREPARED_NO_SUBMISSION','path':str(d),'files':expected}))
