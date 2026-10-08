python3 - <<'PY'
import pathlib,tarfile,json,hashlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools');a=r/'g1-hidden-input-replay-v1.tar.gz';d=r/'g1-hidden-input-replay-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a)=='0530988609db3a4328c58c78b5fa56257cf1c50fc82d7026de25f85287dc219c'
with tarfile.open(a) as t:
 ms=t.getmembers();assert len(ms)==4
 for m in ms:
  parts=pathlib.PurePosixPath(m.name).parts;assert m.isfile() and len(parts)==2 and parts[0]==d.name and parts[1] not in ('.','..')
 if not d.exists():t.extractall(r)
assert not d.is_symlink() and sha(d/'files.json')=='3b62f6b3a7ca1f88a967c96ed18272e02e826abcf50b1e4405fa800de5b4adca'
f=json.loads((d/'files.json').read_text())['files'];assert {p.name for p in d.iterdir()}==set(f)|{'files.json'}
for n,v in f.items():assert not (d/n).is_symlink() and sha(d/n)==v['sha256'] and (d/n).stat().st_size==v['size']
print(json.dumps({'scope':'SOURCE_DEPLOYMENT_ONLY_NOT_ACTUAL_B_REPLAY','archive_sha256':sha(a),'manifest_sha256':sha(d/'files.json'),'directory':str(d),'files':f,'executed_replay':False}))
PY
