import pathlib,subprocess,hashlib,tarfile,json
B=pathlib.Path(__file__).resolve().parent.parent
out=B/'evidence/ray-envfix-hardware3287-raw-evidence.tar.gz';dest=B/'evidence/ray-envfix-hardware3287-raw-evidence';expected='a47b0c645fa27e646a6e7e5fd4ab1e1332a3c9381700063bc465f48c717e0ec7'
if not out.exists():
 with out.open('xb')as f:
  r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','180','cat /dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/logs/hardware3287-raw-evidence.tar.gz'],stdout=f,stderr=subprocess.PIPE,timeout=190)
 (B/'evidence/ray-envfix-hardware3287-download.stderr').write_bytes(r.stderr);assert r.returncode==0
assert out.stat().st_size==5888778 and hashlib.sha256(out.read_bytes()).hexdigest()==expected
assert not dest.exists()
with tarfile.open(out,'r:gz')as t:
 members=t.getmembers();names=set()
 for m in members:
  p=pathlib.PurePosixPath(m.name);assert not p.is_absolute()and '..'not in p.parts and m.name not in names and (m.isfile()or m.isdir())and not m.issym()and not m.islnk();names.add(m.name)
 dest.mkdir();t.extractall(dest)
record=dict(status='ACTUAL_ARCHIVE_BYTES_SHA_SAFE_UNPACK_ONLY_NOT_ACCEPTANCE',archive_path=str(out),archive_bytes=out.stat().st_size,archive_sha256=expected,members=len(members),files=sum(m.isfile()for m in members),directory=str(dest))
(B/'evidence/ray-envfix-hardware3287-download-record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
