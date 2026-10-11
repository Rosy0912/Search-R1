import subprocess,pathlib,hashlib,json,datetime,tarfile,io,sys,shlex
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/mixed-first-200')
bare=pathlib.Path('work/search-g1-mixed-publication-readback-584db0b.git').resolve()
def git(args,cwd):return subprocess.check_output(['git']+args,cwd=cwd)
commit=git(['rev-parse','HEAD'],repo).decode().strip();assert commit.startswith('653712b')
subprocess.run(['git','fetch','git@github.com:shaoshitong/Search-R1.git','codex/search-g1-mixed-first-200'],cwd=bare,check=True,capture_output=True)
remote_commit=git(['rev-parse','FETCH_HEAD'],bare).decode().strip();assert remote_commit==commit
rows=git(['ls-tree','-rz','HEAD'],repo).split(b'\0');files={};archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as tar:
 for row in rows:
  if not row:continue
  meta,name=row.split(b'\t',1);mode,kind,blob=meta.split();assert kind==b'blob'
  name=name.decode();local=git(['cat-file','blob',blob.decode()],repo);other=git(['cat-file','blob',blob.decode()],bare)
  digest=hashlib.sha256(local).hexdigest();assert digest==hashlib.sha256(other).hexdigest()
  assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest
  files[name]=digest;entry=tarfile.TarInfo(name);entry.size=len(local);entry.mode=int(mode,8)&0o777;tar.addfile(entry,io.BytesIO(local))
proof=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=commit,remote_commit=remote_commit,full_blob_sha256_match=True,files=files,verified_file_count=len(files))
pathlib.Path('work/search-g1-publication-653712b-independent-readback.json').write_text(json.dumps(proof,indent=2))
base='/mnt/data/shaoshitong/search-g1-speed-resume5-653712b-20261012'
sys.path.insert(0,'work');from search_g1_restore5_transport import remote
cmd=f'test ! -e {base} && mkdir -p {base}/repo && tar -xzf - -C {base}/repo'
remote(cmd,'outputs/gigpo-e1-instruct200/SEARCH-G1-SPEED-653712b-DEPLOY-v1.json',stdin=archive.getvalue())
# Separate full-file independent target readback and reversible asset binding.
code=f"""import pathlib,json,hashlib,sys,os
base=pathlib.Path({base!r});files=json.load(sys.stdin)
for name,sha in files.items():
 assert hashlib.sha256((base/'repo'/name).read_bytes()).hexdigest()==sha,name
sys.path.insert(0,str(base/'repo'))
from reproduction.common8.bind_native_assets import bind
proof=bind(base)
(base/'preparation/published-source-files.json').write_text(json.dumps(files))
v=os.statvfs(base)
print(json.dumps(dict(verified_files=len(files),full_file_sha256_match=True,asset_binding=proof,ordinary_free_bytes=v.f_bavail*v.f_frsize,physical_free_bytes=v.f_bfree*v.f_frsize)))
"""
remote('/usr/bin/python3 -c '+shlex.quote(code),'outputs/gigpo-e1-instruct200/SEARCH-G1-SPEED-653712b-TARGET-SHA-ASSETS-v1.json',stdin=json.dumps(files).encode())
print(json.dumps(dict(commit=commit,files=len(files),base=base)))
