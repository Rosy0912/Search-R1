import hashlib,json,pathlib,shlex,subprocess,time
from remote import route
stage=pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/bapo-tree-instruct300-aa8e725')
package=json.loads(pathlib.Path('work/tree-instruct300-runtime/manifest.json').read_bytes())
records=package['model_identity']['files'];root='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725'
for name,expected in records.items():
    assert '/' not in name and not name.startswith('.')
    source=stage/name;size=source.stat().st_size
    query='import pathlib,json; p=pathlib.Path('+repr(root)+')/'+repr(name)+'; t=p.with_name(p.name+".part"); print(json.dumps(dict(done=p.exists(),offset=t.stat().st_size if t.exists() else 0)))'
    observed=json.loads(subprocess.check_output(route('/usr/bin/python3.10 -c '+shlex.quote(query)),text=True))
    offset=observed['offset'];assert 0<=offset<=size
    code='''import pathlib,hashlib,sys,json,os,fcntl
root=pathlib.Path(ROOT);root.mkdir(parents=True,exist_ok=True);assert not root.is_symlink()
lock=(root.parent/(root.name+'.tree-upload.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=root/NAME;t=p.with_name(p.name+'.part');size=SIZE;expected=SHA;offset=OFFSET
if not p.exists():
 assert (t.stat().st_size if t.exists() else 0)==offset
 with t.open('ab' if offset else 'xb') as f:
  while True:
   b=sys.stdin.buffer.read(8*1024*1024)
   if not b:break
   f.write(b)
  f.flush();os.fsync(f.fileno())
 assert t.stat().st_size==size
else:t=p
h=hashlib.sha256()
with t.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert t.stat().st_size==size and h.hexdigest()==expected
if t!=p:os.link(t,p);t.unlink()
print(json.dumps(dict(status='TREE_H1009_OFFICIAL_FILE_VERIFIED',file=NAME,bytes=size,sha256=expected)),flush=True)
'''.replace('ROOT',repr(root)).replace('NAME',repr(name)).replace('SIZE',repr(size)).replace('SHA',repr(expected)).replace('OFFSET',repr(offset))
    print(json.dumps(dict(status='UPLOAD_STARTED',file=name,offset=offset,already_done=observed['done'],epoch=time.time())),flush=True)
    with source.open('rb') as stream:
        stream.seek(offset)
        subprocess.run(route('/usr/bin/python3.10 -c '+shlex.quote(code)),stdin=subprocess.DEVNULL if observed['done'] else stream,check=True)
print('TREE_H1009_ALL_OFFICIAL_MODEL_FILES_SHA_VERIFIED',flush=True)
