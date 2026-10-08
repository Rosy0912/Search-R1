import pathlib,json,subprocess,shlex,hashlib,sys
stage=pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/bapo-tree-instruct300-aa8e725')
assert hashlib.sha256((stage/'VERIFIED-MODEL-MANIFEST.json').read_bytes()).hexdigest()=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
m=json.loads((stage/'VERIFIED-MODEL-MANIFEST.json').read_text());assert m['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
root='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004/assets/qwen2.5-3b-instruct-aa8e725'
route=[sys.executable,'work/h100_exec.py','--socks']
records=m['files']+[{ 'file':n,'bytes':(stage/n).stat().st_size,'sha256':hashlib.sha256((stage/n).read_bytes()).hexdigest()}for n in ('VERIFIED-MODEL-MANIFEST.json','PUBLIC-SOURCE-API.json')]
for r in records:
 name=r['file'];assert '/'not in name
 query="import pathlib,json; p=pathlib.Path("+repr(root)+")/"+repr(name)+"; t=p.with_name(p.name+'.part'); print(json.dumps({'done':p.exists(),'offset':t.stat().st_size if t.exists() else 0}))"
 observed=subprocess.check_output(route+['--timeout','60','python3 -c '+shlex.quote(query)],text=True);observed=json.loads(observed)
 code='''import pathlib,hashlib,sys,json,os
root=pathlib.Path(ROOT);root.mkdir(parents=True,exist_ok=True);assert not root.is_symlink()
p=root/NAME;t=p.with_name(p.name+'.part');size=SIZE;expected=SHA;offset=OFFSET
if not p.exists():
 assert (t.stat().st_size if t.exists()else 0)==offset
 with t.open('ab'if offset else'xb')as f:
  while True:
   b=sys.stdin.buffer.read(8*1024*1024)
   if not b:break
   f.write(b)
 assert t.stat().st_size==size,(str(t),t.stat().st_size,size)
else:t=p
h=hashlib.sha256()
with t.open('rb')as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert t.stat().st_size==size and h.hexdigest()==expected
if t!=p:os.rename(t,p)
print(json.dumps(dict(state='REMOTE_OFFICIAL_FILE_VERIFIED',path=str(p),bytes=size,sha256=expected)),flush=True)
'''.replace('ROOT',repr(root)).replace('NAME',repr(name)).replace('SIZE',repr(r['bytes'])).replace('SHA',repr(r['sha256'])).replace('OFFSET',repr(observed['offset']))
 print(json.dumps(dict(state='UPLOAD_STARTED',file=name,offset=observed['offset'],already_done=observed['done'])),flush=True)
 args=route+['--timeout','7200']
 if not observed['done']:args+=['--stdin-file',str(stage/name),'--stdin-offset',str(observed['offset'])]
 subprocess.run(args+['python3 -c '+shlex.quote(code)],check=True)
print('H1005_OFFICIAL_INSTRUCT_ALL_FILES_VERIFIED',flush=True)
