import pathlib,hashlib,json,shlex,io,tarfile
H=pathlib.Path(__file__).resolve().parent;L=H/'rebound-a0968c2-candidate/first5';R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';M='b4d0cc217253bccaf3aa53c467ed6103ee2ee9375935b84c81a96d9814dede66';D=R+'/tools/first5-b4d0cc217253'
ns={'__file__':str(H/'deploy_formal_reviewed.py')};s=pathlib.Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns);remote=ns['remote']
m=json.loads((L/'manifest.json').read_bytes());assert hashlib.sha256((L/'manifest.json').read_bytes()).hexdigest()==M
for n,pin in m['tools'].items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==pin
remote('mkdir '+shlex.quote(D));b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for n in m['tools']:t.add(L/n,arcname=n)
remote('tar --no-same-owner -xf - -C '+shlex.quote(D),b.getvalue());remote('test ! -e '+shlex.quote(D+'/manifest.json')+' && cat > '+shlex.quote(D+'/manifest.json'),(L/'manifest.json').read_bytes())
code=f"""import pathlib,hashlib,json,os,subprocess
p=pathlib.Path({D!r});r=pathlib.Path({R+'/repo-a0968c2d24be'!r});raw=(p/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=={M!r};m=json.loads(raw)
assert subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip()==m['source_commit'];assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip()
for n,h in m['tools'].items():assert (p/n).stat().st_uid==0 and hashlib.sha256((p/n).read_bytes()).hexdigest()==h
for n,h in m['source_files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
print(json.dumps(dict(source=m['source_commit'],manifest_sha256={M!r},tools=str(p),uid=os.getuid(),scope='tool deployment only, no actual first5 acceptance')))
"""
q=remote('python3 -c '+shlex.quote(code));(H/'first5-a096-deployment-actual.json').write_bytes(q.stdout);print(q.stdout.decode())
