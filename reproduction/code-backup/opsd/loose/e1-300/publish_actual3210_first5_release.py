"""Root signer for actual verified3210 first-five payload; private key stays local."""
import argparse,pathlib,json,hashlib,subprocess,shlex,importlib.util
H=pathlib.Path(__file__).resolve().parent;R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210'
p=argparse.ArgumentParser();p.add_argument('--payload',required=True);p.add_argument('--sha256',required=True);a=p.parse_args();f=pathlib.Path(a.payload);raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.sha256;v=json.loads(raw)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
m=json.loads((H/'actual3210-cpu-runner-v5/actual-small-20261004T174500Z/032-manifest.json').read_bytes());ident=('method','run_root','source_commit','config_sha256','global_step','actual_outer_updates','next_global_step','horizon','warmup_steps','world_size','driver_pid','uid','job_id','pause_nonce');assert all(v[k]==m[k] for k in ident)
assert v['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW' and v['checkpoint_manifest_sha256']=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b' and v['checkpoint_files']==m['files']
expected=json.loads((H/'actual3210-final5role-evidence-spec-root.json').read_bytes());assert all(v['evidence'][k]==r for k,r in expected.items());assert v['evidence']['root_review']['sha256']=='bd6ecd338d698544805a915280588f8ca0b8b6eb9f8ba46fde13e13ea6c4f21d';assert v['independent_review']['sha256']=='060415b1da78999a1d13be1007773dce6a9988750f46e7196755bdca5421208c'
for name,pin in [('ACTUAL3210_FIRST5_ROOT_REVIEW_RECORD.json','bd6ecd338d698544805a915280588f8ca0b8b6eb9f8ba46fde13e13ea6c4f21d'),('ACTUAL3210_FIRST5_INDEPENDENT_REVIEWER_ACCEPTED_RECORD.json','060415b1da78999a1d13be1007773dce6a9988750f46e7196755bdca5421208c')]:
 rr=(H/name).read_bytes();assert hashlib.sha256(rr).hexdigest()==pin;d=json.loads(rr);assert d['status']=='PASS_ACTUAL_FIRST5_EVIDENCE' and d['evidence']==expected and d['pause_nonce']==m['pause_nonce']
source=(H/'publish_pre_gpu_approval3210.py').read_text();ns={'__file__':str(H/'publish_pre_gpu_approval3210.py')};exec(source[:source.index("review=H/")],ns);remote=ns['remote']
fresh_records=list(v['evidence'].values())+[v['independent_review']]
livecode=f'''import pathlib,json,hashlib,subprocess
r=pathlib.Path({R!r});p=pathlib.Path('/proc/591546');assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=='1819179500';assert 'job_3210/' in p.joinpath('cgroup').read_text();assert p.stat().st_uid==0;assert p.joinpath('status').read_text().split('Uid:',1)[1].splitlines()[0].split()==['0','0','0','0'];assert not (r/'acceptance-5-receipt.json').exists();assert hashlib.sha256((r/'full_state/global_step_5/manifest.json').read_bytes()).hexdigest()=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b';repo=r.parent.parent/'repo-a0968c2d24be';assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='a0968c2d24beeaa0400f60e4e34677fe6deef9c9';assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True);s=subprocess.check_output(['scontrol','show','job','3210','-o'],text=True);assert 'JobState=RUNNING' in s and 'NodeList=h100-10' in s;assert 'TresPerNode=gres:gpu:h100:8' in s
assert p.joinpath('status').read_text().split('State:',1)[1].strip().split()[0] in ('R','S','D','I')
for rec in {fresh_records!r}:
 x=pathlib.Path(rec['path']);assert x.resolve(strict=True)==x and x.is_file() and x.stat().st_uid==0;x.relative_to(r);assert hashlib.sha256(x.read_bytes()).hexdigest()==rec['sha256']
print(s)
'''
check=remote('python3 -c '+shlex.quote(livecode));(H/'ACTUAL3210_FINAL_SIGNING_LIVE_ORIGINAL.txt').write_bytes(check.stdout)
sig=subprocess.run(['openssl','dgst','-sha256','-sign',str(H/'review-signing/private.pem')],input=canonical(v),capture_output=True,check=True).stdout
sp=importlib.util.spec_from_file_location('root_gate',H.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g);public=json.loads((H/'review-signing/public.json').read_bytes());g.verify_rsa_signature(v,sig.hex(),public)
env=canonical({'payload':v,'signature_hex':sig.hex()});out=H/'ACTUAL3210_FIRST5_SIGNED_RELEASE.json'
with out.open('xb') as handle:handle.write(env)
# Atomically publish only the actual signed envelope; unknown transport outcome must be inspected, never blindly rerun.
publishcode=f'''import os,pathlib,sys,secrets,hashlib,json
r=pathlib.Path({R!r});assert r.resolve(strict=True)==r and r.stat().st_uid==0
final=r/'acceptance-5-receipt.json';temp=r/('.acceptance-5-'+secrets.token_hex(16)+'.tmp')
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(env).hexdigest()!r}
fd=os.open(str(temp),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
os.link(temp,final,follow_symlinks=False);temp.unlink();d=os.open(str(r),os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(d)
finally:os.close(d)
print(json.dumps(dict(receipt_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),published_exclusively=True)))
'''
result=remote('python3 -c '+shlex.quote(publishcode),env);(H/'ACTUAL3210_FIRST5_PUBLISH_ORIGINAL.json').write_bytes(result.stdout)
print(json.dumps({'job_id':'3210','receipt_sha256':hashlib.sha256(env).hexdigest(),'scope':'actualfirst5 signed release only; no10/300/effect completion'}))
