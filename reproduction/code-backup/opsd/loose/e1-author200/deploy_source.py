"""Root deployment of the separately reviewed, clean author200 commit."""
import pathlib,subprocess,json,hashlib,shlex
from transport import remote,publish
H=pathlib.Path(__file__).resolve().parent;L=H.parent/'opsd-e1-300';ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006';OLD='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';BASE='a0968c2d24beeaa0400f60e4e34677fe6deef9c9'
new=subprocess.check_output(['git','-C',str(L),'rev-parse','HEAD'],text=True).strip();assert new!=BASE and not subprocess.check_output(['git','-C',str(L),'status','--porcelain'],text=True).strip()
review=H/'INDEPENDENT_SOFTWARE_REVIEW.md';assert review.is_file() and 'PASS' in review.read_text()
remote('mkdir -p '+shlex.quote(ROOT+'/tools')+' '+shlex.quote(ROOT+'/logs')+' '+shlex.quote(ROOT+'/runs')+' '+shlex.quote(ROOT+'/review-signing'))
bundle=H/('source-'+new[:12]+'.bundle');subprocess.run(['git','-C',str(L),'bundle','create',str(bundle),'HEAD','^'+BASE],check=True);pin=hashlib.sha256(bundle.read_bytes()).hexdigest();bp=ROOT+'/tools/'+bundle.name;publish(bp,bundle.read_bytes());repo=ROOT+'/repo-'+new[:12]
code=f'''import pathlib,subprocess,hashlib,os,json
r=pathlib.Path({repo!r});b=pathlib.Path({bp!r});assert not r.exists();assert b.stat().st_uid==os.getuid()==0 and hashlib.sha256(b.read_bytes()).hexdigest()=={pin!r}
subprocess.run(['git','clone','--no-hardlinks','--no-checkout',{OLD+'/repo-'+BASE[:12]!r},str(r)],check=True)
subprocess.run(['git','-C',str(r),'fetch',str(b),{new!r}],check=True);subprocess.run(['git','-C',str(r),'checkout','--detach',{new!r}],check=True)
assert subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip()=={new!r};assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip()
print(json.dumps({{'source_commit':{new!r},'source_repo':str(r),'bundle_sha256':{pin!r},'uid':os.getuid()}}))
'''
q=remote('python3 -c '+shlex.quote(code));(H/'source-deployment-original.json').write_bytes(q.stdout);(H/'source-deployment.stderr').write_bytes(q.stderr)
key=H.parent/'e1-300/review-signing/public.json';kp=hashlib.sha256(key.read_bytes()).hexdigest();assert kp=='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c';publish(ROOT+'/review-signing/public.json',key.read_bytes());print(q.stdout.decode())
