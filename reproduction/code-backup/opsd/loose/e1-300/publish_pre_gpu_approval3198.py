"""Root-only actual review signing. Local private key never enters upload stream."""
import pathlib,json,hashlib,subprocess,shlex,datetime
H=pathlib.Path(__file__).resolve().parent;R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3198'
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=none']
def route(c):
 w='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['h100-10',c]);h='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',w])
 return ['ssh']+opts+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',h]
def remote(c,data=None):return subprocess.run(route(c),input=data,capture_output=True,check=True,timeout=60)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def publish(path,raw):
 c='test ! -e '+shlex.quote(path)+' && cat > '+shlex.quote(path+'.partial')+' && mv '+shlex.quote(path+'.partial')+' '+shlex.quote(path)
 remote(c,raw)
review=H/'FORMAL3198_PRE_GPU_ACTUAL_INDEPENDENT_REVIEW.md';assert review.is_file()
# Reviewer must explicitly accept actual3198; no CPU fixture or oldjob approval.
text=review.read_text();assert '3198' in text and 'PASS' in text
readyraw=remote('cat '+shlex.quote(R+'/approval-ready.json')).stdout;ready=json.loads(readyraw);assert ready['job_id']=='3198' and ready['source_commit']=='89beac273f783d8e48d7d3e34da040f753514d90' and ready['run_root']==R
livecode=f'''import pathlib,json,hashlib,os,subprocess
r=pathlib.Path({R!r});v=json.loads((r/'approval-ready.json').read_bytes());assert v['job_id']=='3198';pid=v['pid'];a=pathlib.Path('/proc')/str(pid);assert a.stat().st_uid==v['uid']==0;assert a.joinpath('stat').read_text().split(') ',1)[1].split()[19]==str(v['process_start_ticks']);assert 'job_3198/' in a.joinpath('cgroup').read_text()
raw=subprocess.check_output(['scontrol','show','job','3198','-o'],text=True);assert 'JobState=RUNNING' in raw and 'NodeList=h100-10' in raw and 'TresPerNode=gres:gpu:h100:8' in raw
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
b=r/'bindings/manifest.json';assert hashlib.sha256(b.read_bytes()).hexdigest()==v['binding_sha256']
for rec in json.loads(b.read_bytes())['files'].values():assert hashlib.sha256(pathlib.Path(rec['path']).read_bytes()).hexdigest()==rec['sha256']
print(raw)
'''
live=remote('python3 -c '+shlex.quote(livecode));(H/'formal3198-root-live-pre-sign.txt').write_bytes(live.stdout)
reviewraw=review.read_bytes();rp=R+'/evidence/independent-pre-gpu-review.md';publish(rp,reviewraw)
p={k:ready[k] for k in ('source_commit','job_id','uid','pid','process_start_ticks','run_root','nonce','binding_sha256','public_key_sha256')};p.update(version=1,authorization='EXPLICIT_OPSD_E1_GPU_INITIALIZATION_ONLY',ready_sha256=hashlib.sha256(readyraw).hexdigest(),independent_review={'path':rp,'sha256':hashlib.sha256(reviewraw).hexdigest()})
raw=canonical(p);sig=subprocess.run(['openssl','dgst','-sha256','-sign',str(H/'review-signing/private.pem')],input=raw,capture_output=True,check=True).stdout
envelope=canonical({'payload':p,'signature_hex':sig.hex()});(H/'formal3198-pre-gpu-approval-actual.json').write_bytes(envelope)
publish(R+'/bindings/gpu-init-approval.json',envelope);publish(R+'/bindings/gpu-init-approval.sha256',(hashlib.sha256(envelope).hexdigest()+'\n').encode());print(json.dumps({'job_id':3198,'approval_sha256':hashlib.sha256(envelope).hexdigest(),'signed_after_actual_independent_review':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
