import sys, subprocess, shlex, pathlib, json, base64
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
code=r'''import pathlib,subprocess,json,socket,datetime,os,hashlib,base64
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006');run=root/'runs/gigpo-e1-fresh200-live-job3323-1791355362986358231'
if socket.gethostname().split('.')[0]!='h100-8':raise RuntimeError('Wrong node')
out={'hostname':socket.gethostname(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'3323','run':str(run),'files':[],'raw_metadata':{}}
for name in ['run-result.json','baseline-evaluation.json','actual-official-zero-initialization.json','fresh200-initialization.json']:
 p=run/name;raw=p.read_bytes();out['files'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});out['raw_metadata'][name]=base64.b64encode(raw).decode()
result=json.loads((run/'run-result.json').read_bytes())
for p in [root/'logs/fresh200-live-3323.log',run/'calibration.log',run/'ray/session_2026-10-07_14-44-45_262144_3893337/logs/worker-b4e01b36c94fd4a7d46d74659f2ef316698ddd988d4da153a7757ff4-02000000-3897043.out']:
 st=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 st2=p.stat()
 if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(st2.st_dev,st2.st_ino,st2.st_size,st2.st_mtime_ns):raise RuntimeError('Original log changed during terminal hash')
 out['files'].append({'path':str(p),'bytes':st.st_size,'sha256':h.hexdigest(),'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns})
out['owned_pid_exists']={str(pid):pathlib.Path('/proc/'+str(pid)).exists() for pid in [3897043,3897438,3897761,3897762,3897764,3897765,3897763,3897766,3897767]}
out['recorded_cgroup']=result['cgroup'];out['recorded_cgroup_exists']=pathlib.Path(result['cgroup']).exists()
out['audit_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in (run/'audit').rglob('*') if p.is_file()]
print(json.dumps(out))
'''
inner='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(code)
mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 '+shlex.quote(mid)]
a=subprocess.run(route,capture_output=True,text=True,timeout=60)
if a.returncode:raise RuntimeError(a.stderr)
d=json.loads(a.stdout);out=pathlib.Path('outputs/gigpo-e1-instruct200/job3323-terminal-originals-v1');out.mkdir()
for name,encoded in d.pop('raw_metadata').items():
 raw=base64.b64decode(encoded);(out/name).write_bytes(raw)
(out/'TERMINAL-ORIGINALS-RECEIPT.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
