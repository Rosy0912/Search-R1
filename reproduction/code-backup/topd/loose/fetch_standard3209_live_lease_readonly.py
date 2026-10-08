import sys,json,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
code=r'''import pathlib,json,os,hashlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v14/evidence');p=root/'terminal-binding.json';raw=p.read_bytes();v=json.loads(raw);pid=v['server_identity']['pid']
def identity():
 q=pathlib.Path('/proc')/str(pid);s=(q/'stat').read_text();f=s[s.rfind(')')+2:].split();st=(q/'status').read_text();return dict(pid=pid,starttime_ticks=int(f[19]),state=f[0],uids=next(x for x in st.splitlines() if x.startswith('Uid:')).split()[1:],cgroup=(q/'cgroup').read_text())
a=identity();assert a['starttime_ticks']==v['server_identity']['starttime_ticks'] and a['cgroup']==v['server_identity']['cgroup'];rows=[]
for q in (pathlib.Path('/proc')/str(pid)/'fd').iterdir():
 try:target=os.readlink(q)
 except FileNotFoundError:continue
 if 'GPU-' not in target or not target.endswith('.lock'):continue
 st=q.stat();info=(pathlib.Path('/proc')/str(pid)/'fdinfo'/q.name).read_text();rows.append(dict(fd=int(q.name),target=target,device=st.st_dev,inode=st.st_ino,fdinfo_raw=info))
b=identity();assert a==b;assert len(rows)==8 and len({x['target'] for x in rows})==8
print(json.dumps(dict(scope='READ_ONLY_LIVE_PARENT_FD_LOCK_SNAPSHOT_NOT_FUTURE_LOCK_GUARANTEE',job_id=3209,epoch=time.time(),binding_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),identity_before=a,identity_after=b,actual_gpu_lock_fds=rows)))'''
r=send(code,attempts=3,retry_safe=True,timeout=90);assert r.returncode==0,(r.stdout,r.stderr);json.loads(r.stdout);pathlib.Path('outputs/standard3209-live-parent-eight-lease-readonly.json').write_bytes(r.stdout);print(r.stdout.decode())
