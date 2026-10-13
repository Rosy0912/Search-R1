"""Reconcile only the real failed ba9 run; submit the new bound native5 packet once."""
import argparse,json,shlex
from pathlib import Path
import ssh_retained18 as ssh
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reconcile','submit'],required=True);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--rendered',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.deployment.read_text());rendered=json.loads(a.rendered.read_text());pkg=r['package']
if a.mode=='reconcile':
 code=r'''import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'instruct300';source='ba9f6627ec9a4e9749830d7b6acca53f6feec6b4';run=root/'runs'/('tree-nested-branch-fresh200-not-slurm;systemd:'+source);unit='sst-tree-nested-branch-fresh200-ba9f6627.service'
terminal=run/'run-result.json';snapshot=run/'output/ray-fault-20261014-outer6-terminal.json'
for path,expected in ((terminal,'1c20f2d7eff01a56cb6f713556552184383b0855e31f427ac505258a54820a1f'),(snapshot,'55f013d537819d4c7b84eafa0790e8de205fe7d5256e6f79945aae60610e5718')):assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
fd=os.open(base/'FRESH200_TRAIN_DISPATCH.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
try:
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=subprocess.check_output(['systemctl','show',unit,'--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,ControlGroup,InvocationID'],text=True);v=dict(line.split('=',1) for line in state.splitlines())
 assert v['MainPID']=='0' and v['ActiveState']=='failed' and v['Result']=='exit-code' and v['ExecMainStatus']=='1' and v['InvocationID']=='61ae43043c244f1aa10d1ca3f53a8003',state
 cg=Path('/sys/fs/cgroup/system.slice')/unit
 assert not cg.exists() or not any(q.read_text().strip() for q in cg.rglob('cgroup.procs')),'Own failed cgroup remains populated'
 ports=subprocess.check_output(['ss','-ltnp'],text=True)
 assert not any((':'+str(port)+' ') in line for line in ports.splitlines() for port in (16382,18082)),'Original local ports still occupied'
 intent=base/('SUBMISSION_ONCE-nested-branch-fresh200-'+source+'.json');archive=base/'submission-history'/('TERMINAL_RAY_THREAD_FAILURE-'+intent.name);archive.parent.mkdir(exist_ok=True)
 assert list(base.glob('SUBMISSION_ONCE-nested-branch-fresh200-*.json'))==[intent], 'Other nested submission exists; reconcile before dispatch'
 value=json.loads(intent.read_text());assert value['code_commit']==source and value['status']=='SUBMITTED' and value['unit']==unit
 sha=hashlib.sha256(intent.read_bytes()).hexdigest();assert not archive.exists();intent.rename(archive)
 receipt=dict(status='CONFIRMED_OWN_RAY_THREAD_FAILURE_JOURNAL_ARCHIVED',epoch=time.time(),source_commit=source,original_journal=str(intent),preserved_journal=str(archive),journal_sha256=sha,systemd=v,terminal_path=str(terminal),terminal_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest(),fault_snapshot_path=str(snapshot),fault_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),recovered_completed_outer=5,failed_outer6_optimizer_writes_unknown=True,foreign_jobs_modified=False,training_submitted=False)
 with Path(PACKAGE+'/parent-dispatch-reconciled.json').open('x') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
'''.replace('PACKAGE',repr(pkg))
 cmd='env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code)
else:
 cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python',pkg+'/submit_shared_once.py','--package',pkg,'--manifest-sha',r['manifest_sha256'],'--release-path',pkg+'/release.json','--release-sha',r['release_sha256'],'--script',rendered['path'],'--script-sha',rendered['sha256']])
x=ssh.run(cmd,timeout=120);a.output.with_suffix('.command.json').write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n');x.check_returncode();j=json.loads(x.stdout);a.output.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
