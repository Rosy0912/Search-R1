"""Reconcile a confirmed own terminal journal, then invoke exact once submitter."""
import argparse,json,shlex
from pathlib import Path
import ssh_retained18 as ssh
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reconcile','submit'],required=True);a=p.parse_args()
d=Path('outputs/tree-nested-branch-handoff-20261013');r=json.loads((d/'RECOVERY_DELTA_DEPLOYMENT.json').read_text());rendered=json.loads((d/'RECOVERY_DISPATCH_RENDERED.json').read_text());pkg=r['package']
if a.mode=='reconcile':
 code='''import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'instruct300';source='0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
m=json.loads(Path(PACKAGE+'/manifest.json').read_text());terminal=m['nested_native_continuation']['failed_terminal'];snapshot=m['nested_native_continuation']['fault_snapshot']
for desc in (terminal,snapshot): assert hashlib.sha256(Path(desc['path']).read_bytes()).hexdigest()==desc['sha256']
fd=os.open(base/'FRESH200_TRAIN_DISPATCH.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
try:
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=subprocess.run(['systemctl','show','sst-tree-nested-branch-fresh200-0befbaf9.service','--property=MainPID,ActiveState,SubState,Result,ControlGroup'],capture_output=True,text=True,check=True).stdout
 values=dict(line.split('=',1) for line in state.splitlines())
 assert values['MainPID']=='0' and values['ActiveState']=='failed' and values['Result']=='oom-kill',state
 cg=Path('/sys/fs/cgroup')/values['ControlGroup'].lstrip('/')
 if values['ControlGroup'] and cg.exists(): assert not any(p.read_text().strip() for p in cg.rglob('cgroup.procs')),'Own failed cgroup still populated'
 intent=base/('SUBMISSION_ONCE-nested-branch-fresh200-'+source+'.json')
 assert list(base.glob('SUBMISSION_ONCE-nested-branch-fresh200-*.json'))==[intent], 'Other existing nested dispatch requires reconciliation'
 value=json.loads(intent.read_text());assert value['code_commit']==source and value['status']=='SUBMITTED' and value['unit']=='sst-tree-nested-branch-fresh200-0befbaf9.service'
 archive=base/'submission-history'/('TERMINAL_CPU_OOM-'+intent.name);archive.parent.mkdir(exist_ok=True);assert not archive.exists()
 sha=hashlib.sha256(intent.read_bytes()).hexdigest();intent.rename(archive)
 receipt=dict(status='CONFIRMED_OWN_TERMINAL_CPU_OOM_JOURNAL_ARCHIVED',epoch=time.time(),source_commit=source,original_journal=str(intent),preserved_journal=str(archive),journal_sha256=sha,systemd=values,failed_terminal=terminal,fault_snapshot=snapshot,foreign_jobs_modified=False,training_submitted=False)
 with Path(PACKAGE+'/parent-dispatch-reconciled.json').open('x') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
'''.replace('PACKAGE',repr(pkg))
 cmd='env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code);name='RECOVERY_PARENT_JOURNAL_RECONCILED.json'
else:
 cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python',pkg+'/submit_shared_once.py','--package',pkg,'--manifest-sha',r['manifest_sha256'],'--release-path',pkg+'/release.json','--release-sha',r['release_sha256'],'--script',rendered['path'],'--script-sha',rendered['sha256']]);name='RECOVERY_SUBMISSION.json'
x=ssh.run(cmd,timeout=120);(d/(name+'.command.json')).write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n');print(x.stdout,x.stderr);x.check_returncode();j=json.loads(x.stdout);(d/name).write_text(json.dumps(j,indent=2)+'\n')
