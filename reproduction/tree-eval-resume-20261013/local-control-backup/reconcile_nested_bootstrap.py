"""Archive only a proven own pre-training failed submission; no training launch."""
import argparse,json,shlex
from pathlib import Path
import ssh_retained18 as ssh
p=argparse.ArgumentParser();p.add_argument('--fault',type=Path,required=True);p.add_argument('--package',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();fault=json.loads(a.fault.read_text())
code=r'''import fcntl,json,hashlib,os,subprocess,time
from pathlib import Path
fault=FAULT;source=fault['source_commit'];root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'instruct300';run=root/'runs'/('tree-nested-branch-fresh200-not-slurm;systemd:'+source);unit='sst-tree-nested-branch-fresh200-'+source[:8]+'.service'
assert hashlib.sha256(Path(fault['snapshot_path']).read_bytes()).hexdigest()==fault['snapshot_sha256']
assert fault['status']=='OWN_BOOTSTRAP_EXIT_BEFORE_TRAINING' and fault['child_run_created'] is False and fault['native_load_attempted'] is False
fd=os.open(base/'FRESH200_TRAIN_DISPATCH.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
try:
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=subprocess.check_output(['systemctl','show',unit,'--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,ControlGroup,InvocationID'],text=True);v=dict(line.split('=',1) for line in state.splitlines());prior=dict(line.split('=',1) for line in fault['systemd'].splitlines())
 assert v['MainPID']=='0' and v['ActiveState']=='failed' and v['Result']=='exit-code' and v['ExecMainStatus']=='1' and v['InvocationID']==prior['InvocationID'],state
 cg=Path('/sys/fs/cgroup/system.slice')/unit
 assert not cg.exists() and not run.exists(),'Bootstrap no longer terminal before runcreation'
 ports=subprocess.check_output(['ss','-ltnp'],text=True);assert not any((':'+str(port)+' ') in line for line in ports.splitlines() for port in (16382,18082))
 intent=base/('SUBMISSION_ONCE-nested-branch-fresh200-'+source+'.json');assert list(base.glob('SUBMISSION_ONCE-nested-branch-fresh200-*.json'))==[intent]
 value=json.loads(intent.read_text());assert value['status']=='SUBMITTED' and value['code_commit']==source and value['unit']==unit
 archive=base/'submission-history'/('TERMINAL_BOOTSTRAP_FAILURE-'+intent.name);assert not archive.exists();sha=hashlib.sha256(intent.read_bytes()).hexdigest();intent.rename(archive)
 receipt=dict(status='CONFIRMED_OWN_PRETRAIN_BOOTSTRAP_FAILURE_JOURNAL_ARCHIVED',epoch=time.time(),source_commit=source,journal_sha256=sha,preserved_journal=str(archive),systemd=v,fault_snapshot_path=fault['snapshot_path'],fault_snapshot_sha256=fault['snapshot_sha256'],actual_optimizer_writes_this_instance=0,child_run_created=False,in_effective_training_history=False,last_complete_boundary=5,foreign_jobs_modified=False,training_submitted=False)
 with Path(PACKAGE+'/bootstrap-predecessor-reconciled.json').open('x') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
'''.replace('PACKAGE',repr(a.package)).replace('FAULT',repr({k:fault[k] for k in ('source_commit','status','child_run_created','native_load_attempted','systemd','snapshot_path','snapshot_sha256')}))
x=ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code),timeout=60);a.output.with_suffix('.command.json').write_text(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),indent=2)+'\n');
if x.returncode:raise RuntimeError(x.stderr[-2500:]+x.stdout[-1000:])
o=json.loads(x.stdout);a.output.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))
