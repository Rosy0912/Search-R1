import base64,hashlib,importlib.util,json,shlex
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
p=Path('outputs/tree-eval-resume-20261013/POLL-SECOND.json');data=p.read_bytes();value=json.loads(data)
assert 'JobId=3360 ' in value['old_slurm'] and 'JobState=CANCELLED ' in value['old_slurm'] and 'ExitCode=0:15 ' in value['old_slurm']
remote='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-branpo-fresh200-3360/output/eval-handoff-acceptance-inputs'
code='''import base64,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(REMOTE);root.mkdir(exist_ok=True)
def publish(p,b):
 if p.exists():assert p.read_bytes()==b;return
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
raw=base64.b64decode(DATA);receipt=json.loads(raw);h=hashlib.sha256(raw).hexdigest()
publish(root/'original-scontrol-observation.json',raw)
run=root.parent.parent;active=subprocess.check_output(['squeue','--local','-h','-o','%i'],text=True).split()
assert '3360' not in active
snapshot=dict(job_id='3360',state='CANCELLED',exit_code='0:15',observed_epoch=receipt['observed_epoch'],source='scontrol-preserved',raw=receipt['old_slurm'],preserved_source_path=str(root/'original-scontrol-observation.json'),preserved_source_sha256=h,original_run_result_absent=not (run/'run-result.json').exists(),original_train_terminal_absent=not (run/'output/train200-terminal.json').exists(),absence_checked_epoch=time.time(),current_queue_absent=True,accounting_storage='disabled; earlier actual scontrol observation retained')
assert snapshot['original_run_result_absent'] and snapshot['original_train_terminal_absent']
payload=(json.dumps(snapshot,indent=2)+'\\n').encode();publish(root/'original-job-terminal.json',payload)
print(json.dumps(dict(path=str(root/'original-job-terminal.json'),sha256=hashlib.sha256(payload).hexdigest(),snapshot=snapshot)))
'''.replace('REMOTE',repr(remote)).replace('DATA',repr(base64.b64encode(data).decode()))
r=ssh.run('python3 -c '+shlex.quote(code),timeout=60)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
Path('outputs/tree-eval-resume-20261013/ORIGINAL_TERMINAL_BINDING.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n');print(r.stdout)
