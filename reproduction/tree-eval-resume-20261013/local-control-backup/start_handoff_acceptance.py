"""One CPU-only systemd submission after actual final evaluation and cleanup."""
import importlib.util, json, shlex
from pathlib import Path

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ssh',HERE/'ssh_retained18.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
deployment=json.loads(Path('outputs/tree-eval-resume-20261013/HANDOFF_HF_DEPLOYMENT.json').read_text())
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
run=root+'/runs/tree-branpo-fresh200-3360'
inputs=run+'/output/eval-handoff-acceptance-inputs'
evaltools=root+'/instruct300/eval-shared-0e00998fdc645281986250cd8632bb70c6e2fa9b'
unit='sst-tree-handoff-hf-'+deployment['source_commit'][:7]+'.service'
args=[root+'/env/bin/python',deployment['tools']+'/verify_handoff_hf.py','--original-job-terminal',inputs+'/original-job-terminal.json','--handoff-journal',run+'/output/eval-resume/dispatch-once.json','--manifest',run+'/output/eval-resume/asset-manifest.json','--release',evaltools+'/release.json','--review',evaltools+'/SHARED_CODE_REVIEW.json']
code='''import hashlib,json,os,subprocess,time
from pathlib import Path
unit=UNIT; args=ARGS; inputs=Path(INPUTS); run=Path(RUN)
wrapper=Path(args[1])
assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==WRAPPER_SHA
state=subprocess.check_output(['systemctl','show','sst-tree-eval-shared-0e00998.service','--property=ActiveState,MainPID,ExecMainStatus'],text=True)
fields=dict(x.split('=',1) for x in state.splitlines() if '=' in x)
assert fields['ActiveState']=='inactive' and fields['MainPID']=='0' and fields['ExecMainStatus']=='0',state
result=json.loads(Path(EVAL_RESULT).read_text())
assert result['status']=='COMPLETE' and result['worker_exit_code']==0 and result['own_cleanup_residual']=={}
final=json.loads((run/'output/eval-resume/scores/final-metrics.json').read_text())
assert final['evaluated_count']==51713
assert not (run/'output/eval-handoff-hf-acceptance').exists()
intent=dict(status='SUBMISSION_INTENT',epoch=time.time(),unit=unit,args=args,wrapper_sha256=WRAPPER_SHA,evaluation_terminal_sha256=hashlib.sha256(Path(EVAL_RESULT).read_bytes()).hexdigest(),physical_gpus=0,slurm_reserved_gpus=0)
path=inputs/'cpu-dispatch-once.json'
with path.open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
command=['systemd-run','--unit',unit,'--property=CPUQuota=800%','--property=MemoryMax=96G','--property=TimeoutStopSec=120','--property=RuntimeMaxSec=7200','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=HF_HUB_OFFLINE=1','--setenv=TRANSFORMERS_OFFLINE=1','--setenv=PYTHONOPTIMIZE=0','--setenv=PYTHONNOUSERSITE=1','--setenv=OMP_NUM_THREADS=8','--setenv=MKL_NUM_THREADS=8','--setenv=OPENBLAS_NUM_THREADS=8','/bin/bash','-c',SHELL_COMMAND]
r=subprocess.run(command,text=True,capture_output=True)
receipt=dict(intent,status='SUBMISSION_RESULT',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,observed_epoch=time.time())
with (inputs/'cpu-submission-result.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''
values=dict(UNIT=unit,ARGS=args,INPUTS=inputs,RUN=run,WRAPPER_SHA=deployment['files']['verify_handoff_hf.py'],EVAL_RESULT=root+'/runs/sst-tree-eval-shared-0e00998/run-result.json',SHELL_COMMAND='exec '+shlex.join(args)+' > '+shlex.quote(inputs+'/cpu-wrapper.log')+' 2>&1')
for name,value in values.items():code=code.replace(name,repr(value))
r=ssh.run('python3 -c '+shlex.quote(code),timeout=120)
Path('outputs/tree-eval-resume-20261013/HANDOFF_HF_DISPATCH.txt').write_text(r.stdout+'\n'+r.stderr+'\nexit '+str(r.returncode)+'\n')
print(r.stdout,r.stderr,r.returncode)
