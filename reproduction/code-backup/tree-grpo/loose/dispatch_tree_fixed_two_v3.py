import hashlib,json,re,shlex,subprocess
from pathlib import Path
from remote import route
p=Path('work/tree-fixed-two-review/V3_PRE_DISPATCH_RECEIPT.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='e1244804cbdede8b4f6243b665d3b88a7e1f413deeaa02072f088a23645547e3'
r=json.loads(p.read_text());assert r['status']=='READY_FOR_FRESH_TWO_DISPATCH_ONLY'
assert hashlib.sha256(Path('work/start_tree_fixed_two_296217a_v3.sh').read_bytes()).hexdigest()==r['wrapper_sha256']
assert hashlib.sha256(Path('work/tree-fixed-two-296217a-v3/manifest.json').read_bytes()).hexdigest()==r['tool_manifest_sha256']
assert hashlib.sha256(Path('work/check_fixed_two_afterany.py').read_bytes()).hexdigest()=='9db6826e2c286c1c65daa5d3b176a01b73a957b99385055b9e88cf1375ef9ae2'
output=Path('outputs/evidence/tree-fixed-two-v3-dispatch.json');assert not output.exists(),'Never dispatch duplicate two'
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
def submit(args):
 answer=subprocess.check_output(route(shlex.join(['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--ntasks=1',*args]),'h100-16'),text=True).strip();assert re.fullmatch(r'\d+(;[^\n]+)?',answer),answer;return int(answer.split(';')[0])
job=submit(['--gres=gpu:8','--cpus-per-task=32','--mem=256G','--exclusive','--time=02:00:00','--job-name=tree-fixed-two-296217a-v3','--output='+root+'/logs/tree-fixed-two-%j.out','--wrap=bash '+root+'/tools/start_tree_fixed_two_296217a_v3.sh'])
record=dict(training_job_id=job,run=root+f'/runs/tree-fixed-two-{job}',source=r['candidate_commit'],pre_dispatch_receipt_sha256=h,state='SUBMITTED_UNKNOWN_NOT_ACCEPTED',scope='Fresh fixed two only, budget3 actual2 frombase; no automaticfive/formal')
output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
cpu=submit(['--gres=gpu:0','--cpus-per-task=8','--mem=64G','--time=00:30:00','--dependency=afterany:'+str(job),'--job-name=tree-fixed-two-hf','--output='+root+'/logs/tree-fixed-two-hf-%j.out','--wrap='+root+'/env/bin/python '+root+'/tools/check_fixed_two_afterany_296217a.py '+str(job)])
record['afterany_cpu_job_id']=cpu;record['cpu_run']=root+f'/runs/tree-fixed-two-cpu-{job}-{cpu}';output.write_text(json.dumps(record,indent=2)+'\n')
terminal=submit(['--gres=gpu:0','--cpus-per-task=1','--mem=2G','--time=00:02:00','--dependency=afterany:'+str(cpu),'--job-name=tree-fixed-two-cpu-terminal','--output='+root+'/logs/tree-fixed-two-cpu-terminal-%j.out','--wrap=scontrol show job '+str(cpu)+' > '+record['cpu_run']+'/slurm-cpu-terminal.txt'])
record['cpu_terminal_job_id']=terminal;output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
