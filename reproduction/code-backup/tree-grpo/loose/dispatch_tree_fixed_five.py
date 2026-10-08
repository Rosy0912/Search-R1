import hashlib,json,re,shlex,subprocess
from pathlib import Path
from remote import route
p=Path('work/tree-fixed-five-review/PRE_DISPATCH_RECEIPT.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='52c3a26a48db9bfa98a4bd9b2ec130d1d64b4c954ed656552b41f6042059c3e7'
r=json.loads(p.read_text());assert r['status']=='READY_FOR_FRESH_FIVE_DISPATCH_ONLY'
assert hashlib.sha256(Path('work/start_tree_fixed_five_296217a.sh').read_bytes()).hexdigest()==r['wrapper_sha256']
assert hashlib.sha256(Path('work/tree-fixed-five-296217a/manifest.json').read_bytes()).hexdigest()==r['manifest_sha256']
assert hashlib.sha256(Path('work/check_fixed_five_afterany.py').read_bytes()).hexdigest()=='a6a73cf43a659d55402d56005d975749a5766536ceec54bffa8e0f364f3b96d3'
output=Path('outputs/evidence/tree-fixed-five-dispatch.json');assert not output.exists(),'Never dispatch duplicate five'
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
def submit(args):
 answer=subprocess.check_output(route(shlex.join(['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--ntasks=1',*args]),'h100-16'),text=True).strip();assert re.fullmatch(r'\d+(;[^\n]+)?',answer),answer;return int(answer.split(';')[0])
job=submit(['--gres=gpu:8','--cpus-per-task=32','--mem=256G','--exclusive','--time=02:00:00','--job-name=tree-fixed-five-296217a','--output='+root+'/logs/tree-fixed-five-%j.out','--wrap=bash '+root+'/tools/start_tree_fixed_five_296217a.sh'])
record=dict(training_job_id=job,run=root+f'/runs/tree-fixed-five-{job}',source=r['code_commit'],pre_dispatch_receipt_sha256=h,state='SUBMITTED_UNKNOWN_NOT_ACCEPTED',scope='Fresh fixed five only, budget6 actual5 frombase; no automaticformal')
output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
cpu=submit(['--gres=gpu:0','--cpus-per-task=8','--mem=64G','--time=00:30:00','--dependency=afterany:'+str(job),'--job-name=tree-fixed-five-hf','--output='+root+'/logs/tree-fixed-five-hf-%j.out','--wrap='+root+'/env/bin/python '+root+'/tools/check_fixed_five_afterany_296217a.py '+str(job)])
record['afterany_cpu_job_id']=cpu;record['cpu_run']=root+f'/runs/tree-fixed-five-cpu-{job}-{cpu}';output.write_text(json.dumps(record,indent=2)+'\n')
terminal=submit(['--gres=gpu:0','--cpus-per-task=1','--mem=2G','--time=00:02:00','--dependency=afterany:'+str(cpu),'--job-name=tree-fixed-five-cpu-terminal','--output='+root+'/logs/tree-fixed-five-cpu-terminal-%j.out','--wrap=scontrol show job '+str(cpu)+' > '+record['cpu_run']+'/slurm-cpu-terminal.txt'])
record['cpu_terminal_job_id']=terminal;output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
