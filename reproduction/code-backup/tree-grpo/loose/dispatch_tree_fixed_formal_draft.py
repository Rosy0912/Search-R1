"""Not runnable until signed independent formal preparation receipt is present."""
import hashlib,json,re,shlex,subprocess
from pathlib import Path
from remote import route
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path('work/tree-fixed-formal-review/PRE_DISPATCH_RECEIPT.json');r=json.loads(p.read_text())
assert r['status']=='READY_FOR_FRESH_FORMAL500_DISPATCH_ONLY'
assert r['code_commit']=='296217a4b676a26a92fc729f5d9882b785dbfa96' and r['actual_updates_authorized']==500
assert sha('work/tree-fixed-formal-296217a/manifest.json')==r['manifest_sha256']
assert sha('work/start_tree_fixed_formal_296217a.sh')==r['wrapper_sha256']
assert sha('work/check_fixed_formal_afterany_296217a.py')==r['cpu_afterany_sha256']
assert sha('work/summarize_tree_fixed_formal_296217a.py')==r['summary_sha256']
output=Path('outputs/evidence/tree-fixed-formal-dispatch.json');assert not output.exists(),'Never duplicate formal submission'
def submit(args):
 answer=subprocess.check_output(route(shlex.join(['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--ntasks=1',*args]),'h100-16'),text=True).strip();assert re.fullmatch(r'\d+(;[^\n]+)?',answer),answer;return int(answer.split(';')[0])
job=submit(['--gres=gpu:8','--cpus-per-task=32','--mem=256G','--exclusive','--time=3-00:00:00','--job-name=tree-fixed-formal-296217a','--output='+root+'/logs/tree-fixed-formal-%j.out','--wrap=bash '+root+'/tools/start_tree_fixed_formal_296217a.sh'])
record=dict(training_job_id=job,run=root+f'/runs/tree-fixed-formal-{job}',source=r['code_commit'],pre_dispatch_receipt_sha256=sha(p),state='SUBMITTED_UNKNOWN_NOT_ACCEPTED',scope='Fresh fixed formal500 from base; total501; no exact optimizer/RNG/dataloader resume')
def save():output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
save()
cpu=submit(['--gres=gpu:0','--cpus-per-task=8','--mem=64G','--time=00:30:00','--dependency=afterany:'+str(job),'--job-name=tree-fixed-formal-hf500','--output='+root+'/logs/tree-fixed-formal-hf-%j.out','--wrap='+root+'/env/bin/python '+root+'/tools/check_fixed_formal_afterany_296217a.py '+str(job)])
record.update(afterany_cpu_job_id=cpu,cpu_run=root+f'/runs/tree-fixed-formal-cpu-{job}-{cpu}');save()
terminal=submit(['--gres=gpu:0','--cpus-per-task=1','--mem=2G','--time=00:02:00','--dependency=afterany:'+str(cpu),'--job-name=tree-fixed-formal-cpu-terminal','--output='+root+'/logs/tree-fixed-formal-cpu-terminal-%j.out','--wrap=scontrol show job '+str(cpu)+' > '+record['cpu_run']+'/slurm-cpu-terminal.txt'])
record['cpu_terminal_job_id']=terminal;save()
summary=submit(['--gres=gpu:0','--cpus-per-task=4','--mem=16G','--time=00:30:00','--dependency=afterok:'+str(cpu)+':'+str(terminal),'--job-name=tree-fixed-formal-summary','--output='+root+'/logs/tree-fixed-formal-summary-%j.out','--wrap='+root+'/env/bin/python '+root+'/tools/summarize_tree_fixed_formal_296217a.py --run '+record['run']+' --hf-cpu-run '+record['cpu_run']+' --output '+record['run']+'/formal-summary.json'])
record['summary_cpu_job_id']=summary;save()
