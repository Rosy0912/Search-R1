import subprocess,shlex,json
from pathlib import Path
from remote import route
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'; run=root+'/runs/tree-formal-2971'
def submit(name,dependency,cpus,mem,minutes,command):
 args=['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--gres=gpu:0','--ntasks=1','--cpus-per-task='+str(cpus),'--mem='+mem,'--time='+minutes,'--dependency='+dependency,'--job-name='+name,'--output='+root+'/logs/'+name+'-%j.out','--wrap='+command]
 out=subprocess.check_output(route(shlex.join(args),'h100-16'),text=True).strip();assert out.isdigit(),out;return int(out)
archive=submit('tree-formal-parent-terminal','afterany:2971',1,'2G','00:02:00','scontrol show job 2971 > '+run+'/slurm-parent-terminal.txt')
summary=submit('tree-formal-summary','afterok:2972',4,'16G','00:30:00',root+'/env/bin/python '+root+'/tools/summarize_tree_formal.py --run '+run+' --output '+run+'/formal-summary.json')
terminal=submit('tree-formal-cpu-terminal','afterany:2972',1,'2G','00:02:00','scontrol show job 2972 > '+run+'/slurm-cpu-terminal.txt')
result=dict(training=2971,hf_cpu=2972,parent_terminal_capture_job=archive,summary_cpu_job=summary,cpu_terminal_capture_job=terminal,scope='Zero-GPU same-node dependent evidence collectors only; no changes to running training, source or protocol.')
Path('outputs/evidence/shared-formal-2971-dependent-collectors.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
