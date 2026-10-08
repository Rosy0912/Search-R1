"""Launch exactly one reviewed P5 to total300 continuation on owned exclusive H100s."""
import fcntl,hashlib,json,os,subprocess,shlex,shutil,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
RUN='author-native-continue-300-v';SOURCE='author-native-continue-300-u'
def main():
 lock=(ROOT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 claim=ROOT/(RUN+'-launch-claim.json')
 if claim.exists():print(json.dumps(dict(status='already_claimed',claim=json.loads(claim.read_text()))));return
 assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
 assert not (ROOT/'runs'/RUN).exists()
 assert shutil.disk_usage(ROOT).free>5_000_000_000_000
 assert 'MaxTime=UNLIMITED' in subprocess.check_output(['scontrol','show','partition','gpu'],text=True)
 verified=json.loads((ROOT/'tools/source-checkpoint-verification.json').read_text());STEP=verified['source_step']
 assert verified['source_run']==SOURCE and verified['actual_full_file_shas_reverified'] and verified['ranks']==8 and verified['source_resources_released']
 assert str(verified['source_job']) not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
 assert 0<STEP<300 and hashlib.sha256((Path(verified['checkpoint'])/'data.pt').read_bytes()).hexdigest()==verified['data_sha256']
 ready=json.loads((ROOT/'tools/preparation-ready.json').read_text());assert ready['run']==RUN and ready['cpu_preparation_passed']
 for n,h in ready['tools_sha256'].items():assert hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest()==h,n
 assert json.loads((ROOT/'tools/cpu-continue-gate.json').read_text())['passed']
 assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
 assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
 command=['srun','--immediate=60','-p','gpu','-w','h100-16','-N1','--ntasks=1','--gres=gpu:h100:8','--exclusive','--cpus-per-task=128','--mem=768G','--time=7-00:00:00','--job-name=cosearch-author-continue','env','COSEARCH_NATIVE_RUN_ID='+RUN,'COSEARCH_NATIVE_SOURCE_RUN_ID='+SOURCE,'COSEARCH_NATIVE_RESUME_STEP='+str(STEP),'COSEARCH_NATIVE_CONTINUE_TO=300','COSEARCH_NATIVE_RELOAD_ONLY=0','COSEARCH_NATIVE_GPU_PROBE_ONLY=0','bash',str(ROOT/'tools/start_native_entry.sh')]
 with claim.open('x') as f:json.dump(dict(command=command,status='claimed',created_at=time.time(),source=SOURCE,source_step=STEP,target_total_step=300,strict_accepted=False,user_authorized_despite_speed_probe_failures=True),f)
 logs=ROOT/'logs';logs.mkdir(exist_ok=True)
 subprocess.run(['tmux','new-session','-d','-s',RUN,shlex.join(command)+' > '+shlex.quote(str(logs/(RUN+'.log')))+' 2>&1'],check=True)
 print(json.dumps(dict(status='submitted',run=RUN,source=SOURCE,target=300)),flush=True)
if __name__=='__main__':main()
