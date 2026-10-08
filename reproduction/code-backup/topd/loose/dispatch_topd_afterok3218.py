"""Inside ROOT's one afterok:3218 student allocation; obtain future actual refs and exec."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,os,pathlib,re,socket,subprocess,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
TEACHER=ROOT/'logs/r1-standard32b-grpo100-common-v3-r2/evidence'
def atomic_new(path,record):
 with path.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-parent');a=p.parse_args()
 job=os.environ.get('SLURM_JOB_ID','')
 if socket.gethostname()!='h100-11' or not re.fullmatch('[1-9][0-9]*',job) or job=='3218':raise ValueError('Own subsequent h11 student Slurm allocation required')
 parent=pathlib.Path(a.run_parent).resolve() if a.run_parent else ROOT/'logs'/('topd-instruct300-after-teacher3218-'+job)
 if ROOT/'logs' not in parent.parents or parent.is_symlink():raise ValueError('Own fresh student parent required')
 parent.mkdir(mode=0o700,exist_ok=False)
 # Actual result does not exist until the original teacher supervisor completes.
 result_path=TEACHER/'train-result.json';raw=result_path.read_bytes();result=json.loads(raw)
 if str(result.get('job'))!='3218' or result.get('evidence_root')!=str(TEACHER) or result.get('status')!='COMPLETE_UNREVIEWED' or result.get('cleanup_passed') is not True:raise ValueError('3218 actual100 is not completed')
 query=subprocess.run(['scontrol','-o','show','job','3218'],capture_output=True,text=True,timeout=15)
 terminal={'observation_kind':'SCONTROL_RAW_SAME_JOB_TERMINAL','job_id':'3218','returncode':query.returncode,'stdout':query.stdout,'stderr':query.stderr,'epoch':time.time()}
 scontrol_path=parent/'teacher3218-scontrol-original.json';atomic_new(scontrol_path,terminal)
 if query.returncode==0 and re.search(r'\bJobId=3218\b',query.stdout):
  terminal_path=scontrol_path
 else:
  # Accounting is distinct raw evidence, never rewritten into synthetic scontrol text.
  accounting=subprocess.run(['sacct','-n','-P','-j','3218','--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=15)
  terminal={'observation_kind':'SACCT_RAW_SAME_JOB_TERMINAL','job_id':'3218','returncode':accounting.returncode,'stdout':accounting.stdout,'stderr':accounting.stderr,'epoch':time.time(),'original_scontrol':{'path':str(scontrol_path),'sha256':hashlib.sha256(scontrol_path.read_bytes()).hexdigest()}}
  terminal_path=parent/'teacher3218-sacct-original.json';atomic_new(terminal_path,terminal)
 argv=[str(ROOT/'env/bin/python'),str(ROOT/'tools/run_topd_instruct300_after_teacher100_v3.py'),'--infrastructure',str(TEACHER/'effective-runtime-config.json'),'--teacher-result',str(result_path),'--teacher-result-sha256',hashlib.sha256(raw).hexdigest(),'--teacher-job','3218','--teacher-terminal',str(terminal_path),'--teacher-terminal-sha256',hashlib.sha256(terminal_path.read_bytes()).hexdigest(),'--run-directory',str(parent/'evidence'),'--output',str(parent/'static-config.json')]
 atomic_new(parent/'dispatcher-inputs.json',{'status':'BOUND_ACTUAL_INPUTS_NOT_STUDENT_TRAINING_COMPLETE','teacher_job':'3218','student_job':job,'teacher_result':{'path':str(result_path),'sha256':hashlib.sha256(raw).hexdigest()},'terminal':{'path':str(terminal_path),'sha256':hashlib.sha256(terminal_path.read_bytes()).hexdigest()},'argv':argv})
 os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUNBUFFERED']='1'
 os.execv(argv[0],argv)
if __name__=='__main__':main()
