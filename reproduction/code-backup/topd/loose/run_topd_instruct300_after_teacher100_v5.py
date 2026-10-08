"""Run inside ROOT's unique afterok Slurm allocation; no submission or Base fallback."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,os,pathlib,re,socket,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def bound(ref):
 p=pathlib.Path(ref['path'])
 if ROOT not in p.resolve().parents or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Actual teacher evidence drift')
 return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--infrastructure',required=True);p.add_argument('--teacher-result',required=True);p.add_argument('--teacher-result-sha256',required=True);p.add_argument('--teacher-job',required=True);p.add_argument('--teacher-terminal',required=True);p.add_argument('--teacher-terminal-sha256',required=True);p.add_argument('--run-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 if socket.gethostname()!='h100-11' or not os.environ.get('SLURM_JOB_ID'):raise ValueError('ROOT-owned h11 Slurm allocation required')
 if not re.fullmatch('[1-9][0-9]*',a.teacher_job):raise ValueError('Actual teacher job required')
 result=bound({'path':a.teacher_result,'sha256':a.teacher_result_sha256})
 if str(result['job'])!=a.teacher_job or result['status']!='COMPLETE_UNREVIEWED' or result['cleanup_passed'] is not True:raise ValueError('Actual teacher100 did not complete')
 terminal=bound({'path':a.teacher_terminal,'sha256':a.teacher_terminal_sha256})
 raw=terminal.get('stdout','')
 if terminal.get('observation_kind')=='SACCT_RAW_SAME_JOB_TERMINAL':
  rows=[line.split('|') for line in raw.splitlines() if line.strip()]
  matched=[row for row in rows if len(row)>=3 and row[0].strip()==a.teacher_job]
  terminal_ok=terminal.get('job_id')==a.teacher_job and terminal.get('returncode')==0 and len(matched)==1 and matched[0][1].strip()=='COMPLETED' and matched[0][2].strip()=='0:0'
 else:
  terminal_ok=terminal.get('returncode',0)==0 and bool(re.search(r'\bJobId='+re.escape(a.teacher_job)+r'\b',raw) and re.search(r'\bJobState=COMPLETED\b',raw) and re.search(r'\bExitCode=0:0\b',raw))
 if not terminal_ok:raise ValueError('Actual teacher Slurm terminal is not COMPLETED0:0')
 actual=bound(result['actual_outer_evidence'])
 if actual['actual_completed_outer_steps']!=list(range(1,101)) or actual['export_step']!=100:raise ValueError('Need full actual100, not Base/profile/2+98')
 cleanup=bound(result['cleanup'])
 for key in ('own_processes_remaining','baseline_identity_missing','original_gpu_baseline_missing','own_gpu_records_remaining','parent_gpu_records','sampler_errors','guardian_errors'):
  if cleanup.get(key)!=[]:raise ValueError('Teacher cleanup incomplete '+key)
 if cleanup.get('parent_cuda_initialized') is not False or cleanup.get('error') is not None:raise ValueError('Actual teacher cleanup failed')
 export=result['export'];directory=pathlib.Path(export['directory']).resolve()
 if ROOT not in directory.parents or directory==ROOT/'assets/qwen25-32b':raise ValueError('Actual owned trained HF required')
 files=export['files']
 if export['tensor_count']!=771 or not export['shards'] or 'config.json' not in files:raise ValueError('Complete teacher32B HF export required')
 if 'model.safetensors.index.json' not in files and export['shards']!=['model.safetensors']:raise ValueError('HF index or complete single-file weights required')
 for name,record in files.items():
  f=directory/name
  if f.is_symlink() or not f.is_file() or f.stat().st_size!=record['size'] or not re.fullmatch('[0-9a-f]{64}',record['sha256']):raise ValueError('Teacher HF export missing '+name)
 # Full fresh HF SHA is checked by the existing supervisor before any model startup.
 binder=ROOT/'tools/bind_topd_instruct300_v7.py'
 command=[str(ROOT/'env/bin/python'),str(binder),'--infrastructure',a.infrastructure,'--teacher-result',a.teacher_result,'--teacher-result-sha256',a.teacher_result_sha256,'--run-directory',a.run_directory,'--output',a.output]
 subprocess.run(command,check=True)
 os.environ.update(AGENT_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='backend:native,max_split_size_mb:128',PYTHONPATH=str(ROOT/'repo-terminal-fixed413')+':'+str(ROOT/'tools/topd-instruct300-launch-v5/runtime'))
 argv=[str(ROOT/'env/bin/python'),str(ROOT/'repo-terminal-fixed413/search_r1/baselines/preflight.py'),'--',str(ROOT/'env/bin/python'),str(ROOT/'tools/topd-instruct300-launch-v5/runtime/supervise.py'),'--config',a.output]
 os.execv(argv[0],argv)
if __name__=='__main__':main()
