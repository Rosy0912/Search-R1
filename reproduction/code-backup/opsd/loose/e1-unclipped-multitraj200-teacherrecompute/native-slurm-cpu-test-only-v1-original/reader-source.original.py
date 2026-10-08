import subprocess,json,os,socket,pathlib,datetime
assert socket.gethostname()=='h100-10' and os.getuid()==0
repo='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/repo-fcecbea5d939'
env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SRUN_','SLURM_'))}
argv=['sbatch','--test-only','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=128G','--nodelist=h100-10','--export=NONE','--chdir='+repo,'--wrap=/usr/bin/true']
assert '--test-only' in argv and not any(a.startswith(('--gres','--gpu','--exclusive','--cluster')) for a in argv)
r=subprocess.run(argv,env=env,capture_output=True,timeout=40)
p=pathlib.Path('/proc/self');st=p.joinpath('stat').read_text()
print(json.dumps({'kind':'ACTUAL_NATIVE_SBATCH_TEST_ONLY_PARSER_PROJECTION_NOT_NEWJOB_OR_GPU_ALLOCATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'pid':os.getpid(),'start':st.rsplit(')',1)[1].split()[19],'uids':os.getresuid(),'cgroup':p.joinpath('cgroup').read_text(),'argv':argv,'removed_environment_names':sorted(k for k in os.environ if k.startswith(('SBATCH_','SRUN_','SLURM_'))),'returncode':r.returncode,'waited':True,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'actual_submission':False,'actual_new_job_id':None,'actual_gpu_allocation':None,'scope':'No real script, submission, step, GPU or actual200. Representative CPU4/mem128G/nodelist/export/chdir syntax only; scheduling projection is not an actual execution ID.'}))
