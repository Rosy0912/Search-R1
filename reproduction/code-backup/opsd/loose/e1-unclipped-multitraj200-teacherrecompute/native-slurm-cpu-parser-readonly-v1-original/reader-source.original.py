import json,subprocess,datetime,os,socket,pathlib
commands=[['sbatch','--version'],['sbatch','--help'],['srun','--help'],['scontrol','show','partition','-o'],['scontrol','show','config']]
rows=[]
for a in commands:
 r=subprocess.run(a,capture_output=True,timeout=25)
 rows.append({'argv':a,'returncode':r.returncode,'waited':True,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()})
p=pathlib.Path('/proc/self');st=p.joinpath('stat').read_text();start=st.rsplit(')',1)[1].split()[19]
print(json.dumps({'kind':'ACTUAL_READ_ONLY_NATIVE_SLURM_HELP_DEFAULTS_NOT_SUBMISSION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'pid':os.getpid(),'start':start,'uids':os.getresuid(),'cgroup':p.joinpath('cgroup').read_text(),'commands':rows,'job_submitted':False,'test_only_executed':False}))
