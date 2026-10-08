import pathlib,os,json,subprocess,datetime,stat
R=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006');RUN=R/'runs/formal-3304';REPO=R/'repo-fcecbea5d939'
def c(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return {'argv':a,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
files={}
for n in ['evidence/train-process-exit.json','job-exit.txt','approval-ready.json','evaluation.json','evidence/final200-cpu-inspection-once-v1','evidence/postterminal-source-bindings-once-v1','evidence/final-archive-3304-once-v1']:
 p=RUN/n
 if not p.exists():files[n]={'exists':False};continue
 s=p.lstat();q={'exists':True,'size':s.st_size,'mode':s.st_mode,'uid':s.st_uid,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns}
 if stat.S_ISREG(s.st_mode) and s.st_size<2*1024*1024:q['text']=p.read_text()
 elif stat.S_ISDIR(s.st_mode):q['members']=[{'name':x.name,'size':x.lstat().st_size} for x in p.iterdir()]
 files[n]=q
pidfiles={}
for pid in [2548319,2548974,2549227,2549228,2549229,2549231,2549232,2549230,2549233,2543106,2533919]:
 p=pathlib.Path('/proc')/str(pid);q={'exists':p.exists()}
 if q['exists']:
  for n in ['stat','status','cgroup']:
   try:q[n]=p.joinpath(n).read_text()
   except OSError as e:q[n+'_error']=repr(e)
 pidfiles[str(pid)]=q
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':c(['hostname']),'slurm':c(['scontrol','show','job','-o','3304']),'sacct':c(['sacct','-j','3304','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList']),'source_head':c(['git','--no-optional-locks','-C',str(REPO),'rev-parse','HEAD']),'source_status':c(['git','--no-optional-locks','-C',str(REPO),'status','--porcelain']),'files':files,'original_pids':pidfiles,'gpu_apps':c(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']),'run_members':[{'name':p.name,'is_dir':p.is_dir(),'size':p.lstat().st_size} for p in RUN.iterdir()]},sort_keys=True))
