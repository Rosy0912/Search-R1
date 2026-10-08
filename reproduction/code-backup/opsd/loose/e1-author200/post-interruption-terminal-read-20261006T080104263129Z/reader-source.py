
import base64,datetime,hashlib,json,os,stat,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006');run=root/'runs/formal-3261';out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'READ_ONLY_ACTUAL_EXIT_AND_SCHEDULER_NOT_ACCEPTANCE','commands':{},'files':{},'pids':{},'checkpoint200_presence':{}}
for name,argv in [('scontrol',['scontrol','show','job','3261','-o']),('sacct',['sacct','-j','3261','--format=JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed,NodeList,AllocTRES','--parsable2','--noheader']),('nvidia',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])]:
 try:
  p=subprocess.run(argv,capture_output=True,timeout=25);out['commands'][name]={'argv':argv,'returncode':p.returncode,'stdout_base64':base64.b64encode(p.stdout).decode(),'stderr_base64':base64.b64encode(p.stderr).decode()}
 except Exception as e:out['commands'][name]={'error_type':type(e).__name__,'error':str(e)}
for pid in [1533312,1533910,1534229,1534230,1534231,1534232,1534233,1534234,1534235,1531763,1518778]:
 p=Path('/proc')/str(pid)/'stat';out['pids'][str(pid)]={'present':p.exists()}
 if p.exists():out['pids'][str(pid)]['stat_original']=p.read_text()
for relative in ['job-exit.txt','evidence/train-process-exit.json','evidence/sampler-stop.json','evidence/resource-data-audit.json','evidence/owned-child-cleanup.json','evidence/process-finished-review-required.json']:
 p=run/relative
 if not p.exists():out['files'][relative]={'missing':True};continue
 assert p.resolve()==p and p.is_file() and p.stat().st_uid==0 and p.stat().st_size<10*1024*1024
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:b=os.fstat(f.fileno());raw=f.read();a=os.fstat(f.fileno())
 z=p.stat();fields=('st_dev','st_ino','st_uid','st_size','st_mtime_ns','st_ctime_ns');out['files'][relative]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode(),'unchanged':all(tuple(getattr(v,k) for k in fields)==tuple(getattr(b,k) for k in fields) for v in [a,z]),'device':b.st_dev,'inode':b.st_ino}
for p in [run/'train.log',run/'formal-wrapper.log',root/'logs/formal-3261.err']:
 if p.exists():
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  with os.fdopen(fd,'rb') as f:b=os.fstat(f.fileno());raw=f.read();a=os.fstat(f.fileno())
  out['files'][str(p)]={'whole_bytes_observed':len(raw),'whole_sha256':hashlib.sha256(raw).hexdigest(),'tail_base64':base64.b64encode(raw[-24000:]).decode(),'outer_marker_count':raw.count(b'OPSD_E1_OUTER_JSON='),'final_marker_count':raw.count(b'BASELINE_FINAL_EVALUATION_JSON='),'unchanged_size':b.st_size==a.st_size==p.stat().st_size}
actor=run/'checkpoints/actor/global_step_200';paths=[actor/n for n in ['config.json','generation_config.json','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json']]
paths += [actor/'training_state'/('rank_'+str(r)+s) for r in range(8) for s in ['.pt','.json']]+[run/'full_state/global_step_200'/n for n in ['driver.pt','driver.json']]+[run/'checkpoints/algorithm/global_step_200/state.json']
out['checkpoint200_presence']={str(p):p.stat().st_size if p.is_file() else None for p in paths}
print(json.dumps(out,sort_keys=True))
