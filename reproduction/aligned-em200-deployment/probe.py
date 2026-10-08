from pathlib import Path
import json,subprocess,datetime,os,shutil
v={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename}
for label,args in [('slurm',['scontrol','show','job','3315','-o']),('node',['scontrol','show','node','h100-5','-o']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);v[label]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for path in ['/dataset/shaoshitong/search-r1-rl','/mnt/shared','/mnt/shared/workspace','/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data']:
 p=Path(path);r=subprocess.run(['findmnt','-T',path,'-J'],capture_output=True,text=True,timeout=10);v[path]={'exists':p.exists(),'realpath':str(p.resolve()),'findmnt':r.stdout,'findmnt_exit':r.returncode,'writable':os.access(p,os.W_OK),'disk':shutil.disk_usage(p)._asdict() if p.exists() else None}
 if p.is_dir() and path.endswith('common-data'):v[path]['files']=[x.name for x in p.iterdir()]
print(json.dumps(v))
