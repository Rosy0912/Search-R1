#!/usr/bin/env python3
"""Terminal-only hardware evidence seal. Does not sign independent acceptance."""
import argparse,base64,csv,io,hashlib,importlib.util,json,pathlib,re,resource,sys,tarfile,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006')
SOURCE='b7119791b86951727cea36a1881e8a525c932494'
WATCH='e56781fefcd42b3d97f06af824312ac69fd318eb480093a66c4194d1f0fcaf74'
HELPER='2a714082c22675b975eff07dd060b13ca8eb9da9394d9b6f8bea605ffff46960'

def need(ok,msg):
 if not ok:raise ValueError(msg)

def digest(p):
 p=pathlib.Path(p);need(p.is_file()and not p.is_symlink(),'Nonordinary evidence '+str(p))
 before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 after=p.stat();need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Evidence changed '+str(p))
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':after.st_size}

def field(text,key):
 values=re.findall(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+)',text)
 need(len(values)==1,'Missing/duplicate Slurm field '+key);return values[0]

def terminal_cost(raw,job):
 text=raw.decode('utf-8')
 for k,v in [('JobId',str(job)),('JobName','bapo-instruct300-hardware'),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:need(field(text,k)==v,'Wrong terminal '+k)
 value=field(text,'RunTime');m=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)',value);need(m is not None,'Invalid actual RunTime')
 days,hours,mins,seconds=map(int,(m[1]or'0',m[2],m[3],m[4]));need(mins<60 and seconds<60,'Invalid actual RunTime units')
 elapsed=days*86400+hours*3600+mins*60+seconds;need(elapsed>0,'Invalid terminal cost')
 gpu_terminal_allocation(text)
 return elapsed

def tres_items(text,key):
 values=field(text,key).split(',');pairs=[v.split('=',1)for v in values]
 need(all(len(v)==2 for v in pairs)and len({v[0]for v in pairs})==len(pairs),'Duplicate/invalid '+key)
 return dict(pairs)

def gpu_terminal_allocation(text):
 need(field(text,'NumNodes')=='1'and field(text,'NodeList')=='h100-5','Expected exact single-node allocation')
 need(field(text,'TresPerNode')=='gres:gpu:h100:8','Expected actual TresPerNode eight H100 GPUs')
 need(field(text,'JOB_GRES')=='gpu:h100:8','JOB_GRES conflicts with TresPerNode')
 tres=tres_items(text,'TRES');need(tres.get('node')=='1','TRES node count differs')
 for key,value in tres.items():
  if key.startswith('gres/gpu'):need(value=='8','TRES GPU allocation conflicts')
 optional=re.findall(r'(?:^|\s)AllocTRES=([^\s]+)',text)
 need(len(optional)<=1,'Duplicate optional AllocTRES')
 if optional:
  alloc=tres_items(text,'AllocTRES');need(alloc.get('gres/gpu')=='8','Optional AllocTRES GPU allocation conflicts')
  if 'node'in alloc:need(alloc['node']=='1','Optional AllocTRES node conflict')
 return {'gpu_count':8,'node_count':1,'node':'h100-5','gpu_count_source':'ACTUAL_TresPerNode_AND_JOB_GRES','tres_per_node':field(text,'TresPerNode'),'job_gres':field(text,'JOB_GRES')}

def allocated_cost_binding(allocation,gpus,summary,result,job):
 def env(key):
  values=re.findall(r'^'+re.escape(key)+r'=([^\r\n]+)$',allocation,re.M)
  need(len(values)==1,'Missing/duplicate actual allocation environment '+key);return values[0]
 need(env('SLURM_JOB_ID')==str(job)and env('SLURM_JOB_NODELIST')=='h100-5'and env('SLURM_JOB_NUM_NODES')=='1','Actual allocation identity differs')
 visible=env('CUDA_VISIBLE_DEVICES').split(',');need(len(visible)==8 and len(set(visible))==8 and all(re.fullmatch(r'[0-9]+',v)is not None for v in visible),'Expected eight actual visible device indices')
 need(env('SLURM_JOB_GPUS').split(',')==visible and env('SLURM_GPUS_ON_NODE')=='8','Slurm/CUDA visibility differs')
 rows=list(csv.DictReader(io.StringIO(gpus),skipinitialspace=True));mapping={}
 for row in rows:
  index=row['index'].strip();uuid=row['uuid'].strip()
  need(index not in mapping and uuid.startswith('GPU-'),'Duplicate/invalid physical GPU map');mapping[index]=uuid
 uuids=[mapping.get(index)for index in visible];need(all(isinstance(v,str)for v in uuids)and len(set(uuids))==8,'Physical visible UUID map incomplete')
 need(summary['allocated_uuids']==sorted(uuids)and result['allocated_uuids']==sorted(uuids),'Actual journal/fixture UUID leases differ from visible allocation')
 return {'gpu_count':8,'actual_cuda_visible_devices':visible,'actual_allocated_uuids':sorted(uuids),'binding_source':'ORIGINAL_slurm-allocation.txt_AND_gpus-before.csv_AND_actual_guard_replay_AND_fixture_inherited_UUID_leases'}

def verify_head(proof,terminal,job):
 need(type(proof)is dict and set(proof)=={'files'},'Invalid original head proof envelope')
 d=ROOT/'scheduler-head';names=('hardware-job-id.txt','hardware-submit-intent.txt','submit.sh','hardware.sbatch','approved-launch-hashes.json',f'hardware-{job}-terminal.txt')
 need(set(proof['files'])==set(names),'Incomplete/extraneous original head ledger')
 decoded={}
 for name,x in proof['files'].items():
  need(x['path']==str(d/name),'Head original path differs')
  raw=base64.b64decode(x['base64'],validate=True)
  need(type(x['bytes'])is int and len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256'],'Head original byte hash differs')
  decoded[name]=raw
 need(decoded['hardware-job-id.txt'].decode().strip()==str(job),'Actual head job ID differs')
 need(decoded['hardware-submit-intent.txt'].decode().strip()=='hardware','Actual head intent differs')
 need(decoded[f'hardware-{job}-terminal.txt']==terminal,'Original watcher terminal differs from stdin')
 approved=json.loads(decoded['approved-launch-hashes.json'])
 for name in ('submit.sh','hardware.sbatch'):
  value=hashlib.sha256(decoded[name]).hexdigest()
  need(approved[name]==value==digest(ROOT/'tools'/name)['sha256'],'Actual head/worker/approved launch hash differs')
 return proof

def create_or_compare(path,raw):
 if path.exists():need(path.is_file()and not path.is_symlink()and path.read_bytes()==raw,'Existing original differs '+str(path))
 else:
  with path.open('xb')as f:f.write(raw)

def run_paths(run):
 out=[]
 for p in run.rglob('*'):
  parts=p.relative_to(run).parts
  # Every fixture file is included, even if its directory happens to use a cache-like name.
  fixture=parts[0]=='freeze-scalability-gpu'
  if not fixture and any(n in ('runtime','cache','tmp','__pycache__')for n in parts):continue
  need(not p.is_symlink(),'Run evidence symlink '+str(p))
  if p.is_file():out.append(p)
 return out

def seal(job,terminal,headproof):
 start=time.monotonic();reserved=terminal_cost(terminal,job);run=ROOT/'runs'/f'bapo-instruct300-hardware-{job}'
 need(run.is_dir()and run.resolve()==run and not run.is_symlink(),'Actual run path differs')
 need((run/'source-commit.txt').read_text().strip()==SOURCE,'Actual source differs')
 need(json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[],'Actual own cleanup incomplete')
 need((run/'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY').is_file(),'Missing actual measurement-success marker')
 proof=verify_head(headproof,terminal,job)
 fixture=run/'freeze-scalability-gpu';result=json.loads((fixture/'result.json').read_text())
 need(result['status']=='PASS_ACTUAL_NCCL8_301CPU_FORCED_OWN_FREEZE_ONLY'and result['job_id']==job and result['source_commit']==SOURCE and result['run_root']==str(run),'Actual NCCL/scalability fixture identity/result differs')
 inv=run/'actual-fullhash-inventory.json';archive=ROOT/'logs'/f'hardware{job}-raw-evidence.tar.gz'
 need(not inv.exists()and not archive.exists(),'Seal output already exists; do not overwrite')
 create_or_compare(run/'slurm-terminal.txt',terminal)
 create_or_compare(run/'head-terminal-ledger-proof.json',(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode())
 helperpath=ROOT/'tools/guard_journal.py';watchpath=ROOT/'tools/resource_watch.py'
 need(digest(helperpath)['sha256']==HELPER and digest(watchpath)['sha256']==WATCH,'Frozen helper/watch differs')
 spec=importlib.util.spec_from_file_location('actual_bound_guard_journal',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 summary=helper.replay(run/'resource-watch.jsonl',job,str(run),SOURCE,watchpath)
 cost_binding=allocated_cost_binding((run/'slurm-allocation.txt').read_text(),(run/'gpus-before.csv').read_text(),summary,result,job)
 cost_source=gpu_terminal_allocation(terminal.decode('utf-8'))
 summarypath=run/'guard-summary.json'
 if summarypath.exists():
  need(not summarypath.is_symlink()and summarypath.is_file()and json.loads(summarypath.read_text())==summary,'Existing actual replay summary differs')
 else:create_or_compare(summarypath,(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode())
 paths=run_paths(run)
 paths.extend(ROOT/n for n in ('source-expected.txt','common-data/manifest.json','assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json'))
 paths.extend(p for p in (ROOT/'tools').iterdir()if p.is_file())
 # Dynamic final approval/report file names: preserve all ordinary root-log files, not directories.
 paths.extend(p for p in (ROOT/'logs').iterdir()if p.is_file()and p!=archive)
 paths.extend(ROOT/'repo'/n for n in ('search_r1/baselines/distillation.py','search_r1/baselines/objectives.py','search_r1/baselines/recovery.py','search_r1/baselines/actor_metrics.py','verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/config/ppo_bapo_e1_300.yaml','verl/workers/sharding_manager/fsdp_vllm.py'))
 paths=sorted(set(paths));items=[];small=[]
 for p in paths:
  need(p==p.resolve()and ROOT in p.parents,'Noncanonical evidence path '+str(p));item=digest(p);items.append(item)
  if item['bytes']<=20*1024*1024 or fixture in p.parents:small.append(p)
 record={'status':'ACTUAL_COMPLETED_HARDWARE_FULL_HASH_INVENTORY_NOT_INDEPENDENT_ACCEPTANCE','job_id':job,'source_commit':SOURCE,'run_root':str(run),'evidence':items,'wall_seconds':time.monotonic()-start,'process_cpu_seconds':time.process_time(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'slurm_reserved_wall_seconds':reserved,'slurm_reserved_gpu_hours':reserved*8/3600,'guard_summary':summary,'large_files_preserved_remotely':len(items)-len(small),'fixture_all_raw_files':len([p for p in paths if fixture in p.parents]),'fixture_archived_files':len([p for p in small if fixture in p.parents]),'terminal_head_ledger_bound':True,'reserved_cost_source':{'runtime_source':'ACTUAL_ORIGINAL_HEAD_TERMINAL_RunTime',**cost_source,**cost_binding}}
 need(record['fixture_all_raw_files']==record['fixture_archived_files']and record['fixture_all_raw_files']>0,'Fixture raw archive incomplete')
 with inv.open('x')as f:json.dump(record,f,indent=2,allow_nan=False)
 with tarfile.open(archive,'x:gz')as t:
  for p in small+[inv]:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
 # Prove archived original small files did not change between full hash and archive reads.
 bypath={x['path']:x for x in items}
 for p in small:need(digest(p)==bypath[str(p)],'Evidence changed while archived '+str(p))
 packed=digest(archive)
 print(json.dumps({'status':record['status'],'archive':packed['path'],'bytes':packed['bytes'],'sha256':packed['sha256'],'files':len(items),'large_files':record['large_files_preserved_remotely'],'fullhash_wall_seconds':record['wall_seconds'],'guard_summary':summary,'fixture_all_raw_archived':True,'slurm_reserved_wall_seconds':reserved}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-id',required=True);p.add_argument('--head-ledger',required=True)
 a=p.parse_args();need(re.fullmatch(r'[1-9][0-9]*',a.job_id)is not None,'Invalid actual job ID')
 terminal=sys.stdin.buffer.read();seal(int(a.job_id),terminal,json.loads(pathlib.Path(a.head_ledger).read_text()))
if __name__=='__main__':main()
