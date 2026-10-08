"""Independent three-worker fixed-weight diagnostic controller, GPU4–6 only."""
import fcntl,hashlib,json,os,subprocess,time,urllib.request,uuid
from pathlib import Path
BASE=Path('/data/shaoshitong/search-e1-opsd-debug-20261008')
ORIGINAL=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
RETRIEVER=Path('/data/shaoshitong/search-e1-retriever-resident-20261008')
UUIDS=['GPU-1b18ee4f-7486-4183-bf3e-3290d8133db6','GPU-0f337443-2d5c-cac7-51e9-1f628f90a2bf','GPU-2d1c2f87-07d3-0c7a-ef9c-20f7ee656e45']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def rows(kind,fields):
 raw=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True)
 return [line.split(', ') for line in raw.splitlines() if line.strip()]
def health():
 with urllib.request.urlopen('http://10.120.17.101:18005/health',timeout=30) as r:return json.load(r)
def main():
 assert os.getuid()==1005
 lock=(BASE/'experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((ORIGINAL/'receipts/user-stop-eight-evaluation-for-debug-20261008.json').read_text())['status']=='STOPPED_BY_LATEST_HUMAN_FOR_DEBUG'
 m=json.loads((BASE/'receipts/gpu-source-manifest.json').read_text())
 for relative,h in m.items():assert sha(BASE/'source/opsd_debug'/relative)==h,relative
 admission_control=json.loads((RETRIEVER/'receipts/admit-requests.json').read_text())
 gate_path=Path(admission_control['full_gpu_http_gate_path']);gate=json.loads(gate_path.read_text())
 assert gate['accepted'] is True and sha(gate_path)==admission_control['full_gpu_http_gate_sha256']
 live=health();assert live['ready'] is True
 for k in ['pid','uid','starttick','source_identity_sha256']:assert live[k]==admission_control[k] and live[k]==gate['candidate_health'][k]
 available={int(x[0]):x for x in rows('gpu','index,uuid,memory.free')}
 assert [available[g][1] for g in [4,5,6]]==UUIDS
 assert all(int(available[g][2])>=20000 for g in [4,5,6])
 assert not any(x[0] in UUIDS for x in rows('compute-apps','gpu_uuid,pid,used_gpu_memory'))
 import shutil;assert shutil.disk_usage(BASE).free >= 100*(1<<30)
 cases=json.loads((BASE/'cases/manifest.json').read_text());assert cases['selected_cases']==192
 allcases=[json.loads(l) for l in (BASE/'cases/cases.jsonl').read_text().splitlines()]
 fullcases=[]
 for d in ['nq','hotpotqa']:
  for group in ['correct_correct','correct_wrong','all5_wrong']:
   groupcases=[c for c in allcases if c['dataset']==d and c['group']==group]
   fullcases.append(min(groupcases,key=lambda c:sum(len(x['row']['response']) for x in c['original_siblings']) if c['original_pair'] is None else len(c['original_pair']['row']['student']))['case_id'])
 run_id=uuid.uuid4().hex;output=BASE/'runs'/('debug-'+run_id);output.mkdir()
 admitted=dict(accepted=True,retriever_url='http://10.120.17.101:18005/retrieve',**{k:live[k] for k in ['pid','uid','starttick','source_identity_sha256']},full_gpu_gate_path=str(gate_path),full_gpu_gate_sha256=sha(gate_path),verified_unix=time.time())
 admission_path=output/'retriever-admission.json';write(admission_path,admitted)
 versions={'source100':'best_adapter','update1':'adapter_update_1','update5':'adapter_update_5','epoch1':'adapter_epoch_1'}
 argv=[str(ORIGINAL/'env/bin/python'),str(BASE/'source/opsd_debug/worker.py'),'--runtime',str(ORIGINAL/'runtime/opsd-preparation3-xformers'),'--cases-dir',str(BASE/'cases'),'--model',str(ORIGINAL/'assets/archive100'),'--weight-manifest',str(BASE/'receipts/weight-manifest.json'),'--retriever-admission',str(admission_path),'--output',str(output),'--workers','3','--full-vocab-versions','source100,epoch1','--full-vocab-max-bytes',str(24*(1<<30))]
 for name,d in versions.items():argv+=['--adapter-'+name,str(ORIGINAL/'runs/opsd'/d)]
 for case in fullcases:argv+=['--full-vocab-case-id',case]
 config=dict(run_id=run_id,output=str(output),controller_pid=os.getpid(),controller_uid=os.getuid(),source_manifest=m,retriever=admitted,physical_gpus=[4,5,6],gpu_uuids=UUIDS,cases_sha256=cases['cases']['sha256'],weight_manifest_sha256=sha(BASE/'receipts/weight-manifest.json'),full_vocab_cases=fullcases,full_vocab_global_cap_bytes=24*(1<<30),worker_argv_common=argv,science='Fixed source100/update1/update5/epoch1, no updates; greedy HF debug, diagnostic strata not benchmark',started_unix=time.time())
 write(output/'config.json',config);children=[]
 for rank,gpu in enumerate([4,5,6]):
  env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(gpu),E1_RUN_ID=run_id,PYTHONUNBUFFERED='1',PYTHONPATH=str(BASE/'source/opsd_debug')+':'+str(ORIGINAL/'runtime/opsd-preparation3-xformers')+':'+str(ORIGINAL/'runtime/opsd-preparation3-xformers/upstream/Search-R1'),TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  child_argv=argv+['--worker-index',str(rank)];log=BASE/'logs'/('debug-'+run_id+'-worker-'+str(rank)+'.log');handle=log.open('x')
  child=subprocess.Popen(child_argv,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True);handle.close()
  tick=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19])
  children.append(dict(process=child,pid=child.pid,uid=os.getuid(),starttick=tick,physical_gpu=gpu,gpu_uuid=UUIDS[rank],argv=child_argv,log=str(log)))
 public=[{k:v for k,v in c.items() if k!='process'} for c in children]
 pointer=dict(status='RUNNING',run_id=run_id,output=str(output),controller_pid=os.getpid(),children=public)
 write(BASE/'receipts/current-debug-run.json',pointer);write(output/'children-start.json',public)
 print(json.dumps(pointer),flush=True)
 while True:
  exits={str(c['pid']):c['process'].poll() for c in children}
  if all(code is not None for code in exits.values()):break
  write(BASE/'receipts/current-debug-run.json',dict(pointer,child_exit_codes=exits,updated_unix=time.time()));time.sleep(10)
 status='WORKERS_COMPLETE' if all(code==0 for code in exits.values()) else 'WORKER_FAILED_REQUIRES_RECOVERY'
 final=dict(pointer,status=status,child_exit_codes=exits,ended_unix=time.time(),no_process_signals=True)
 write(BASE/'receipts/current-debug-run.json',final);write(output/'controller-final.json',final)
 print(json.dumps(final),flush=True)
if __name__=='__main__':main()
