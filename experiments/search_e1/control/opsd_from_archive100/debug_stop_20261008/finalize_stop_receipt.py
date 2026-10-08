"""Finish stop evidence after the /proc exit race; never sends any signal."""
import hashlib,json,os,subprocess,time
from pathlib import Path
BASE=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
PRES=BASE/'preserved/user-stop-eight-evaluation-debug-20261008T034935Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
  return {'pid':pid,'state':s[0],'starttick':int(s[19])}
 except FileNotFoundError:return None
def main():
 assert os.getuid()==1005
 a=json.loads((PRES/'authorization-before-signals.json').read_text())
 assert a['run_id']=='914bfff786c84ede8d82414dff65f60e'
 targets=[a['controller']]+a['owned_members']
 assert {x['pid'] for x in targets}=={2459676,3960546,3960643,3960647,3960648,3960649,3960650,3960651,3960652,3960653}
 terminal={str(x['pid']):identity(x['pid']) for x in targets}
 assert all(v is None or v['state']=='Z' or v['starttick']!=next(x['starttick'] for x in targets if str(x['pid'])==k) for k,v in terminal.items())
 kept=identity(2239529);assert kept and kept['starttick']==914664099 and kept['state']!='Z'
 shards=[]
 for f in sorted((BASE/'runs/preparation-3-final-test.jsonl.parts').glob('rollouts-*.jsonl')):
  with f.open('rb') as stream:rows=sum(line.endswith(b'\n') for line in stream)
  shards.append({'path':str(f),'rows':rows,'bytes':f.stat().st_size,'sha256':sha(f)})
 e={'status':'STOPPED_BY_LATEST_HUMAN_FOR_DEBUG','time':time.time(),'run_id':a['run_id'],'authorization_path':str(PRES/'authorization-before-signals.json'),'authorization_sha256':sha(PRES/'authorization-before-signals.json'),'source_commit':'e820ed6d9811e1001091a42cd0a73579be307923','targets_terminal':terminal,'retriever_kept':kept,'completed_opsd_updates':3156,'partial_rows':sum(x['rows'] for x in shards),'final_seven_partial_shards':shards,'partial_is_not_complete_final_evaluation':True,'initial_stop_evidence_failure':'After initial exact SIGTERM actions, exited /proc environ became inaccessible; all original targets independently confirmed absent. This finalizer sends no signals.','compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines(),'actual_memory_free':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
 with (BASE/'receipts/user-stop-eight-evaluation-for-debug-20261008.json').open('x') as f:json.dump(e,f,indent=2)
 t=BASE/'receipts/pipeline-stage.user-stop.tmp';t.write_text(json.dumps({'phase':'STOPPED_BY_USER_FOR_GPU_DEBUG','run_id':a['run_id'],'time':time.time(),'receipt':str(BASE/'receipts/user-stop-eight-evaluation-for-debug-20261008.json'),'scope':'Latest human stop of old K1 evaluation only; independent diagnostics continue. No automatic baseline/raw seven restart.'},indent=2));t.replace(BASE/'receipts/pipeline-stage.json')
 print(json.dumps({'status':e['status'],'partial_rows':e['partial_rows'],'targets_absent':all(x is None for x in terminal.values()),'retriever_kept':kept}))
if __name__=='__main__':main()
