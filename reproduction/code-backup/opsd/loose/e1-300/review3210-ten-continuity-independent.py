"""Read actual own-driver snapshots; never manufactures rows or signs/releases."""
from pathlib import Path
import argparse,json,hashlib,importlib.util,math,base64
W=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def need(v,message):
 if not v:raise ValueError(message)
def run(snapshot,require10):
 gatepath=W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py';need(H(gatepath)=='74c3c13929c4f5cb784b90b7fd568ee80dfe1d37c8efaa28f97aae60b3492632','Frozen gate changed');sp=importlib.util.spec_from_file_location('continuity_gate',gatepath);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
 meta=g.load_json(snapshot/'observed-original.json');record=meta['files']['driver-direct.out'];raw=(snapshot/'driver-direct.out').read_bytes();need(H(snapshot/'driver-direct.out')==record['sha256'] and len(raw)==record['bytes'] and raw==base64.b64decode(record['base64']),'Original FD snapshot differs')
 need(meta['job_id']=='3210' and meta['actual_driver_start_ticks']=='1819179500' and record['fd_source']=='/proc/591546/fd/1' and record['physical_path'].startswith('/tmp/o4e-3210/ray/session_') and Path(record['physical_path']).name=='worker-cf21a80fc2417a39395bc3d60f1aa38ce3a118a2cd40cf1eb7b469de-02000000-591546.out','Own original stdout identity differs')
 need(all(v in meta['slurm'] for v in ['JobId=3210','JobState=RUNNING','Restarts=0','NodeList=h100-10','TresPerNode=gres:gpu:h100:8']),'Allocation changed')
 M=W/'actual3210-cpu-runner-v5/actual-small-20261004T174500Z/032-manifest.json';need(H(M)=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b','Actual accepted checkpoint changed');m=g.load_json(M);workers=set()
 for rank in range(8):
  p=M.parent/f'{rank+6:03}-rank_{rank}.json';need(H(p)==m['files'][f'actor/training_state/rank_{rank}.json']['sha256'],'Pinned rank report changed');workers.add(g.load_json(p)['pid'])
 R=W/'formal3210-release-log-independent-20261004T181009Z/2-acceptance-5-receipt.json';need(H(R)=='25b41a15d3e70958b6335f43350ece1c63472478f5d23d9a8c6f78b1a4af433f','Actual receipt changed');receipt=g.load_json(R);key=W/'review-signing/public.json';need(H(key)=='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c','Original public key changed');g.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],g.load_json(key));need(receipt['payload']==g.load_json(W/'actual3210-unsigned-actual-small/unsigned-step5-payload.json'),'Signed payload differs from reviewed actual unsigned content')
 for field in ['driver_pid','job_id','source_commit','config_sha256','pause_nonce']:need(receipt['payload'][field]==m[field],'Signed identity differs')
 need(receipt['payload']['checkpoint_manifest_sha256']==H(M) and receipt['payload']['checkpoint_files']==m['files'],'Signed checkpoint differs')
 rows=[];release=[];pause=[];positions={}
 for pos,line in enumerate(raw.decode().splitlines()):
  for marker,dest in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_RELEASED_JSON=',release),('OPSD_E1_STEP5_PAUSED_JSON=',pause)]:
   if marker in line:
    v=g.parse_log_marker(line,marker,driver_pid=m['driver_pid'],job_id='3210',worker_pids=workers,checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py');dest.append(v)
    if marker=='OPSD_E1_OUTER_JSON=':positions[v['actual_outer_updates']]=pos
    else:positions[marker]=pos
 need(1<=len(rows)<=300 and [v['actual_outer_updates'] for v in rows]==list(range(1,len(rows)+1)),'Missing/duplicated original actual outer')
 need(len(release)==len(pause)==1 and release[0]['receipt_sha256']==H(R) and release[0]['actual_outer_updates']==5 and release[0]['next_global_step']==6 and pause[0]['manifest_sha256']==H(M) and pause[0]['pause_nonce']==m['pause_nonce'],'Actual unique release/pause differs')
 need(positions[5]<positions['OPSD_E1_STEP5_PAUSED_JSON=']<positions['OPSD_E1_STEP5_RELEASED_JSON='] and (len(rows)<6 or positions['OPSD_E1_STEP5_RELEASED_JSON=']<positions[6]),'Release order is not actual5->6')
 compact=[]
 for step,row in enumerate(rows,1):
  need(row['global_step']==step and all(row[k]==m[k] for k in ['driver_pid','job_id','source_commit','config_sha256']),'Actual source/driver/job/config changed');metrics=row['metrics'];g.validate_outer(metrics);need(math.isclose(metrics['actor/lr'],1e-6*min(step,57)/57,rel_tol=1e-14),'Warmup LR differs')
  for rank in range(8):need(metrics[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}']==step,'Teacher actual clock differs')
  compact.append(dict(step=step,per_rank_optimizer_steps=[metrics[f'actor/optimizer_steps/rank_{i}'] for i in range(8)],teacher_cuda_entries=[metrics[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{i}'] for i in range(8)],lr=metrics['actor/lr'],loss=metrics['actor/opsd_loss'],reward=metrics['critic/score/mean'],finish=metrics['env/finish_ratio']))
 first=g.load_json(W/'ACTUAL3210_FIRST5_LOG_REVIEW.json');need([row['metrics']['actor/opsd_loss'] for row in rows[:5]]==[row['opsd_loss'] for row in first['outer_rows']],'First5 original log/directed evidence changed')
 original_first=[]
 for line in (W/'actual3210-cpu-runner-v5/actual-small-20261004T174500Z/034-train.log').read_text().splitlines():
  if 'OPSD_E1_OUTER_JSON=' in line:original_first.append(g.parse_log_marker(line,'OPSD_E1_OUTER_JSON=',driver_pid=m['driver_pid'],job_id='3210',worker_pids=workers,checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'))
 need(len(original_first)==5 and rows[:5]==original_first,'Full original accepted first5 differs from direct records')
 # Audit aggregate separately. Missing Ray rows are retained, never injected.
 aggregate=[]
 for line in (snapshot/'train.log').read_text().splitlines():
  if 'OPSD_E1_OUTER_JSON=' in line:
   v=g.parse_log_marker(line,'OPSD_E1_OUTER_JSON=',driver_pid=m['driver_pid'],job_id='3210',worker_pids=workers,checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py');need(1<=v['actual_outer_updates']<=len(rows) and v==rows[v['actual_outer_updates']-1],'Aggregate/direct conflict');aggregate.append(v['actual_outer_updates'])
 need(len(set(aggregate))==len(aggregate),'Aggregate duplicate records')
 if require10:need(len(rows)>=10,'Actual10 is not yet complete')
 return dict(scope='ORIGINAL_DRIVER_STDOUT_CONTINUITY_PROGRESS_NOT_FINAL300_OR_HF10',source_commit=m['source_commit'],job_id='3210',driver_pid=m['driver_pid'],driver_start_ticks='1819179500',config_sha256=m['config_sha256'],manifest_sha256=H(M),signed_actual_release_receipt_sha256=H(R),original_direct_stdout_sha256=H(snapshot/'driver-direct.out'),original_aggregate_sha256=H(snapshot/'train.log'),actual_completed_outers=len(rows),actual10_logs_complete=len(rows)>=10,aggregate_observed_steps=aggregate,direct_actual_steps=list(range(1,len(rows)+1)),release=release[0],rows=compact,pending='Final10 additionally requires fresh8PID/startUUID/resources/localretriever/sourceclean audit; no storedHF10/Adam200/cursor5120/fullnewprocessRLresume/effect claim',signature_created=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--snapshot',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--require10',action='store_true');x=a.parse_args();result=run(x.snapshot,x.require10)
 with x.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'actual_completed_outers':result['actual_completed_outers'],'actual10_logs_complete':result['actual10_logs_complete'],'scope':result['scope']}))
