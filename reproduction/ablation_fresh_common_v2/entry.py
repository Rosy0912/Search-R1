"""Lumia allocation header; commonv2 worker and collection body unchanged."""
import argparse,json,multiprocessing as mp,os,subprocess,time
from pathlib import Path
from reproduction.official_eval_v2.driver import once,atomic
from reproduction.official_eval_v2.protocol import SCIENCE,PROFILE,TOTAL,sha,validate_rows,em,answer,metrics
from reproduction.official_eval_v2.port_recovery import startup_worker
from .contract import verify,require
from .progress import ProgressObserver

def run(planfile):
 planfile=Path(planfile);plan=json.loads(planfile.read_text());rows,root=verify(plan)
 jid=os.environ['SLURM_JOB_ID'];sub=json.loads((root/'submission.json').read_text())
 require(sub['job_id']==jid and sub['plan_sha256']==sha(planfile) and sub['job']==plan['job'],'Allocation binding differs')
 actual=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
 require('JobName='+plan['job']+' ' in actual and 'JobState=RUNNING' in actual and 'gres/gpu=8' in actual and 'mem=600G' in actual and 'UserId=jmzhang(1066)' in actual,'Wrong actual resources')
 gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
 gpu=[v.split(',') for v in gpu_lines]
 require(len(gpu)==8 and len({v[1].strip() for v in gpu})==8 and {int(v[0]) for v in gpu}==set(range(8)) and all(float(v[-1])>=44000 for v in gpu),'Eight compatible48GB+ GPUs required')
 require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'No GPU preemption')
 once(root/'entry-once.json',dict(at=time.time(),job_id=jid,job=plan['job'],plan_sha256=sha(planfile),gpu_inventory=gpu_lines,actual=actual))
 out=root/'output';out.mkdir(exist_ok=False);(out/'plan.json').write_bytes(planfile.read_bytes())
 once(out/'evaluation-config.json',dict(science=SCIENCE,profile=PROFILE,model=plan['model'],data_sha256=plan['data_sha256'],
  eight_gpu_data_parallel=True,runtime_batch_size_per_gpu=128,maximum_model_context=4596,
  hardware_memory_fraction_policy={'below44000MiB':.85,'otherwise':.65},
  official_legacy_metadata_sampling_loss_not_reproduced='Greedy is explicitly enforced on all five calls.'))
 ctx=mp.get_context('spawn');processes=[]
 for rank,record in enumerate(gpu):
  proc=ctx.Process(target=startup_worker,args=(plan,rank,record[1].strip(),float(record[-1]),int(record[0])));proc.start();processes.append(proc)
 observer=ProgressObserver()
 try:
  while any(q.is_alive() for q in processes):
   if any(q.exitcode not in (None,0) for q in processes):raise RuntimeError('A frozen inference worker failed; preserve partial outputs, no automatic retry')
   progress=observer.read(out)
   atomic(out/'progress.json',dict(at=time.time(),completed=sum(q['completed'] for q in progress),total=TOTAL,job=jid))
   time.sleep(5)
  for q in processes:q.join()
  assert all(q.exitcode==0 for q in processes)
 except BaseException:
  for q in processes:
   if q.is_alive():q.terminate()
  for q in processes:q.join(timeout=30)
  raise
 final=[];workers=[]
 for rank in range(8):
  f=out/f'rank-{rank}.predictions.jsonl';w=json.loads((out/f'rank-{rank}.complete.json').read_text())
  assert sha(f)==w['predictions_sha256'];workers.append(w);final.extend(json.loads(v) for v in f.open())
 final.sort(key=lambda q:q['final_index']);validate_rows(final)
 for expected,observed in zip(rows,final):
  assert all(observed[k]==expected[k] for k in ['final_index','question_id','source','gold_answers'])
  assert observed['em']==em(answer(observed['model_response_spans']),observed['gold_answers'])
 with (out/'predictions.jsonl').open('x') as f:
  for r in final:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 assert len({w['gpu_uuid'] for w in workers})==8
 # Model payloads are immutable: recheck all file SHA after all workers exit.
 for n,h in plan['model_files'].items():assert sha(Path(plan['model'])/n)==h,n
 summary=dict(at=time.time(),profile=PROFILE,job_id=jid,count=TOTAL,metrics=metrics(final),workers=workers,
  predictions_sha256=sha(out/'predictions.jsonl'),raw_emitted_tokens=sum(w['raw_emitted_tokens'] for w in workers),
  retained_generated_tokens=sum(w['generated_tokens'] for w in workers),retrieval_calls=sum(w['search_calls'] for w in workers),
  model_files_post_sha_verified=True,model_identity=plan['model_identity'],data_sha256=plan['data_sha256'])
 once(out/'summary.json',summary);atomic(out/'progress.json',dict(at=time.time(),completed=TOTAL,total=TOTAL,job=jid))
 once(out/'entry-exit.json',dict(at=time.time(),job_id=jid,exit_code=0,summary_sha256=sha(out/'summary.json')))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();run(a.plan)
