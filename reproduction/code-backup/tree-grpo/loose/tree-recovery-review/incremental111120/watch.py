"""Bounded readonly same-node incremental collector; no scheduler writes."""
import subprocess,time,json,importlib.util,sys,hashlib
from pathlib import Path
folder=Path(__file__).resolve().parent;workspace=folder.parents[2]
spec=importlib.util.spec_from_file_location('owned_readonly_route',workspace/'work/remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
start=time.monotonic();iteration=0;accepted={}
# Existing independent artifacts make a restarted reader idempotent; never rerun accepted heavy payloads.
for step in range(111,121):
 f=folder/f'ROUND{step}_RECEIPT.json'
 if f.exists():accepted[step]=json.loads(f.read_text())
def execute(script,path,step=None):
 env="CUDA_VISIBLE_DEVICES='' "
 if step is not None:env+="INDEPENDENT_REVIEW_STEPS='["+str(step)+"]' "
 command=env+'/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -'
 with script.open('rb') as stream:
  r=subprocess.run(remote.route(command),stdin=stream,capture_output=True,timeout=240)
 path.write_bytes(r.stdout);path.with_suffix('.stderr').write_bytes(r.stderr)
 if r.returncode:raise RuntimeError('Readonly reader failed '+str(script)+' rc='+str(r.returncode))
 return json.loads(r.stdout)
while time.monotonic()-start<7200:
 iteration+=1;d=execute(folder/'discover.py',folder/f'discovery-{iteration:03d}.json');state=d['slurm'].split('JobState=')[1].split()[0]
 print(json.dumps(dict(iteration=iteration,slurm=state,rounds=d['rounds'],accepted=sorted(accepted))),flush=True)
 if state!='RUNNING':break
 for step in range(111,121):
  if step in accepted:continue
  x=d['rounds'][str(step)]
  if not ((x['console_completed'] or step==120) and x['ranks']==8 and x['profiles']==8 and x['ratios']==80 and x['tree']==2 and x['trajectory_manifest']):continue
  r=execute(folder/'verify_rounds.py',folder/f'ROUND{step}_CPU_PROOF.json',step);p=execute(folder/'verify_profiles.py',folder/f'ROUND{step}_PROFILE_PROOF.json',step)
  assert r['status']=='PASS_INCREMENTAL_NEW_ROUNDS_ONLY' and len(r['records'])==1 and r['records'][0]['step']==step and len(p['records'])==8
  receipt=dict(status='PASS_INCREMENTAL_SINGLE_OUTER_ONLY',job_id='3234',code_commit='502e9fafa8e3aba722dbfcb493abc2b9d3ece326',outer_step=step,record=r['records'][0],evidence=r['evidence']+[dict(path=x['path'],sha256=x['sha256']) for x in p['records']],cpu_cost={k:r[k] for k in ['wall_seconds','process_cpu_seconds','peak_rss_kib','torch_threads','cuda_initialized']},proof_sha256=hashlib.sha256((folder/f'ROUND{step}_CPU_PROOF.json').read_bytes()).hexdigest(),profile_proof_sha256=hashlib.sha256((folder/f'ROUND{step}_PROFILE_PROOF.json').read_bytes()).hexdigest(),periodic120_accepted=False,full_checkpoint120_accepted=False,final300_accepted=False,limits='Effective input finiteness and existing preclip norms/8192 fixed sample coordinates; not all live parameter elements. Logratio tails explicitly recorded. No final300 or executor bitwise resume claim.')
  target=folder/f'ROUND{step}_RECEIPT.json'
  with target.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
  accepted[step]=receipt;print(json.dumps(dict(outer_accepted=step,receipt_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),record=r['records'][0])),flush=True)
 if len(accepted)==10:break
 time.sleep(60)
print(json.dumps(dict(status='BOUNDED_READONLY_WATCH_FINISHED',accepted=sorted(accepted),elapsed_seconds=time.monotonic()-start)),flush=True)
