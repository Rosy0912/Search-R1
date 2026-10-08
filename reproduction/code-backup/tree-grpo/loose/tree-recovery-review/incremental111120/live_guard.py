import json,hashlib,time,subprocess,os
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');p=root/'resource-samples.jsonl';guardraw=p.read_bytes();lines=guardraw.splitlines();samples=[]
for line in lines:
 try:samples.append(json.loads(line))
 except json.JSONDecodeError:assert line==lines[-1]
assert samples and all('event' not in x for x in samples)
assert time.time()-samples[-1]['epoch']<20 and all(x['job_id']=='3234' and len(x['gpus'])==8 for x in samples)
assert all(g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15 for x in samples for g in x['gpus'])
pids=[1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446]
ps=subprocess.check_output(['ps','-p',','.join(map(str,pids)),'-o','pid=,lstart='],text=True)
assert len(ps.strip().splitlines())==8
for line in ps.strip().splitlines():
 parts=line.split();pid=int(parts[0]);assert ' '.join(parts[1:])==('Wed Sep 30 10:23:26 2026' if pid in pids[:4] else 'Wed Sep 30 10:31:12 2026')
stats={}
for pid in pids:
 s=Path(f'/proc/{pid}/stat').read_text();fields=s[s.rfind(')')+2:].split();stats[str(pid)]=dict(state=fields[0],starttime=fields[19]);assert fields[0]!='Z'
slurm=subprocess.check_output(['scontrol','show','job','3234'],text=True)
fields=dict(token.split('=',1) for token in slurm.split() if '=' in token);assert fields['JobId']=='3234' and fields['JobState']=='RUNNING' and fields['Restarts']=='0' and fields['NodeList']==fields['BatchHost']=='h100-9'
log=root/'stage.log';lograw=log.read_bytes();loglines=lograw.decode().splitlines();actual=[s for s in loglines if 'step:' in s and 'training/actual_outer_updates:' in s]
bindings=[]
for step in range(111,121):
 mp=root/f'output/trajectories/outer-{step:06d}/manifest.json';raw=mp.read_bytes();m=json.loads(raw);i=m['identity']
 assert i==dict(outer_step=step,job_id='3234',source='502e9fafa8e3aba722dbfcb493abc2b9d3ece326',config='e54d322ba3635beb8d930c4f8282997f1867b4dbf5a0ca727e54a569bb198f35',data='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24')
 bindings.append(dict(path=str(mp),sha256=hashlib.sha256(raw).hexdigest(),identity=i))
 for rank in range(8):
  r=json.loads((root/f'output/tree-rank-audit/step-{step:06d}-rank-{rank:02d}.json').read_bytes());assert r['data_identity']==i['data'] and r['run_id']=='tree-recovery300-3234'
entry=root/'output/entry-audit/entry.jsonl';entryraw=entry.read_bytes();entryrecords=[json.loads(x) for x in entryraw.splitlines()];bindings.append(dict(path=str(entry),sha256=hashlib.sha256(entryraw).hexdigest(),records=entryrecords))
resolved=root/'resolved.yaml';resolvedraw=resolved.read_bytes();bindings.append(dict(path=str(resolved),sha256=hashlib.sha256(resolvedraw).hexdigest()))
print(json.dumps(dict(bindings=bindings,epoch=time.time(),status='PASS_LIVE_GUARD_PREFIX_ONLY',slurm=slurm,guard_sha256=hashlib.sha256(guardraw).hexdigest(),guard_bytes=len(guardraw),samples=len(samples),first_epoch=samples[0]['epoch'],last_epoch=samples[-1]['epoch'],foreign_memory_max=max(g['foreign_memory_fraction'] for x in samples for g in x['gpus']),foreign_sm_max=max(g['foreign_sm_upper'] for x in samples for g in x['gpus']),protected_ps=ps,protected_proc=stats,stage_log_sha256=hashlib.sha256(lograw).hexdigest(),actual_outer_log_lines=actual,scope='Live prefix resource guard and original protected PID identities; not final cleanup or Slurm terminal.')),flush=True)
