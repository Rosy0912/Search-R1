"""Read-only foreground monitor; exit on failure or both formal runs reach5."""
import json,subprocess,time,re
from pathlib import Path
nodes=[('hkust4','/weka_3/shaoshitong/search-e1-config-study-20261004','archive','smoke-035'),('hkust1','/data/shaoshitong/search-e1-config-study-20261004','r1','smoke-025')]
for poll in range(1440):
 reports=[]
 for host,root,profile,gate in nodes:
  script="""import json,re,time
from pathlib import Path
base=Path(ROOT)
phase=(base/'receipts/pipeline-stage.txt').read_text().strip()
formal=base/'runs'/PROFILE/'round-1-grpo/train.log'
log=formal if formal.exists() else base/'runs'/PROFILE/GATE/'train.log'
text=log.read_text(errors='replace') if log.exists() else ''
text=re.sub(r'\\x1b\\[[0-9;]*m','',text)
steps=[int(x) for x in re.findall(r'\\bstep:(\\d+) -',text)]
active='round-1-grpo' if formal.exists() else GATE
receipt=base/'receipts'/(PROFILE+'-'+active+'.json')
record=json.loads(receipt.read_text()) if receipt.exists() else {}
priority=base/'logs/formal-priority.log'
priority_text=priority.read_text(errors='replace') if priority.exists() else ''
errors=bool(re.search(r'FloatingPointError|CUDA out of memory|RuntimeError:|AssertionError|Traceback',priority_text))
nonfinite=bool(re.search(r'actor/(?:pg_loss|grad_norm|kl_loss)\\s*[:=]\\s*(?:nan|inf)',text,re.I))
print(json.dumps(dict(profile=PROFILE,phase=phase,formal=formal.exists(),iteration=max(steps,default=0),status=record.get('status'),run_id=record.get('run_id'),peak_mib=max(record.get('peaks_mib',{}).values(),default=0),heartbeat_age=round(time.time()-record.get('updated_unix',time.time())),priority_error=errors,nonfinite=nonfinite,last_line=text.splitlines()[-1][-160:] if text else '',priority_tail=priority_text.splitlines()[-1][-160:] if priority_text else '')))
""".replace('ROOT',repr(root)).replace('PROFILE',repr(profile)).replace('GATE',repr(gate))
  for attempt in range(6):
   try:
    completed=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',host,root+'/env/bin/python -'],input=script,text=True,capture_output=True,timeout=35)
    if completed.returncode:raise RuntimeError(host+': '+completed.stderr[-500:])
    report=json.loads(completed.stdout);break
   except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as error:
    print('READ_ONLY_MONITOR_CONNECTION_RETRY',host,attempt+1,str(error)[-160:],flush=True)
    if attempt==5:
     report=dict(profile=profile,formal=False,iteration=0,status='CONNECTION_UNKNOWN',heartbeat_age=999999,priority_error=False,nonfinite=False,peak_mib=0,connection_error=str(error)[-500:]);break
    time.sleep(5)
  reports.append(report)
 print(json.dumps(dict(check=poll,reports=reports),ensure_ascii=False),flush=True)
 if any(x['priority_error'] or x['nonfinite'] or x['status'] in ['FAILED','FAILED_EXCEPTION'] for x in reports):raise SystemExit('INTERVENTION_REQUIRED')
 if all(x['formal'] and x['iteration']>=5 and x['status']=='RUNNING' and x['heartbeat_age']<120 for x in reports):
  Path('outputs/search-e1-formal-five-iterations.json').write_text(json.dumps(reports,indent=2));print('BOTH_FORMAL_FIVE_ITERATIONS_ACCEPTED',flush=True);break
 time.sleep(45)
else:raise RuntimeError('Monitor deadline reached; acceptance incomplete')
