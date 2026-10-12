import subprocess,json,datetime
v=dict(scope='EXISTING_USER_AUTHORIZED_OLD3315_STOP_SCOPED_3315_0_ONLY_NO_DAEMON_OR_FOREIGN_OPERATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization=dict(instruction='停掉现在的训练吧',recorded_at_utc='2026-10-08T18:07:39.855509+00:00',target_job=3315,scope='original3315only; newalignedgoal remainsACTIVE; do not resume3315',provenance='outputs/STATUS.json latest_human_stop and stop-current3315-user-20261009.actual.jsonl'))
p=subprocess.run(['scontrol','show','job','3315','-o'],capture_output=True,text=True,timeout=15);v['job_before']=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
expected='BAPO_FRESH200:formal200:60f566b3ddabc88ce9de56ead5ad30a3d6795dd2:bab8981d9fd14afc98923eb4ebb89e33'
if p.returncode!=0 or expected not in p.stdout or 'JobState=COMPLETING' not in p.stdout:raise RuntimeError('own original completing3315 required')
p=subprocess.run(['scontrol','show','step','3315.0'],capture_output=True,text=True,timeout=15);v['step_before']=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
if p.returncode!=0 or 'StepId=3315.0 ' not in p.stdout or 'NodeList=h100-5' not in p.stdout:raise RuntimeError('own original3315.0 required')
p=subprocess.run(['scancel','--signal=KILL','3315.0'],capture_output=True,text=True,timeout=20);v['scoped_step_cancel']=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(v))
