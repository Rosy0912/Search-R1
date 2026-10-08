import subprocess,time,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'work'));import remote
out=Path.cwd()/'work/tree-fixed-formal-3112-initial-review';start=time.monotonic()
command="python3 - <<'REMOTE'\nimport json,subprocess,pathlib,time,re\np=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fixed-formal-3112');s=subprocess.run(['scontrol','show','job','3112'],capture_output=True,text=True);t=(p/'stage.log').read_text() if (p/'stage.log').exists() else '';print(json.dumps({'epoch':time.time(),'slurm_exit':s.returncode,'slurm':s.stdout+s.stderr,'step_lines':[x for x in t.splitlines() if re.search(r'\\bstep:\\d+ - ',x)],'install':[x for x in t.splitlines() if 'TREE_VALIDATION_COST_INSTALLED_JSON=' in x],'rank_count':len(list((p/'output/tree-rank-audit').glob('step-*-rank-*.json'))),'profile_count':len(list((p/'output/tree-rank-audit/memory-profile').glob('*.json'))),'tree_count':len(list((p/'output/tree-audit').glob('*.json'))),'result':json.loads((p/'run-result.json').read_text()) if (p/'run-result.json').exists() else None}))\nREMOTE"
i=0
while time.monotonic()-start<45*60:
 i+=1;p=subprocess.run(remote.route(command),capture_output=True,text=True,timeout=100);(out/f'ACTUAL3112_WATCH_{i:02d}.json').write_text(p.stdout or json.dumps({'error':p.stderr,'exit':p.returncode}))
 try:
  j=json.loads(p.stdout);print(json.dumps({'iteration':i,'rank':j['rank_count'],'profile':j['profile_count'],'tree':j['tree_count'],'completed_metric_steps':len(j['step_lines']),'observer_installed':bool(j['install']),'result':j['result']['status'] if j['result'] else None}),flush=True)
  if len(j['step_lines'])>=2 or j['result'] or 'JobState=FAILED' in j['slurm'] or 'JobState=CANCELLED' in j['slurm'] or 'JobState=COMPLETED' in j['slurm']:break
 except Exception as e: print(str(e),flush=True)
 time.sleep(120)
