"""One explicit execution refinement after a completed low-headroom gate."""
import os,json,time,subprocess,shutil
from pathlib import Path
from verify_checkpoint import verify
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];control=base/'source/experiments/search_e1/control';python=str(base/'env/bin/python')
run_id=os.environ['E1_GATE_RUN_ID'];session=os.environ['E1_GATE_SESSION'];continuation=os.environ['E1_OLD_CONTINUATION_SESSION'];stage=os.environ.get('E1_GATE_STAGE','smoke-025')
assert os.environ['E1_LOGIT_CHUNK']=='16'
for _ in range(1440):
 if all(subprocess.run(['tmux','has-session','-t',s],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode for s in [session,continuation]):break
 time.sleep(30)
else:raise RuntimeError('Previous execution still active; no refinement performed')
receipts=base/'receipts';record=json.loads((receipts/f'{profile}-{stage}.json').read_text())
assert record['run_id']==run_id and record['status']=='COMPLETED' and record['exit_code']==0
for key in ['E1_PHYSICAL_MICRO','E1_PHYSICAL_SCORE','E1_JIT_OPTIMIZER']:assert record['execution_caps'].get(key)==os.environ.get(key)
assert record['execution_caps'].get('E1_LOGIT_CHUNK') is None
assert any(record['peaks_mib'][row[0]]>int(row[2])-6144 for row in record['admission']['devices']),'No measured need for this refinement'
assert not (receipts/f'{profile}-memory-tuning.json').exists() and not (receipts/f'{profile}-gpu-gate.accepted').exists()
subprocess.run([python,str(control/'verify_runtime.py')],check=True)
out=base/'runs'/profile/stage
assert 'SMOKE_STOP_AFTER_REAL_UPDATES 2' in (out/'train.log').read_text()
for step in [1,2]:assert (out/'actor'/f'global_step_{step}'/'config.json').is_file()
report=verify(base/'assets/model',out/'actor/global_step_2');assert report['accepted']
proof=(base/'source/experiments/search_e1/docs/receipts/fsdp-statistics-check.log').read_text()
for rank in range(4):assert f'"rank": {rank}, "result": "FSDP_ROW_STATISTICS_EQUIVALENCE_OK"' in proof
saved=base/'preserved'/f'{profile}-completed-before-row-statistics';saved.mkdir(parents=True,exist_ok=False)
(saved/'completed-checkpoint.json').write_text(json.dumps(report,indent=2))
shutil.move(str(base/'runtime'/profile),str(saved/'runtime'))
shutil.move(str(out),str(saved/'run'))
for p in [receipts/f'{profile}-runtime.sha256.json',receipts/f'{profile}-overlay.patch',*receipts.glob(f'{profile}-{stage}*.json')]:
 if p.exists():shutil.move(str(p),str(saved/p.name))
print('PRESERVED_COMPLETED_LOW_HEADROOM_GATE',run_id,flush=True)
subprocess.run([python,str(control/'build_runtime.py'),'--base',str(base),'--profile',profile],check=True)
env=dict(os.environ,E1_REUSE_RUNTIME='1',E1_RESUME_GATES='0',RAY_enable_metrics_collection='0',TZ='UTC')
subprocess.run(['bash',str(control/'pipeline.sh')],env=env,check=True)
