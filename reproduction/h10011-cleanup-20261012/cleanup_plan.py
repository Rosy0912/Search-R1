import pathlib,json,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl')
plan=[]
for root in r.glob('bapo-aligned-cohost-node11-v*-20261009'):
 for d in [root/'cache']+list((root/'runs').glob('*/diagnostics/worker-events'))+list((root/'runs').glob('*/training-em-audit')):
  if d.exists():plan.append(dict(path=str(d),reason='Idle BAPO disposable cache or verbose diagnostics'))
 for run in (root/'runs').glob('*'):
  cp=run/'checkpoints'
  if not cp.exists():continue
  if root.name!='bapo-aligned-cohost-node11-v11-20261009':plan.append(dict(path=str(cp),reason='Completed failed/superseded BAPO experiment checkpoint'))
  else:
   for category in ['actor','algorithm','fresh200_state']:
    for d in (cp/category).glob('*'):
     if d.name in ['global_step_200','outer_200_COMPLETE']:continue
     plan.append(dict(path=str(d),reason='Old checkpoint; best150 remote backup required before cleanup'))
for x in plan:
 d=pathlib.Path(x['path']);x['symlink']=d.is_symlink();x['bytes']=int(subprocess.check_output(['du','-sb',str(d)]).decode().split()[0])
print(json.dumps(dict(epoch=time.time(),scope='BAPO only',required_hf_status='ALL_SELECTED_HF_FILES_SHA_VERIFIED',preserved=['outer200 complete recovery','all source checkouts','necessary logs and metrics','TOPD and all shared models/retrieval'],delete=plan,total_bytes=sum(x['bytes'] for x in plan)),indent=2))
