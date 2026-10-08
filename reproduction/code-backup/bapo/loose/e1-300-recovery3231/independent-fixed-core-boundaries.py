import sys,json,tempfile,importlib.util
from pathlib import Path
from pytest import MonkeyPatch
sys.path.insert(0,str(Path('work/bapo-e1-300-recovery').resolve()))
sys.path.insert(0,str(Path('work/bapo-e1-300-recovery/tests/recovery').resolve()))
from search_r1.baselines import recovery as r
from test_recovery_state import native,report
from test_gpu_proof import setup_proof
results=[]
for case in ('dummy','nonjson','state_none','oldpid_bool','empty_groups','empty_lr','fractional_adam'):
 with tempfile.TemporaryDirectory() as tmp,MonkeyPatch.context()as mp:
  try:
   if case in ('empty_groups','empty_lr','fractional_adam'):
    x=native()
    if case=='empty_groups':x['optimizer']['param_groups']=[]
    if case=='empty_lr':x['scheduler']['_last_lr']=[]
    if case=='fractional_adam':x['optimizer']['state'][0]['step'].fill_(500.5)
    r.validate_native(x,report(x),0,8)
   else:
    m,p,c,root,run,ranks,d,bind=setup_proof(Path(tmp).resolve(),mp)
    if case=='dummy':p['gpu_witnesses']=[bind(run/'dummy.json',{'actual':True})]
    elif case=='nonjson':p['gpu_witnesses']=[bind(run/'dummy.txt','not GPU JSON',False)]
    else:
     if case=='state_none':ranks[0]['process']['state']=None
     if case=='oldpid_bool':ranks[0]['old_pid']=True
     d['rank_witnesses']=ranks
     p['gpu_witnesses']=[bind(run/'recovery-witness'/f'rank_{i}.json',w)for i,w in enumerate(ranks)]+[bind(run/'recovery-witness/driver.json',d)]
     raw=p['gpu_validation']['raw_artifacts'];raw['train_log']=bind(run/'train.log','\n'.join('BAPO_RECOVERY_RANK_JSON='+json.dumps(w)for w in ranks)+'\nBAPO_RECOVERY_DRIVER_JSON='+json.dumps(d)+'\n',False)
    r.validate_gpu_proof(p,m,c,root)
  except (ValueError,KeyError,TypeError)as e:results.append(dict(case=case,rejected=True,error=str(e)));continue
  results.append(dict(case=case,rejected=False))
out=dict(scope='INDEPENDENT_LOCAL_SCHEMA_ONLY_NOT_GPU_ACCEPTANCE',all_rejected=all(x['rejected']for x in results),results=results)
Path('work/evidence/recovery3231-independent-fixed-core-boundaries.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['all_rejected']
