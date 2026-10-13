"""Actual completed outer receipts, published only after original actor RPC and logger."""
import json,pathlib,time,math
from pipeline import atomic

def record_outer(path,step,metrics,checkpoint_root):
 p=pathlib.Path(path)
 old=json.loads(p.read_text()) if p.exists() else {'actual_completed_outer_steps':[],'records':[]}
 if old['actual_completed_outer_steps']!=list(range(1,step)):raise ValueError('Actual outer sequence missing/repeated')
 # These metrics came from the completed original actor RPC, not owner ACKs.
 required=('actor/grad_norm','actor/lr')
 if any(k not in metrics for k in required):raise ValueError('Original actor update evidence absent')
 if any(not math.isfinite(float(metrics[k])) or float(metrics[k])<0 for k in required):raise ValueError('Nonfinite/negative original actor update evidence')
 old['actual_completed_outer_steps'].append(step)
 old['records'].append({'outer_step':step,'completed_epoch':time.time(),'metrics':metrics})
 checkpoint=pathlib.Path(checkpoint_root)/'actor'/('global_step_'+str(step))
 if (checkpoint/'resume-complete.json').is_file():
  old['export_step']=step;old['checkpoint_directory']=str(checkpoint)
 else:
  old.setdefault('export_step',None);old.setdefault('checkpoint_directory',None)
 old['scope']='ORIGINAL_ACTOR_RPC_AND_LOGGER_RETURNED_OUTER_SEQUENCE_NOT_RANK_GEOMETRY_APPROVAL'
 atomic(p,old)
