"""Whole-job finally always attempts owned cleanup, preserves primary/raw failures."""
import json,os,pathlib,sys

def finalize_owned_pipeline(*,tools,owned,scope,baseline,gpu_original,baseline_gpu_missing,roles,observed,guardian,sampler_stop,monitor,sampler_errors,terminal_drained,terminal_connection,listener,root,primary_error):
 errors=[];error=primary_error
 if error:errors.append({'stage':'original-runtime','error_type':type(error).__name__,'error':str(error)})
 def failed(stage,exc):
  nonlocal error
  errors.append({'stage':stage,'error_type':type(exc).__name__,'error':str(exc)})
  if error is None:error=exc
 def capture(stage,call):
  try:return call()
  except BaseException as exc:failed(stage,exc);return None
 for name,obj in [('terminal-connection',terminal_connection),('listener',listener)]:
  if obj:capture(name+'-close',obj.close)
 if not terminal_drained:capture('guardian-finish',guardian.finish)
 capture('guardian-close',guardian.close)
 capture('sampler-stop',sampler_stop.set);capture('sampler-join',lambda:monitor.join(45))
 alive=capture('sampler-alive',monitor.is_alive)
 if alive is not False or sampler_errors:failed('sampler-final-state',RuntimeError('Owned sampler not cleanly drained: '+str(sampler_errors)))
 current=capture('precleanup-actual-census',lambda:tools.cgroup_pids(scope)-owned.baseline)
 if current is not None:
  for pid in current:
   identity=capture('precleanup-identity-'+str(pid),lambda pid=pid:tools.process_identity(pid))
   observed[pid]=identity if identity is not None else {'pid':pid,'identity_unavailable':True}
 # This operation is unconditional even if ALL preceding diagnostics failed.
 capture('actual-owned-cleanup',owned.cleanup)
 remaining=capture('postcleanup-actual-census',lambda:sorted(tools.cgroup_pids(scope)-owned.baseline))
 missing=[]
 for pid,r in baseline.items():
  actual=capture('baseline-identity-'+str(pid),lambda pid=pid:tools.process_identity(pid))
  if actual is None or actual.get('starttime_ticks')!=r.get('starttime_ticks'):missing.append({'pid':pid,'expected':r,'actual':actual,'identity_unavailable':actual is None})
 gpu=capture('postcleanup-actual-gpu-query',tools.gpu_sample)
 gmissing=capture('original-global-gpu-baseline',lambda:baseline_gpu_missing(tools,gpu,gpu_original)) if gpu is not None else None
 own_gpu=capture('actual-owned-gpu-rows',lambda:[line for line in gpu['processes'] if int(line.split(',')[1]) in observed]) if gpu is not None else None
 parent_gpu=capture('actual-parent-gpu-rows',lambda:[line for line in gpu['processes'] if int(line.split(',')[1])==os.getpid()]) if gpu is not None else None
 history_path=pathlib.Path(root)/'owned-cleanup-history.jsonl'
 history=capture('original-cleanup-history',lambda:[json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else [])
 cuda=capture('parent-cuda-context',lambda:bool(sys.modules['torch'].cuda.is_initialized()) if 'torch' in sys.modules else False)
 guardian_errors=capture('guardian-errors',lambda:list(guardian.errors))
 if any(value!=[] for value in (remaining,missing,gmissing,own_gpu,parent_gpu)) or cuda is not False or guardian_errors!=[]:failed('actual-final-cleanup-gate',RuntimeError('Actual owned cleanup/baseline evidence incomplete'))
 record={'scope':'ACTUAL_OWNED_PIPELINE_EXIT_UNREVIEWED','roles':roles,'identities':observed,'own_processes_remaining':remaining,'baseline_identity_missing':missing,'original_gpu_baseline_missing':gmissing,'own_gpu_records_remaining':own_gpu,'cleanup':getattr(owned,'cleanup_record',None),'prior_owned_cleanup_history':history,'parent_torch_imported':'torch' in sys.modules,'parent_cuda_initialized':cuda,'parent_gpu_records':parent_gpu,'sampler_errors':sampler_errors,'guardian_errors':guardian_errors,'terminal_monitors_drained_before_authorized_exit':terminal_drained,'error':str(error) if error else None,'primary_error':{'error_type':type(error).__name__,'error':str(error)} if error else None,'failure_history':errors,'gpu_after':gpu}
 return error,record
