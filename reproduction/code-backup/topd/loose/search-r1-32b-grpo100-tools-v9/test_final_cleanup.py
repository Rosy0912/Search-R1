import types,pathlib,pytest
from final_cleanup import finalize_owned_pipeline

def context(tmp_path):
 done=[];tools=types.SimpleNamespace(cgroup_pids=lambda scope:set(),process_identity=lambda pid:{'pid':pid,'starttime_ticks':pid},gpu_sample=lambda:{'processes':[]})
 owned=types.SimpleNamespace(baseline=set(),cleanup=lambda:done.append('cleanup'),cleanup_record={'actual_cgroup_empty':True})
 guard=types.SimpleNamespace(finish=lambda:None,close=lambda:None,errors=[])
 args=dict(tools=tools,owned=owned,scope='scope',baseline={},gpu_original=[],baseline_gpu_missing=lambda *a:[],roles={},observed={},guardian=guard,sampler_stop=types.SimpleNamespace(set=lambda:None),monitor=types.SimpleNamespace(join=lambda n:None,is_alive=lambda:False),sampler_errors=[],terminal_drained=False,terminal_connection=None,listener=None,root=tmp_path,primary_error=None)
 return args,done

def test_success_actual_finalizer_keeps_empty(tmp_path):
 args,done=context(tmp_path);error,record=finalize_owned_pipeline(**args)
 assert done==['cleanup'] and error is None and record['own_processes_remaining']==[] and record['failure_history']==[]

def test_census_failure_still_wholejob_cleanup_and_unknown_not_empty(tmp_path):
 args,done=context(tmp_path)
 def unavailable(scope):raise RuntimeError('actual census unavailable')
 args['tools'].cgroup_pids=unavailable
 error,record=finalize_owned_pipeline(**args)
 assert done==['cleanup'] and str(error)=='actual census unavailable' and record['own_processes_remaining'] is None
 assert record['failure_history'][0]['stage']=='precleanup-actual-census'

def test_secondary_cleanup_and_identity_error_never_masks_primary(tmp_path):
 args,done=context(tmp_path);primary=ValueError('original actor failed');args['primary_error']=primary;args['baseline']={7:{'starttime_ticks':7}}
 def cleanup():done.append('cleanup');raise RuntimeError('secondary cleanup error')
 def identity(pid):raise OSError('secondary identity missing')
 args['owned'].cleanup=cleanup;args['tools'].process_identity=identity
 error,record=finalize_owned_pipeline(**args)
 assert done==['cleanup'] and error is primary and record['primary_error']['error']=='original actor failed'
 assert record['baseline_identity_missing'][0]['identity_unavailable'] and len(record['failure_history'])>=3

def test_precleanup_monitor_error_still_cleanup(tmp_path):
 args,done=context(tmp_path)
 def bad():raise RuntimeError('guardian close failed')
 args['guardian'].close=bad
 error,record=finalize_owned_pipeline(**args)
 assert done==['cleanup'] and str(error)=='guardian close failed' and record['failure_history'][0]['stage']=='guardian-close'
