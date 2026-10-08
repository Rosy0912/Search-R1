import json,sys,pathlib,threading,types
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import window_guardian as w

def test_actual_constructor_fail_closed_without_native_review(tmp_path):
 owned=types.SimpleNamespace(cleanup=lambda:None)
 with pytest.raises(ValueError,match='pending'):w.Guardian(tmp_path,{'runtime_sha256':w.RUNTIME,'guard_sha256':w.GUARD},tmp_path,None,owned)

def test_actual_start_sampling_failure_cancels_only_supplied_owned_scope(tmp_path):
 g=object.__new__(w.Guardian);g.stop=threading.Event();g.errors=[];g.thread=None;events=[]
 g.owned=types.SimpleNamespace(cleanup=lambda:events.append('OWN_ONLY_CLEANUP'))
 def sample():raise RuntimeError('Foreign aggregate SM exceeds15% or unknown')
 g.sample=sample;g.start();g.thread.join(2)
 assert not g.thread.is_alive() and events==['OWN_ONLY_CLEANUP'] and g.stop.is_set()
 with pytest.raises(RuntimeError,match='SM exceeds15'):g.check()

def test_actual_control_unknown_operation_rejected_before_identity_or_gpu(tmp_path):
 control=tmp_path/'phase-control';control.mkdir();(control/'bad.request.json').write_text(json.dumps(dict(operation='ASSUME_READY',phase='A',outer_step=0)))
 g=types.SimpleNamespace();tools=types.SimpleNamespace(process_identity=lambda p:pytest.fail('Unknown request reached identity'))
 with pytest.raises(ValueError,match='Unknown supervised stage'):w.poll_phase_controls(g,{},tmp_path,tools)
 assert not list(control.glob('*.ack.json'))

def test_actual_control_profile_or_training_step_rejected_in_initial_scope(tmp_path):
 control=tmp_path/'phase-control';control.mkdir();(control/'bad.request.json').write_text(json.dumps(dict(operation='BEGIN',phase='A',outer_step=25)))
 with pytest.raises(ValueError,match='Unknown supervised stage'):w.poll_phase_controls(object(),{},tmp_path,object())

def test_guardian_finish_rechecks_error_after_real_thread_join():
 g=object.__new__(w.Guardian);g.errors=[];g.stop=threading.Event();g.runtime=None
 def tail():g.stop.wait();g.errors.append('TAIL_FOREIGN_FAILURE')
 g.thread=threading.Thread(target=tail);g.thread.start()
 with pytest.raises(RuntimeError,match='TAIL_FOREIGN_FAILURE'):g.finish()
 assert not g.thread.is_alive()
