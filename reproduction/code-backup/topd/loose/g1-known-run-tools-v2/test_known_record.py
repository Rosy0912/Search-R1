import json,pathlib,sys,types
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import known_record as k

def test_unknown_is_durable_before_actual_submit_and_reuses_known_job(tmp_path):
 record=tmp_path/'submission.json';calls=[];binding={'canonical':'actual-reviewed-config'}
 def submit(cmd,**kwargs):
  assert json.loads(record.read_text())['status']=='UNKNOWN_BEFORE_SUBMISSION';calls.append(cmd)
  return types.SimpleNamespace(returncode=0,stdout='3123;cluster\n',stderr='')
 actual=k.submit_once(record,binding,['sbatch','entry'],{},submit,lambda:{'physical_gate':'actual-read'})
 assert actual['status']=='SUBMITTED' and actual['job_id']==3123
 assert k.submit_once(record,binding,['sbatch','entry'],{},lambda *a,**kw:pytest.fail('duplicate submit'),lambda:pytest.fail('known receipt recannotadmit'))==actual
 assert len(calls)==1

def test_lost_submit_ack_cannot_start_a_second_job(tmp_path):
 record=tmp_path/'submission.json';binding={'canonical':'reviewed'}
 def submit(*args,**kwargs):raise OSError('transport lost after possible acceptance')
 with pytest.raises(OSError):k.submit_once(record,binding,['sbatch'],{},submit,lambda:{})
 assert json.loads(record.read_text())['status']=='UNKNOWN_BEFORE_SUBMISSION'
 with pytest.raises(RuntimeError,match='no retry'):k.submit_once(record,binding,['sbatch'],{},lambda *a,**kw:pytest.fail('unsafe resubmit'),lambda:{})

def test_non_numeric_ack_stays_unknown_and_binding_change_refused(tmp_path):
 record=tmp_path/'submission.json';binding={'canonical':'reviewed'}
 with pytest.raises(RuntimeError,match='UNKNOWN'):k.submit_once(record,binding,['sbatch'],{},lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='not a job',stderr=''),lambda:{})
 with pytest.raises(RuntimeError,match='mismatched'):k.submit_once(record,{'canonical':'other'},['sbatch'],{},lambda *a,**kw:pytest.fail('unsafe'),lambda:{})

def test_failed_physical_precheck_never_calls_submit_or_writes_unknown(tmp_path):
 record=tmp_path/'submission.json'
 def blocked():raise ValueError('8 leases pending')
 with pytest.raises(ValueError):k.submit_once(record,{},['sbatch'],{},lambda *a,**kw:pytest.fail('unsafe'),blocked)
 assert not record.exists()
