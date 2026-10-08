import importlib.util,pathlib,types,json
spec=importlib.util.spec_from_file_location('observed_native',pathlib.Path(__file__).with_name('observe_native.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def response(state='RUNNING',job=9,exitcode='0:0'):return types.SimpleNamespace(returncode=0,stdout='JobId=%d JobState=%s ExitCode=%s NodeList=h100-11'%(job,state,exitcode),stderr='')
def test_same_call_saves_raw_before_terminal_TTL_and_captures_proof(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'logs').mkdir();times=[0.];calls=[]
 def run(argv,**kw):
  calls.append(argv)
  if argv[0]=='scontrol':return response('RUNNING' if len(calls)==1 else 'COMPLETED')
  return types.SimpleNamespace(returncode=0,stdout='actual bound proof',stderr='')
 result=m.observe(9,tmp_path/'helper',runner=run,clock=lambda:times[0],pause=lambda n:times.__setitem__(0,times[0]+n))
 assert result['completed_zero'] and len(result['observations'])==2 and result['binding_capture']['returncode']==0
 raw=json.loads(pathlib.Path(result['raw']['path']).read_text());assert raw['scontrol_stdout'].find('JobState=COMPLETED')>=0
 again=m.observe(9,tmp_path/'helper',runner=lambda *a,**k:(_ for _ in ()).throw(AssertionError('no new submission or lost terminal query')))
 assert again==result

def test_failed_or_wrong_job_never_invokes_binding_capture(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'logs').mkdir();calls=[]
 def run(argv,**kw):calls.append(argv);return response('FAILED',exitcode='1:0')
 result=m.observe(9,tmp_path/'helper',runner=run)
 assert result['terminal_observed'] and not result['completed_zero'] and 'binding_capture' not in result and len(calls)==1
 assert not m.parse(9,response('COMPLETED',job=10))['terminal_observed']

def test_pending_budget_then_same_submission_continues_actual_observation(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'logs').mkdir();times=[0.];calls=[]
 def running(argv,**kw):calls.append(argv);return response()
 pending=m.observe(9,tmp_path/'helper',runner=running,clock=lambda:times[0],pause=lambda n:times.__setitem__(0,times[0]+n),budget=3)
 assert times[0]==3 and not pending['terminal_observed'] and len(calls)==2
 done=m.observe(9,tmp_path/'helper',runner=lambda *a,**kw:response('FAILED',exitcode='1:0'))
 assert done['terminal_observed'] and not done['completed_zero']
 assert all(argv[0]=='scontrol' for argv in calls)
 source=pathlib.Path(__file__).with_name('remote_bootstrap.py').read_text()
 assert source.index("result['observation']=observe")>source.index("if record.exists():")
