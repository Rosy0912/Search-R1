import ast,os,pathlib,types,pytest
from stage_baseline import protected_stage_baseline,post_cleanup_remaining
HERE=pathlib.Path(__file__).parent

def row(pid,parent,group='/job',uid=None):return {'pid':pid,'ppid':parent,'uid':os.getuid() if uid is None else uid,'starttime_ticks':pid,'cgroup':['0::'+group]}
def test_continuous_roots_and_gpu_baseline_exclude_ephemeral_query():
 rows={1:row(1,0,'/'),10:row(10,1),20:row(20,10),30:row(30,10),40:row(40,10),50:row(50,1,'/foreign')}
 gpu=[{'pid':40,'identity':rows[40]},{'pid':50,'identity':rows[50]}]
 baseline=protected_stage_baseline(None,'scope','/job',gpu,read=lambda p:rows[p],start_pid=20)
 assert set(baseline)=={10,20,40} and 30 not in baseline # parent query is not a continuous root

@pytest.mark.parametrize('kind',['ancestor','resident'])
def test_continuous_uid_mismatch_fails(kind):
 rows={1:row(1,0,'/'),10:row(10,1),20:row(20,10),40:row(40,10)}
 original=dict(rows[40]);rows[10 if kind=='ancestor' else 40]['uid']=-1
 with pytest.raises(RuntimeError):protected_stage_baseline(None,'scope','/job',[{'pid':40,'identity':original}],read=lambda p:rows[p],start_pid=20)

def test_no_closed_pidfd_census_and_failed_cleanup_remaining_preserved():
 class Closed:
  ledger={20:row(20,10)}
  def pids(self,scope):raise AssertionError('closed pidfd authority must not be used')
 owned=types.SimpleNamespace(stage_scope=Closed(),cleanup_record={'actual_cgroup_empty':False,'remaining':[row(20,10)]})
 tools=types.SimpleNamespace(cgroup_pids=lambda scope:{20,30},process_identity=lambda p:row(p,10))
 remaining,evidence=post_cleanup_remaining(owned,tools,'scope')
 assert remaining==[20] and evidence['error'] and evidence['actual_job_pids']==[20,30]

def test_successful_cleanup_actual_ledger_absent():
 owned=types.SimpleNamespace(stage_scope=types.SimpleNamespace(ledger={20:row(20,10)}),cleanup_record={'actual_cgroup_empty':True,'remaining':[]})
 tools=types.SimpleNamespace(cgroup_pids=lambda scope:{30})
 remaining,evidence=post_cleanup_remaining(owned,tools,'scope');assert remaining==[] and 'error' not in evidence

def test_actual_three_nested_paths_have_scoped_cleanup_and_scoped_live_census():
 for filename,names in [('owner.py',('run_stage','main')),('refit_with_owner.py',('owned_command',))]:
  tree=ast.parse((HERE/filename).read_text())
  for name in names:
   node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
   calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)]
   assert any(isinstance(n.func,ast.Name) and n.func.id=='owned_stage' for n in calls)
   assert not any(isinstance(n.func,ast.Attribute) and n.func.attr=='OwnedProcesses' for n in calls)
   assert any(isinstance(n.func,ast.Name) and n.func.id=='post_cleanup_remaining' for n in calls)
   if name!='main':assert any(isinstance(n.func,ast.Attribute) and n.func.attr=='pids' and ast.unparse(n.func.value)=='owned.stage_scope' for n in calls)

@pytest.mark.parametrize('failed',[False,True])
def test_actual_run_stage_success_and_primary_failure_still_cleanup(failed,tmp_path,monkeypatch):
 import json,owner,phase_control
 cleaned=[False];calls=[0]
 class Scope:
  ledger={123:row(123,20)}
  def pids(self,scope):
   calls[0]+=1
   if failed and calls[0]>1:raise RuntimeError('secondary census fault')
   return set() if cleaned[0] else {123}
 class Child:
  pid=123;returncode=1 if failed else 0
  def __init__(self):self.polls=0
  def poll(self):self.polls+=1;return None if self.polls==1 else self.returncode
 class Owned:
  baseline=set();stage_scope=Scope();cleanup_record=None
  def spawn(self,*a,**kw):
   (tmp_path/'phase-A-receipt.json').write_text(json.dumps({'pid':123,'config_sha256':owner.sha(config_path),'source_sha256':owner.DUMP_SHA,'uuids':['u']}));return Child()
  def cleanup(self):
   if failed:
    raw=json.loads((tmp_path/'phase-A-primary-error.json').read_text());assert raw['error']=='owned phaseA exited 1'
   assert (tmp_path/'phase-A-registration-before-cleanup.json').is_file()
   cleaned[0]=True;self.cleanup_record={'actual_cgroup_empty':True,'remaining':[]}
 obj=Owned();tools=types.SimpleNamespace(cgroup_scope=lambda job:('scope','/job'),cgroup_pids=lambda scope:set() if cleaned[0] else {123},process_identity=lambda p:row(p,20),process_sample=lambda p:{'pid':p},gpu_sample=lambda:{'processes':[] if cleaned[0] else ['u,123,1']})
 monkeypatch.setattr(owner,'owned_stage',lambda *a,**kw:obj);monkeypatch.setattr(owner,'gpu_rows',lambda:{'u':0});monkeypatch.setattr(owner,'leases',lambda x:());monkeypatch.setattr(phase_control,'notify',lambda *a:None);monkeypatch.setattr(owner.time,'sleep',lambda n:None)
 config_path=tmp_path/'config.json';config_path.write_text('{}');config={'job':'1','uuids':['u'],'leases':[],'stage_timeout_seconds':30}
 if failed:
  with pytest.raises(RuntimeError,match='owned phaseA exited 1'):owner.run_stage('A',tmp_path,config_path,config,tools,{},[])
  raw=json.loads((tmp_path/'phase-A-owner-barrier.json').read_text());assert raw['cleanup_census_error']['error']=='secondary census fault' and raw['primary_error']['error']=='owned phaseA exited 1'
 else:
  raw=owner.run_stage('A',tmp_path,config_path,config,tools,{},[]);assert raw['own_processes_empty'] and raw['gpu_after']=={'processes':[]}
 assert cleaned[0]

def test_continuous_baseline_uid_and_cgroup_fresh_check(monkeypatch):
 import stage_baseline as b
 previous=row(os.getpid(),1);previous['cgroup']=['0::/job']
 tools=types.SimpleNamespace(process_identity=lambda pid:dict(previous))
 assert b.continuous_baseline_missing(tools,{os.getpid():previous},read=lambda pid:dict(tools.process_identity(pid),uid=os.getuid()))==[]
 previous['uid']=-1;assert b.continuous_baseline_missing(tools,{os.getpid():previous},read=lambda pid:dict(tools.process_identity(pid),uid=os.getuid()))


def test_unknown_post_cleanup_census_is_none_not_empty():
 owned=types.SimpleNamespace(stage_scope=types.SimpleNamespace(ledger={}),cleanup_record={'remaining':[]})
 def unavailable(scope):raise RuntimeError('actual census unavailable')
 remaining,evidence=post_cleanup_remaining(owned,types.SimpleNamespace(cgroup_pids=unavailable),'scope')
 assert remaining is None and evidence['error']=='actual census unavailable'
