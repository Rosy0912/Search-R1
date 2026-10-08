import hashlib,json,pathlib,sys
import pytest
import native_run as n

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value) if not isinstance(value,str) else value)
 return {'path':str(path),'sha256':n.sha(path)}

def fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(n,'ROOT',tmp_path)
 source=tmp_path/'tools/final-isolated-pipeline';source.mkdir(parents=True)
 put(source/'supervise.py','# CPU fixture source\n')
 manifest={'files':{'supervise.py':{'size':(source/'supervise.py').stat().st_size,'sha256':n.sha(source/'supervise.py')}}}
 put(source/'files.json',manifest);digest=n.sha(source/'files.json')
 logs=tmp_path/'logs';logs.mkdir()
 terminal=put(logs/'terminal.json',{'job_id':12,'final_state_observed':True,'scontrol_returncode':0,'scontrol_stdout':'JobId=12 JobState=COMPLETED ExitCode=0:0 NodeList=h100-11'})
 log=put(logs/'cpu.log','EXACT_SOURCE_MANIFEST '+str(source/'files.json')+' '+digest+'\n170 passed in 2.5s\n')
 binding=logs/'binding.json';put(binding,{'directory':str(source),'manifest_sha256':digest,'native_expected':170,'native_cpu':{'job_id':12,'passed':170,'log':log,'terminal':terminal}})
 return source,binding

def test_actual_source_native_binding(tmp_path,monkeypatch):
 source,binding=fixture(tmp_path,monkeypatch)
 try:assert n.source_gate(binding)['files']['supervise.py']['sha256']==n.sha(source/'supervise.py')
 finally:
  if str(source) in sys.path:sys.path.remove(str(source))

@pytest.mark.parametrize('mutation',['source_drift','extra_pth','skip','wrong_manifest','failed_terminal'])
def test_proof_rejects_real_mutations(tmp_path,monkeypatch,mutation):
 source,binding=fixture(tmp_path,monkeypatch);data=json.loads(binding.read_text());proof=data['native_cpu']
 if mutation=='source_drift':(source/'supervise.py').write_text('mutated')
 elif mutation=='extra_pth':(source/'added.pth').write_text('/wrong/path')
 elif mutation in ('skip','wrong_manifest'):
  p=pathlib.Path(proof['log']['path']);p.write_text('170 passed, 1 skipped in 2.5s\n' if mutation=='skip' else 'EXACT_SOURCE_MANIFEST /other/files.json '+data['manifest_sha256']+'\n170 passed in 2.5s\n');proof['log']['sha256']=n.sha(p)
 else:
  p=pathlib.Path(proof['terminal']['path']);v=json.loads(p.read_text());v['scontrol_stdout']=v['scontrol_stdout'].replace('COMPLETED','FAILED');put(p,v);proof['terminal']['sha256']=n.sha(p)
 put(binding,data)
 with pytest.raises(ValueError):n.source_gate(binding)


def test_initial_wrong_job_rejected(tmp_path,monkeypatch):
 import types
 source,binding=fixture(tmp_path,monkeypatch);logs=tmp_path/'logs'
 config=logs/'initial-config.json';put(config,{'job':'13','evidence_root':str(logs/'initial-evidence')})
 args=types.SimpleNamespace(initial_terminal=str(logs/'terminal.json'),initial_job=12,initial_config=str(config))
 with pytest.raises(ValueError,match='another job'):n.initial_gate(args)

def profile_fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(n,'ROOT',tmp_path);root=tmp_path/'logs/profile/evidence';root.mkdir(parents=True)
 outer=put(root.parent/'actual-outer2.json',{'actual_completed_outer_steps':[1,2],'export_step':2})
 cfg=root/'effective-runtime-config.json';put(cfg,{'job':'55','evidence_root':str(root),'actual_outer_receipt_path':outer['path']})
 held=root/'trainer-terminal-held.json';put(held,{'job':'55','effective_config_sha256':n.sha(cfg),'actual_outer_evidence':outer})
 auth=put(root/'trainer-terminal-authorization.json',{'job':'55','status':'AUTHORIZED_OWN_SHUTDOWN','held_sha256':n.sha(held),'actual_outer_evidence':outer})
 result=root/'profile2-result.json';put(result,{'terminal_supervision_evidence':auth,'actual_outer_evidence':outer})
 put(root/'cleanup.json',dict(own_processes_remaining=[],baseline_identity_missing=[],original_gpu_baseline_missing=[],own_gpu_records_remaining=[],sampler_errors=[],guardian_errors=[],error=None,cleanup={'actual_cgroup_empty':True}))
 return result

def test_actual_profile_job_and_root_binding(tmp_path,monkeypatch):
 result=profile_fixture(tmp_path,monkeypatch);assert n.profile_run_binding(result,55)['actual_outer_evidence']
 with pytest.raises(ValueError,match='another job/root'):n.profile_run_binding(result,56)

def test_profile_unrelated_authorization_job_fatal(tmp_path,monkeypatch):
 result=profile_fixture(tmp_path,monkeypatch);p=result.parent/'trainer-terminal-authorization.json';v=json.loads(p.read_text());v['job']='56';link=put(p,v)
 r=json.loads(result.read_text());r['terminal_supervision_evidence']=link;put(result,r)
 with pytest.raises(ValueError,match='job differs'):n.profile_run_binding(result,55)

def test_explicit_final_source_suite_count_mismatch(tmp_path,monkeypatch):
 source,binding=fixture(tmp_path,monkeypatch);v=json.loads(binding.read_text());v['native_expected']=171;put(binding,v)
 with pytest.raises(ValueError,match='expected suite'):n.source_gate(binding)
