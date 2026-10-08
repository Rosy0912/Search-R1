import copy,json,pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from caller_contracts import authorization,context_receipt,same_cpu_identity,verify_bundle
from score_gate import PARAMETERS,digest

def test_actual_authorization_binds_phase_pid_generation_source_and_job():
 identity=dict(pid=13,starttime_ticks=78)
 raw=dict(phase='SCORING_ONLY',**identity,job='44',provenance_sha256='source')
 authorization(raw,'SCORING_ONLY',identity,'source','44')
 for k,v in [('phase','CONTEXT_ONLY'),('pid',14),('starttime_ticks',79),('job','45'),('provenance_sha256','other')]:
  bad=dict(raw);bad[k]=v
  with pytest.raises(RuntimeError):authorization(bad,'SCORING_ONLY',identity,'source','44')

def test_context_requires_exact_device_uuid_mapping_and_zero_allocator():
 uuids=['gpu'+str(i) for i in range(8)]
 raw=dict(pid=13,gpu_pid_uuid=[dict(pid=13,uuid=u) for u in uuids],allocators=[dict(device=i,uuid=u,allocated_bytes=0,reserved_bytes=0) for i,u in enumerate(uuids)])
 context_receipt(raw,dict(pid=13),uuids)
 for mode in ['device','uuid','reserved','foreignpid','missing']:
  bad=copy.deepcopy(raw)
  if mode=='device':bad['allocators'][7]['device']=6
  elif mode=='uuid':bad['allocators'][7]['uuid']=uuids[0]
  elif mode=='reserved':bad['allocators'][7]['reserved_bytes']=32*1024**2
  elif mode=='foreignpid':bad['gpu_pid_uuid'][7]['pid']=14
  else:bad['gpu_pid_uuid'].pop()
  with pytest.raises(RuntimeError):context_receipt(bad,dict(pid=13),uuids)

def test_cpu_identity_compares_buffer_and_frozen_flags_not_only_param_sha(tmp_path):
 snapshot=dict(parameter_count=PARAMETERS,parameter_bytes=PARAMETERS*4,parameter_sha256='parameters',buffer_sha256='buffers',parameter_devices=['cpu'],buffer_devices=['cpu'],training_modules=0,requires_grad_parameters=0,grad_parameters=0)
 p=tmp_path/'identity.json';p.write_text(json.dumps(dict(snapshot=snapshot)))
 provenance=dict(cpu_identity=dict(path=str(p)))
 same_cpu_identity(snapshot,provenance)
 for k,v in [('buffer_sha256','changed'),('parameter_bytes',1),('training_modules',1),('grad_parameters',1),('parameter_devices',['cuda:0'])]:
  bad=dict(snapshot);bad[k]=v
  with pytest.raises(RuntimeError):same_cpu_identity(bad,provenance)

def test_bundle_missing_payload_and_postfreeze_drift_refused(tmp_path):
 names=['score_gate.py','caller_contracts.py','worker.py','supervisor.py','native-score.sbatch','test_score_gate.py','test_caller_contracts.py']
 for n in names:(tmp_path/n).write_text(n)
 manifest={n:digest(tmp_path/n) for n in names};p=tmp_path/'tool-files.json';p.write_text(json.dumps(manifest))
 assert verify_bundle(tmp_path)['files']==manifest
 (tmp_path/'worker.py').write_text('changed')
 with pytest.raises(RuntimeError):verify_bundle(tmp_path)
 manifest.pop('worker.py');p.write_text(json.dumps(manifest))
 with pytest.raises(RuntimeError):verify_bundle(tmp_path)

def test_actual_supervisor_check_only_never_calls_slurm_or_spawns(monkeypatch,tmp_path):
 import supervisor
 monkeypatch.setattr(sys,'argv',['supervisor','--provenance',str(tmp_path/'proof.json'),'--candidate','candidate','--check-only'])
 (tmp_path/'proof.json').write_text('{}')
 monkeypatch.setattr(supervisor,'verify_bundle',lambda directory:{'files':'frozen'})
 monkeypatch.setattr(supervisor,'check_identity',lambda proof,candidate:tmp_path)
 monkeypatch.setattr(supervisor,'slurm',lambda:pytest.fail('Check-only reached Slurm/GPU lifecycle'))
 supervisor.main()

def test_actual_scorer_measurement_adapter_always_closes_identity_retry(tmp_path):
 import supervisor,importlib.util
 base=pathlib.Path(__file__).parents[1]
 folder=base/'qwen25-32b-window-safety-candidate-v3'
 if not folder.exists():folder=base/'qwen25-32b-window-safety-candidate-v3'
 path=folder/'startup_guard.py'
 assert digest(path)==supervisor.MEASUREMENT_SHA
 spec=importlib.util.spec_from_file_location('real_window_scorer_guard',path);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
 # Use real adapter constructor; fake only native telemetry module functions.
 import types
 safety_path=folder/'runtime_safety.py'
 safety=types.SimpleNamespace(__file__=str(safety_path),nvml_snapshot=lambda:{},active_gate=lambda runtime:{},pid_identity=lambda p:{},collect_process_windows=lambda snapshot,foreign,own:{})
 guard=supervisor.measurement_guard(adapter,safety,safety_path,tmp_path/'trace.jsonl')
 assert guard.ready_check() is True
 guard._poll_ready()
 assert guard.closed is True
 assert guard.deadline is None
 guard.close_startup()
 safety.nvml_snapshot=guard.original_snapshot

@pytest.mark.parametrize('mode',['valid','eval_drift','import_shadow'])
def test_clean_subprocess_explicit_window_dependency(mode,tmp_path):
 import subprocess,os
 base=pathlib.Path(__file__).parents[1]
 folder=base/'qwen25-32b-window-safety-candidate-v3'
 if not folder.exists():folder=base/'qwen25-32b-window-safety-candidate-v3'
 original=base/'qwen25-32b-terminal-fix-refreeze/eval64.py'
 if mode=='eval_drift':
  altered=tmp_path/'eval64.py';altered.write_text(original.read_text()+'\n# drift\n');original=altered
 code="import sys,types;sys.path.insert(0,"+repr(str(pathlib.Path(__file__).parent))+" );import supervisor;"
 if mode=='import_shadow':code+="sys.modules['eval64']=types.SimpleNamespace(__file__='/wrong/eval64.py');"
 code+="s=supervisor.load_window_safety("+repr(str(folder))+","+repr(str(original))+" );assert s.require is sys.modules['eval64'].require;assert s.FRACTION==.4;print('CLEAN_WINDOW_IMPORT_NO_GPU')"
 env=dict(os.environ);env.pop('PYTHONPATH',None)
 result=subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,env=env,text=True,capture_output=True)
 if mode=='valid':assert result.returncode==0 and 'CLEAN_WINDOW_IMPORT_NO_GPU' in result.stdout,result.stderr
 else:assert result.returncode!=0 and ('source drift' in result.stderr or 'import shadow' in result.stderr),result.stderr
