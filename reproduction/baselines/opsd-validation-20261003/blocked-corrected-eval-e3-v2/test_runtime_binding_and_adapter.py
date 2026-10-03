"""Native-relevant CPU helper tests; no genuine Slurm/GPU proof is created."""
import ast,copy,hashlib,importlib.util,json,os,sys,types
from pathlib import Path
import pytest
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
import runtime_binding as binding
import native_main_adapter as adapter
import owned_runtime as own

def synthetic_bundle(tmp_path):
 root=tmp_path/'A';root.mkdir();script=root/'eval_job.py';script.write_text('# CPU helper A only')
 helper=root/'runtime_binding.py';helper.write_text('# CPU helper binding only')
 cfg=tmp_path/'actual.yaml';cfg.write_text('CPU config only')
 bundle={'directory':str(root),'file_sources':{p.name:{'path':str(p),'sha256':binding.digest(p)} for p in (script,helper)},'config_binding':{'path':str(cfg),'sha256':binding.digest(cfg)}}
 return bundle,script,cfg

def test_actual_directory_file_config_binding_positive(tmp_path):
 bundle,script,cfg=synthetic_bundle(tmp_path)
 result=binding.verify_runtime(bundle,script,cfg,modules={})
 assert result['executing_file']['path']==str(script)

@pytest.mark.parametrize('mutation',['foreign_directory','foreign_module','source_changed','wrong_config_path','config_changed'])
def test_prove_A_execute_B_config_or_import_shadow_rejected(tmp_path,mutation):
 bundle,script,cfg=synthetic_bundle(tmp_path);modules={}
 if mutation=='foreign_directory':script=tmp_path/'B.py';script.write_text('# CPU helper A only')
 elif mutation=='foreign_module':modules={'runtime_binding':types.SimpleNamespace(__file__=str(tmp_path/'foreign_binding.py'))}
 elif mutation=='source_changed':script.write_text('# altered A')
 elif mutation=='wrong_config_path':cfg=tmp_path/'other.yaml';cfg.write_text('CPU config only')
 elif mutation=='config_changed':cfg.write_text('altered config')
 with pytest.raises(ValueError):binding.verify_runtime(bundle,script,cfg,modules)

def fake_preflight(tmp_path):
 p=tmp_path/'preflight.py';p.write_text("calls=[]\ndef validate_inherited_gpu_leases(expected_count=None):\n calls.append(expected_count)\n return 'ORIGINAL CPU FUNCTION'\n")
 spec=importlib.util.spec_from_file_location('cpu_fake_preflight',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m,p

@pytest.mark.parametrize('failure',[False,True])
def test_original_main_call_config_unchanged_and_restore_finally(tmp_path,monkeypatch,failure):
 module,p=fake_preflight(tmp_path);original=module.validate_inherited_gpu_leases;calls=[];config={'trainer':{'val_only':True},'generation':'unchanged'};saved=copy.deepcopy(config)
 monkeypatch.setattr(adapter,'reassert_owned',lambda preflight,count:calls.append((preflight,count)) or {'scope':'CPU own proof fixture'})
 record=tmp_path/'adapter.json'
 def native_main_fixture(cfg):
  assert cfg is config
  module.validate_inherited_gpu_leases(8)
  if failure:raise ValueError('CPU main failure')
 if failure:
  with pytest.raises(ValueError,match='main failure'):
   with adapter.MainLeaseWindow(module,record,binding.digest(p),{'scope':'CPU binding fixture'}):native_main_fixture(config)
 else:
  with adapter.MainLeaseWindow(module,record,binding.digest(p),{'scope':'CPU binding fixture'}):native_main_fixture(config)
 assert module.validate_inherited_gpu_leases is original and module.calls==[] and config==saved
 assert calls==[(module,8)]
 raw=json.loads(record.read_text());assert raw['restored_original'] and raw['replacement_intact']
 assert raw['original']['code_sha256']==adapter.compiled_function_code(p,'validate_inherited_gpu_leases')
 assert raw['replacement']['code_sha256']==adapter.compiled_function_code(HERE/'native_main_adapter.py','replacement')
 assert module.validate_inherited_gpu_leases(8)=='ORIGINAL CPU FUNCTION' and module.calls==[8]

def test_changed_original_code_cannot_borrow_source_filename_label(tmp_path):
 module,p=fake_preflight(tmp_path)
 def foreign(expected_count=None):return 'ALTERED CODE'
 module.validate_inherited_gpu_leases.__code__=foreign.__code__.replace(co_filename=str(p),co_name='validate_inherited_gpu_leases')
 with pytest.raises(ValueError,match='compiled frozen source'):adapter.MainLeaseWindow(module,tmp_path/'record',binding.digest(p),{})

def test_foreign_fourth_uid_and_wrong_cgroup_reject():
 row={'uid':123,'uids':[123]*4,'cgroup':'/slurm/job_101/step_0'}
 assert own.assert_owned(row,'101',123)==row
 for changed in [dict(row,uids=[123,123,123,999]),dict(row,cgroup='/slurm/job_102/step_0')]:
  with pytest.raises(ValueError):own.assert_owned(changed,'101',123)

def test_poststart_does_not_reinvoke_idle_thresholds_ast():
 source=(HERE/'owned_runtime.py').read_text();tree=ast.parse(source)
 names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 assert 'check_selected_gpus' not in names and 'validate_inherited_gpu_leases' not in names
 source=(HERE/'eval_job.py').read_text();assert source.index('preflight.check_selected_gpus')<source.index("child(['ray'")<source.index("'upstream-pilot/retrieval_compare.py'")
 assert 'check_selected_gpus' not in (HERE/'eval_supervisor.py').read_text()

@pytest.mark.parametrize('failure',['foreign_uid','unreadable','pid_reused','stale_fd',None])
def test_late_owned_busy_gpu_reassert_rejects_foreign_unreadable_stale_pid_fd(monkeypatch,failure):
 uid=os.getuid();uuids=['GPU-'+f'{i:08x}'+'-0000-0000-0000-000000000000' for i in range(8)]
 inventory={str(i):{'uuid':u,'utilization':95.,'memory_used':50000.,'memory_total':80000.} for i,u in enumerate(uuids)}
 preflight=types.SimpleNamespace(gpu_snapshot=lambda:(inventory,set(uuids)))
 monkeypatch.setenv('SLURM_JOB_ID','101');monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(inventory));monkeypatch.setenv('SEARCH_R1_GPU_PREFLIGHT','1');monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([{'uuid':u,'fd':20+i} for i,u in enumerate(uuids)]));monkeypatch.setenv('OPSD_CORRECTED_EVAL_STAGE_SHA256','a'*64)
 monkeypatch.setattr(own.subprocess,'check_output',lambda *args,**kw:uuids[0]+', 12345\n')
 calls={}
 def identity(pid):
  if failure=='unreadable' and pid==12345:raise FileNotFoundError('CPU proc disappearance')
  calls[pid]=calls.get(pid,0)+1
  changed=failure=='pid_reused' and pid==12345 and calls[pid]>1
  return {'pid':pid,'uid':uid,'uids':[uid,uid,uid,9999999] if failure=='foreign_uid' and pid==12345 else [uid]*4,'process_start_ticks':'999' if changed else '123','cgroup':'/slurm/job_101/step_0'}
 monkeypatch.setattr(own,'process_identity',identity)
 def lease(fd,path):
  if failure=='stale_fd' and fd==20:raise ValueError('CPU fixture wrong lease inode')
  return {'scope':'synthetic CPU lease only','fd':fd,'path':str(path)}
 monkeypatch.setattr(own,'capture_lease',lease)
 if failure:
  with pytest.raises((ValueError,FileNotFoundError)):own.reassert_owned(preflight,8)
 else:
  # High busy95% is owned workload, not a second idle check.
  result=own.reassert_owned(preflight,8);assert len(result['gpu_leases'])==8 and result['active_gpu_processes'][0]['pid']==12345


def test_owned_child_exception_cleanup_is_bounded_and_original_error_preserved(monkeypatch):
 import subprocess,time
 import owned_children
 child=subprocess.Popen([sys.executable,'-u','-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('READY',flush=True);time.sleep(60)"],stdout=subprocess.PIPE)
 assert child.stdout.readline()==b'READY\n';original_wait=child.wait;attempts=[];original_stop=owned_children.stop_own_child
 def wait(timeout=None):
  if timeout is None and not attempts:attempts.append(True);raise ValueError('Original CPU execution failure')
  return original_wait(timeout=timeout)
 monkeypatch.setattr(child,'wait',wait);monkeypatch.setattr(owned_children.subprocess,'Popen',lambda *args,**kwargs:child);monkeypatch.setattr(owned_children,'stop_own_child',lambda p,timeout:original_stop(p,timeout=.15))
 started=time.monotonic()
 try:
  with pytest.raises(ValueError,match='Original CPU execution failure'):owned_children.run_owned(['CPU-fixture-existing-own-child'])
  assert time.monotonic()-started<2 and child.poll() is not None
 finally:
  if child.poll() is None:child.kill();original_wait(timeout=2)
  child.stdout.close()
