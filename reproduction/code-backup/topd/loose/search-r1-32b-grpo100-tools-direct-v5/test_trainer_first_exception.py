import pathlib,json,os
import pytest
import train_entry

def config(tmp_path):
 (tmp_path/'effective-runtime-config.json').write_text('{"CPU_CONTRACT_ONLY":true}')
 return {'evidence_root':str(tmp_path)}
def test_actual_main_exception_durable_before_shutdown(tmp_path,monkeypatch):
 monkeypatch.setenv('SLURM_JOB_ID','3163');c=config(tmp_path);events=[]
 def main(_):events.append('main');raise RuntimeError('first-real-main-error')
 def shutdown():
  raw=json.loads((tmp_path/'trainer-first-exception.json').read_text());assert raw['message']=='first-real-main-error';assert 'RuntimeError: first-real-main-error' in raw['traceback'];assert raw['pid']==os.getpid() and raw['effective_config_sha256']==train_entry.sha(tmp_path/'effective-runtime-config.json');events.append('shutdown-after-durable')
 with pytest.raises(RuntimeError,match='first-real-main-error'):train_entry.run_original_main(main,None,shutdown,c)
 assert events==['main','shutdown-after-durable'];assert (tmp_path/'trainer-first-exception.json').stat().st_mode&0o777==0o600

def test_normal_authorized_hold_return_no_failure(tmp_path):
 c=config(tmp_path);events=[]
 def main(_):events.extend(['fit','terminal-held','authorized']);return 42
 assert train_entry.run_original_main(main,None,lambda:events.append('shutdown'),c)==42
 assert events==['fit','terminal-held','authorized','shutdown'];assert not (tmp_path/'trainer-first-exception.json').exists()

def test_persistence_fault_remains_failclosed_shutdown_runs(tmp_path,monkeypatch,capsys):
 c=config(tmp_path);events=[]
 def failure(_):raise ValueError('original-before-finally')
 def write(*_):raise OSError('injected-fsync-disk-fault')
 monkeypatch.setattr(train_entry,'persist_first_exception',write)
 with pytest.raises(ValueError,match='original-before-finally') as raised:train_entry.run_original_main(failure,None,lambda:events.append('shutdown'),c)
 assert 'injected-fsync-disk-fault' in raised.value.first_exception_persistence_error
 assert events==['shutdown'];assert 'PERSISTENCE_FAILED' in capsys.readouterr().err

def test_exclusive_first_fault_never_overwritten(tmp_path):
 c=config(tmp_path)
 train_entry.persist_first_exception(c,RuntimeError('first'))
 before=(tmp_path/'trainer-first-exception.json').read_bytes()
 with pytest.raises(FileExistsError):train_entry.persist_first_exception(c,RuntimeError('second'))
 assert (tmp_path/'trainer-first-exception.json').read_bytes()==before
