import json,types,pytest
import first_fault as f

def owned():return types.SimpleNamespace(stage_scope=types.SimpleNamespace(roots={10:{'pid':10}},ledger={10:{'pid':10,'uid':7}},events=[{'event':'actual registration'}],fds={10:4},sealed=False))
def test_first_original_fault_exclusive_and_later_error_cannot_replace(tmp_path):
 first=ValueError('original fault');a=f.before_cleanup(tmp_path,'phase-A',first,owned(),'cfgsha')
 raw=json.loads((tmp_path/'phase-A-primary-error.json').read_text());assert raw['error_type']=='ValueError' and raw['error']=='original fault' and raw['config_sha256']=='cfgsha'
 b=f.before_cleanup(tmp_path,'phase-A',RuntimeError('later cleanup'),owned(),'cfgsha')
 assert a['primary_error_ref']==b['primary_error_ref'] and json.loads((tmp_path/'phase-A-primary-error.json').read_text())['error']=='original fault'
 assert json.loads((tmp_path/'phase-A-registration-before-cleanup.json').read_text())['open_pidfd_registered_pids']==[10]

def test_original_exception_publication_failure_is_secondary_not_raised(tmp_path,monkeypatch):
 def unavailable(*a):raise OSError('fsync unavailable')
 monkeypatch.setattr(f,'exclusive',unavailable);error=ValueError('primary remains')
 result=f.before_cleanup(tmp_path,'phase-A',error,owned(),'cfgsha')
 assert len(result['publication_errors'])==2 and result['primary_error_ref'] is None and str(error)=='primary remains'

def test_success_records_registration_only_no_synthetic_primary(tmp_path):
 result=f.before_cleanup(tmp_path,'heads',None,owned(),'cfgsha')
 assert result['primary_error_ref'] is None and not (tmp_path/'heads-primary-error.json').exists()
 assert result['registration_ref'] and f.after_cleanup(tmp_path,'heads',owned())['registration_ref']

def test_exclusive_symlink_rejected_without_touching_target(tmp_path):
 target=tmp_path/'target';target.write_text('untouched');(tmp_path/'phase-A-primary-error.json').symlink_to(target)
 result=f.before_cleanup(tmp_path,'phase-A',ValueError('real'),owned(),'cfgsha')
 assert result['publication_errors'][0]['event']=='primary-before-cleanup' and target.read_text()=='untouched'
