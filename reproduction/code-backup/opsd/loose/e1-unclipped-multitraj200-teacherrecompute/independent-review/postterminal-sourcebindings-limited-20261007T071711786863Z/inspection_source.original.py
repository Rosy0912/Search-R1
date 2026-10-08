from pathlib import Path
import hashlib,json,types,unittest
H=Path(__file__).resolve().parents[2];C=H/'postterminal-source-binding-candidate'
pins={'collect_postterminal_source_bindings.py':'2cbcb83a4b54bafa0f042121850b09089bc80414b6a852c0fb72af15051df7fe','test_postterminal_source_bindings.py':'f8bab74716506788b7142ddff0a4f54e1902d09af04df2b2ec59d40c5e21d4d8','SOFTWARE_READINESS.json':'b9f27f0dceb7f80ef9f91142d9ce3d4f90cb5207628e7129bb8e1dcebc25c1dc','DESIGN.md':'f1d002b6d1511b40d9a15a8528042c427965b5bdd79fa85321bc0141a5f8dc90'}
for name,pin in pins.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==pin
ready=json.loads((C/'SOFTWARE_READINESS.json').read_bytes());directory=Path(ready['tests']['original_directory'])
for name,record in ready['tests']['originals'].items():
 raw=(directory/name).read_bytes();assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
wait=json.loads((directory/'wait.original.json').read_bytes());assert wait['waited'] is True and wait['returncode']==0 and wait['stable'] is True and len(wait['before'])==32 and wait['before']==wait['after']
assert b'Ran 20 tests' in (directory/'stderr.original').read_bytes() and b'OK' in (directory/'stderr.original').read_bytes()
test=types.ModuleType('postterminal_source_limited_independent');test.__file__=str(C/'test_postterminal_source_bindings.py');test.__package__=''
exec(compile((C/'test_postterminal_source_bindings.py').read_bytes(),test.__file__,'exec',dont_inherit=True,optimize=0),test.__dict__)
suite=unittest.TestSuite(test.Boundaries(name) for name in ('test_literal_nonzero_wrapper_preserved','test_failure_keeps_request_terminal_and_once'))
result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful() and result.testsRun==2
for name,pin in pins.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==pin
print(json.dumps({'status':'PASS_LIMITED_TERMINAL_LITERAL_AND_PARTIAL_ONCE_REPLAY','independent_selected_local_tests':2,'author20_originals_only_audited_not_rerun':True,'author32_before_after_pins_stable':True,'actual3304_terminal':None,'production_main':False,'remote':False,'gpu':False,'current_native_import':False,'large_model_or_index_read':False},sort_keys=True))
