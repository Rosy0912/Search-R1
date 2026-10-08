from pathlib import Path
import hashlib,json,sys,types,unittest,time
H=Path(__file__).resolve().parents[2];P=H/'final200-raw-replay-consumer-candidate'
expected={'replay_final200_raw.py':'1f6fac7e829d5918a5d15d44ff144acd15a360b4361edbcfcf1a6091688501d2','test_model_layout_delta.py':'c8c9f8d27b7addce0387bed06eb33e3647330e35d0c26bdcd4f1d17ed217b26d','READINESS_V3.json':'b4b9737e8e4be04ac7efe15166daebfdbb2b2aa5f2766b7515e469572015ac64','retained-705-v2/replay_final200_raw.py':'705c286dc7ac51353d4eb4de3ec21439e4459bd4b14df0c313af1ac9a31ca28a'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={n:sha(P/n) for n in expected};assert before==expected
p=P/'test_model_layout_delta.py';m=types.ModuleType('independent_layout_delta');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),m.__dict__)
start=time.time();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
after={n:sha(P/n) for n in expected};assert before==after==expected
print(json.dumps(dict(kind='INDEPENDENT_MODEL_LAYOUT_ONLY_DELTA_CPU',tests_run=result.testsRun,success=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.time()-start,pins_before=before,pins_after=after,scope='SYNTHETIC_LAYOUT_NOT_REAL_MODEL_OR_NATIVE_STATE_QUALIFICATION',production_main_executed=False,actual_final51713_replayed=False,remote_executed=False,gpu_executed=False),sort_keys=True));raise SystemExit(0 if result.wasSuccessful() else 1)
