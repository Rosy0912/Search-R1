import hashlib,importlib.machinery,json,os,subprocess,sys,time,types,unittest
from pathlib import Path
from unittest import mock
H=Path(__file__).resolve().parents[2];P=H/'final200-raw-replay-consumer-candidate';REPO=H.parent/'opsd-final200-paired-eval-source'
expected={'replay_final200_raw.py':'705c286dc7ac51353d4eb4de3ec21439e4459bd4b14df0c313af1ac9a31ca28a','test_raw_replay.py':'9b6a56cc09dcd723f985c9867a0160988e906636a6fac8a884b35229fb090e80','test_review_delta.py':'f963d6bcb0167f697242986f69aa9b8eff795c174c104b793c36d3081f558a8b'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,v in expected.items():assert sha(P/n)==v
pins=json.loads((P/'FROZEN_SOURCE_INTERFACE_PINS.json').read_bytes());paths=[P/n for n in expected]+[P/'FROZEN_SOURCE_INTERFACE_PINS.json',P/'READINESS_V2.json',P/'V2_MINIMAL_RUNTIME_DIFF.patch',H/'final200-paired-launch-candidate/paired_eval_contract.py']+[REPO/n for n in pins['files']]
for n,v in pins['files'].items():assert sha(REPO/n)==v
before={str(p):sha(p) for p in paths};git_before=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True);assert not git_before
compiled={};original_get_code=importlib.machinery.SourceFileLoader.get_code
def exact_get_code(loader,fullname):
 p=Path(loader.path).resolve()
 if p.is_relative_to(REPO) or p.is_relative_to(P) or p==paths[6]:
  raw=p.read_bytes();compiled[str(p)]=hashlib.sha256(raw).hexdigest();return compile(raw,str(p),'exec',dont_inherit=True)
 return original_get_code(loader,fullname)
start=time.time()
with mock.patch.object(importlib.machinery.SourceFileLoader,'get_code',exact_get_code):
 sys.path.insert(0,str(P));modules=[]
 for name in ['test_raw_replay','test_review_delta']:
  p=P/(name+'.py');m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),m.__dict__);modules.append(m)
 subject=modules[0].c
 class IndependentOriginalMalformedNPZ(unittest.TestCase):
  def test_exact_910_byte_original_rejected_before_numpy_load(self):
   p=H/'independent-review/raw-replay-npy-header-boundary/small-malformed-original.npz';assert sha(p)=='639b903ec4f00f7ac0f65523efcd555cdd5cdac48e105502c3fad60b180b1aad'
   with mock.patch.object(subject.np,'load',side_effect=AssertionError('allocation forbidden')) as loader:
    with self.assertRaisesRegex(ValueError,'NPY shape exceeds real scoring tensor budget'):subject.npz_arrays(subject.Originals(os.getuid()),str(p),sha(p))
    loader.assert_not_called()
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules]+[unittest.defaultTestLoader.loadTestsFromTestCase(IndependentOriginalMalformedNPZ)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
after={str(p):sha(p) for p in paths};git_after=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
assert before==after and git_before==git_after==''
record=dict(kind='INDEPENDENT_RAW_REPLAY_V2_LOCAL_CPU_SOFTWARE_TESTS',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),elapsed_seconds=time.time()-start,pins_before=before,pins_after=after,compiled_actual_sources=compiled,candidate_source_compile='EXACT_VERIFIED_BYTES_NO_PYC',production_main_executed=False,actual_final51713_replayed=False,remote_executed=False,gpu_executed=False,cuda_is_initialized=modules[0].torch.cuda.is_initialized())
print(json.dumps(record,sort_keys=True));assert not record['cuda_is_initialized'];raise SystemExit(0 if result.wasSuccessful() else 1)
