"""Actual own-child CPU guard/HF schema tests; never GPU execution."""
from diagnostic_contract import *
import time
require(not sys.flags.optimize and sys.platform=='linux'and os.uname().nodename=='h100-5'and sys.prefix==str(SOURCE_ROOT/'env'),'Actual target diagnostic CPU interpreter required')
require(sha(ROOT/'hf-probe.py')==sha(ROOT/'tools/hf-probe.py'),'Tested and executed HF probe must be exact same bytes')
manifest=read(ROOT/'logs/tools-manifest.json')
for name,digest in manifest['files'].items():require(sha(ROOT/'tools'/name)==digest,'Diagnostic code changed')
files={p.name:sha(p)for p in sorted((ROOT/'tests').glob('test_*.py'))};require(len(files)==5,'Exact five diagnostic CPU suites required')
log=ROOT/'logs/target-tests.actual.log';start=time.monotonic()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'tools')+':'+str(SOURCE_ROOT/'repo'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
env.pop('PYTHONOPTIMIZE',None)
with log.open('x')as stream:
 result=subprocess.run([str(SOURCE_ROOT/'env/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
record=dict(source_commit=SOURCE,host=os.uname().nodename,exit_code=result.returncode,scope='ACTUAL_ZERO_ADAM_DIAGNOSTIC_LINUX_CPU_TESTS_NOT_GPU',tools_manifest_sha256=sha(ROOT/'logs/tools-manifest.json'),files=files,raw_log_sha256=sha(log),elapsed_seconds=time.monotonic()-start)
with (ROOT/'logs/target-tests.actual.json').open('x')as f:json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps(record));raise SystemExit(result.returncode)
