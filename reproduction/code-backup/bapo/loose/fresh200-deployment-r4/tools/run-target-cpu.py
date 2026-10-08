"""Actual target CPU tests and all-row audit; no GPU execution."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r4-20261006')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
require(sys.platform=='linux' and os.uname().nodename=='h100-5','wrong target')
require(sys.prefix==str(R/'env'),'parent interpreter prefix mismatch')
sys.path.insert(0,str(R/'repo'))
import search_r1.baselines.fresh200_contract as parent_contract
require(Path(parent_contract.__file__).resolve()==R/'repo/search_r1/baselines/fresh200_contract.py','parent contract source origin mismatch')
source=(R/'source-expected.txt').read_text().strip()
env=dict(os.environ,PYTHONPATH=str(R/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SPT_NOENV='1')
env.pop('PYTHONOPTIMIZE',None)
started=time.monotonic();log=R/'logs/target-cpu-gate.raw.log'
commands=[['reproduction/h20/check_environment.py'],['-m','pytest','-q','tests/baselines/test_fresh200_contract.py','tests/utils/test_fsdp_optimizer_transfer.py','tests/baselines/test_fresh200_skip_contract.py','tests/baselines/test_optimizer_transaction.py','tests/baselines/test_gradient_structure_collective.py','tests/baselines/test_gradient_numerics.py','tests/baselines/test_numeric_safety.py','tests/baselines/test_fresh200_driver.py','tests/baselines/test_fresh200_driver_fit.py','tests/baselines/test_fresh200_gate_skip.py','tests/baselines/test_fresh200_sampling_evidence.py','tests/baselines/test_nonfinite_sampling_actor.py','tests/baselines/test_fresh200_native_evidence.py','tests/baselines/test_fresh200_policy_safety.py','tests/baselines/test_fresh200_bapo_sampling_chain.py','tests/test_fresh200_sampling_generation.py','tests/test_fresh200_sampling_trace.py']]
exit_code=0
with log.open('x') as stream:
 for args in commands:
  stream.write(json.dumps(dict(command=args))+'\n');stream.flush()
  result=subprocess.run([str(R/'env/bin/python'),'-B',*args],cwd=R/'repo',env=env,stdout=stream,stderr=subprocess.STDOUT)
  exit_code=result.returncode
  if exit_code:break
if not exit_code:
 audit=R/'logs/context-budget-audit.raw.log'
 with audit.open('x') as out,(R/'logs/context-budget-audit.progress.log').open('x') as err:
  result=subprocess.run([str(R/'env/bin/python'),'-B',str(R/'tools/context-budget-audit.py')],cwd=R/'repo',env=env,stdout=out,stderr=err)
 exit_code=result.returncode
 if not exit_code:
  lines=audit.read_text().splitlines();record=json.loads(lines[-1]);require(record['source_commit']==source,'audit source mismatch')
  (R/'logs/context-budget-audit.json').write_text(json.dumps(record,indent=2))
from search_r1.baselines.fresh200_contract import critical_source_hashes
record=dict(source_commit=source,exit_code=exit_code,platform=sys.platform,host=os.uname().nodename,
 log_path=str(log),log_sha256=sha(log),tools_manifest_sha256=sha(R/'logs/tools-manifest.json'),critical_source_sha256=critical_source_hashes(),
 context_audit_sha256=sha(R/'logs/context-budget-audit.json') if not exit_code else None,elapsed_seconds=time.monotonic()-started,scope='ACTUAL_CPU_IMPORT_TEST_CONTEXT_GATE_NOT_NATIVE_GPU')
with (R/'logs/target-cpu-gate.json').open('x')as stream:json.dump(record,stream,indent=2)
print(json.dumps(record));raise SystemExit(exit_code)
