"""Independent read-only zero-Adam diagnostic identity, never a training release."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
SOURCE_ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006')
ROOT=SOURCE_ROOT/'numeric-diagnostic-3310-20261006'
SOURCE='90755a561abd239fbb23882939410e643bb913ae'
PROTOCOL='BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1'
RAW=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/runs/formal200-6a29642035ff40b2b7787de206f886c0-3307/checkpoints/diagnostics/formal-sampling-attempt_1/formal-first-sampling-batch.pt')
RAW_SHA='2544e34b622b59e5fadd802cb4d950e7d122073dac2fd1e2c51ff2d525e6c77c'
MODEL=SOURCE_ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
PHASE='numeric-zero-adam-replay'
REFERENCE=ROOT/'logs/reference3310.actual.json'
REFERENCE_SHA='058e52ab43a05f82f29e14849d8f03eaf0f0910b466f8ddd6aaaa403569e88ec'
WITNESS=ROOT/'logs/prefix-witness-v2.actual.json'
WITNESS_SHA='12ed8011eb1d94f0076551ce1e420db1b824cce0637ab8d097c022d8da1c6af1'
HELPER_SHA='ada012340292269b7aef3f6c3bd322067328b3dd92ba3ded18bb9d13585a3ce5'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):
 p=Path(path);require(p.is_file()and not p.is_symlink()and p.resolve()==p,'Canonical independent diagnostic record required');return json.loads(p.read_text())
def bind(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def validate_scope(a):
 for key in ('optimizer_writes','formal_outer_updates','scheduler_updates'):
  require(type(a.get(key))is int and a[key]==0,'Zero-Adam/formal/scheduler scope required')
 require(a.get('native_acceptance')is False and a.get('production_release')is False and type(a.get('allocation_gpu_count'))is int and a['allocation_gpu_count']==8 and type(a.get('probe_gpu_count'))is int and a['probe_gpu_count']==1,'Only8allocated/1compute diagnostic, no acceptance')
def driver_library_identity(directory=None,resolved_directory=None):
 directory=SOURCE_ROOT/'lib/driver' if directory is None else Path(directory)
 expected=Path('/dataset/shaoshitong/search-r1-rl/bapo/lib/driver')if resolved_directory is None else Path(resolved_directory)
 require(directory.is_dir()and directory.resolve()==expected,'Exact observed read-only driver directory alias required')
 library=directory/'libcuda.so';require(library.is_file(),'Actual libcuda.so link target required')
 resolved=library.resolve();stat=resolved.stat()
 return dict(directory=str(directory),resolved_directory=str(expected),library=str(library),resolved_path=str(resolved),bytes=stat.st_size,sha256=sha(resolved))
def verify_driver_library(record,require_env=False):
 require(record==driver_library_identity(),'Actual driver library path/bytes changed')
 if require_env:
  require(os.environ.get('TRITON_LIBCUDA_PATH')==record['directory']and os.environ.get('LIBRARY_PATH')==record['directory'],'Job-local Triton/CUDA linker environment required')
def verify(require_runtime=True,require_job=False):
 require(not sys.flags.optimize,'Optimized assertions forbidden')
 require(sys.platform=='linux'and os.uname().nodename=='h100-5'and sys.prefix==str(SOURCE_ROOT/'env'),'Actual target own R5 interpreter required')
 require(ROOT.resolve()==ROOT and not ROOT.is_symlink(),'Independent diagnostic ROOT required')
 require((ROOT/'source-expected.txt').read_text().strip()==SOURCE,'Diagnostic source marker changed')
 require(subprocess.check_output(['git','-C',str(SOURCE_ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(SOURCE_ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Frozen read-only training source differs')
 a=read(ROOT/'logs/diagnostic-approval.json')
 require(a.get('protocol')==PROTOCOL and a.get('status')=='ALLOW_ZERO_ADAM_NUMERIC_REPLAY','Separate diagnostic approval required; production/E1 receipts refused')
 require(a.get('root')==str(ROOT)and a.get('repo')==str(SOURCE_ROOT/'repo')and a.get('source_commit')==SOURCE and a.get('raw_batch')==str(RAW)and a.get('raw_sha256')==RAW_SHA and a.get('model')==str(MODEL),'Exact source/actual failed-run raw/model/root bindings required')
 validate_scope(a)
 driver=read(ROOT/'logs/driver-library.actual.json');require(a.get('driver_library')==bind(ROOT/'logs/driver-library.actual.json'),'Actual driver receipt differs');verify_driver_library(driver,require_env=require_job)
 manifest=read(ROOT/'logs/tools-manifest.json');require(sha(ROOT/'logs/tools-manifest.json')==a['tools_manifest_sha256']and manifest['source_commit']==SOURCE,'Independent diagnostic tools manifest differs')
 require(set(manifest['files'])=={p.name for p in (ROOT/'tools').iterdir()if p.is_file()},'Exact diagnostic tool inventory required')
 for name,digest in manifest['files'].items():require(sha(ROOT/'tools'/name)==digest,'Reviewed diagnostic code changed '+name)
 require(manifest['files']['numeric-probe.py']==a['numeric_probe_sha256']and sha(RAW)==RAW_SHA,'Numeric probe/actual raw bytes differ')
 plan=read(ROOT/'logs/numeric-cpu-plan.json');require(a.get('numeric_cpu_plan_sha256')==sha(ROOT/'logs/numeric-cpu-plan.json')and plan.get('status')=='CPU_ACTUAL_PREFIX_WITNESS_NOT_GPU'and plan.get('source_commit')==SOURCE and plan.get('numeric_probe_sha256')==manifest['files']['numeric-probe.py']and plan.get('reference_result_sha256')==REFERENCE_SHA and plan.get('actual_prefix_witness_sha256')==WITNESS_SHA,'Actual new numeric CPU plan binding required')
 require(manifest['files']['hf-probe-helpers.py']==HELPER_SHA==a['helper_sha256']and sha(REFERENCE)==REFERENCE_SHA==a['reference_result_sha256']and sha(WITNESS)==WITNESS_SHA==a['prefix_witness_sha256'],'Exact readonly actual reference/witness/helper inputs required')
 review=a['independent_review'];require(sha(review['path'])==review['sha256'],'Independent diagnostic review changed')
 r=read(review['path']);require(r.get('status')=='PASS_CODE_ONLY_ZERO_ADAM_DIAGNOSTIC_TOOLS_REVIEW'and r.get('mustfix')==[]and r.get('native_acceptance')is False and r.get('source_commit')==SOURCE and r.get('tools_manifest_sha256')==a['tools_manifest_sha256'],'Independent diagnostic code review unresolved')
 sys.path.insert(0,str(SOURCE_ROOT/'repo'))
 from search_r1.baselines.fresh200_contract import verify_official_model_assets,critical_source_hashes
 require(verify_official_model_assets(MODEL)=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44','Official model assets changed')
 pre=read(ROOT/'logs/diagnostic-cpu-precheck.json');require(sha(ROOT/'logs/diagnostic-cpu-precheck.json')==a['cpu_precheck_sha256']and pre.get('status')=='ACTUAL_ZERO_ADAM_CPU_CONTEXT_INPUT_PRECHECK_NOT_NATIVE'and pre.get('source_commit')==SOURCE and pre.get('raw_sha256')==RAW_SHA and pre.get('critical_source_sha256')==critical_source_hashes(),'Actual new CPU context/input precheck required')
 tests=read(ROOT/'logs/target-tests.actual.json');text=(ROOT/'logs/target-tests.actual.log').read_text()
 require(sha(ROOT/'logs/target-tests.actual.json')==a['target_tests_sha256']and tests.get('source_commit')==SOURCE and tests.get('host')=='h100-5'and type(tests.get('exit_code'))is int and tests['exit_code']==0 and tests.get('tools_manifest_sha256')==a['tools_manifest_sha256']and tests.get('raw_log_sha256')==sha(ROOT/'logs/target-tests.actual.log'),'Actual new diagnostic CPU test proof changed')
 require(tests.get('files')=={p.name:sha(p)for p in sorted((ROOT/'tests').glob('test_*.py'))}and len(tests['files'])==6,'Actual new diagnostic CPU suite bytes changed')
 require(re.search(r'^Ran 87 tests in [\d.]+s$',text,re.M)is not None and re.search(r'^OK$',text,re.M)is not None and not re.search(r'^FAILED|^OK \(|skipped=',text,re.M),'Actual new diagnostic Linux87 checks required')
 if require_runtime:
  b=a['runtime_executable'];require(b==bind(ROOT/'logs/runtime-executable.json'),'Independent actual runtime bytes differ')
  runtime=read(b['path']);exe=(SOURCE_ROOT/'env/bin/python').resolve()
  require(runtime.get('source_commit')==SOURCE and runtime.get('root')==str(ROOT)and runtime.get('sys_prefix')==str(SOURCE_ROOT/'env')and runtime.get('host')=='h100-5'and runtime.get('scope')=='ACTUAL_ZERO_ADAM_DIAGNOSTIC_INTERPRETER_NOT_NATIVE_ACCEPTANCE','New diagnostic runtime identity differs')
  require(runtime['executable']['resolved_path']==str(exe)and runtime['executable_sha256']==sha(exe),'Actual diagnostic executable bytes differ')
 if require_job:
  from launch_transaction import identity,scheduler
  receipt_path=ROOT/'scheduler-head'/f'{PHASE}-receipt.json';receipt=read(receipt_path)
  require(a.get('job_receipt_path')==str(receipt_path)and os.environ.get('BAPO_DIAGNOSTIC_RECEIPT')==str(receipt_path),'Exact new diagnostic receipt path required')
  require(receipt.get('source')==SOURCE and receipt.get('phase')==PHASE and str(receipt.get('job_id'))==os.environ.get('SLURM_JOB_ID'),'Own actual diagnostic allocation receipt required')
  identity(receipt,scheduler(receipt['job_id']))
  run=ROOT/'runs'/f"{PHASE}-{receipt['nonce']}-{receipt['job_id']}"
  require(os.environ.get('BAPO_RUN_ID')==str(run),'Actual nonce/job diagnostic run differs')
 return a
