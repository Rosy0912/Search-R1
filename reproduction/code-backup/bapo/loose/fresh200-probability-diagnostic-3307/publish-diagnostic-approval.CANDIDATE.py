"""Publish only independently reviewed zero-Adam diagnostic permission."""
import argparse,json,secrets,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from diagnostic_contract import *
def build(review_path,expected_review_sha,expected_tools_sha):
 require(not sys.flags.optimize and sys.platform=='linux'and os.uname().nodename=='h100-5'and sys.prefix==str(SOURCE_ROOT/'env'),'Actual target own R5 environment required')
 for value in (expected_review_sha,expected_tools_sha):require(re.fullmatch('[0-9a-f]{64}',value)is not None,'Trusted independent SHA inputs required')
 require(sha(review_path)==expected_review_sha and sha(ROOT/'logs/tools-manifest.json')==expected_tools_sha,'Trusted independent tool/review bytes changed')
 review=read(review_path);require(review.get('status')=='PASS_CODE_ONLY_ZERO_ADAM_DIAGNOSTIC_TOOLS_REVIEW'and review.get('mustfix')==[] and review.get('native_acceptance')is False and review.get('source_commit')==SOURCE and review.get('tools_manifest_sha256')==expected_tools_sha,'Independent zero-Adam code review unresolved')
 manifest=read(ROOT/'logs/tools-manifest.json')
 require(manifest['source_commit']==SOURCE and set(manifest['files'])=={p.name for p in (ROOT/'tools').iterdir()if p.is_file()},'Exact independent diagnostic inventory required')
 for name,digest in manifest['files'].items():require(sha(ROOT/'tools'/name)==digest,'Reviewed diagnostic file changed')
 require(sha(RAW)==RAW_SHA and subprocess.check_output(['git','-C',str(SOURCE_ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(SOURCE_ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Actual immutable raw/source differs')
 pre=read(ROOT/'logs/diagnostic-cpu-precheck.json');runtime=read(ROOT/'logs/runtime-executable.json')
 require(pre['status']=='ACTUAL_ZERO_ADAM_CPU_CONTEXT_INPUT_PRECHECK_NOT_NATIVE'and pre['source_commit']==SOURCE and pre['raw_sha256']==RAW_SHA and pre['rows']==2560 and pre['audit']['context_verified']is True,'New actual diagnostic CPU2560 context gate required')
 require(runtime['source_commit']==SOURCE and runtime['root']==str(ROOT)and runtime['sys_prefix']==str(SOURCE_ROOT/'env')and runtime['scope']=='ACTUAL_ZERO_ADAM_DIAGNOSTIC_INTERPRETER_NOT_NATIVE_ACCEPTANCE','New actual diagnostic runtime required')
 tests=read(ROOT/'logs/target-tests-v2.actual.json');test_log=ROOT/'logs/target-tests-v2.actual.log'
 require(tests.get('source_commit')==SOURCE and tests.get('host')=='h100-5'and type(tests.get('exit_code'))is int and tests['exit_code']==0 and tests.get('scope')=='ACTUAL_ZERO_ADAM_DIAGNOSTIC_LINUX_CPU_TESTS_NOT_GPU'and tests.get('tools_manifest_sha256')==expected_tools_sha and tests.get('raw_log_sha256')==sha(test_log),'Actual new diagnostic Linux CPU tests required')
 require(tests.get('files')=={p.name:sha(p)for p in sorted((ROOT/'tests').glob('test_*.py'))}and len(tests['files'])==5,'Exact new diagnostic five CPU test files required')
 text=test_log.read_text();require(re.search(r'^Ran 80 tests in [\d.]+s$',text,re.M)is not None and re.search(r'^OK$',text,re.M)is not None and not re.search(r'^FAILED|^OK \(|skipped=',text,re.M),'Actual Linux80 zero-skip diagnostic checks required')
 return dict(protocol=PROTOCOL,status='ALLOW_ZERO_ADAM_PROBABILITY_REPLAY',source_commit=SOURCE,root=str(ROOT),repo=str(SOURCE_ROOT/'repo'),raw_batch=str(RAW),raw_sha256=RAW_SHA,model=str(MODEL),nonce=secrets.token_hex(16),hf_probe_sha256=manifest['files']['hf-probe.py'],tools_manifest_sha256=expected_tools_sha,independent_review=dict(path=str(review_path),sha256=expected_review_sha),cpu_precheck_sha256=sha(ROOT/'logs/diagnostic-cpu-precheck.json'),runtime_executable=bind(ROOT/'logs/runtime-executable.json'),target_tests_sha256=sha(ROOT/'logs/target-tests-v2.actual.json'),job_receipt_path=str(ROOT/'scheduler-head'/f'{PHASE}-receipt.json'),allocation_gpu_count=8,probe_gpu_count=1,optimizer_writes=0,formal_outer_updates=0,scheduler_updates=0,native_acceptance=False,production_release=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--expected-review-sha',required=True);p.add_argument('--expected-tools-sha',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args();value=build(a.review,a.expected_review_sha,a.expected_tools_sha)
 if a.publish:
  from launch_transaction import durable
  durable(ROOT/'logs/diagnostic-approval.json',value)
 else:print(json.dumps(value,allow_nan=False))
