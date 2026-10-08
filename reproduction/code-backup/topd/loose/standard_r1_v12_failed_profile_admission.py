"""Specific failed3197 permits only new profile2-v12; never formal admission."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,re
TERMINAL_SHA='fb8f493c69b1f68aa9b1d19437dac74831d7c99e277ff6fcf0a7affaee9fe9d3'
CLEANUP_SHA='51deb7e3645db562cc3f76adacc3a0b1a22d099106f8d03bd8261653165c5a91'
FAILED_RESULT_SHA='e44536779e4133f3a8d426f949fb291a6c0cb805d67531d2114b98c5c319eee7'
TOOLS_MANIFEST='abf316a0018dd36c09a46162e8219033b0c66f53b937498c6cd826cac71a0ef8' # Immutable source-only freeze; actual admission still required
NATIVE_TOOLS_COUNT=87 # Frozen native expectation, still requires actual bound result/log/review
SOURCE_MANIFEST='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be'
NATIVE26_SHA='8ecb69cc6bfc5ac8970625a2e7d379b3e49201937fe117b4aa3d7e47b05377d3'
def need(ok,message):
 if not ok:raise ValueError(message)
def validate_failed3197(c,g,attempt,root):
 need(TOOLS_MANIFEST is not None and NATIVE_TOOLS_COUNT is not None,'PENDING immutable V12 tools and actual native CPU result/review; execution forbidden')
 root=pathlib.Path(root)
 def ref(r):
  p=pathlib.Path(r['path']);need(not p.is_symlink() and root/'logs' in p.resolve().parents,'Ownlogs raw required')
  raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==r['sha256'],'Actual raw SHA drift');return json.loads(raw)
 need(c['mode']=='profile2' and attempt=='profile2-v12' and g.get('allowed_attempt')==attempt and c['expected_outer_steps']==2,'3197 failure permits named diagnostic profile2-v12 only')
 need(g['prior_failed_job_id']==3197,'Wrong actual failure')
 need(c.get('cuda_allocator')=={'environment_variable':'PYTORCH_CUDA_ALLOC_CONF','value':'backend:native,max_split_size_mb:128','backend':'native'},'Only immutable V12 allocator engineering policy')
 need(c['tools']==str(root/'tools/search-r1-32b-grpo100-tools-v12') and c['tools_manifest']['path']==c['tools']+'/files.json' and c['tools_manifest']['sha256']==TOOLS_MANIFEST,'Only exact V12 tools')
 need(c['candidate']==str(root/'tools/search-r1-32b-grpo100-source-v4') and c['source_manifest']['path']==c['candidate']+'/r1-files.json' and c['source_manifest']['sha256']==SOURCE_MANIFEST,'Only exact streamed sourceV4')
 need(g['actual_native_passed']==NATIVE_TOOLS_COUNT and g['actual_native_skipped']==0 and g['tools_manifest']==c['tools_manifest'] and g['source_manifest']==c['source_manifest'],'Actual V12 expected-count zero-skip new source admission required')
 review=ref(g['independent_review'])
 need(review['status']=='PASS' and review['scope']=='SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_REVIEW' and review['reviewed_static_recipe_sha256']==g['reviewed_static_recipe_sha256'],'Actual config/source independent review differs')
 for key in ('actual_native_passed','actual_native_skipped','actual_source_native_job_id','tools_manifest','source_manifest','native_tools_result','native_tools_log','streamed_source_native26','prior_failed_result','source_device_correctness'):
  need(review[key]==g[key],'Independent review native/source identity differs '+key)
 need(g['telemetry_policy']==c['telemetry_policy'] and review['telemetry_policy']==c['telemetry_policy'],'Independent peerWindowV5 policy binding differs')
 from source_binding import check as check_latest_source,check_device_correctness
 check_latest_source(c,read_native=True)
 need(g['source_device_correctness']==c['source_device_correctness'],'Actual CUDA device proof binding differs')
 check_device_correctness(c)
 from window_guardian import validate_policy_review
 validate_policy_review(root,c['telemetry_policy'])
 need(g['source_streamed_adamw_native_job_id']==3200 and g['source_streamed_adamw_native_result_sha256']==NATIVE26_SHA,'Wrong streamed source native identity')
 need(g['streamed_source_native26']['sha256']==NATIVE26_SHA,'Only actual source3200 native26 receipt')
 n=ref(g['streamed_source_native26']);need(n['job']=='3200' and n['tests']==26 and n['skips']==n['failures']==n['exit_code']==0 and n['cuda_initialized'] is False and n['source_manifest_sha256']==SOURCE_MANIFEST and n['torch'].split('+')[0]=='2.4.0','Actual source3200 twentysix contracts failed')
 n=ref(g['native_tools_result']);need(n['scope']=='STANDARD_R1_SOURCE_CPU_CONTRACTS_ONLY_NO_GPU_PROOF' and str(n['job'])==str(g['actual_source_native_job_id']) and type(g['actual_source_native_job_id']) is int and g['actual_source_native_job_id']>0 and n['returncode']==0 and n['expected_count']==NATIVE_TOOLS_COUNT and n['zero_skip'] is True and n['source_manifests']=={'source':SOURCE_MANIFEST,'tools':TOOLS_MANIFEST} and n['cuda_visible_devices']=='','Actual V12 native V12 expected-count incomplete/wrong source')
 raw_log=pathlib.Path(g['native_tools_log']['path']);need(root/'logs' in raw_log.resolve().parents and not raw_log.is_symlink() and hashlib.sha256(raw_log.read_bytes()).hexdigest()==g['native_tools_log']['sha256']==n['log_sha256'],'Actual native V12 expected-count full log differs')
 text=raw_log.read_text();need(re.findall(r'\b(\d+) passed\b',text)==[str(NATIVE_TOOLS_COUNT)] and not re.search(r'\b\d+ (?:failed|skipped|error)',text),'Actual native V12 expected-count log not unique zero-skip PASS')
 need(g['prior_terminal']['sha256']==TERMINAL_SHA and g['prior_terminal']['path']==str(root/'logs/standard3197-captured-controller-terminal.json'),'Only captured3197 terminal')
 t=ref(g['prior_terminal']);need(t['job_id']==3197 and t['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',t['stdout'])==['3197'] and re.search(r'\bJobState=FAILED\b',t['stdout']) and re.search(r'\bExitCode=1:0\b',t['stdout']),'Actual3197 FAILED1:0 required')
 need(g['prior_run_cleanup']['sha256']==CLEANUP_SHA and g['prior_failed_result']['sha256']==FAILED_RESULT_SHA,'Only actual failed3197 raw result/cleanup')
 failed=ref(g['prior_failed_result']);need(str(failed['job'])=='3197' and failed['status']=='FAILED' and failed['cleanup_passed'] is False and failed['error'] and failed['cleanup']==g['prior_run_cleanup'],'Failed3197 may not be relabelled or exported as approval')
 cleanup=ref(g['prior_run_cleanup']);need(cleanup['cleanup']['actual_cgroup_empty'] is True and cleanup['parent_cuda_initialized'] is False,'Actual prior own cgroup/CPU supervisor cleanup incomplete')
 for key in ('own_processes_remaining','own_gpu_records_remaining','original_gpu_baseline_missing','baseline_identity_missing','parent_gpu_records'):need(cleanup[key]==[],'Actual prior cleanup incomplete '+key)
 # Historical errors are retained. They are not erased or treated as profile PASS.
 return cleanup
