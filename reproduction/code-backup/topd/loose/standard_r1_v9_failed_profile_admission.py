"""Specific failed3188 permits only new profile2-v9; never formal admission."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,re
TERMINAL_SHA='a660f8101c4d940d13bab43ca32dd8f0f4b1734ef34b58f76d197513711480bc'
CLEANUP_SHA='8b7b06f780469a5a33b264082d709d2763a911c4d1dea29247119dd23afb0688'
FAILED_RESULT_SHA='358b2286ed69056a2b6a2dceb7de5353846e75ec9cb2b589dbfcd4fbcb409fdb'
TOOLS_MANIFEST='ad422ea17014180a88500446faf73175908b3c2fad9a1c8bd50e071ca51fa68d'
NATIVE_TOOLS_COUNT=76 # Frozen native expectation, still requires actual bound result/log/review
SOURCE_MANIFEST='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
NATIVE17_SHA='cdbc2f1374057e61446735f9c7a879bc2a9b2538297b52ce8691d351c0285eb0'
def need(ok,message):
 if not ok:raise ValueError(message)
def validate_failed3188(c,g,attempt,root):
 need(TOOLS_MANIFEST is not None and NATIVE_TOOLS_COUNT is not None,'PENDING immutable V9 tools and actual native CPU result/review; execution forbidden')
 root=pathlib.Path(root)
 def ref(r):
  p=pathlib.Path(r['path']);need(not p.is_symlink() and root/'logs' in p.resolve().parents,'Ownlogs raw required')
  raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==r['sha256'],'Actual raw SHA drift');return json.loads(raw)
 need(c['mode']=='profile2' and attempt=='profile2-v9' and g.get('allowed_attempt')==attempt and c['expected_outer_steps']==2,'3188 failure permits named diagnostic profile2-v9 only')
 need(g['prior_failed_job_id']==3188,'Wrong actual failure')
 need(c.get('cuda_allocator')=={'environment_variable':'PYTORCH_CUDA_ALLOC_CONF','value':'backend:native,max_split_size_mb:128','backend':'native'},'Only immutable V9 allocator engineering policy')
 need(c['tools']==str(root/'tools/search-r1-32b-grpo100-tools-v9') and c['tools_manifest']['path']==c['tools']+'/files.json' and c['tools_manifest']['sha256']==TOOLS_MANIFEST,'Only exact V9 tools')
 need(c['candidate']==str(root/'tools/search-r1-32b-grpo100-source-v2') and c['source_manifest']['path']==c['candidate']+'/r1-files.json' and c['source_manifest']['sha256']==SOURCE_MANIFEST,'Only exact streamed sourceV2')
 need(g['actual_native_passed']==NATIVE_TOOLS_COUNT and g['actual_native_skipped']==0 and g['tools_manifest']==c['tools_manifest'] and g['source_manifest']==c['source_manifest'],'Actual V9 expected-count zero-skip new source admission required')
 review=ref(g['independent_review'])
 need(review['status']=='PASS' and review['scope']=='SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_REVIEW' and review['reviewed_static_recipe_sha256']==g['reviewed_static_recipe_sha256'],'Actual config/source independent review differs')
 for key in ('actual_native_passed','actual_native_skipped','actual_source_native_job_id','tools_manifest','source_manifest','native_tools_result','native_tools_log','streamed_source_native17','prior_failed_result'):
  need(review[key]==g[key],'Independent review native/source identity differs '+key)
 need(g['telemetry_policy']==c['telemetry_policy'] and review['telemetry_policy']==c['telemetry_policy'],'Independent peerWindowV5 policy binding differs')
 from window_guardian import validate_policy_review
 validate_policy_review(root,c['telemetry_policy'])
 need(g['source_streamed_adamw_native_job_id']==3172 and g['source_streamed_adamw_native_result_sha256']==NATIVE17_SHA,'Wrong streamed source native identity')
 need(g['streamed_source_native17']['sha256']==NATIVE17_SHA,'Only actual source3172 native17 receipt')
 n=ref(g['streamed_source_native17']);need(n['job']=='3172' and n['tests']==17 and n['skips']==n['failures']==n['exit_code']==0 and n['cuda_initialized'] is False and n['source_manifest_sha256']==SOURCE_MANIFEST and n['torch'].split('+')[0]=='2.4.0','Actual source3172 seventeen contracts failed')
 n=ref(g['native_tools_result']);need(n['scope']=='STANDARD_R1_SOURCE_CPU_CONTRACTS_ONLY_NO_GPU_PROOF' and str(n['job'])==str(g['actual_source_native_job_id']) and type(g['actual_source_native_job_id']) is int and g['actual_source_native_job_id']>0 and n['returncode']==0 and n['expected_count']==NATIVE_TOOLS_COUNT and n['zero_skip'] is True and n['source_manifests']=={'source':SOURCE_MANIFEST,'tools':TOOLS_MANIFEST} and n['cuda_visible_devices']=='','Actual V9 native V9 expected-count incomplete/wrong source')
 raw_log=pathlib.Path(g['native_tools_log']['path']);need(root/'logs' in raw_log.resolve().parents and not raw_log.is_symlink() and hashlib.sha256(raw_log.read_bytes()).hexdigest()==g['native_tools_log']['sha256']==n['log_sha256'],'Actual native V9 expected-count full log differs')
 text=raw_log.read_text();need(re.findall(r'\b(\d+) passed\b',text)==[str(NATIVE_TOOLS_COUNT)] and not re.search(r'\b\d+ (?:failed|skipped|error)',text),'Actual native V9 expected-count log not unique zero-skip PASS')
 need(g['prior_terminal']['sha256']==TERMINAL_SHA and g['prior_terminal']['path']==str(root/'logs/standard3188-captured-controller-terminal.json'),'Only captured3188 terminal')
 t=ref(g['prior_terminal']);need(t['job_id']==3188 and t['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',t['stdout'])==['3188'] and re.search(r'\bJobState=FAILED\b',t['stdout']) and re.search(r'\bExitCode=1:0\b',t['stdout']),'Actual3188 FAILED1:0 required')
 need(g['prior_run_cleanup']['sha256']==CLEANUP_SHA and g['prior_failed_result']['sha256']==FAILED_RESULT_SHA,'Only actual failed3188 raw result/cleanup')
 failed=ref(g['prior_failed_result']);need(str(failed['job'])=='3188' and failed['status']=='FAILED' and failed['cleanup_passed'] is False and failed['error'] and failed['cleanup']==g['prior_run_cleanup'],'Failed3188 may not be relabelled or exported as approval')
 cleanup=ref(g['prior_run_cleanup']);need(cleanup['cleanup']['actual_cgroup_empty'] is True and cleanup['parent_cuda_initialized'] is False,'Actual prior own cgroup/CPU supervisor cleanup incomplete')
 for key in ('own_processes_remaining','own_gpu_records_remaining','original_gpu_baseline_missing','baseline_identity_missing','parent_gpu_records'):need(cleanup[key]==[],'Actual prior cleanup incomplete '+key)
 # Historical errors are retained. They are not erased or treated as profile PASS.
 return cleanup
