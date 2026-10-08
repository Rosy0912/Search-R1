"""Specific failed3176 permits only new profile2-v7; never formal admission."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,re
TERMINAL_SHA='4dd682d726e3027a108c06baf4f1dafe3737dcaa1d3b32057abdc8c7ad1fd0a1'
CLEANUP_SHA='13903bf0d544fa0dd7964e8af259400a92f1f91471069e113728897b714740f1'
FAILED_RESULT_SHA='138f06ce3c07336cc1b748ec512196e059db70cceca250a32af7b179c8937b0a'
TOOLS_MANIFEST='7a379f1a30308b75caaf380d28d74125bd90b9639c173b716c84d4a3cb91e86a'
NATIVE_TOOLS_COUNT=56 # Frozen native expectation, still requires actual bound result/log/review
SOURCE_MANIFEST='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
NATIVE17_SHA='cdbc2f1374057e61446735f9c7a879bc2a9b2538297b52ce8691d351c0285eb0'
def need(ok,message):
 if not ok:raise ValueError(message)
def validate_failed3176(c,g,attempt,root):
 need(TOOLS_MANIFEST is not None and NATIVE_TOOLS_COUNT is not None,'PENDING immutable V7 tools and actual native CPU result/review; execution forbidden')
 root=pathlib.Path(root)
 def ref(r):
  p=pathlib.Path(r['path']);need(not p.is_symlink() and root/'logs' in p.resolve().parents,'Ownlogs raw required')
  raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==r['sha256'],'Actual raw SHA drift');return json.loads(raw)
 need(c['mode']=='profile2' and attempt=='profile2-v7' and g.get('allowed_attempt')==attempt and c['expected_outer_steps']==2,'3176 failure permits named diagnostic profile2-v7 only')
 need(g['prior_failed_job_id']==3176,'Wrong actual failure')
 need(c['tools']==str(root/'tools/search-r1-32b-grpo100-tools-v7') and c['tools_manifest']['path']==c['tools']+'/files.json' and c['tools_manifest']['sha256']==TOOLS_MANIFEST,'Only exact V7 tools')
 need(c['candidate']==str(root/'tools/search-r1-32b-grpo100-source-v2') and c['source_manifest']['path']==c['candidate']+'/r1-files.json' and c['source_manifest']['sha256']==SOURCE_MANIFEST,'Only exact streamed sourceV2')
 need(g['actual_native_passed']==NATIVE_TOOLS_COUNT and g['actual_native_skipped']==0 and g['tools_manifest']==c['tools_manifest'] and g['source_manifest']==c['source_manifest'],'Actual V7 expected-count zero-skip new source admission required')
 review=ref(g['independent_review'])
 need(review['status']=='PASS' and review['scope']=='SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_REVIEW' and review['reviewed_static_recipe_sha256']==g['reviewed_static_recipe_sha256'],'Actual config/source independent review differs')
 for key in ('actual_native_passed','actual_native_skipped','actual_source_native_job_id','tools_manifest','source_manifest','native_tools_result','native_tools_log','streamed_source_native17','prior_failed_result'):
  need(review[key]==g[key],'Independent review native/source identity differs '+key)
 need(g['source_streamed_adamw_native_job_id']==3172 and g['source_streamed_adamw_native_result_sha256']==NATIVE17_SHA,'Wrong streamed source native identity')
 need(g['streamed_source_native17']['sha256']==NATIVE17_SHA,'Only actual source3172 native17 receipt')
 n=ref(g['streamed_source_native17']);need(n['job']=='3172' and n['tests']==17 and n['skips']==n['failures']==n['exit_code']==0 and n['cuda_initialized'] is False and n['source_manifest_sha256']==SOURCE_MANIFEST and n['torch'].split('+')[0]=='2.4.0','Actual source3172 seventeen contracts failed')
 n=ref(g['native_tools_result']);need(n['scope']=='STANDARD_R1_SOURCE_CPU_CONTRACTS_ONLY_NO_GPU_PROOF' and str(n['job'])==str(g['actual_source_native_job_id']) and type(g['actual_source_native_job_id']) is int and g['actual_source_native_job_id']>0 and n['returncode']==0 and n['expected_count']==NATIVE_TOOLS_COUNT and n['zero_skip'] is True and n['source_manifests']=={'source':SOURCE_MANIFEST,'tools':TOOLS_MANIFEST} and n['cuda_visible_devices']=='','Actual V7 native V7 expected-count incomplete/wrong source')
 raw_log=pathlib.Path(g['native_tools_log']['path']);need(root/'logs' in raw_log.resolve().parents and not raw_log.is_symlink() and hashlib.sha256(raw_log.read_bytes()).hexdigest()==g['native_tools_log']['sha256']==n['log_sha256'],'Actual native V7 expected-count full log differs')
 text=raw_log.read_text();need(re.findall(r'\b(\d+) passed\b',text)==[str(NATIVE_TOOLS_COUNT)] and not re.search(r'\b\d+ (?:failed|skipped|error)',text),'Actual native V7 expected-count log not unique zero-skip PASS')
 need(g['prior_terminal']['sha256']==TERMINAL_SHA and g['prior_terminal']['path']==str(root/'logs/standard3176-captured-controller-terminal.json'),'Only captured3176 terminal')
 t=ref(g['prior_terminal']);need(t['job_id']==3176 and t['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',t['stdout'])==['3176'] and re.search(r'\bJobState=FAILED\b',t['stdout']) and re.search(r'\bExitCode=1:0\b',t['stdout']),'Actual3176 FAILED1:0 required')
 need(g['prior_run_cleanup']['sha256']==CLEANUP_SHA and g['prior_failed_result']['sha256']==FAILED_RESULT_SHA,'Only actual failed3176 raw result/cleanup')
 failed=ref(g['prior_failed_result']);need(str(failed['job'])=='3176' and failed['status']=='FAILED' and failed['error'] and failed['cleanup']==g['prior_run_cleanup'],'Failed3176 may not be relabelled or exported as approval')
 cleanup=ref(g['prior_run_cleanup']);need(cleanup['cleanup']['actual_cgroup_empty'] is True and cleanup['parent_cuda_initialized'] is False,'Actual prior own cgroup/CPU supervisor cleanup incomplete')
 for key in ('own_processes_remaining','own_gpu_records_remaining','original_gpu_baseline_missing','baseline_identity_missing','parent_gpu_records'):need(cleanup[key]==[],'Actual prior cleanup incomplete '+key)
 # Historical errors are retained. They are not erased or treated as profile PASS.
 return cleanup
