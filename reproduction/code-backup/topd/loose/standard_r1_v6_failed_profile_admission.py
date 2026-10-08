"""Specific failed3170 permits only new profile2-v6; never formal admission."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,re
TERMINAL_SHA='db636febd85c9866e3046e5027b105b1963b94baf28801e8c492990b696cb52d'
CLEANUP_SHA='38bb81b87d073059fa9071c5eb1756d9de4ea7bbcc261e298fd912d6a7bcc48b'
FAILED_RESULT_SHA='8366b7dde23932edb160742c45e8744289c8897a115f255c93c5e5481fa0ac15'
TOOLS_MANIFEST='cf5e43abae51233fc4c9857de01176d987c553cc4a756be29e375cd3e04fb80c'
SOURCE_MANIFEST='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
NATIVE17_SHA='cdbc2f1374057e61446735f9c7a879bc2a9b2538297b52ce8691d351c0285eb0'
def need(ok,message):
 if not ok:raise ValueError(message)
def validate_failed3170(c,g,attempt,root):
 root=pathlib.Path(root)
 def ref(r):
  p=pathlib.Path(r['path']);need(not p.is_symlink() and root/'logs' in p.resolve().parents,'Ownlogs raw required')
  raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==r['sha256'],'Actual raw SHA drift');return json.loads(raw)
 need(c['mode']=='profile2' and attempt=='profile2-v6' and g.get('allowed_attempt')==attempt and c['expected_outer_steps']==2,'3170 failure permits named diagnostic profile2-v6 only')
 need(g['prior_failed_job_id']==3170,'Wrong actual failure')
 need(c['tools']==str(root/'tools/search-r1-32b-grpo100-tools-v6') and c['tools_manifest']['path']==c['tools']+'/files.json' and c['tools_manifest']['sha256']==TOOLS_MANIFEST,'Only exact V6 tools')
 need(c['candidate']==str(root/'tools/search-r1-32b-grpo100-source-v2') and c['source_manifest']['path']==c['candidate']+'/r1-files.json' and c['source_manifest']['sha256']==SOURCE_MANIFEST,'Only exact streamed sourceV2')
 need(g['actual_native_passed']==43 and g['actual_native_skipped']==0 and g['tools_manifest']==c['tools_manifest'] and g['source_manifest']==c['source_manifest'],'Actual43 zero-skip new source admission required')
 review=ref(g['independent_review'])
 need(review['status']=='PASS' and review['scope']=='SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_REVIEW' and review['reviewed_static_recipe_sha256']==g['reviewed_static_recipe_sha256'],'Actual config/source independent review differs')
 for key in ('actual_native_passed','actual_native_skipped','actual_source_native_job_id','tools_manifest','source_manifest','native_tools_result','native_tools_log','streamed_source_native17','prior_failed_result'):
  need(review[key]==g[key],'Independent review native/source identity differs '+key)
 need(g['source_streamed_adamw_native_job_id']==3172 and g['source_streamed_adamw_native_result_sha256']==NATIVE17_SHA,'Wrong streamed source native identity')
 need(g['streamed_source_native17']['sha256']==NATIVE17_SHA,'Only actual source3172 native17 receipt')
 n=ref(g['streamed_source_native17']);need(n['job']=='3172' and n['tests']==17 and n['skips']==n['failures']==n['exit_code']==0 and n['cuda_initialized'] is False and n['source_manifest_sha256']==SOURCE_MANIFEST and n['torch'].split('+')[0]=='2.4.0','Actual source3172 seventeen contracts failed')
 n=ref(g['native_tools_result']);need(n['scope']=='STANDARD_R1_SOURCE_CPU_CONTRACTS_ONLY_NO_GPU_PROOF' and str(n['job'])==str(g['actual_source_native_job_id']) and type(g['actual_source_native_job_id']) is int and g['actual_source_native_job_id']>0 and n['returncode']==0 and n['expected_count']==43 and n['zero_skip'] is True and n['source_manifests']=={'source':SOURCE_MANIFEST,'tools':TOOLS_MANIFEST} and n['cuda_visible_devices']=='','Actual V6 native43 incomplete/wrong source')
 raw_log=pathlib.Path(g['native_tools_log']['path']);need(root/'logs' in raw_log.resolve().parents and not raw_log.is_symlink() and hashlib.sha256(raw_log.read_bytes()).hexdigest()==g['native_tools_log']['sha256']==n['log_sha256'],'Actual native43 full log differs')
 text=raw_log.read_text();need(re.findall(r'\b(\d+) passed\b',text)==['43'] and not re.search(r'\b\d+ (?:failed|skipped|error)',text),'Actual native43 log not unique zero-skip PASS')
 need(g['prior_terminal']['sha256']==TERMINAL_SHA and g['prior_terminal']['path']==str(root/'logs/standard3170-captured-controller-terminal.json'),'Only captured3170 terminal')
 t=ref(g['prior_terminal']);need(t['job_id']==3170 and t['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',t['stdout'])==['3170'] and re.search(r'\bJobState=FAILED\b',t['stdout']) and re.search(r'\bExitCode=1:0\b',t['stdout']),'Actual3170 FAILED1:0 required')
 need(g['prior_run_cleanup']['sha256']==CLEANUP_SHA and g['prior_failed_result']['sha256']==FAILED_RESULT_SHA,'Only actual failed3170 raw result/cleanup')
 failed=ref(g['prior_failed_result']);need(str(failed['job'])=='3170' and failed['status']=='FAILED' and failed['error'] and failed['cleanup']==g['prior_run_cleanup'],'Failed3170 may not be relabelled or exported as approval')
 cleanup=ref(g['prior_run_cleanup']);need(cleanup['cleanup']['actual_cgroup_empty'] is True and cleanup['parent_cuda_initialized'] is False,'Actual prior own cgroup/CPU supervisor cleanup incomplete')
 for key in ('own_processes_remaining','own_gpu_records_remaining','original_gpu_baseline_missing','baseline_identity_missing','parent_gpu_records'):need(cleanup[key]==[],'Actual prior cleanup incomplete '+key)
 # Historical errors are retained. They are not erased or treated as profile PASS.
 return cleanup
