"""Only diagnostic profile retry after real failure, never formal admission."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,re
TERMINAL_SHA='e90b001a087de2b217c44f88177d98264cb48aca626c05d24bbeb6f1dc7c51a4'
CLEANUP_SHA='b8a7bce576862f79523f1e9b713305072dad37509be399197de48176808e508e'
TOOLS_MANIFEST='6a0895a621ca3810cb8179a42a18b0019e3a9e2c8830bf37e7554a37016079f9'
def validate_failed3163(c,g,attempt,root):
 def need(ok,message):
  if not ok:raise ValueError(message)
 def ref(r):
  p=pathlib.Path(r['path']);need(not p.is_symlink() and pathlib.Path(root)/'logs' in p.resolve().parents,'Ownlogs raw required')
  raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==r['sha256'],'Actual raw SHA drift');return json.loads(raw)
 need(c['mode']=='profile2' and attempt=='profile2-v5' and g.get('allowed_attempt')==attempt,'3163 failure only permits named diagnostic profile2')
 need(g['actual_native_passed']==40 and g['actual_native_skipped']==0 and g['tools_manifest']==c['tools_manifest'] and g['source_manifest']==c['source_manifest'],'Only actual40-zero-skip V5 source admission')
 review=ref(g['independent_review']);need(review['status']=='PASS' and review['scope']=='SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_REVIEW' and review['reviewed_static_recipe_sha256']==g['reviewed_static_recipe_sha256'] and review['actual_native_passed']==40 and review['actual_native_skipped']==0 and review['actual_source_native_job_id']==g['actual_source_native_job_id'],'Actual source/config independent review differs')
 need(g['prior_failed_job_id']==3163 and c['expected_outer_steps']==2,'Wrong diagnostic prior/budget')
 need(c['tools_manifest']['sha256']==TOOLS_MANIFEST and c['tools']==str(pathlib.Path(root)/'tools/search-r1-32b-grpo100-tools-v5'),'Only frozen V5 diagnostic source')
 need(g['prior_terminal']['sha256']==TERMINAL_SHA and g['prior_terminal']['path']==str(pathlib.Path(root)/'logs/standard3163-captured-controller-terminal.json'),'Only captured actual3163 terminal')
 t=ref(g['prior_terminal']);need(t['job_id']==3163 and t['returncode']==0 and re.findall(r'\bJobId=([0-9]+)\b',t['stdout'])==['3163'],'Actual terminal job differs')
 need(re.search(r'\bJobState=FAILED\b',t['stdout']) and re.search(r'\bExitCode=1:0\b',t['stdout']),'Prior profile is not actual FAILED1:0')
 need(g['prior_run_cleanup']['sha256']==CLEANUP_SHA,'Only actual3163 cleanup bytes');cl=ref(g['prior_run_cleanup'])
 need(cl['cleanup']['actual_cgroup_empty'] is True,'Actual prior cgroup not empty')
 for key in ('own_processes_remaining','own_gpu_records_remaining','original_gpu_baseline_missing','baseline_identity_missing'):
  need(cl[key]==[],'Actual prior identity/GPU cleanup incomplete '+key)
 return cl
