"""Actual evidence assembly only; external independent reviewer signs approval."""
import sys
sys.dont_write_bytecode=True
import argparse,copy,hashlib,json,pathlib
BASE=pathlib.Path(__file__).resolve().parent.parent
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
O=BASE/'outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 data=(json.dumps(v,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==data,'Immutable output already differs'
 else:p.write_bytes(data)
 return sha(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--window-review',type=pathlib.Path);args=ap.parse_args()
 pending=O/'standard-r1-profile2-v14-static-config-pending-review.json';c=read(pending)
 assert c['tools_manifest']['sha256']=='783709270e21d4da3fa999dae2830d2c053fb28414175f5f2a1e53b5f8af4ba7'
 native=O/'remote-evidence/standard-r1-native3208';refs=read(native/'fetch-refs.json')
 for r in refs:
  p=native/r['name'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
 n=read(native/'result.json');assert n['job']=='3208' and n['returncode']==0 and n['expected_count']==87 and n['zero_skip'] is True and n['cuda_visible_devices']==''
 assert n['source_manifests']=={'source':c['source_manifest']['sha256'],'tools':c['tools_manifest']['sha256']}
 c['source_device_correctness']=read(O/'standard-r1-profile2-v12-bound-static-config.json')['source_device_correctness']
 prior=O/'remote-evidence/standard-r1-profile3204-small';raw=read(prior/'fetch-refs.json')['files'];prior_refs={pathlib.Path(x['path']).name:x for x in raw}
 actual_result=read(prior/'train-result.json');assert actual_result['job']=='3204' and actual_result['status']=='FAILED' and actual_result['cleanup_passed'] is False
 refs_by_name={r['name']:r for r in refs}
 r=copy.deepcopy(read(O/'standard-r1-profile2-v12-bound-independent-review.json'))
 r.update(status='DRAFT_NOT_APPROVAL',actual_source_native_job_id=3208,actual_native_passed=87,actual_native_skipped=0,source_manifest=c['source_manifest'],tools_manifest=c['tools_manifest'],raw_refs=refs,telemetry_policy=c['telemetry_policy'],fresh_evidence_root=c['evidence_root'],failed_prior_profile_job=3204,prior_failed_job_id=3204,allowed_attempt='profile2-v14',prior_terminal=prior_refs['standard3204-captured-controller-terminal.json'],prior_run_cleanup=prior_refs['owned-cleanup.json'],prior_failed_result=prior_refs['train-result.json'],native_tools_result=refs_by_name['result.json'],native_tools_log=refs_by_name['pytest.log'],source_device_correctness=c['source_device_correctness'])
 r['review_md']=None
 if args.window_review:
  q=read(args.window_review)
  assert q['status']=='CPU_REPLAY_POLICY_PASS_NOT_GPU_TELEMETRY_APPROVAL' and q['schema']=='WINDOW_V6_CPU_QUERY_CONCURRENCY_REVIEW_V1' and q['actual_gpu_telemetry_approval'] is False
  assert q['runtime_source_sha256']==c['telemetry_policy']['runtime_sha256'] and q['adapter_source_sha256']==c['telemetry_policy']['guard_sha256']
  assert q['approval_scope']=='SOURCE_AND_NATIVE_CPU_CONTRACTS_ONLY'
  c['telemetry_policy']['actual_independent_native_review']={'path':str(ROOT/'logs'/args.window_review.name),'sha256':sha(args.window_review)}
  r['telemetry_policy']=copy.deepcopy(c['telemetry_policy'])
  bound=O/'standard-r1-profile2-v14-bound-static-config.json';r['recipe_file_sha256']=write(bound,c)
  r['reviewed_static_recipe_sha256']=hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!='independent_gate'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  r['bound_recipe_local_path']=str(bound)
 else:
  r['recipe_file_sha256']=None;r['reviewed_static_recipe_sha256']=None;r['pending_refs']=['Actual WindowV6 native117 independent CPU review; external specific source/allocation review/gate']
  r['pending_recipe_sha256']=sha(pending)
 out=O/('standard-r1-v14-actual-review-inputs.json' if args.window_review else 'standard-r1-v14-review-inputs-pending-window.json');write(out,r)
 print(json.dumps({'scope':'EVIDENCE_INPUT_ASSEMBLY_NOT_APPROVAL','inputs':str(out),'sha256':sha(out),'bound_config':r.get('bound_recipe_local_path'),'pending':r.get('pending_refs',[])}))
if __name__=='__main__':main()
