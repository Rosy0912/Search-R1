"""Make pinned review release locally or remotely; no dispatch/network actions."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-manifest',type=Path,required=True);p.add_argument('--remote-source-sha',required=True);p.add_argument('--verified-at',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--operational-recovery',action='store_true');p.add_argument('--prior-release',type=Path);a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads((a.package/'manifest.json').read_text());receipt=json.loads((a.package/'package-build.json').read_text())
if a.remote_source_sha!=m['code_commit'] or not re.fullmatch('[0-9a-f]{40}',a.remote_source_sha):raise ValueError('Actual pushed remote SHA must equal packaged source')
if sha(a.package/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('Package build manifest bytes changed')
if 'STATIC_SOURCE_RUNTIME_REVIEW_ACCEPTED' not in a.review.read_text():raise ValueError('Independent static review is not accepted')
review=json.loads(a.review_manifest.read_text())
if review.get('blocking_findings')!=[] or review.get('status')!='STATIC_SOURCE_RUNTIME_REVIEW_ACCEPTED':raise ValueError('Structured accepted independent review required')
reviewed=review['reviewed_files_sha256']
# Bind every new runtime/config and every changed implementation file, without commit self-reference.
required={name.removeprefix('source/repo/') for name in m['files'] if name.startswith('source/repo/reproduction/tree-nested-branch-runtime-20261012/')}
required.update(('search_r1/baselines/rollout.py','search_r1/baselines/controller.py','search_r1/baselines/tree_audit.py','search_r1/baselines/tree_fresh200.py','search_r1/baselines/advantages.py','search_r1/baselines/tree_nested_branch.py','search_r1/baselines/local_runtime.py'))
if a.operational_recovery:
 if not m.get('nested_native_continuation') or a.prior_release is None:raise ValueError('Explicit same-goal native continuation and actual prior release required')
 prior=json.loads(a.prior_release.read_text())
 if prior['code_commit']!='0befbaf99fc44942105831b3fbec94e4bf1b8ccb' or not prior['independent_source_review_complete'] or not prior['independent_runtime_review_complete']:raise ValueError('Actual accepted parent release differs')
 required={'search_r1/baselines/'+name for name in ('rollout.py','controller.py','tree_audit.py','advantages.py','tree_nested_branch.py','local_runtime.py')}
 required.add('reproduction/tree-nested-branch-runtime-20261012/config/ppo_tree_nested_branch_fresh200.yaml')
 if any(reviewed.get(name)!=prior['reviewed_files_sha256'].get(name) for name in required):raise ValueError('Prior independent algorithm hashes differ')
 reviewed={name:reviewed[name] for name in required}
if not required.issubset(reviewed):raise ValueError('Independent review omits changed implementation/runtime/config bytes')
for relative,expected in reviewed.items():
 path=Path(relative)
 if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe review file')
 if m['files'].get('source/repo/'+relative)!=expected or sha(a.package/'source/repo'/path)!=expected:raise ValueError('Reviewed bytes differ from final package: '+relative)
 runtime_prefix='reproduction/tree-nested-branch-runtime-20261012/'
 if relative.startswith(runtime_prefix):
  deployed=relative.removeprefix(runtime_prefix)
  if m['files'].get(deployed)!=expected or sha(a.package/deployed)!=expected:raise ValueError('Deployed runtime differs from reviewed source: '+deployed)
if not all(sha(a.package/name)==expected for name,expected in m['files'].items()):raise ValueError('Package bytes changed after review/build')
release=dict(scope=m['scope'],code_commit=m['code_commit'],manifest_sha256=receipt['manifest_sha256'],human_training_authority_sha256=m['human_training_authority']['sha256'],cluster_name='wuhan-h100',node_list='h100-10',dispatch_intent_path=m['dispatch']['once_intent'],independent_source_review_complete=True,independent_runtime_review_complete=True,dispatch_renderer_sha256=m['files']['render_dispatch.py'],dispatch_submitter_sha256=m['files']['submit_once.py'],independent_review_sha256=sha(a.review),independent_review_manifest_sha256=sha(a.review_manifest),reviewed_files_sha256=reviewed,actual_remote_source_sha=a.remote_source_sha,publication_verified_at=a.verified_at,production_thresholds_accepted=False,independent_acceptance=False,training_accepted=False,original_sampler_actor_gate_accepted=False)
if a.operational_recovery:
 release.update(review_policy='latest_human_launch_first_operational_recovery_no_redundant_review',
  independent_source_review_complete=False,independent_runtime_review_complete=False,
  operational_changes_independently_reviewed=False,unchanged_algorithm_files_sha256=reviewed,
  prior_reviewed_release=dict(path='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/nested-branch-fresh200-0befbaf99fc44942105831b3fbec94e4bf1b8ccb/release.json',sha256=sha(a.prior_release)),
  same_goal_native_continuation=m['nested_native_continuation'])
if m['dispatch'].get('kind')=='shared_systemd':
 d=m['dispatch']
 release.update(dispatch_kind='shared_systemd',resource_protocol='training_memory_headroom_shared_v1',physical_gpus=8,slurm_reserved_gpus=0,
  dispatch_renderer_sha256=m['files']['render_shared_dispatch.py'],dispatch_submitter_sha256=m['files']['submit_shared_once.py'],
  shared_memory_contract={key:d[key] for key in ('initial_required_free_gib','runtime_required_headroom_gib','own_memory_cap_gib','measured_training_budget_gib','expected_gpu_uuids','initial_required_host_mem_available_gib','minimum_host_cpu_capacity','memory_gib')},
  completed200_memory_peak_report=d['completed200_memory_peak_report'],gpu_sharing_authority=d['gpu_sharing_authority'],
  sampled_peak_caveat='Five-second whole-card sampled occupancy is a lower bound; admission8GiB margin and continuous1s physical free guard')
with a.output.open('x') as stream:stream.write(json.dumps(release,indent=2)+'\n')
print(json.dumps(dict(release_path=str(a.output.resolve()),release_sha256=sha(a.output),slurm_submitted=False)))
