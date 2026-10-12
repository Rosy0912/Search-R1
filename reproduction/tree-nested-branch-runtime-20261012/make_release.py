"""Make pinned review release locally or remotely; no dispatch/network actions."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-manifest',type=Path,required=True);p.add_argument('--remote-source-sha',required=True);p.add_argument('--verified-at',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
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
required.update(('search_r1/baselines/rollout.py','search_r1/baselines/controller.py','search_r1/baselines/tree_audit.py','search_r1/baselines/tree_fresh200.py','search_r1/baselines/advantages.py','search_r1/baselines/tree_nested_branch.py'))
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
with a.output.open('x') as stream:stream.write(json.dumps(release,indent=2)+'\n')
print(json.dumps(dict(release_path=str(a.output.resolve()),release_sha256=sha(a.output),slurm_submitted=False)))
