"""Make pinned review release locally or remotely; no dispatch/network actions."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--remote-source-sha',required=True);p.add_argument('--verified-at',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads((a.package/'manifest.json').read_text());receipt=json.loads((a.package/'package-build.json').read_text())
if a.remote_source_sha!=m['code_commit'] or not re.fullmatch('[0-9a-f]{40}',a.remote_source_sha):raise ValueError('Actual pushed remote SHA must equal packaged source')
if sha(a.package/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('Package build manifest bytes changed')
if 'STATIC_SOURCE_RUNTIME_REVIEW_ACCEPTED' not in a.review.read_text():raise ValueError('Independent static review is not accepted')
if not all(sha(a.package/name)==expected for name,expected in m['files'].items()):raise ValueError('Package bytes changed after review/build')
release=dict(scope=m['scope'],code_commit=m['code_commit'],manifest_sha256=receipt['manifest_sha256'],human_training_authority_sha256=m['human_training_authority']['sha256'],cluster_name='wuhan-h100',node_list='h100-10',dispatch_intent_path=m['dispatch']['once_intent'],independent_source_review_complete=True,independent_runtime_review_complete=True,dispatch_renderer_sha256=m['files']['render_dispatch.py'],dispatch_submitter_sha256=m['files']['submit_once.py'],independent_review_sha256=sha(a.review),actual_remote_source_sha=a.remote_source_sha,publication_verified_at=a.verified_at,production_thresholds_accepted=False,independent_acceptance=False,training_accepted=False,original_sampler_actor_gate_accepted=False)
with a.output.open('x') as stream:stream.write(json.dumps(release,indent=2)+'\n')
print(json.dumps(dict(release_path=str(a.output.resolve()),release_sha256=sha(a.output),slurm_submitted=False)))
