#!/usr/bin/env bash
# Resource/time directives are supplied by the separately reviewed renderer.
set -euo pipefail
: "${TREE_INSTRUCT_TOOLS:?Reviewed fresh200 training package required}"
: "${TREE_INSTRUCT_PACKAGE_SHA256:?Exact manifest SHA required}"
: "${TREE_INSTRUCT_REVIEW_SHA256:?Exact final independent release SHA required}"
: "${TREE_PRODUCTION_RELEASE_PATH:?External final release path required}"
export PYTHONPATH="$TREE_INSTRUCT_TOOLS:/dataset/shaoshitong/search-r1-rl/tree-grpo/tools:$TREE_INSTRUCT_TOOLS/source/repo"
export PYTHONOPTIMIZE=0
cd "$TREE_INSTRUCT_TOOLS/source/repo"
/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python - <<'PY'
import os,pathlib,hashlib,sys
from production_contract import read_bound,production_inputs,require,ROOT,submitted_intent
require(sys.flags.optimize==0,'Production requires unoptimized Python')
package=pathlib.Path(os.environ['TREE_INSTRUCT_TOOLS']).resolve()
require(package.is_relative_to(ROOT/'instruct300/candidates'),'Production package outside own h100-9 task candidates')
m,manifest_sha,_=read_bound(package/'manifest.json',os.environ['TREE_INSTRUCT_PACKAGE_SHA256'])
require(m['scope']=='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW' and m['stage']=='fresh_mu_train200','Unreleased/nonproduction package')
require(m['training_protocol']=='tree_fresh200_five_leaf_v1' and m['bootstrap_thresholds_accepted'] is True and m['production_thresholds_accepted'] is False and m['original_sampler_actor_gate_accepted'] is False,'Explicit production profile/thresholds missing')
require(not any(k in m for k in ('parent_manifest','healthy110_metadata')),'Fresh production forbids inherited checkpoints')
require(not any(k in os.environ for k in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Production forbids zero-observation branches')
for relative,expected in m['files'].items():
 path=pathlib.Path(relative);require(not path.is_absolute() and '..' not in path.parts,'Unsafe package member')
 require(hashlib.sha256((package/path).read_bytes()).hexdigest()==expected,'Package bytes changed: '+relative)
_,pins=production_inputs(m,manifest_sha256=manifest_sha,dispatch_review_descriptor={'path':os.environ['TREE_PRODUCTION_RELEASE_PATH'],'sha256':os.environ['TREE_INSTRUCT_REVIEW_SHA256']})
release,_,_=read_bound(os.environ['TREE_PRODUCTION_RELEASE_PATH'],os.environ['TREE_INSTRUCT_REVIEW_SHA256'])
intent=m['dispatch']['once_intent'];require(type(intent) is str and pathlib.Path(intent).is_relative_to(ROOT/'instruct300') and release['dispatch_intent_path']==intent,'Reviewed new train intent missing')
from slurm_identity import verify_runtime,install_local_environment
identity=verify_runtime()
submitted_intent(intent,manifest_sha,os.environ['TREE_INSTRUCT_REVIEW_SHA256'],os.environ['SLURM_JOB_ID'],cluster_name=identity['cluster_name'],node_list=identity['node_list'],run_id='tree-fresh200-train-'+os.environ['SLURM_JOB_ID'])
# Bash is replaced by this verified process so sanitized routing reaches preflight.
install_local_environment(os.environ)
os.execv(sys.executable,[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,str(package/'supervisor.py')])
PY

