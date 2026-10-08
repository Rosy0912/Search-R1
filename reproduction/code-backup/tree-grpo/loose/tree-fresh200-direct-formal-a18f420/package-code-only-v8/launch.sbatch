#!/usr/bin/env bash
# Resource/time directives are supplied by the separately reviewed renderer.
set -euo pipefail
# Initial allocation admission always uses the unchanged whole-card preflight.
unset TREE_DIRECT_OWNED_MODELS_LOADED TREE_DIRECT_ALLOCATION_IDENTITY_PATH TREE_DIRECT_ALLOCATION_IDENTITY_SHA256
: "${TREE_INSTRUCT_TOOLS:?Human-authorized direct fresh200 package required}"
: "${TREE_INSTRUCT_PACKAGE_SHA256:?Exact manifest SHA required}"
: "${TREE_INSTRUCT_REVIEW_SHA256:?Exact human direct release SHA required}"
: "${TREE_PRODUCTION_RELEASE_PATH:?External human direct release path required}"
export PYTHONPATH="$TREE_INSTRUCT_TOOLS:/dataset/shaoshitong/search-r1-rl/tree-grpo/tools:$TREE_INSTRUCT_TOOLS/source/repo"
export PYTHONOPTIMIZE=0
cd "$TREE_INSTRUCT_TOOLS/source/repo"
/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python - <<'PY'
import os,pathlib,hashlib,sys
from production_contract import read_bound,production_inputs,require,ROOT,submitted_intent,DIRECT_MODE,CHAIN_SCOPE
require(sys.flags.optimize==0,'Production requires unoptimized Python')
package=pathlib.Path(os.environ['TREE_INSTRUCT_TOOLS']).resolve()
require(package.is_relative_to(ROOT/'instruct300/candidates'),'Production package outside own h100-9 task candidates')
m,manifest_sha,_=read_bound(package/'manifest.json',os.environ['TREE_INSTRUCT_PACKAGE_SHA256'])
require(m['scope']==CHAIN_SCOPE and m['execution_mode']==DIRECT_MODE and m['packet_revision']==8 and m['stage']=='fresh_mu_train200','Explicit human direct200 package required')
require(m['training_protocol']=='tree_fresh200_five_leaf_v1' and m['bootstrap_thresholds_accepted'] is False and m['production_thresholds_accepted'] is False and m['original_sampler_actor_gate_accepted'] is False,'Truthful direct profile/prospective thresholds missing')
require(not any(k in m for k in ('parent_manifest','healthy110_metadata')),'Fresh production forbids inherited checkpoints')
require(not any(k in os.environ for k in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Production forbids zero-observation branches')
for relative,expected in m['files'].items():
 path=pathlib.Path(relative);require(not path.is_absolute() and '..' not in path.parts,'Unsafe package member')
 require(hashlib.sha256((package/path).read_bytes()).hexdigest()==expected,'Package bytes changed: '+relative)
_,pins=production_inputs(m,manifest_sha256=manifest_sha,dispatch_review_descriptor={'path':os.environ['TREE_PRODUCTION_RELEASE_PATH'],'sha256':os.environ['TREE_INSTRUCT_REVIEW_SHA256']})
release,_,_=read_bound(os.environ['TREE_PRODUCTION_RELEASE_PATH'],os.environ['TREE_INSTRUCT_REVIEW_SHA256'])
intent=m['dispatch']['once_intent'];require(type(intent) is str and pathlib.Path(intent).is_relative_to(ROOT/'instruct300') and release['dispatch_intent_path']==intent,'Reviewed new train intent missing')
from slurm_identity import verify_runtime,install_local_environment,verify_runtime_cgroup
verify_runtime_cgroup()
identity=verify_runtime()
submitted_intent(intent,manifest_sha,os.environ['TREE_INSTRUCT_REVIEW_SHA256'],os.environ['SLURM_JOB_ID'],cluster_name=identity['cluster_name'],node_list=identity['node_list'],run_id='tree-current200-continue-'+os.environ['SLURM_JOB_ID'])
from current_chain_runtime import install_chain_environment
install_chain_environment(m,os.environ)
# Bash is replaced by this verified process so sanitized routing reaches preflight.
install_local_environment(os.environ)
os.execv(sys.executable,[sys.executable,'-c','from slurm_identity import guarded_preflight_main; guarded_preflight_main()','--',sys.executable,str(package/'supervisor.py')])
PY
