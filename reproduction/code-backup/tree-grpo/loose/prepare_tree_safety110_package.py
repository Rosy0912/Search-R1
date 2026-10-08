"""Verify a new source/metadata package and resolve CPU-only configuration."""
import hashlib, json, os, socket, sys, tarfile
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PACKAGE_SHA = '84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
target = ROOT/'instruct300/candidates'/('safety110-a632ba9-'+PACKAGE_SHA[:8])
archive = ROOT/'instruct300/transfers'/('safety110-a632ba9-'+PACKAGE_SHA[:8]+'.tar.gz')
expected_archive = os.environ['TREE_SAFETY110_ARCHIVE_SHA256']
assert socket.gethostname().split('.')[0] == 'h100-9'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_archive
target.mkdir(exist_ok=False)
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    bundle.extractall(target)
assert hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest() == PACKAGE_SHA
manifest = json.loads((target/'manifest.json').read_bytes())
for relative, expected in manifest['files'].items():
    assert hashlib.sha256((target/relative).read_bytes()).hexdigest() == expected,relative
for path, expected in manifest['external_files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,path
assert hashlib.sha256(Path(manifest['parent_manifest']['path']).read_bytes()).hexdigest() == manifest['parent_manifest']['sha256']
os.environ.update(SEARCH_R1_REPO=str(target/'source/repo'),TREE_COMMON_DATA=str(ROOT/'instruct300/common-data'),
    TREE_INSTRUCT_MODEL=str(ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'),
    SEARCH_R1_ASSETS=str(ROOT/'assets-root'),SEARCH_R1_BASELINE_OUTPUT=str(target/'CPU_PREP_CONFIG/output'),
    CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1')
sys.path[:0] = [str(target),str(ROOT/'tools'),str(target/'source/repo')]
import supervisor
prep = target/'CPU_PREP_CONFIG';prep.mkdir()
value, steps = supervisor.configuration(prep,'calibration110')
assert steps == 300
result = dict(status='SAME_NODE_CODE_AND_CPU_CONFIG_VERIFIED_NOT_SUBMITTED',source_commit=manifest['code_commit'],
    package=str(target),manifest_sha256=PACKAGE_SHA,files_verified=len(manifest['files']),
    parent_manifest_sha256=manifest['parent_manifest']['sha256'],parent_completed=110,
    train_batch=value.data.train_batch_size,mini_batch=value.actor_rollout_ref.actor.ppo_mini_batch_size,
    global_micro=value.actor_rollout_ref.actor.ppo_micro_batch_size,safety_dual_clip=value.actor_rollout_ref.actor.policy_safety.dual_clip,
    sampling_enabled=value.actor_rollout_ref.rollout.sampling_consistency.enabled,
    sampling_enforced=value.actor_rollout_ref.rollout.sampling_consistency.enforce,
    trial_thresholds=manifest['policy_safety_thresholds'],actual_3b_training_accepted=False,submitted=False)
(prep/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
