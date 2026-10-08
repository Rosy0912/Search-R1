from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
stamp = datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(relative, expected):
    p = ROOT / relative
    actual = sha(p)
    if actual != expected:
        raise ValueError('Frozen review bytes changed: ' + str(p))
    return {'path': str(p), 'sha256': actual}

source_reject = record('work/tree-fresh200-af67976-v10-v9-context-free-review/UTF8_AND_V9_SUBREVIEW.json', '82ddc9720bd8371df4d8ad1a550e99351e67645e7edba6b67503ca1747a04ea0')
source_probe = record('work/tree-fresh200-af67976-v10-v9-context-free-review/UTF8_INDEPENDENT_PROBE_RESULT.json', '4ef978d9279012a99945bb346dcb833ed82eba37a71a82bfb97574777c919f69')
wrapper_accept = record('work/tree-fresh200-zero-v10-context-free-review/WRAPPER_V10_RECEIPT.json', 'c0b3bfc5d3bee6cbed2f54025fc69240d1b9f35b7d7f2a2274a892f8f568b1bd')
formal_accept = record('work/tree-fresh200-production-v3-context-free-review/RECEIPT.json', '0941423564ec6c4fccddb17801d2dc9494a4a46d665b5c77a23c814f6a3a49d6')
helper_scope = record('work/tree-fresh200-af67976-v10-v9-context-free-review/UTF8_REVIEWER_HELPER_V9_RECEIPT.json', 'e3d7b81cb94275a7cf3c4c249c7576ebd2bf390530894708dd900bd1159d0ea3')

status_path = OUT / 'STATUS.json'
d = json.loads(status_path.read_text())
d.update(status='FRESH200_UTF8_V1_SEMANTIC_BINDING_REJECTED_V2_REPAIR_ACTIVE_NO_GPU',
         phase='ISOLATED_UTF8_V2_TRUSTED_TOKENIZER_BINDING_AND_NEW_NONE_REVIEW',
         updated_at=stamp, active_training_job=None, active_source_commit=None,
         current_job_scope='Historical completed3296 zero-write observation; not active fresh200 job')
d['pending'] = [
    'Finish isolated UTF8 V2 actual-tokenizer registration and legacy semantic binding; preserve af67976 P1 rejection.',
    'Freeze new clean source and repinned zero helper, then new fork:none exact-combination SOURCE/CPU review.',
    'Only after acceptance: code-only same-h1009 staging, CPU config/assets verification and full official-tokenizer historical5120 recheck.',
    'Actual zero-write8GPU observation and independent release remain required before same-live-instance5 through200/final51713/HF/cost/cleanup.'
]
f = d['fresh200']
f.update(new_completed_outer=0, actual_new_outer_updates=0, actual_new_training_attempts=0,
         new_optimizer_calls_per_rank=0, actual_new_optimizer_write_calls_per_rank=0,
         new_gpu_job_submitted=False, new_gpu_execution_released=False,
         latest_new_evaluation=None, new_score_updated=False, evaluation_updated=False,
         source_state='F7_SOURCE_CPU_ACCEPTED_UTF8_V1_REJECTED_NEW_V2_REPAIR_ACTIVE',
         runtime_version='V10_SOURCE_CPU_ACCEPTED_NO_ACTUAL_READY')
f['utf8_source_compatibility_repair'].update(
    status='AF67976_FROZEN_NONE_REVIEW_P1_REJECTED_NEW_V2_IMPLEMENTATION_ACTIVE',
    new_source_commit=None, previous_rejected_source_commit='af67976ec3b544eddaaa68c35411a46b9db7af68',
    independent_accepted=False, previous_independent_review=source_reject,
    previous_independent_real_counterexample=source_probe,
    actual_official_all5120_recheck_accepted=False,
    runtime_change_scope='raw helper trusted actual-tokenizer registry plus FSDP Tree opt-in registration of existing tokenizer; ledger cannot auto-load assets',
    next_required='New clean V2 source, fork:none strict review and actual same-node full5120 tokenizer recheck; no policy-token reencoding/filtering')
f['zero_runtime_entry'].update(v10='SOURCE_CPU_INDEPENDENT_ACCEPTED_ONLY',
    v10_independent_wrapper_receipt=wrapper_accept,
    helper_v8_status='ZERO_P05_REJECTED_MUTABLE_SBATCH_PATH',
    helper_v9_scope='Immutable stdin helper logic independently CPU reviewed; bound af67976 source rejected; no combined dispatch release',
    helper_v9_independent_receipt=helper_scope, actual_ready=False, GPU_dispatch_allowed=False)
f['production_entry'].update(status='V3_SOURCE_CPU_INDEPENDENT_ACCEPTED_ONLY_NEW_SOURCE_COMBINATION_UNREVIEWED',
    repair_path='work/tree-fresh200-train-runtime-v3',
    v3_inventory_sha256='8955830bbd5a8a75acfc50938be1668d002c04ad56b033b8a591d612b2fea2e7',
    v3_owner_receipt_sha256='ad0d4b611ee6e287a00fc7b259a12bc22229d091761a63a3fa20e8cbe43bf690',
    v3_independent_receipt=formal_accept, blocking_findings=[],
    source_binding_scope='V3 immutable stdin repair accepted with prior f7 source only; future UTF8 V2 exact-combination review and actual calibration remain required',
    thresholds_accepted=False, independent_release=False, actual_training=False)
f['dispatch_candidate'].update(version='V9_HELPER_LOGIC_CPU_REVIEWED_SOURCE_AF67976_REJECTED_NEW_SOURCE_REPIN_PENDING',
    code_commit='af67976ec3b544eddaaa68c35411a46b9db7af68', runtime_version='V10_SOURCE_CPU_ACCEPTED_ONLY',
    independent_accepted=False,
    blocker='UTF8 V1 semantic binding counterexample; new V2 source and repinned helper require exact-combination review and same-node preflight')
status_path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

report = OUT / 'REPORT.md'
with report.open('a', encoding='utf-8') as stream:
    stream.write('\n\n### ' + stamp + ' - Actual independent UTF8 semantic binding rejection; isolated V2 repair\n\n'
        'The new fork:none review rejects clean af67976: a real CPU counterexample changes canonical semantic IDs and reward text, recomputes action/trajectory bindings, and semantic_spans accepts WRONG while original policy IDs, mu, canonical text and UTF8 proof stay unchanged. P1 receipt82ddc972 and actual probe4ef978d9 are retained. Independent UTF8 boundary matrix305 passed and V9 helper10 additional actual-function CPU mock groups passed; those scoped passes do not override the source rejection.\n\n'
        'V10 wrapper24 independent CPU groups accepted after cgroup-before-query repair; old V8 helper ZERO_P05 rejected. Formal V3 immutable stdin repair independently SOURCE/CPU accepted, P01-P05 resolved within prior f7 scope. These do not establish actual GPU/READY/threshold/five/200 acceptance.\n\n'
        'A separate UTF8 V2 adds trusted existing-tokenizer registration and actual canonical encode/default-decode semantic validation across helper, trajectory/inheritance and mu; unregistered/identity-drift paths fail closed and no ledger-driven asset loading is allowed. Original freezes remain unchanged. New clean source, repinned helper, a newly spawned fork:none strict review, and complete same-h1009 official5120 recheck are required before GPU calibration.\n\n'
        'Latest human accounting remains20 Tree opportunities with successful Adam writes separate; all20 processed plus at least1 actual write credits one outer/warmup, allskip/abandoned credits none. Fresh completed0/200, actual writes0, new score none, no new Slurm/GPU job. Goal active; hourly automation unchanged PAUSED.\n')

bundles = {
    'utf8-v1-independent-rejected': ROOT/'work/tree-fresh200-af67976-v10-v9-context-free-review',
    'zero-v10-independent': ROOT/'work/tree-fresh200-zero-v10-context-free-review',
    'production-v3-independent': ROOT/'work/tree-fresh200-production-v3-context-free-review',
}
destinations = [Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'),
    ROOT/'work/tree-grpo/reproduction/baselines/tree-validation-20261002']
for target in [OUT, *destinations]:
    target.mkdir(parents=True, exist_ok=True)
    if target != OUT:
        for name in ('STATUS.json','REPORT.md'):
            shutil.copy2(OUT/name, target/name)
    for name, origin in bundles.items():
        dest = target/'tree-fresh200'/name
        dest.mkdir(parents=True, exist_ok=True)
        for p in origin.iterdir():
            if p.is_file() and p.suffix in {'.json','.md','.py','.stdout','.stderr'} and p.stat().st_size < 2*1024**2:
                shutil.copy2(p, dest/p.name)
print(json.dumps({'updated_at': stamp, 'fresh_outer': 0, 'actual_adam_writes': 0,
                  'source_utf8_v1': 'P1_REJECTED', 'v2': 'ACTIVE_REPAIR',
                  'new_slurm_job': False, 'destinations': [str(p) for p in destinations]}))
