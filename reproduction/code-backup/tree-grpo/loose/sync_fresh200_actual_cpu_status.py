import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / 'outputs'
F = ROOT / 'work/tree-fresh200-freeze-f7-v8'
R = ROOT / 'work/tree-fresh200-f7-v8-v6-context-free-review'
P = ROOT / 'work/tree-fresh200-production200-context-free-review'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require_sha(p, expected):
    actual = sha(p)
    if actual != expected:
        raise ValueError(f'Immutable evidence mismatch: {p}: {actual}')

pins = {
    R/'RECEIPT.json': '6f2ceaebd1ff6c04e9330b404e17ff7db9df7493db5c686cbc3366cb3253fb44',
    R/'CPU_EXECUTION_BINDING_REVIEW.json': 'fdf074bdfe8ae64d20d60c01553103dcb9f8df34323618fc3d5b5d2b585bec37',
    F/'actual-cpu/PREPARATION.json': 'a6ce9101dc2529d451123f4c23f65dca090b1db76517d446df5cec51d9705df8',
    F/'actual-cpu/resolved.yaml': '77570d7e7806407c2262c75b73b94ca23786edb52732257b3b251fb025a684a9',
    P/'RECEIPT.json': 'a0682eeebcef5a762b2c2fd5eeafe767d0e6d2c3013cd2520244a03406498507',
}
for p, expected in pins.items():
    require_sha(p, expected)
prep = json.loads((F/'actual-cpu/PREPARATION.json').read_text())
if prep['submitted'] or prep['optimizer_writes'] or prep['cuda_initialized']:
    raise ValueError('CPU preparation is not zero-write/no-CUDA')
now = datetime.now(timezone.utc).isoformat()
s = json.loads((O/'STATUS.json').read_text())
fresh = s['fresh200']
s['status'] = 'FRESH200_H1009_CPU_PREPARATION_VERIFIED_TOKENIZER_RUNNING_PRODUCTION_V2_REPAIR_NO_GPU'
s['phase'] = 'FRESH200_ACTUAL_SAME_NODE_CPU_PREFLIGHT_AND_PRODUCTION_V2_REPAIR'
s['updated_at'] = now
s['current_job_scope'] = 'Historical completed zero-write3296; no current fresh200 Slurm/GPU job'
fresh['source_commit_scope'] = 'Clean frozen f7 independently SOURCE/CPU accepted; actual same-h1009 CPU assets/config verified; GPU/calibration/training not accepted'
fresh['independent_reviewer'].update(status='F7_V8_V6_SOURCE_CPU_AND_EXACT_CPU_EXECUTION_BINDING_ACCEPTED', receipt_sha256=pins[R/'RECEIPT.json'], execution_released=False, binding_review_sha256=pins[R/'CPU_EXECUTION_BINDING_REVIEW.json'])
fresh['dispatch_candidate'].update(version='V6_LOCAL_CLUSTER_INDEPENDENT_SOURCE_CPU_ACCEPTED', independent_accepted=True, actual_remote_executed=True, actual_remote_execution_scope='Bound stage and CPU preparation only; no submit helper/Slurm/GPU execution', blocker='Actual historical tokenizer preflight and independent zero-only READY still required')
fresh['actual_local_package'].update(status='ACTUAL_CODE_ONLY_PACKAGE_BYTE_BINDING_ACCEPTED_AND_SAME_NODE_CPU_PREPARED', actual_remote_executed=True, actual_remote_execution_scope='Code/audit metadata staging and CPU config/assets verification only', slurm_submitted=False)
fresh['actual_same_node_cpu_preparation_executed'] = True
fresh['actual_same_node_cpu_preparation'] = {'node':'h100-9', 'report_sha256':pins[F/'actual-cpu/PREPARATION.json'], 'resolved_config_sha256':pins[F/'actual-cpu/resolved.yaml'], 'status':prep['status'], 'model_or_dataset_transfer':False, 'optimizer_writes':0, 'cuda_initialized':False, 'actual_training_accepted':False}
fresh['actual_historical_tokenizer_preflight'] = {'status':'RUNNING_NO_FINAL_REPORT', 'tool_session_id':20316, 'expected_historical_rows':5120, 'scope':'Existing same-node historical3295/3296 JSON and local official tokenizer only; no model/GPU/Adam; no fresh rollout or original probability gate acceptance', 'last_tool_observation_at':now}
fresh['production_entry'].update(status='ORIGINAL_PRODUCTION24_INDEPENDENT_REJECTED_V2_REPAIR_RUNNING', original_path='work/tree-fresh200-train-runtime-draft', repair_path='work/tree-fresh200-train-runtime-v2', independent_review_receipt_sha256=pins[P/'RECEIPT.json'], blocking_findings=['P01 cluster lineage','P02 query/cancellation identity','P03 same-live-first5 threshold assessment','P04 partial10 physical retention'], independent_release=False, actual_training=False)
fresh['new_completed_outer'] = 0
fresh['actual_new_outer_updates'] = 0
fresh['actual_new_optimizer_write_calls_per_rank'] = 0
fresh['new_optimizer_calls_per_rank'] = 0
fresh['new_gpu_job_submitted'] = False
fresh['new_gpu_execution_released'] = False
fresh['latest_new_evaluation'] = None
fresh['new_score_updated'] = False
(O/'STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (O/'REPORT.md').open('a') as out:
    out.write(f'\n\n### {now} - Actual same-h1009 CPU preparation verified; no fresh training\n\n')
    out.write('The independently bound code/audit staging and CPU preparation actually completed on h100-9. Full existing local official weights/tokenizer and fixed169028/512/51713 data identities matched; actual Hydra config includes20 opportunities, minimum1 actual write for credited outer, separate write accounting, and no credit/warmup for zero-write or abandoned attempts. No weights/data/index transfer, CUDA initialization, optimizer write, Slurm or GPU submission occurred. Actual PREPARATION SHA a6ce9101 and resolved config77570d7e are preserved. First connection failure was preserved as UNKNOWN then read-only absence/no-live-helper proof established not started before the same CPU argv reexecution succeeded.\n\n')
    out.write('Historical5120 tokenizer CPU preflight is actually running and its final outcome is unknown. It does not validate fresh native24 sampling or the original sampler probability gate. The no-context production24 review independently rejected four real counterexamples P01-P04; original artifacts remain immutable and a separate V2 repair is active. Frozen source f7/V8/V6 SOURCE/CPU and execution-binding acceptance does not release GPU/zero calibration/formal training. New official completed0/200, actual Adam writes0, no new score. Active goal unchanged.\n')
groups = [
    (R, O/'tree-fresh200/f7-v8-v6-independent', ['RECEIPT.json','BYTE_AUDIT.json','HELPER_V6_RECEIPT.json','CPU_EXECUTION_BINDING_REVIEW.json']),
    (P, O/'tree-fresh200/production200-independent-rejected', ['RECEIPT.json','BYTE_AUDIT.json','REVIEW.md','REQUIREMENTS_AUDIT.md']),
    (F, O/'tree-fresh200/actual-node-cpu', ['CPU_STAGE_ACTUAL_TOOL_RESULT.json','CPU_PREPARATION_ACTUAL_TOOL_RESULT.json','CPU_PREPARATION_CONNECTION_UNKNOWN_TOOL_RESULT.json','CPU_PREPARATION_FIRST_READONLY_PROBE.json','CPU_PREPARATION_NOT_STARTED_READONLY_PROOF.json','CPU_TOKENIZER_CURRENT_TOOL_OBSERVATION.json','actual-cpu/PREPARATION.json','actual-cpu/resolved.yaml']),
]
for src, dst, names in groups:
    for name in names:
        p = src/name
        if not p.is_file():
            continue
        d = dst/name
        d.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,d)
for base in [Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo'),ROOT/'work/tree-grpo/reproduction/baselines/tree-validation-20261002']:
    base.mkdir(parents=True,exist_ok=True)
    for name in ['STATUS.json','REPORT.md']:
        shutil.copyfile(O/name,base/name)
    for _,dst,_ in groups:
        rel=dst.relative_to(O)
        for p in dst.rglob('*'):
            if p.is_file():
                target=base/rel/p.relative_to(dst)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,target)
print(json.dumps({'status':s['status'],'updated_at':now,'new_outers':0,'new_adam_writes':0,'actual_cpu_prepared':True,'tokenizer_status':'RUNNING_NO_FINAL_REPORT','production_v2':'REPAIR_RUNNING','GPU_or_Slurm_submitted':False}))
