"""Future reviewed32B dependency gate; never produces readiness/approvals."""
import json
from pathlib import Path
from .protocol import canonical,digest,file_sha,require
ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
METHOD='e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d'
PIPELINE='8d7a847daa6ffc84d62c24c0cdb6250c285c8433883fc37fdfe72b6f0254edde'


def artifact(ref):
    path=Path(ref['path']).resolve()
    require(ROOT/'logs' in path.parents and path.is_file() and file_sha(path)==ref['sha256'], 'Fresh own raw proof artifact missing/drift')
    return json.loads(path.read_text())


def validate_training_readiness(manifest):
    proof=artifact(manifest['production_readiness'])
    require(proof['status']=='REVIEWED' and proof['scope']=='SHARED32B_PIPELINE_NUMERICAL_RELEASE_COEXISTENCE_NOT_TOPD_COMPLETION',
            'New shared32B independent readiness required; old3B/CPU proofs forbidden')
    require(proof['pipeline_sha256']==PIPELINE and proof['original_method_sha256']==METHOD
            and proof['parameter_count']==32763876352 and proof['teacher_vocabulary']==152064
            and proof['model_dtype']=='float32' and proof['workers']==8,'Actual full32B original sharedpipeline required')
    require(proof['teacher_receipt_sha256']==manifest['teacher_receipt_sha256']
            and proof['leased_uuids']==manifest['leased_uuids'],'Teacher/source/physicalGPU identity mismatch')
    review=artifact(proof['independent_review'])
    unsigned={k:v for k,v in proof.items() if k!='independent_review'}
    require(review['status']=='PASS' and review['approval_scope']==proof['scope']
            and review['reviewed_payload_sha256']==digest(canonical(unsigned)), 'Independent reviewed payload binding')
    provenance=artifact(proof['teacher_provenance'])
    require(proof['teacher_provenance']['sha256']==manifest['teacher_receipt_sha256'],
            'Canonical teacher provenance receipt SHA mismatch')
    gate_ref=proof['teacher_gate_source']
    path=Path(gate_ref['path']).resolve()
    require(path==ROOT/'tools/topd-32b-native-score-gate-r1-v3/score_gate.py'
            and gate_ref['sha256']=='f00962e5915f9497597596a35fbd68af8f84490b88970b73e9499d908b47f2fb'
            and file_sha(path)==gate_ref['sha256'],
            'Actual reviewed canonical teacher gate source missing/drift')
    import importlib.util
    spec=importlib.util.spec_from_file_location('canonical32b_teacher_identity_gate',path)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    model=gate.check_identity(provenance,ROOT/'tools/topd-32b-pipeline-candidate')
    require(provenance['expected_fp32_parameter_sha256']==proof['fp32_parameter_sha256']
            and provenance['model_inventory']['sha256']==proof['hf_inventory_sha256'],
            'Canonical CPU/model identity does not match native32B proof')
    # Raw32B scorer receipts, not only normalized Boolean PASS.
    numerical=artifact(proof['native_numerical_release'])
    require(numerical['status']=='SCORER_NUMERICAL_RELEASE_COMPLETE_UNREVIEWED_NOT_TRAINING_READY'
            and max(numerical['actual_tokens_per_row'])==6144,'Actual longest-context native evidence')
    phases=numerical['phase_records']
    parallel=[p for p in phases if p['kind']=='PIPELINE']
    require(len(parallel)==2 and [p['repeat'] for p in parallel]==[0,1]
            and all(p['allclose'] is True and p['atol']==p['rtol']==1e-4 for p in parallel),'Two strict originalGPU parity rounds')
    for phase in phases:
        raw=phase['record'];initial=raw['initial_cpu'];final=raw['released_cpu'];alloc=raw['allocators_released']
        require(raw['status']=='RELEASED' and raw['original_method_sha256']==METHOD
                and initial['parameter_count']==32763876352 and initial['parameter_sha256']==proof['fp32_parameter_sha256']
                and initial['parameter_sha256']==final['parameter_sha256']
                and initial['buffer_sha256']==final['buffer_sha256'] and final['parameter_devices']==['cpu']
                and len(alloc)==8 and {a['uuid'] for a in alloc}==set(manifest['leased_uuids'])
                and all(a['allocated_bytes']==a['reserved_bytes']==0 for a in alloc),'Full frozen/all8 actual release evidence')
    coexist=artifact(proof['native_training_phase_coexistence'])
    require(coexist['status']=='INDEPENDENTLY_REVIEWED_SHARED32B_TRAINING_PHASE_COEXISTENCE'
            and coexist['pipeline_sha256']==PIPELINE and coexist['teacher_fp32_parameter_sha256']==proof['fp32_parameter_sha256']
            and coexist['leased_uuids']==manifest['leased_uuids']
            and coexist['teacher_phase_rollout_kv_released'] is True
            and coexist['next_rollout_kv_rebuilt_after_teacher_release'] is True
            and coexist['local_gpu_e5_full8_faiss'] is True
            and coexist['actual_own_cleanup'] is True,'Actual native actor/rollout/retrieval transition proof missing')
    return proof
