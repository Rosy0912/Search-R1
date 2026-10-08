"""Template: freeze only full real production originals; null evidence refuses before write."""
import copy
import json
from pathlib import Path
from deployment_contract_v1 import (SOURCE,SCOPE,READY,record,read_bound,sha,require,write_once,proposal,validate_ready)


def freeze(candidate_record,physical_record,review_record,target=READY):
    require(Path(target)==READY,'Only this new versioned production READY path')
    candidate=read_bound(candidate_record);ready=proposal(candidate)
    physical=read_bound(physical_record);review=read_bound(review_record)
    require(physical['status']=='PHYSICAL_FORMAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU'
        and physical['candidate_sha256']==candidate_record['sha256'] and physical['code_commit']==SOURCE
        and physical['cuda_initialized'] is False and physical['actual_formal_GPU_accepted'] is False,
        'Wrong original formal deployment physical scope')
    for field in ('files','evidence'):
        require(len(physical[field])==len({r['path'] for r in physical[field]})
            and [{k:r[k] for k in ('path','bytes','sha256')} for r in physical[field]]==candidate[field],
            'Physical path/bytes/SHA closure differs or duplicates')
    preview=read_bound(candidate['expected_config_preview'])
    require(physical['resolved_config']==preview['resolved'] and physical['config_preview_sha256']==candidate['expected_config_preview']['sha256']
        and physical['guard_identity']==candidate['guard_identity']
        and physical['pad_token_id']==candidate['probability_identity']['pad_token_id']
        and physical['eos_token_id']==candidate['probability_identity']['eos_token_id'],
        'Physical config/guard/PAD/EOS differs from actual accepted probability source')
    require(review['status']=='PASS' and review['scope']==SCOPE
        and review['candidate_sha256']==candidate_record['sha256'] and review['physical_sha256']==physical_record['sha256']
        and review['freeze_helper_sha256']==sha(Path(__file__)) and review['source_commit']==SOURCE
        and review['native_acceptance_sha256']==candidate['native_acceptance']['review_sha256']
        and review['calibration_acceptance_sha256']==read_bound(candidate['production_artifact'])['actual_proofs']['calibration']['sha256']
        and review['runtime_manifest_sha256']==candidate['runtime_manifest']['sha256']
        and review['actual_formal_GPU_accepted'] is False and review['extra_AB_diagnostic_Adam_required']==0,
        'No exact independent static/physical production launch review; not actual formal GPU acceptance')
    ready['deployment_freeze']=dict(candidate=candidate_record,physical=physical_record,review=review_record)
    validate_ready(ready);write_once(target,ready)
    return dict(binding=str(target),sha256=sha(target),submitted=False,actual_formal_GPU_accepted=False)


if __name__=='__main__':
    import sys
    require(len(sys.argv)==4,'Need actual candidate/physical/review immutable record JSON files; no optional defaults')
    print(json.dumps(freeze(*[json.loads(Path(x).read_text()) for x in sys.argv[1:]])))
