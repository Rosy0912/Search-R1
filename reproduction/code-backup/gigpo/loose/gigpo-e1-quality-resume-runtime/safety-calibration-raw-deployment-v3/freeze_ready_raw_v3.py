"""Freeze a new immutable READY only after actual native and physical independent reviews."""
import copy
import json
import sys
from pathlib import Path

from deployment_contract_raw_v3 import (SOURCE, SCOPE, validate_candidate, validate_ready, read_native_acceptance,
    require, sha, read_bound, write_once, match_closure, validate_inheritance)


def freeze(candidate_path, candidate_sha, physical_path, physical_sha, review_path, review_sha, native_spec_path, target):
    c = read_bound(dict(path=str(candidate_path), sha256=candidate_sha))
    validate_candidate(c)
    physical = read_bound(dict(path=str(physical_path), sha256=physical_sha))
    review = read_bound(dict(path=str(review_path), sha256=review_sha))
    acceptance = json.loads(Path(native_spec_path).read_text())
    read_native_acceptance(acceptance, c)  # Null, old3267/source52 and any fresh fixture without actual packing acceptance refuse here.
    match_closure(physical['metadata'],c['files'],'actual files')
    match_closure(physical['evidence'],c['evidence'],'actual evidence')
    parent=read_bound(c['native_parent_candidate']);origin=read_bound(c['inherited_physical'])
    validate_inheritance(c,parent,origin)
    require(physical['candidate_sha256'] == candidate_sha and physical['code_commit'] == SOURCE
            and physical['status'] == 'PHYSICAL_CODE_PROOF_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',
            'Physical proof not this candidate')
    require(physical['cuda_initialized'] is False and physical['target_resolved_config']==read_bound(c['expected_config_preview'])['resolved']
            and physical['config_preview_sha256']==c['expected_config_preview']['sha256']
            and physical['actual_runner_inputs_observed'] is False and physical['actual_probability_accepted'] is False,
            'Physical config/raw diagnostic scope differs')
    require(review['status'] == 'PASS' and review['scope'] == SCOPE and review['candidate_sha256'] == candidate_sha
            and review['physical_sha256'] == physical_sha and review['freeze_helper_sha256'] == sha(Path(__file__))
            and review['native_result_sha256'] == acceptance['result_sha256']
            and review['native_review_sha256'] == acceptance['review_sha256']
            and review['native_terminal_sha256'] == acceptance['terminal_sha256']
            and review['native_physical_payload_sha256'] == acceptance['physical_payload_sha256'],
            'Independent deployment acceptance differs')
    # A separately reviewed physical proof cannot substitute different bytes for
    # an immutable candidate path, even when the path and file length are equal.
    def closure(records, expected, label):
        require(len(records) == len({r['path'] for r in records}), 'Duplicate actual '+label+' closure path')
        by_path = {r['path']:r for r in records}
        require(set(by_path) == {r['path'] for r in expected}, 'Incomplete actual '+label+' closure')
        for entry in expected:
            actual = by_path[entry['path']]
            require({key:actual[key] for key in ('path','bytes','sha256')} == entry,
                    'Actual '+label+' bytes/SHA differs from immutable candidate: '+entry['path'])
    closure(physical['metadata'], c['files'], 'file')
    closure(physical['evidence'], c['evidence'], 'evidence')
    ready = copy.deepcopy(c)
    for path, expected in ((physical_path, physical_sha), (review_path, review_sha),
                           (acceptance['result_path'], acceptance['result_sha256']),
                           (acceptance['review_path'], acceptance['review_sha256']),
                           (acceptance['native_binding_path'], acceptance['native_binding_sha256']),
                           (acceptance['terminal_path'], acceptance['terminal_sha256']),
                           (acceptance['physical_payload_path'], acceptance['physical_payload_sha256'])):
        p = Path(path)
        require(sha(p) == expected, 'Acceptance proof changed before freeze')
        record = dict(path=str(p), bytes=p.stat().st_size, sha256=expected)
        found = [r for r in ready['evidence'] if r['path'] == str(p)]
        require(not found or found == [record], 'Conflicting existing proof')
        if not found:
            ready['evidence'].append(record)
    ready.update(status='REVIEWED_FOR_NATIVE_ONLY', readiness='READY_FOR_RAW_PROBABILITY_PROBE_THEN_WAIT_ONLY',
        source_native_acceptance=acceptance, pad_token_id=physical['pad_token_id'], eos_token_id=physical['eos_token_id'],
        candidate_sha256=candidate_sha, physical_path=str(physical_path), physical_sha256=physical_sha,
        deployment_review_path=str(review_path), deployment_review_sha256=review_sha,
        actual_runtime_config_path='Assigned isolated run/resolved-config.yaml after unique Slurm launch; SHA recorded by frozen launcher')
    validate_ready(ready)
    write_once(target, ready)
    return dict(binding=str(target), binding_sha256=sha(target), ready=True, submitted=False)


if __name__ == '__main__':
    require(len(sys.argv) == 9, 'Need candidate/SHA, physical/SHA, review/SHA, real native-spec and new output path')
    print(json.dumps(freeze(*sys.argv[1:])))
