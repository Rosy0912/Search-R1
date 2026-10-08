"""New immutable calibration draft from actual read-only evidence/tokenizer proof; never READY."""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.with_name('safety-calibration-deployment-v2')
sys.path.insert(0, str(V2))
from deployment_contract_v2 import ROOT, HELPERS, SOURCE, validate_candidate, require, sha, write_once
from prepare_candidate_v2 import make_archive

LOCAL = HERE.parents[2]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
PREDECESSOR = '8e8c0e624c0e96f5a53cba405efa2ddd4cf3b373ea7e050afbf933716ccd8874'
RAW = 'ba88813d3320d9b670312e00db151537e09dbfccf44b9fa35c41ff8fc258d1de'


def main():
    parent_path = OUT / 'CALIBRATION-CANDIDATE-52BEFE9-v2.json'
    raw_path = OUT / 'CALIBRATION-EVIDENCE-TOKENIZER-READONLY-v3.json'
    require(sha(parent_path) == PREDECESSOR and sha(raw_path) == RAW, 'Immutable predecessor/raw proof changed')
    parent = json.loads(parent_path.read_text())
    value = json.loads(raw_path.read_text())
    require(value['node'] == 'h100-8' and value['source_commit'] == SOURCE and value['inherited_evidence_count'] == 156
            and not value['cuda_initialized'] and not value['weights_loaded'] and not value['remote_files_written']
            and value['large_asset_hashes_performed'] == 0 and value['native_acceptance'] is None,
            'Not authorized actual read-only identity proof')
    by_path = {r['path']: r for r in value['evidence']}
    require(len(by_path) == 156, 'Duplicate actual proof coverage')
    candidate = copy.deepcopy(parent)
    for entry in candidate['evidence']:
        if entry['bytes'] is None:
            observed = by_path[entry['path']]
            require(observed['sha256'] == entry['sha256'], 'Actual inherited proof SHA differs')
            entry['bytes'] = observed['bytes']
    require(all(isinstance(r['bytes'], int) for r in candidate['files'] + candidate['evidence']), 'Unfilled actual byte identity')
    require(value['actual_hf_pad_token_id'] == 151643 and value['actual_hf_eos_token_id'] == 151645,
            'Actual official tokenizer IDs differ from this reviewed preparation')
    candidate.update(pad_token_id=value['actual_hf_pad_token_id'], source_native_acceptance=None,
        native_pending_job='3267', native_pending_job_acceptance=None,
        predecessor_calibration_candidate_sha256=PREDECESSOR,
        preparation_scope='Actual inherited proof bytes/SHA and official CPU tokenizer identity; new full native still pending')
    candidate['actual_tokenizer_identity'] = dict(source_commit=SOURCE, official_revision=value['official_revision'],
        model_path=value['official_model_path'], pad_token_id=value['actual_hf_pad_token_id'], eos_token_id=value['actual_hf_eos_token_id'],
        model_pad_token_id=value['model_pad_token_id'], generation_pad_token_id=value['generation_pad_token_id'],
        tokenizer_import_path=value['tokenizer_import_path'], cuda_initialized=False, weights_loaded=False,
        raw_receipt=dict(path=str(ROOT/'logs'/raw_path.name),sha256=RAW))
    sources = {}
    implementation = json.loads((OUT/'CALIBRATION-IMPLEMENTATION-v1.json').read_text())
    for entry in implementation['files']:
        p = Path(entry['path'])
        require(sha(p) == entry['sha256'], 'Frozen runtime helper changed')
        sources['runtime/'+p.name] = p
    for p in sorted(V2.glob('*.py')):
        if p.name.startswith('test_'):
            continue
        remote = str(Path(candidate['controls'])/p.name)
        record = next(r for r in candidate['files'] if r['path'] == remote)
        require(sha(p) == record['sha256'], 'Frozen v2 controller changed')
        sources['control/'+p.name] = p
    local_evidence = [OUT/'CALIBRATION-STATIC-INDEPENDENT-REVIEW-v2.json',OUT/'CALIBRATION-IMPLEMENTATION-v1.json',
        OUT/'CALIBRATION-DESIGN-v1.md',OUT/'CALIBRATION-CPU-CONTRACT-v1.log',OUT/'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json']
    for p in local_evidence:
        record = next(r for r in candidate['evidence'] if r['path'] == str(ROOT/'logs'/p.name))
        require(sha(p) == record['sha256'], 'Frozen v2 proof changed')
        sources['evidence/'+p.name] = p
    for p in (raw_path,OUT/'CALIBRATION-EVIDENCE-TOKENIZER-READONLY-v3-transport.log',OUT/'CALIBRATION-DEPLOYMENT-v3.md'):
        candidate['evidence'].append(dict(path=str(ROOT/'logs'/p.name),bytes=p.stat().st_size,sha256=sha(p)))
        sources['evidence/'+p.name] = p
    validate_candidate(candidate)
    target = OUT/'CALIBRATION-CANDIDATE-52BEFE9-v3.json'
    write_once(target,candidate)
    sources['candidate/'+target.name] = target
    archive = HERE/'policy-safety-calibration-52befe9-v3-package.tar.gz'
    make_archive(archive,sources)
    write_once(OUT/'CALIBRATION-PACKAGE-v3.json',dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',
        candidate=dict(path=str(target),sha256=sha(target)),archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),
        predecessor_candidate_sha256=PREDECESSOR,readonly_raw_sha256=RAW,native_acceptance=None,submitted_job=None,
        retained_training_updates_contributed=0,all_42343_file_sizes_established=True,all_evidence_sizes_established=True,
        inherited_proofs_actual_sha_checked=156,official_tokenizer_cpu_identity_verified=True,
        runtime_13_helpers_unchanged=True,controllers_v2_unchanged=True,package_members=sorted(sources)))
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),
        files=len(candidate['files']),evidence=len(candidate['evidence']),ready=False,submitted=False)))


if __name__ == '__main__':
    main()
