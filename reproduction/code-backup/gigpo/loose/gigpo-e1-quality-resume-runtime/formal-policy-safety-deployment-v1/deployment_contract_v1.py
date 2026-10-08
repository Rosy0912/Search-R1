"""Formalv4 deployment templates; actual calibration/threshold evidence is mandatory.

No CUDA, Slurm or target mutation in this contract. Runtime/source52 are immutable.
The new directories are separate from every native/calibration/helper history.
"""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
HELPERS=ROOT/'tools/policy-safety-formal-52befe9-v4'
CONTROLS=ROOT/'tools/policy-safety-formal-controls-52befe9-v1'
SOURCE='52befe968c856162d1bc79cd0171bb38309f4f9e'
REPO=ROOT/'repo-policy-safety-52befe9'
SCOPE='GIGPO_POLICY_SAFETY_FORMAL_RESUME25_TO300'
RUNTIME_MANIFEST_SHA='a87fa9d98cc0606a2b7fe8fb4212e1dd64a779bc355e60a7e332209f240674d4'
RUNTIME_REVIEW_SHA='8974dfb61caee1e95eb2637a1c6cfcac11d43c98872af5fe8cf1527ad931019b'
PARENT_CANDIDATE_SHA='264fc8ebdf5f2c8d81c762a744a4ae0d25b39d8a4efdcd137c727e12c648e045'
PARENT_READY_SHA='7cc3098a35d6f56e809e293e2f26642cc365eac8c49c1ddaedb592e989c79ce2'
CANDIDATE=ROOT/'logs/FORMAL-PROTECTION-CANDIDATE-TEMPLATE-52BEFE9-v1.json'
READY=ROOT/'logs/policy-safety-formal-52befe9-reviewed-binding-v1.json'
INTENT=ROOT/'logs/policy-safety-formal-52befe9-submit-intent-v1.json'
RECEIPT=ROOT/'logs/policy-safety-formal-52befe9-submit-receipt-v1.json'
INSTALL_INTENT=ROOT/'logs/policy-safety-formal-52befe9-install-intent-v1.json'
INSTALL_RECEIPT=ROOT/'logs/policy-safety-formal-52befe9-install-receipt-v1.json'
BUDGET=dict(retained25=25,new_outer275=275,stop300=300,mini256=256,adam_per_outer10=10,micro_per_outer160=160,warmup57=57)
DYNAMIC_FIELDS=('production_artifact','probability_identity','rollout_backend','policy_safety','expected_config_preview')
SHA_RE=re.compile(r'[0-9a-f]{64}')


def require(condition,reason):
    if not condition: raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def record(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def valid_record(item):
    require(isinstance(item,dict) and set(item)=={'path','bytes','sha256'},'Required immutable path/bytes/SHA missing')
    path=Path(item['path'])
    require(path.is_absolute() and '..' not in path.parts and str(path)==item['path']
        and type(item['bytes']) is int and item['bytes']>=0 and isinstance(item['sha256'],str)
        and SHA_RE.fullmatch(item['sha256']) is not None,'Invalid immutable record')


def read_bound(item):
    valid_record(item);path=Path(item['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Missing/changed original proof')
    return json.loads(path.read_text())


def closure(items):
    require(isinstance(items,list) and items and len({item['path'] for item in items})==len(items),'Empty/duplicate closure')
    for item in items: valid_record(item)
    return {item['path']:item for item in items}


def write_once(path,value):
    path=Path(path)
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    directory=os.open(str(path.parent),os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def validate_candidate(candidate):
    require(candidate['status']=='CANDIDATE_NOT_READY' and candidate['scope']==SCOPE,'Only a new formal NOT_READY candidate is permitted')
    require(candidate['code_commit']==SOURCE and candidate['repo']==str(REPO)
        and candidate['helpers']==str(HELPERS) and candidate['controls']==str(CONTROLS)
        and candidate['node']=='h100-8' and candidate['formal_budget']==BUDGET,
        'Wrong source/node/directory/25to300 budget')
    require(candidate['training_submitted'] is False and candidate['extra_AB_diagnostic_Adam_required']==0
        and candidate['first_formal_hook_acceptance_stage']=='SAME_PROCESS26_TO30_BEFORE31'
        and candidate['runtime_full_SHA_before_GPU_required'] is True,'Unreviewed launch or mandatory extra experiment')
    files=closure(candidate['files']); evidence=closure(candidate['evidence'])
    manifest_record=candidate['runtime_manifest'];review_record=candidate['runtime_static_review']
    require(manifest_record['sha256']==RUNTIME_MANIFEST_SHA and review_record['sha256']==RUNTIME_REVIEW_SHA
        and evidence.get(manifest_record['path'])==manifest_record and evidence.get(review_record['path'])==review_record,
        'Formalv4 static identity not uniquely closed')
    manifest=read_bound(manifest_record);review=read_bound(review_record)
    require(review['status']=='PASS' and review['scope']=='LOCAL_STATIC_AND_CPU_DRAFT_ONLY_NOT_DEPLOYMENT_READY'
        and review['runtime_manifest_sha256']==RUNTIME_MANIFEST_SHA and review['source_commit']==SOURCE
        and review['actual_formal_GPU_hook_accepted'] is False and manifest['source_commit']==SOURCE,
        'Static original relabeled as actual formal GPU acceptance')
    frozen={Path(item['path']).name:{k:item[k] for k in ('bytes','sha256')} for item in manifest['files']}
    runtime=candidate['runtime_helpers'];require(len(runtime)==29 and len({Path(item['path']).name for item in runtime})==29,'Formal runtime inventory differs')
    require(all(Path(item['path']).parent==HELPERS and files.get(item['path'])==item
        and frozen.get(Path(item['path']).name)=={k:item[k] for k in ('bytes','sha256')} for item in runtime),
        'Changed frozen formalv4 runtime bytes or path')
    require(frozen['metadata_colocation_v1.py']['sha256']=='39e3226c82132524e57ee1adc964f607da6a82314a784c1081c505fd10ea6b9e','Factory identity changed')
    controls=candidate['control_helpers'];require(len(controls)==7 and len({item['path'] for item in controls})==7
        and all(Path(item['path']).parent==CONTROLS and files.get(item['path'])==item for item in controls),'Control inventory not separately isolated')
    parent_record=candidate['inherited_candidate'];ready_record=candidate['inherited_launch_READY_not_actual_calibration_acceptance']
    require(parent_record['sha256']==PARENT_CANDIDATE_SHA and ready_record['sha256']==PARENT_READY_SHA
        and evidence.get(parent_record['path'])==parent_record and evidence.get(ready_record['path'])==ready_record,
        'Parent identity missing: launchREADY is not actual calibration acceptance')
    parent=read_bound(parent_record);parent_ready=read_bound(ready_record)
    require(parent['code_commit']==SOURCE and parent_ready['code_commit']==SOURCE
        and parent_ready['source_native_acceptance']['job_id']=='3267','Wrong inherited source/native')
    removed={item['path'] for item in parent['runtime_helpers']}
    removed|={item['path'] for item in parent['files'] if Path(item['path']).parent==Path(parent['controls'])}
    base=[item for item in parent['files'] if item['path'] not in removed]
    require(len(base)==candidate['inherited_file_count']==42327
        and candidate['inherited_file_records_sha256']==hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        and all(files.get(item['path'])==item for item in base),'Inherited model/data/env/source/retrieval SHA closure changed')
    require(all(evidence.get(item['path'])==item for item in parent['evidence']),'Inherited proof records changed/omitted')
    guard=candidate['guard_identity'];require(guard==parent_ready['source_native_acceptance']['guard_identity'],'Guard/native ownership identity differs')
    for item in guard.values():
        found=files.get(item['path']);require(found is not None and found['sha256']==item['sha256'],'Guard lacks exact unique file closure')
    require(candidate['quality_selection_control']==parent['quality_selection_control']
        and candidate['restore_checkpoint']==parent['restore_checkpoint']
        and candidate['restore_manifest_sha256']==parent['restore_manifest_sha256']
        and candidate['restore_step']==25 and candidate['common_manifest_sha256']==parent['common_manifest_sha256']
        and candidate['fresh_model_revision']==parent['fresh_model_revision'],'Full25/common selection changed')
    require(candidate['native_acceptance']==dict(parent_ready['source_native_acceptance'],native_helpers=str(ROOT/'tools/policy-safety-native-52befe9-v2'))
        and candidate['native_status']=='PASS','Existing native3267 acceptance altered or replaced')
    return candidate


def proposal(candidate):
    """A local object for validation only; no writer and no readiness inference."""
    validate_candidate(candidate)
    require(candidate['preparation_only'] is False and candidate['actual_v9_calibration_acceptance'] is not None
        and candidate['production_artifact'] is not None and candidate['expected_config_preview'] is not None
        and candidate['probability_identity']['acceptance'] is not None
        and candidate['probability_identity']['startup_static_physical_review'] is not None
        and candidate['rollout_backend']['calibration_backend_acceptance'] is not None,
        'Actual v9 probability/10Adam/calibration/threshold/tolerance and startup physical review still pending')
    import copy
    ready=copy.deepcopy(candidate);ready['status']='REVIEWED_FOR_PRODUCTION'
    from formal_protection_contract_v1 import admit_binding
    admit_binding(ready,SOURCE)
    # Both preview and every independent role are explicit originals in evidence.
    evidence=closure(ready['evidence']); preview=read_bound(ready['expected_config_preview'])
    require(evidence.get(ready['expected_config_preview']['path'])==ready['expected_config_preview']
        and preview['source_commit']==SOURCE and preview['mode']=='production','Resolved production config original missing')
    artifact=read_bound(ready['production_artifact'])
    require(ready['actual_v9_calibration_acceptance']==artifact['actual_proofs']['calibration'],'Actual calibration original differs from production artifact')
    proofs=[ready['production_artifact'],ready['probability_identity']['acceptance'],ready['probability_identity']['startup_static_physical_review'],ready['rollout_backend']['calibration_backend_acceptance']]+list(artifact['actual_proofs'].values())
    for item in proofs: require(evidence.get(item['path'])==item,'Actual acceptance not exact evidence closure')
    return ready


def validate_ready(ready):
    require(ready['status']=='REVIEWED_FOR_PRODUCTION' and ready['deployment_freeze'] is not None,'No actual production READY')
    import copy
    candidate=copy.deepcopy(ready);candidate['status']='CANDIDATE_NOT_READY';candidate.pop('deployment_freeze',None)
    validated=proposal(candidate)
    require(all(validated[key]==ready[key] for key in candidate),'ProductionREADY differs from fully accepted candidate')
    freeze=ready['deployment_freeze']
    for key in ('candidate','physical','review'):read_bound(freeze[key])
    require(read_bound(freeze['candidate'])==candidate,'READY differs from independently reviewed original candidate')
    review=read_bound(freeze['review']);physical=read_bound(freeze['physical'])
    require(review['status']=='PASS' and review['scope']==SCOPE and review['candidate_sha256']==freeze['candidate']['sha256']
        and review['physical_sha256']==freeze['physical']['sha256'] and physical['candidate_sha256']==freeze['candidate']['sha256'],
        'Wrong original production deployment freeze proof')
    for field in ('files','evidence'):
        require(len(physical[field])==len({r['path'] for r in physical[field]})
            and [{k:r[k] for k in ('path','bytes','sha256')} for r in physical[field]]==candidate[field],
            'READY physical path/bytes/SHA closure differs')
    require(physical['code_commit']==SOURCE and physical['cuda_initialized'] is False
        and physical['actual_formal_GPU_accepted'] is False and physical['guard_identity']==candidate['guard_identity']
        and review['actual_formal_GPU_accepted'] is False and review['extra_AB_diagnostic_Adam_required']==0
        and review['freeze_helper_sha256']==closure(candidate['files'])[str(CONTROLS/'freeze_ready_v1.py')]['sha256'],
        'READY physical/static scope or frozen helper identity differs')
    return ready
