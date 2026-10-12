"""Explicit h100-8 immutable installation only; no native, model, READY or sbatch."""
import argparse
import hashlib
import json
import os
import socket
import stat
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
HELPERS = ROOT/'tools/policy-safety-calibration-52befe9-v9'
CONTROLS = ROOT/'tools/policy-safety-calibration-controls-52befe9-v9'
CANDIDATE = ROOT/'logs/CALIBRATION-CANDIDATE-52BEFE9-v9.json'
INTENT = ROOT/'logs/policy-safety-calibration-52befe9-install-intent-v9.json'
RECEIPT = ROOT/'logs/policy-safety-calibration-52befe9-install-receipt-v9.json'
PACKAGE_SHA = '4f86493769f14634b34362f4a8a31c425b4c73aa9ee574132e2e250661a87c17'
CANDIDATE_SHA = '264fc8ebdf5f2c8d81c762a744a4ae0d25b39d8a4efdcd137c727e12c648e045'
IMPLEMENTATION_SHA = 'e1fb643f082eac44d3e91c6151c1956f635ff50f4810caa57a93c97243f67143'
RUNTIME_MANIFEST_NAME = 'CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json'
EVIDENCE_NAMES = {'CALIBRATION-DEPLOYMENT-STATIC-COUNTEREXAMPLES-v4.py', 'CALIBRATION-COOLDOWN-CPU-v6-v3.log', 'backends-xformers.py', 'CALIBRATION3281-FAILURE-INDEPENDENT-REVIEW-v1.json', 'CALIBRATION-BACKEND-KNOWN-ATTRIBUTE-CHAIN-v9.json', 'CALIBRATION-DATAROW-TARGET-CPU-v9.json', 'CALIBRATION-DEPLOYMENT-CPU-v9.log', 'CALIBRATION-RUNTIME-DIFF-v9-v2.patch', 'CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json', 'PROBABILITY3281-TRACE-RAW-METADATA-20261006T000029859946Z.json', 'CALIBRATION-RUNTIME-DIFF-v6-v3.patch', 'CALIBRATION-DEPLOYMENT-v7.md', 'CALIBRATION-DATAROW-INDEPENDENT-REVIEW-v8.json', 'policy-safety-calibration-52befe9-reviewed-binding-v8.json', 'CALIBRATION-BACKEND-CPU-v9-v2.log', 'backends-flash_attn.py', 'layers-sampler.py', 'CALIBRATION-DEPLOYMENT-CPU-v7.log', 'CALIBRATION-FACTORY-CPU-v7.stdout.log', 'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json', 'CALIBRATION-DEPLOYMENT-PROOF-INPUTS-v8.json', 'CALIBRATION-DATAROW-TARGET-CPU-v8.json', 'CALIBRATION-DEPLOYMENT-PROOF-INPUTS-v7.json', 'CALIBRATION-BACKEND-INDEPENDENT-REVIEW-v9.json', 'qwen2-modeling_qwen2.py', 'CALIBRATION-DATAROW-CPU-v8.stderr.log', 'CALIBRATION3276-FAILURE-PAYLOAD-v1.json', 'CALIBRATION-DEPLOYMENT-PROOF-INPUTS-v9.json', 'CALIBRATION-IMPLEMENTATION-v1.json', 'CALIBRATION-DEPLOYMENT-CPU-v5.log', 'CALIBRATION-RUNTIME-DIFF-v7.patch', 'CALIBRATION-DEPLOYMENT-v3.md', 'CALIBRATION-EVIDENCE-TOKENIZER-READONLY-v3.json', 'CALIBRATION-DEPLOYMENT-v8.md', 'CALIBRATION3276-STARTUP-MRO-INDEPENDENT-DIAGNOSIS-v1.json', 'CALIBRATION-DEPLOYMENT-CONTROL-STATIC-INDEPENDENT-REVIEW-v3.json', 'test_metadata_colocation_v7_actual.py', 'CALIBRATION-DATAROW-TARGET-CPU-INDEPENDENT-REVIEW-v8.json', 'CALIBRATION-DATAROW-TARGET-CPU-RAW-v9.json', 'CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json', 'CALIBRATION-DEPLOYMENT-v5.md', 'CALIBRATION-CANDIDATE-52BEFE9-v8.json', 'CALIBRATION3272-CGROUP-RELEASE-v2.json', 'CALIBRATION-STATIC-INDEPENDENT-REVIEW-v2.json', 'CALIBRATION-DESIGN-v1.md', 'qwen2.5-3b-instruct-aa8e725-config.json', 'CALIBRATION-DATAROW-CPU-v8.json', 'CALIBRATION-DATAROW-TARGET-CPU-INDEPENDENT-REVIEW-v9.json', 'triton-cross_entropy.py', 'vllm-engine-arg_utils.py', 'CALIBRATION-COOLDOWN-INDEPENDENT-REVIEW-v6-v3.json', 'CALIBRATION-BACKEND-V9-SINGLE-LITERAL-DRAFT-v1.json', 'CALIBRATION-DEPLOYMENT-STATIC-COUNTEREXAMPLES-v4.json', 'CALIBRATION-COLOCATION-INDEPENDENT-REVIEW-v7.json', 'vllm-flash-attn-init.py', 'CALIBRATION-CPU-CONTRACT-v1.log', 'CALIBRATION-RUNTIME-IMPLEMENTATION-v6-v3.json', 'attention-selector.py', 'CALIBRATION-FACTORY-CPU-v7.stderr.log', 'CALIBRATION-DATAROW-TARGET-CPU-RAW-v8.json', 'CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json', 'CALIBRATION-FACTORY-CPU-v7.json', 'CALIBRATION-DEPLOYMENT-PROOF-INPUTS-v6.json', 'CALIBRATION-BACKEND-INSTALLED-EVIDENCE-v9.json', 'models-qwen2.py', 'CALIBRATION-DEPLOYMENT-CPU-v8-v2.log', 'PROBABILITY-MISMATCH-BOUND-SOURCE-RECEIPT-v3.json', 'test_calibration_backend_v9_v2.py', 'PROBABILITY-MISMATCH-BOUND-SOURCE-RECEIPT-v1.json', 'CALIBRATION3272-TERMINAL-PAYLOAD-v1.json', 'PROBABILITY-MISMATCH-BOUND-SOURCE-RECEIPT-v4.json', 'PROBABILITY3281-FAILURE-RAW-METADATA-20261005T235846704624Z.json', 'vllm-config.py', 'CALIBRATION-DEPLOYMENT-CPU-v6.log', 'CALIBRATION-BACKEND-TARGET-CPU-IMPORT-v9.json', 'ops-paged_attn.py', 'CALIBRATION-DEPLOYMENT-v6.md', 'CALIBRATION3281-FAILURE-PAYLOAD-v2.json', 'CALIBRATION-DEPLOYMENT-v9.md', 'layers-rotary_embedding.py', 'xformers-fmha-dispatch.py', 'CALIBRATION3276-FAILURE-INDEPENDENT-REVIEW-v1.json', 'CALIBRATION-RUNTIME-DIFF-v8.patch', 'CALIBRATION-DATAROW-CPU-v8.stdout.log', 'CALIBRATION-DEPLOYMENT-v4.md', 'PROBABILITY-MISMATCH-3281-DIAGNOSIS-v1.json', 'CALIBRATION3272-FAILURE-INDEPENDENT-REVIEW-v2.json', 'CALIBRATION-DEPLOYMENT-CPU-v4.log', 'test_calibration_datarow_v8_actual.py', 'CALIBRATION-EVIDENCE-TOKENIZER-READONLY-v3-transport.log'}
EVIDENCE_COUNT = 363
REQUIRED_EXISTING_COUNT = 273
SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
RUNTIME_NAMES = {
    'calibration_audit_v1.py','calibration_contract_v1.py','calibration_driver_v1.py','calibration_run_v1.py',
    'calibration_state_v1.py','calibration_v1.sbatch','calibration_worker_v1.py','owned_runtime_resume_v1.py',
    'ray_audit_resume_v1.py','safety_audit_v1.py','test_calibration_audit_v1.py','test_calibration_contract_v1.py',
    'test_calibration_state_v1.py','metadata_colocation_v1.py'}
CONTROL_NAMES = {'deployment_contract_v9.py','freeze_ready_v9.py','prepare_candidate_v9.py','submit_once_v9.py','verify_physical_v9.py'}
MEMBERS = ({'runtime/'+name for name in RUNTIME_NAMES} | {'control/'+name for name in CONTROL_NAMES}
           | {'evidence/'+name for name in (EVIDENCE_NAMES or ())}
           | {'candidate/'+CANDIDATE.name,'cpu/test_deployment_v9.py'})


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_archive(archive, expected_sha=None):
    expected_sha = PACKAGE_SHA if expected_sha is None else expected_sha
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, 'Unsealed package identity')
    require(sha_file(archive) == expected_sha, 'Package SHA differs')
    payloads = {}
    with tarfile.open(archive,'r:gz') as stream:
        members = stream.getmembers()
        require(len(members) == len(MEMBERS) and len({m.name for m in members}) == len(members),
                'Unexpected or duplicate tar inventory')
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and len(path.parts) == 2 and '..' not in path.parts
                    and '.' not in path.parts and str(path) == member.name and member.name in MEMBERS,
                    'Unexpected or traversal tar path')
            require(member.isfile() and not member.issym() and not member.islnk() and not member.issparse(),
                    'Symlink/hardlink/sparse/directory/non-file tar member forbidden')
            require(0 <= member.size <= 12*1024*1024, 'Oversized tar member')
            total += member.size
            require(total <= 64*1024*1024, 'Oversized uncompressed package')
            data = stream.extractfile(member).read()
            require(len(data) == member.size, 'Truncated tar bytes')
            payloads[member.name] = data
    require(set(payloads) == MEMBERS, 'Missing exact package member')
    return payloads


def installation_plan(payloads):
    require(all(value is not None for value in (PACKAGE_SHA,CANDIDATE_SHA,IMPLEMENTATION_SHA,
            RUNTIME_MANIFEST_NAME,EVIDENCE_NAMES,EVIDENCE_COUNT,REQUIRED_EXISTING_COUNT)),
            'Unsealed immutable installer; final reviewed runtime proofs are required')
    raw = payloads['candidate/'+CANDIDATE.name]
    require(sha_bytes(raw) == CANDIDATE_SHA, 'Frozen candidate differs')
    candidate = json.loads(raw)
    require(candidate['code_commit'] == SOURCE and candidate['status'] == 'CANDIDATE_NOT_READY'
            and candidate['source_native_acceptance'] is None and candidate['helpers'] == str(HELPERS)
            and candidate['controls'] == str(CONTROLS)
            and candidate['calibration_phase'] == 'probability_probe_then_wait', 'Wrong isolated/null-native draft')
    files = {r['path']:r for r in candidate['files']}
    evidence = {r['path']:r for r in candidate['evidence']}
    require(len(files) == len(candidate['files']) == 42346 and len(evidence) == len(candidate['evidence']) == EVIDENCE_COUNT,
            'Frozen full file/proof closure inventory differs')
    manifest_raw = payloads['evidence/'+RUNTIME_MANIFEST_NAME]
    require(sha_bytes(manifest_raw) == IMPLEMENTATION_SHA, 'Reviewed frozen runtime manifest changed')
    manifest = json.loads(manifest_raw)
    require(manifest['source_commit']==SOURCE
        and manifest['runtime_changed_filenames']==candidate['runtime_changed_filenames']==
            ['calibration_driver_v1.py','calibration_run_v1.py','calibration_worker_v1.py']
        and manifest['backend_change_scope']==candidate['backend_change_scope']==
            'CALIBRATION_FLASH_ATTN_ENV_PROPAGATION_AND_READONLY_ACTUAL_BACKEND_GATE_ONLY'
        and manifest['base_v8_manifest_sha256']==candidate['backend_base_runtime_manifest_sha256']==
            'c2cc68f23c8fbe99387a5c0b0a23d90ec76bbf6a5a9c1795452fa8806f3bb765'
        and manifest['preserved_v8_ready_binding']['sha256']==candidate['preserved_v8_ready_binding_sha256']==
            '180431b9fe8d747e5f33f7e756b7d32181cb55c94cc623ebe788c3d4273d72d5'
        and candidate['predecessor_calibration_candidate_sha256']==
            'e18615c998a69e0212a8aac99fb2a5db32556b97a98cdcfd05fcb87418b618d1'
        and candidate['expected_rollout_attention_backend']=='FLASH_ATTN'
        and candidate['actual_backend_gate_required_before_restore_baseline_adam'] is True
        and manifest['one_outer_full_AST_equal_to_v8'] is True
        and candidate['historical_runtime_proof_versions']==dict(startup='v6-v3',factory='v7',datarow='v8'),
        'Not exact new backend scope descended from actual preserved v8')
    proof_roles={
        'startup_control_proofs':{'runtime_manifest','runtime_diff','runtime_cpu','runtime_review','failure_payload','failure_review','failure_cgroup'},
        'factory_control_proofs':{'runtime_manifest','runtime_diff','runtime_cpu','runtime_review','failure_payload','failure_review','failure_mro_diagnostic'},
        'datarow_control_proofs':{'runtime_manifest','runtime_diff','runtime_cpu','runtime_review','failure_payload','failure_review','failure_sampling_metadata','failure_trace_metadata','runtime_target_cpu','runtime_target_cpu_raw','runtime_target_cpu_review'},
        'backend_control_proofs':{'runtime_manifest','runtime_diff','runtime_cpu','runtime_review','runtime_target_cpu_import','installed_backend_evidence','diagnosis'},
        'current_driver_target_cpu_proofs':{'runtime_target_cpu','runtime_target_cpu_raw','runtime_target_cpu_review'}}
    for field,roles in proof_roles.items():
        proofs=candidate[field]
        require(set(proofs)==roles and len({r['path'] for r in proofs.values()})==len(roles)
            and all(evidence.get(r['path'])==r and Path(r['path']).parent==ROOT/'logs' for r in proofs.values()),
            'Historical/current proof closure incomplete: '+field)
    require(candidate['datarow_control_proofs']['runtime_manifest']['path']==str(ROOT/'logs/CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json')
        and candidate['factory_control_proofs']['runtime_manifest']['path']==str(ROOT/'logs/CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json')
        and candidate['backend_control_proofs']['runtime_manifest']['path']==str(ROOT/'logs'/RUNTIME_MANIFEST_NAME)
        and candidate['backend_control_proofs']['runtime_manifest']['sha256']==IMPLEMENTATION_SHA,
        'Historical proof renamed or new backend manifest not bound')
    frozen={Path(r['path']).name:r for r in manifest['files']}
    require(len(manifest['files'])==len(frozen)==14 and set(frozen)==RUNTIME_NAMES,'Not exact fourteen reviewed runtime files')
    require(frozen['calibration_run_v1.py']['sha256']==candidate['actual_v9_runtime_wrapper_sha256']
        and frozen['metadata_colocation_v1.py']['sha256']=='39e3226c82132524e57ee1adc964f607da6a82314a784c1081c505fd10ea6b9e'
        and candidate['inherited_v8_wrapper_sha256']=='7aa80984aee6a5589ee44f9b72c4bed0d091e30c38e9c181fe2cc2728d1172c9',
        'Actual new wrapper or preserved external factory/history differs')
    planned = []
    for prefix,names,destination,records in (('runtime',RUNTIME_NAMES,HELPERS,files),
            ('control',CONTROL_NAMES,CONTROLS,files),('evidence',EVIDENCE_NAMES,ROOT/'logs',evidence)):
        for name in sorted(names):
            key = prefix+'/'+name;data=payloads[key];target=destination/name
            bound = records[str(target)]
            require(bound['bytes'] == len(data) and bound['sha256'] == sha_bytes(data), 'Target bound bytes/SHA differ: '+name)
            if prefix == 'runtime':
                require(frozen[name]['bytes'] == len(data) and frozen[name]['sha256'] == sha_bytes(data),
                        'Fourteen frozen runtime bytes changed: '+name)
            planned.append(dict(path=str(target),bytes=len(data),sha256=sha_bytes(data),member=key,
                                existing_same_bytes_reuse_allowed=prefix=='evidence'))
    planned.append(dict(path=str(CANDIDATE),bytes=len(raw),sha256=CANDIDATE_SHA,
                        member='candidate/'+CANDIDATE.name,existing_same_bytes_reuse_allowed=True))
    transfers = {r['path'] for r in planned}
    inherited = [r for r in candidate['evidence'] if r['path'] not in transfers]
    require(len(inherited) == REQUIRED_EXISTING_COUNT, 'Inherited evidence coverage differs')
    return dict(status='LOCAL_INSTALL_PLAN_NOT_READY_NOT_DEPLOYED',node='h100-8',package_sha256=PACKAGE_SHA,
        candidate_sha256=CANDIDATE_SHA,code_commit=SOURCE,install_files=planned,required_existing_evidence=inherited,
        all_evidence_count=EVIDENCE_COUNT,transfer_ancillary_proofs=len(EVIDENCE_NAMES),runtime_helpers_count=14,control_count=5,
        unused_cpu_member='cpu/test_deployment_v9.py',native_directory_modified=False,
        source_native_acceptance=None,ready=False,submitted=False,retained_training_updates_contributed=0)


def require_existing_directory(path):
    info = os.lstat(path)
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and Path(path).resolve() == Path(path),
            'Target parent is absent, non-directory or symlink: '+str(path))


def preflight(plan):
    require(socket.gethostname().split('.')[0] == 'h100-8', 'Installation writes permitted only on h100-8')
    for directory in (ROOT,ROOT/'tools',ROOT/'logs'):
        require_existing_directory(directory)
    require(not os.path.lexists(HELPERS) and not os.path.lexists(CONTROLS),
            'New helper/control directory already exists; preserve and inspect, never overwrite or retry')
    require(not os.path.lexists(INTENT) and not os.path.lexists(RECEIPT),
            'Existing or uncertain installation intent/receipt: inspect, never repeat')
    for record in plan['required_existing_evidence']:
        path=Path(record['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes']
                and sha_file(path) == record['sha256'], 'Missing/changed inherited ancillary proof: '+str(path))
    reused=[]
    for record in plan['install_files']:
        path=Path(record['path'])
        if os.path.lexists(path):
            require(record['existing_same_bytes_reuse_allowed'] and path.is_file() and not path.is_symlink()
                    and path.stat().st_size == record['bytes'] and sha_file(path) == record['sha256'],
                    'Existing target must never be overwritten: '+str(path))
            reused.append(str(path))
    return reused


def write_new(path, data, mode=0o644):
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    directory=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def install(archive):
    payloads=read_archive(archive);plan=installation_plan(payloads);reused=preflight(plan)
    claim=dict(status='CLAIMED_NO_OVERWRITE_OR_RETRY',package_path=str(Path(archive).resolve()),
               package_sha256=PACKAGE_SHA,candidate_sha256=CANDIDATE_SHA,plan=plan,reused_same_bytes=reused)
    write_new(INTENT,(json.dumps(claim,indent=2)+'\n').encode())
    installed=[]
    try:
        os.mkdir(HELPERS,0o755);os.mkdir(CONTROLS,0o755)
        for record in plan['install_files']:
            path=Path(record['path'])
            if str(path) not in reused:
                write_new(path,payloads[record['member']]);installed.append(str(path))
            require(path.stat().st_size == record['bytes'] and sha_file(path) == record['sha256'],
                    'Installed bytes differ: '+str(path))
        receipt=dict(status='INSTALLED_IMMUTABLE_DRAFT_NOT_READY',package_sha256=PACKAGE_SHA,
            candidate_path=str(CANDIDATE),candidate_sha256=CANDIDATE_SHA,installed=installed,reused_same_bytes=reused,
            native_directory_modified=False,source_native_acceptance=None,ready=False,submitted=False,
            retained_training_updates_contributed=0)
        write_new(RECEIPT,(json.dumps(receipt,indent=2)+'\n').encode())
        return receipt
    except BaseException as error:
        receipt=dict(status='PARTIAL_OR_UNCERTAIN_INSTALLATION_NO_RETRY',package_sha256=PACKAGE_SHA,
            installed=installed,reused_same_bytes=reused,error_type=type(error).__name__,error=str(error),ready=False,submitted=False)
        if not os.path.lexists(RECEIPT):write_new(RECEIPT,(json.dumps(receipt,indent=2)+'\n').encode())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only',action='store_true');mode.add_argument('--install',action='store_true')
    args=parser.parse_args()
    result=install(args.archive) if args.install else installation_plan(read_archive(args.archive))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
