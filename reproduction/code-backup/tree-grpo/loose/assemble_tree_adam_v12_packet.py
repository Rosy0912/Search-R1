"""Materialize local code/audit bytes after sealed SOURCE review; no route."""
import base64
import datetime
import gzip
import hashlib
import io
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / 'work/tree-fresh200-freeze-02b1-v11'
OUT = ROOT / 'work/tree-fresh200-freeze-53082e-v12'
WRAPPER = ROOT / 'work/tree-fresh200-mu-runtime-v12'
HELPER = ROOT / 'work/tree-fresh200-dispatch-v12'
REVIEW = ROOT / 'work/tree-fresh200-53082e-v12-context-free-review'
COMMIT = '53082e1667a6ab131a036ca2ef66e004989ed1a9'
STAGE = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-53082e-v12-20261006'
TARGET = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-mu-53082e-v12'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    path = OUT / name
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return sha(path)

def copy(source, name):
    with (OUT / name).open('xb') as stream:
        stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    assert sha(OUT / name) == sha(source)

review = json.loads((REVIEW / 'RECEIPT.json').read_bytes())
assert review['reviewer_context']['fork_turns'] == 'none'
assert review['reviewer_context']['inherited_parent_conversation'] is False
assert review['source']['audited_clean_head'] == COMMIT
assert review['source']['initial_clean'] is True
assert review['source_interface_reviewed'] is True
assert review['stages']['preparation']['accepted'] is True
assert review['stages']['fresh_mu_calibration']['source_review_accepted'] is True
assert review.get('actual_gpu_accepted', False) is False
assert review.get('blocking_findings', []) == []
helper_review = json.loads((REVIEW / 'HELPER_SOURCE_REVIEW.json').read_bytes())
assert helper_review['status'] == 'FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED'
assert helper_review['actual_gpu_accepted'] is False
for name, digest in helper_review['helper_files'].items():
    assert sha(HELPER / name) == digest
wrapper_inventory = WRAPPER / 'FILE_HASHES_V12.json'
helper_inventory = HELPER / 'FILE_HASHES_V12.json'
assert sha(wrapper_inventory) == 'c0f95b872771da3e37525c2ec56e837bcc3b53d5167e09e127b0818bca2a7c85'
assert sha(helper_inventory) == 'e179f31ed06e1104c224ca9faae3f4029d82bd51eaec0c8d35c0e64c0bc39138'

for name, source in {
    'SOURCE_CONTEXT_FREE_RECEIPT.json': REVIEW / 'RECEIPT.json',
    'SOURCE_BYTE_AUDIT.json': REVIEW / 'BYTE_AUDIT.json',
    'HELPER_SOURCE_REVIEW.json': REVIEW / 'HELPER_SOURCE_REVIEW.json',
    'SOURCE_INTERFACE_RECEIPT.json': WRAPPER / 'SOURCE_INTERFACE_COMPATIBILITY_V12.json',
    'HISTORICAL_RAW_CPU_PLAN.json': BASE / 'HISTORICAL_RAW_CPU_PLAN.json',
    'RETAINED_TERMINAL3314_PROOF_V2.json': BASE / 'actual-zero-dispatch/RETAINED_TERMINAL3314_PROOF_V2.json',
}.items():
    copy(source, name)
for name in helper_review['helper_files']:
    copy(HELPER / name, name)
proof_ref = {'path': STAGE + '/RETAINED_TERMINAL3314_PROOF_V2.json',
             'sha256': sha(OUT / 'RETAINED_TERMINAL3314_PROOF_V2.json')}
assert proof_ref['sha256'] == 'a5358874bb9ed134196779cd9e9ed5a96479a14f74fc941ef12d701261656ac6'
inputs = json.loads((WRAPPER / 'BUILDER_INPUTS_V12.json').read_bytes())
assert inputs['source']['code_commit'] == COMMIT
assert inputs['source']['source_interface_receipt_sha256'] == sha(OUT / 'SOURCE_INTERFACE_RECEIPT.json')
inputs['final_review'] = {
    'status': 'SOURCE_ZERO_V12_HELPER_V12_COMBINATION_ACCEPTED_ONLY',
    'fork_turns': 'none', 'inherit_old_review_as_dispatch_permission': False,
    'source_review_path': str(REVIEW / 'RECEIPT.json'),
    'source_review_sha256': sha(REVIEW / 'RECEIPT.json'),
    'exact_packet_execution_binding_pending': True,
}
inputs['previous_terminal_evidence'] = {'3314': proof_ref}
inputs['local_preparation_provenance'] = {
    'scope': 'New local code-only package; same-node short CPU config pending exact prospective review. Historical official5120 reused by unchanged helper/tokenizer identities.',
    'new_candidate_target': TARGET, 'new_stage_directory': STAGE,
    'actual_remote_executed': False, 'gpu_or_slurm_execution': False,
    'actual_new_source_gpu_accepted': False, 'exact_packet_execution_binding_pending': True,
    'source_combination_receipt_sha256': sha(REVIEW / 'RECEIPT.json'),
    'wrapper_inventory_sha256': sha(wrapper_inventory),
    'helper_inventory_sha256': sha(helper_inventory),
}
inputs_sha = write('BUILDER_INPUTS.json', inputs)
argv = ['python3', '-B', str(WRAPPER / 'build_package_draft.py'), '--inputs', str(OUT / 'BUILDER_INPUTS.json'),
        '--output', str(OUT / 'package-code-only'), '--wrapper-inventory', str(wrapper_inventory),
        '--wrapper-inventory-sha', sha(wrapper_inventory)]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONOPTIMIZE='0', PYTHONDONTWRITEBYTECODE='1',
           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
start = time.monotonic(); result = subprocess.run(argv, capture_output=True, env=env, cwd=ROOT)
for name, raw in [('BUILDER_ACTUAL.stdout', result.stdout), ('BUILDER_ACTUAL.stderr', result.stderr)]:
    with (OUT / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
write('BUILDER_ACTUAL_EXECUTION.json', {
    'argv': argv, 'returncode': result.returncode, 'wall_seconds': time.monotonic() - start,
    'actual_local_builder_executed': True, 'actual_remote_executed': False,
    'gpu_or_slurm_execution': False, 'source_runtime_imported_or_executed': False,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'stdout_sha256': sha(OUT / 'BUILDER_ACTUAL.stdout'), 'stderr_sha256': sha(OUT / 'BUILDER_ACTUAL.stderr'),
    'training_release': False,
})
assert result.returncode == 0, 'Actual local builder failed; partial package retained, no retry'
package = OUT / 'package-code-only'
manifest_path = package / 'manifest.json'
copy(manifest_path, 'DRAFT_MANIFEST_BEFORE_FREEZE.json')
manifest = json.loads(manifest_path.read_bytes())
assert manifest['code_commit'] == COMMIT and manifest['wrapper_inventory_sha256'] == sha(wrapper_inventory)
assert len(manifest['wrapper_files']) == 15
assert manifest['source_selection']['selected_count'] == len(manifest['source_selection']['selected_files'])
review_names = ('SOURCE_INTERFACE_RECEIPT.json', 'SOURCE_CONTEXT_FREE_RECEIPT.json',
                'SOURCE_BYTE_AUDIT.json', 'HELPER_SOURCE_REVIEW.json')
manifest.update(
    status='FROZEN_CPU_PREPARATION_PENDING_ACTUAL_BYTE_BINDING_REVIEW',
    scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',
    actual_remote_executed=False, actual_gpu_executed=False, actual_slurm_submitted=False,
    actual_ready=False, actual_zero_ready=False, training_release=False,
    source_cpu_review_receipts={name: {'path': STAGE + '/' + name, 'sha256': sha(OUT / name)} for name in review_names},
    source_interface_review_scope='Source53082e/unchanged zero15/helperV12 targeted accounting fallback accepted as SOURCE/CPU capability. Exact execution binding and actual new GPU behavior pending.',
    historical_wrapper_receipt_scope={
        'file': 'SOURCE_CPU_RECEIPT.json', 'sha256': sha(package / 'SOURCE_CPU_RECEIPT.json'),
        'scope': 'Immutable inherited wrapper-owner historical SOURCE/CPU metadata only. New source/zero/helper combination acceptance is separate; no old suite relabelled.',
    },
)
manifest['probability_calibration']['source_interface_reviewed'] = True
manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
manifest_sha = sha(manifest_path)
archive_path = OUT / 'CODE_PACKAGE.tar.gz'
with archive_path.open('xb') as output:
    with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as bundle:
            for name in sorted(set(manifest['files']) | {'manifest.json'}):
                path = package / name
                assert path.is_file() and not path.is_symlink()
                raw = path.read_bytes()
                if name != 'manifest.json': assert hashlib.sha256(raw).hexdigest() == manifest['files'][name]
                member = tarfile.TarInfo(name); member.size = len(raw); member.mode = 0o600; member.mtime = 0
                bundle.addfile(member, io.BytesIO(raw))
archive_sha = sha(archive_path)
prep_inputs = {
    'status': 'EXACT_BYTE_CPU_PREPARATION_INPUTS_PENDING_EXECUTION_BINDING_REVIEW',
    'scope': 'Only new same-h1009 code/CPU config preparation. Existing official5120 report reused; no second5120, GPU, Slurm, READY or training release.',
    'code_commit': COMMIT, 'manifest_sha256': manifest_sha, 'archive_sha256': archive_sha,
    'stage': 'fresh_mu_calibration', 'target': TARGET, 'archive': STAGE + '/CODE_PACKAGE.tar.gz',
    'max_uncompressed_bytes': 32 * 1024 ** 2,
    'source_interface_receipt': {'path': STAGE + '/SOURCE_INTERFACE_RECEIPT.json', 'sha256': sha(OUT / 'SOURCE_INTERFACE_RECEIPT.json')},
    'context_free_review': {'path': STAGE + '/SOURCE_CONTEXT_FREE_RECEIPT.json', 'sha256': sha(OUT / 'SOURCE_CONTEXT_FREE_RECEIPT.json')},
    'context_byte_audit': {'path': STAGE + '/SOURCE_BYTE_AUDIT.json', 'sha256': sha(OUT / 'SOURCE_BYTE_AUDIT.json')},
    'previous_terminal_evidence': {'3314': proof_ref},
    'actual_assets_transferred': False, 'actual_gpu_accepted': False, 'actual_slurm_submitted': False,
}
prep_sha = write('PREPARATION_INPUTS.json', prep_inputs)
names = {'CODE_PACKAGE.tar.gz', 'PREPARATION_INPUTS.json', 'HISTORICAL_RAW_CPU_PLAN.json',
         'fresh200_dispatch_common.py', 'prepare_tree_fresh200_package.py', 'submit_tree_fresh200_once.py',
         'SOURCE_INTERFACE_RECEIPT.json', 'SOURCE_CONTEXT_FREE_RECEIPT.json', 'SOURCE_BYTE_AUDIT.json',
         'HELPER_SOURCE_REVIEW.json', 'RETAINED_TERMINAL3314_PROOF_V2.json'}
envelope = {'status': 'REVIEWED_CODE_METADATA_CPU_STAGE_ONLY', 'files': {
    name: {'sha256': sha(OUT / name), 'bytes': (OUT / name).stat().st_size,
           'base64': base64.b64encode((OUT / name).read_bytes()).decode('ascii')}
    for name in sorted(names)}}
envelope_sha = write('CODE_METADATA_ENVELOPE.json', envelope)
stage_code = (BASE / 'stage_code_metadata_cpu.py').read_text()
old_stage = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-02b1-v11-20261006'
assert stage_code.count(old_stage) == 1
assert stage_code.count("'HELPER_SOURCE_REVIEW.json',") == 1
stage_code = stage_code.replace(old_stage, STAGE).replace("'HELPER_SOURCE_REVIEW.json',", "'HELPER_SOURCE_REVIEW.json', 'RETAINED_TERMINAL3314_PROOF_V2.json',")
with (OUT / 'stage_code_metadata_cpu.py').open('x') as stream:
    stream.write(stage_code)
env_argv = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONOPTIMIZE=0', 'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
            '/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python', '-B']
stage_argv = env_argv + ['-c', stage_code, envelope_sha]
prepare_argv = env_argv + [STAGE + '/prepare_tree_fresh200_package.py', '--archive', STAGE + '/CODE_PACKAGE.tar.gz',
    '--archive-sha', archive_sha, '--target', TARGET, '--manifest-sha', manifest_sha,
    '--inputs', STAGE + '/PREPARATION_INPUTS.json', '--inputs-sha', prep_sha,
    '--helper-review', STAGE + '/HELPER_SOURCE_REVIEW.json', '--helper-review-sha', sha(OUT / 'HELPER_SOURCE_REVIEW.json')]
stage_command = shlex.join(stage_argv); prepare_command = shlex.join(prepare_argv)
assert shlex.split(stage_command) == stage_argv and shlex.split(prepare_command) == prepare_argv
report = BASE / 'actual-cpu/HISTORICAL_RAW_CPU_PREFLIGHT.json'
assert sha(report) == '2cc2366e9e3473276e4386820fa28249db9e1bb28875ca5298889c48d09c0761'
reuse_sha = write('HISTORICAL_CPU_REPORT_REUSE_PLAN.json', {
    'status': 'EXACT_HISTORICAL5120_CPU_REPORT_REUSE_ONLY', 'local_report': str(report), 'report_sha256': sha(report),
    'remote_report': '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-mu-02b1-v10/CPU_PREP_CONFIG/HISTORICAL_RAW_CPU_PREFLIGHT.json',
    'raw_helper_sha256': '15b696607846046f671e30531420c6f48cca321762f77016f4e5bdb7868ea229',
    'tokenizer_sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'new_tokenizer5120_execution_planned': False, 'historical_supported_rows': 0,
    'actual_new_native24_or_fullTree_accepted': False,
})
packet = {
    'status': 'NEW_LOCAL_CODE_CPU_PACKET_PENDING_EXACT_EXECUTION_BINDING_REVIEW',
    'code_commit': COMMIT, 'candidate_target': TARGET, 'stage_directory': STAGE, 'node': 'h100-9',
    'code_archive_path': str(archive_path), 'code_archive_sha256': archive_sha,
    'code_manifest_path': str(manifest_path), 'code_manifest_sha256': manifest_sha,
    'code_archive_bytes': archive_path.stat().st_size,
    'archive_regular_member_count_including_manifest': len(manifest['files']) + 1,
    'builder_inputs_path': str(OUT / 'BUILDER_INPUTS.json'), 'builder_inputs_sha256': inputs_sha,
    'preparation_inputs_path': str(OUT / 'PREPARATION_INPUTS.json'), 'preparation_inputs_sha256': prep_sha,
    'wrapper_inventory_path': str(wrapper_inventory), 'wrapper_inventory_sha256': sha(wrapper_inventory),
    'helper_inventory_path': str(helper_inventory), 'helper_inventory_sha256': sha(helper_inventory),
    'source_review_path': str(REVIEW / 'RECEIPT.json'), 'source_review_sha256': sha(REVIEW / 'RECEIPT.json'),
    'historical5120_reuse_plan_sha256': reuse_sha, 'previous_terminal_evidence': {'3314': proof_ref},
    'local_sources': {name: str(OUT / name) for name in sorted(names)},
    'stage_files': {name: {'sha256': sha(OUT / name), 'bytes': (OUT / name).stat().st_size} for name in sorted(names)},
    'envelope_path': str(OUT / 'CODE_METADATA_ENVELOPE.json'), 'envelope_sha256': envelope_sha,
    'stage_script_path': str(OUT / 'stage_code_metadata_cpu.py'), 'stage_script_sha256': sha(OUT / 'stage_code_metadata_cpu.py'),
    'stage_argv': stage_argv, 'prepare_argv': prepare_argv,
    'stage_remote_command': stage_command, 'prepare_remote_command': prepare_command,
    'prospective_action_names': ['stage_code_metadata_cpu', 'prepare_code_assets_and_cpu_config'],
    'actual_local_builder_executed': True, 'actual_remote_executed': False, 'actual_source_runtime_executed': False,
    'actual_gpu_accepted': False, 'actual_slurm_submitted': False, 'actual_ready': False, 'actual_zero_ready': False,
    'production_thresholds_accepted': False, 'original_sampler_actor_gate_accepted': False,
    'new_tokenizer5120_execution_planned': False, 'asset_transfer': False, 'rawtokens_transfer': False,
    'training_release': False,
}
packet_sha = write('CPU_EXECUTION_PACKET.json', packet)
route = ROOT / 'work/remote.py'
route_record = {
    'status': 'EXACT_PROSPECTIVE_DEFAULT_ROUTE_ARGV_PENDING_REVIEW', 'packet_sha256': packet_sha,
    'cwd': str(ROOT), 'node': 'h100-9', 'route_source': str(route), 'route_source_sha256': sha(route),
    'stage_local_argv': ['python3', '-B', str(route), stage_command, '--stdin', str(OUT / 'CODE_METADATA_ENVELOPE.json')],
    'prepare_local_argv': ['python3', '-B', str(route), prepare_command],
    'stage_stdin_path': str(OUT / 'CODE_METADATA_ENVELOPE.json'), 'stage_stdin_sha256': envelope_sha,
    'actual_remote_executed': False, 'actual_route_entry_or_route_mock_executed': False,
    'asset_transfer': False, 'gpu_or_slurm_execution': False, 'training_release': False,
}
route_sha = write('CPU_MANAGEMENT_ROUTE_ARGV.json', route_record)
write('LOCAL_PACKET_SEAL.json', {
    'status': 'LOCAL_V12_CODE_AUDIT_PACKET_SEALED_NO_REMOTE_RELEASE', 'code_commit': COMMIT,
    'packet_sha256': packet_sha, 'route_sha256': route_sha, 'envelope_sha256': envelope_sha,
    'files': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
    'actual_gpu_or_slurm_executed': False, 'actual_remote_executed': False, 'training_release': False,
})
print(json.dumps({'code_commit': COMMIT, 'manifest_sha256': manifest_sha, 'archive_sha256': archive_sha,
                  'packet_sha256': packet_sha, 'route_sha256': route_sha, 'stage_file_count': len(names),
                  'local_builder_returncode': result.returncode, 'actual_remote_or_gpu_execution': False}))
