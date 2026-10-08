"""SOURCE-ONLY a18f420 assembler draft; main refuses every unbound receipt.

This file was not executed. Existing fifteen runtime bytes and archive/envelope
mechanisms remain unchanged. Only current, independently produced evidence may
bind the changed raw helper; historical5120 is never reused for this helper.
"""
import base64
import datetime
import gzip
import hashlib
import io
import json
import os
import pathlib
import shlex
import subprocess
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/tree-fresh200-freeze-53082e-v12'
OUT = ROOT / 'work/tree-fresh200-freeze-a18f420'
WRAPPER = ROOT / 'work/tree-fresh200-mu-runtime-v12'
COMMIT = 'a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
RAW_PATH = 'search_r1/baselines/tree_raw_tokens.py'
RAW_SHA = '5ebffa13473b111f5b73992679370d1c90603f2f9da5bdfee52db8b5736750c6'
STAGE = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-a18f420-20261007'
TARGET = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-mu-a18f420'
NATIVE_REPORT = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preflight-a18f420-row96-20261007/ACTUAL3317_ROW96_UTF8_PREFLIGHT.json'
SOURCE_REVIEW = ROOT / 'work/tree-fresh200-utf8-cache-none-review/INDEPENDENT_UTF8_CACHE_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = 'b31c86c1b81258200580b0a2bb0346c24aba9e4ba53332e4f732bbfe0b1efdbf'
CORE_MANIFEST = ROOT / 'work/tree-fresh200-utf8-cache-owner/AFFECTED_SOURCE_MANIFEST.json'
CORE_MANIFEST_SHA = '2e6e0fb90a78966e80d1940efd46f34f5bd8b806c2ffae4fa6350890d6d8120d'
BOUNDED_CPU = ROOT / 'work/tree-fresh200-utf8-cache-none-review/INDEPENDENT_UTF8_CACHE_BOUNDED_CPU_RESULT.json'
BOUNDED_CPU_SHA = '806e505fa6c5fd475b50a35e57a9847f72b0be192f42494d9949fa6ebfe30870'
HISTORICAL_REVIEW = ROOT / 'work/tree-fresh200-53082e-v12-context-free-review/RECEIPT.json'
HISTORICAL_REVIEW_SHA = 'ce3ef87928c463db6f47518ca547492b32204c540d709d13c98fef3646489f6d'

# Exact future bytes only; NULL is pending, never acceptance or a guessed pin.
REQUIRED_INPUT_PINS = {
    'helper_inventory': {
        'path': str(ROOT / 'work/tree-fresh200-dispatch-a18f420/FILE_HASHES_A18F420.json'),
        'sha256': '7a1269309c432b363ad32191cc03a82a7ce7a9a45d6345dc9a37ac1350c0bd60'},
    'helper_source_review': {
        'path': str(ROOT / 'work/tree-fresh200-a18f-packet-none-review/HELPER_SOURCE_REVIEW.json'),
        'sha256': 'd525ee289b55311053ed913cadbc1bfa3aecb156f77c8f9b2663ace5ccff8fba'},
    'source_interface_receipt': {
        'path': str(ROOT / 'work/tree-fresh200-utf8-cache-none-review/SOURCE_INTERFACE_RECEIPT.json'),
        'sha256': '2b67f6dd67e58b3d0e0c3d7d34d0dd7c559486d5e50c2fd39a34c9db27108bb7'},
    'context_free_review': {
        'path': str(ROOT / 'work/tree-fresh200-utf8-cache-none-review/SOURCE_CONTEXT_FREE_RECEIPT.json'),
        'sha256': 'b1426019f794f597beed2d29679cad158c5d5aa7d083bf49a0299c794f6a601c'},
    'context_byte_audit': {
        'path': str(ROOT / 'work/tree-fresh200-utf8-cache-none-review/SOURCE_BYTE_AUDIT.json'),
        'sha256': '62463b42be6f6715939e1733b3ecffa638211fbbf39255e14f7a91e20e11fe57'},
    'native3317_preflight': {
        'path': str(ROOT / 'work/tree-fresh200-utf8-cache-actual/ACTUAL3317_ROW96_UTF8_PREFLIGHT.json'),
        'sha256': '2122dc288ab9d0b6ea59cc996fe8251908ce87e7519abb5e7d9e60582cedccc8'},
    'retained_terminal3314_proof': {
        'path': str(BASE / 'RETAINED_TERMINAL3314_PROOF_V2.json'),
        'sha256': 'a5358874bb9ed134196779cd9e9ed5a96479a14f74fc941ef12d701261656ac6'},
    'retained_terminal3317_proof': {
        'path': str(ROOT / 'work/tree-fresh200-a18f420-zero-static-plan/RETAINED_TERMINAL3317_PROOF.json'),
        'sha256': '97c94827a6a7b1c83730c818cf8cf672f7b5051fb9e6bdbf652c061a2816acda'},
}

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

def main():
    # Reject all pending pins before creating outputs or invoking the builder.
    if not __debug__:
        raise ValueError('Assembler source guards require PYTHONOPTIMIZE=0')
    refs = {}
    for name, entry in REQUIRED_INPUT_PINS.items():
        if (type(entry['path']) is not str or not pathlib.Path(entry['path']).is_absolute()
                or type(entry['sha256']) is not str or len(entry['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError('Pending required exact receipt path/SHA: ' + name)
        path = pathlib.Path(entry['path'])
        if not path.is_file() or path.is_symlink() or sha(path) != entry['sha256']:
            raise ValueError('Required receipt bytes differ: ' + name)
        refs[name] = path
    for path, digest in ((SOURCE_REVIEW, SOURCE_REVIEW_SHA), (CORE_MANIFEST, CORE_MANIFEST_SHA),
                         (BOUNDED_CPU, BOUNDED_CPU_SHA), (HISTORICAL_REVIEW, HISTORICAL_REVIEW_SHA)):
        assert sha(path) == digest
    core = json.loads(CORE_MANIFEST.read_bytes())
    source_review = json.loads(SOURCE_REVIEW.read_bytes())
    bounded_cpu = json.loads(BOUNDED_CPU.read_bytes())
    assert core['code_commit'] == COMMIT and core['source_interface_count'] == 20
    assert core['source_files'][RAW_PATH] == RAW_SHA
    assert core['all_other_19_interface_files_byte_identical'] is True
    assert source_review['source_review_accepted'] is True
    assert source_review['reviewer']['fork_turns'] == 'none'
    assert source_review['source']['current_head'] == COMMIT
    assert source_review['source']['current_sha256'] == RAW_SHA
    assert source_review['source']['current_worktree_clean'] is True
    assert bounded_cpu['status'] == 'BOUNDED_LOCAL_CPU_PASS'
    assert bounded_cpu['source']['commit'] == COMMIT and bounded_cpu['source']['sha256'] == RAW_SHA
    assert bounded_cpu['execution']['cases_executed'] == bounded_cpu['execution']['cases_passed'] == 27
    assert bounded_cpu['execution']['cases_failed'] == bounded_cpu['execution']['cases_skipped'] == 0
    assert bounded_cpu['independent_source_review']['sha256'] == SOURCE_REVIEW_SHA
    assert all(bounded_cpu['acceptance'][key] is False for key in
               ('actual_3317_500_token_replay', 'native24', 'full_tree_mu', 'full_zero', 'ready', 'training_release'))
    review = json.loads(refs['context_free_review'].read_bytes())
    assert review['reviewer_context']['fork_turns'] == 'none'
    assert review['reviewer_context']['inherited_parent_conversation'] is False
    assert review['source']['audited_clean_head'] == COMMIT
    assert review['source']['initial_clean'] is True
    assert review['source_interface_reviewed'] is True
    assert review['stages']['preparation']['accepted'] is True
    assert review['execution_scope']['same_h100_9_cpu_only_preparation_allowed'] is True
    assert review['stages']['fresh_mu_calibration']['source_review_completed'] is True
    assert review['stages']['fresh_mu_calibration']['source_review_accepted'] is True
    assert review.get('actual_gpu_accepted', False) is False
    assert review.get('blocking_findings', []) == []
    interface = json.loads(refs['source_interface_receipt'].read_bytes())
    assert interface['status'] == 'FRESH200_REPAIRED_SOURCE_INTERFACE_CPU_OWNER_VERIFIED'
    assert interface['code_commit'] == COMMIT and interface['source_clean'] is True
    assert interface['source_files'] == core['source_files']
    assert all(interface[key] is False for key in ('actual_gpu_executed', 'actual_remote_executed', 'training_release'))
    audit = json.loads(refs['context_byte_audit'].read_bytes())
    assert audit['git_head'] == COMMIT
    assert review['artifacts']['BYTE_AUDIT.json'] == sha(refs['context_byte_audit'])
    assert set(audit['source_files']) == set(core['source_files'])
    assert all(entry['matches_git_head'] is True and entry['sha256'] == core['source_files'][name]
               for name, entry in audit['source_files'].items())
    wrapper_inventory = WRAPPER / 'FILE_HASHES_V12.json'
    assert sha(wrapper_inventory) == 'c0f95b872771da3e37525c2ec56e837bcc3b53d5167e09e127b0818bca2a7c85'
    wrapper = json.loads(wrapper_inventory.read_bytes())
    assert set(audit['wrapper_files']) == set(wrapper['wrapper_files'])
    assert all(entry['matches'] is True and entry['sha256'] == wrapper['wrapper_files'][name]
               for name, entry in audit['wrapper_files'].items())
    assert {name for name, digest in core['source_files'].items()
            if wrapper['source_files'][name] != digest} == {RAW_PATH}
    helper_inventory = refs['helper_inventory']
    HELPER = helper_inventory.parent
    helper_inventory_value = json.loads(helper_inventory.read_bytes())
    assert helper_inventory_value['code_commit'] == COMMIT
    helper_review = json.loads(refs['helper_source_review'].read_bytes())
    assert helper_review['status'] == 'FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED'
    assert helper_review['actual_gpu_accepted'] is False
    assert set(helper_review['helper_files']) == {
        'fresh200_dispatch_common.py', 'prepare_tree_fresh200_package.py', 'submit_tree_fresh200_once.py'}
    assert helper_review['helper_files'] == helper_inventory_value['helper_files']
    for name, digest in helper_review['helper_files'].items():
        assert sha(HELPER / name) == digest
    for name in ('prepare_tree_fresh200_package.py', 'submit_tree_fresh200_once.py'):
        assert (HELPER / name).read_bytes() == (BASE / name).read_bytes()
    assert (HELPER / 'fresh200_dispatch_common.py').read_bytes() == (
        BASE / 'fresh200_dispatch_common.py').read_bytes().replace(
            b'53082e1667a6ab131a036ca2ef66e004989ed1a9', COMMIT.encode('ascii'))
    preflight = json.loads(refs['native3317_preflight'].read_bytes())
    assert preflight['status'] == 'HISTORICAL_RAW_CPU_PREFLIGHT_OBSERVED_NOT_FRESH_TRAINING_ACCEPTED'
    assert preflight['rows'] == 1 and type(preflight['rows']) is int
    assert preflight['raw_helper_sha256'] == RAW_SHA
    assert preflight['tokenizer_identity']['tokenizer_json_sha256'] == 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
    assert all(preflight[key] == 0 for key in
               ('model_weight_loads', 'tensor_asset_loads', 'generation_calls', 'optimizer_writes'))
    assert all(preflight[key] is False for key in
               ('cuda_initialized', 'original_sampler_actor_gate_accepted', 'fresh_tree_rollout_accepted'))
    assert preflight['code_commit'] == COMMIT and preflight['rejected_rows'] == 0
    assert preflight['plan_sha256'] == '52d96784d37274fc9a10de14d62199b25be5519f0793504738d5e2d309e64e88'
    assert preflight['producer_sha256'] == '5063792bd7aee4a538999f26290ae0e42934dfd5e40152fe0810c27fa31f9c05'
    assert preflight['rejection_sha256'] == '91237cb19325f3f8cc5467a550fffd73345b7b41da921065ae0a56d8c57e41fb'
    assert preflight['sample_selector'] == {
        'row_index': 96, 'sample_id': '4a659f5a308a597087edf7206b2c01d09862285fb77b01f04d58d43670cd9259',
        'emitted_tokens': 500, 'native_emissions_in_rejection': 784,
        'first_literal_fffd_token_prefix': 159}
    assert preflight['native_prefix_agreement_count'] == 500
    assert preflight['first_literal_fffd_token_prefix'] == 159
    assert preflight['valid_literal_utf8_efbfbd_present'] is True
    assert preflight['final_strict_original_bytes_complete'] is True
    assert preflight['final_pending_byte_count'] == 0
    assert preflight['original_sample_hashes_before'] == preflight['original_sample_hashes_after']
    assert set(preflight['original_sample_hashes_before']) == {
        'emission_sha256', 'ids_sha256', 'selected_logprobs_sha256', 'context_sha256', 'legacy_text_sha256'}
    assert all(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
               for value in list(preflight['original_sample_hashes_before'].values()) + [
                   preflight['original_emitted_byte_sha256'], preflight['native_prefix_agreement_sha256']])
    assert all(preflight[key] is False for key in
               ('actual_native24_accepted', 'actual_behavior_mu_accepted', 'actual_full_tree_accepted'))
    assert type(preflight['scope']) is str and preflight['scope']
    assert type(preflight['limitations']) is str and preflight['limitations']
    assert not OUT.exists(), 'No-clobber new packet path required'
    OUT.mkdir()

    for name, source in {
        'SOURCE_CONTEXT_FREE_RECEIPT.json': refs['context_free_review'],
        'SOURCE_BYTE_AUDIT.json': refs['context_byte_audit'],
        'HELPER_SOURCE_REVIEW.json': refs['helper_source_review'],
        'SOURCE_INTERFACE_RECEIPT.json': refs['source_interface_receipt'],
        'A18F_BOUNDED_CPU_RECEIPT.json': BOUNDED_CPU,
        'A18F_AFFECTED_SOURCE_MANIFEST.json': CORE_MANIFEST,
        'A18F_SOURCE_ONLY_REVIEW.json': SOURCE_REVIEW,
        'HISTORICAL_53082E_CONTEXT_RECEIPT.json': HISTORICAL_REVIEW,
        'RETAINED_TERMINAL3314_PROOF_V2.json': refs['retained_terminal3314_proof'],
        'RETAINED_TERMINAL3317_PROOF.json': refs['retained_terminal3317_proof'],
    }.items():
        copy(source, name)
    for name in helper_review['helper_files']:
        copy(HELPER / name, name)
    proof_ref = {'path': STAGE + '/RETAINED_TERMINAL3317_PROOF.json',
                 'sha256': sha(OUT / 'RETAINED_TERMINAL3317_PROOF.json')}
    assert proof_ref['sha256'] == REQUIRED_INPUT_PINS['retained_terminal3317_proof']['sha256']
    previous_terminal_evidence = {
        '3314': {'path': STAGE + '/RETAINED_TERMINAL3314_PROOF_V2.json',
                 'sha256': sha(OUT / 'RETAINED_TERMINAL3314_PROOF_V2.json')},
        '3317': proof_ref,
    }
    inputs = json.loads((WRAPPER / 'BUILDER_INPUTS_V12.json').read_bytes())
    inputs['source'].update(path=core['source_directory'], code_commit=COMMIT,
                            source_interface_receipt_sha256=sha(refs['source_interface_receipt']))
    assert inputs['source']['source_interface_receipt_sha256'] == sha(OUT / 'SOURCE_INTERFACE_RECEIPT.json')
    inputs['final_review'] = {
        'status': 'SOURCE_ZERO_V12_HELPER_V12_COMBINATION_ACCEPTED_ONLY',
        'fork_turns': 'none', 'inherit_old_review_as_dispatch_permission': False,
        'source_review_path': str(refs['context_free_review']),
        'source_review_sha256': sha(refs['context_free_review']),
        'exact_packet_execution_binding_pending': True,
    }
    inputs['previous_terminal_evidence'] = previous_terminal_evidence
    inputs['local_preparation_provenance'] = {
        'scope': 'New a18f420 code-only package. Actual bounded27 CPU and exact historical3317 row96 replay are separately bound; no historical5120 helper-byte reuse or fresh rollout acceptance.',
        'new_candidate_target': TARGET, 'new_stage_directory': STAGE,
        'actual_remote_executed': False, 'gpu_or_slurm_execution': False,
        'actual_new_source_gpu_accepted': False, 'exact_packet_execution_binding_pending': True,
        'source_combination_receipt_sha256': sha(refs['context_free_review']),
        'wrapper_inventory_sha256': sha(wrapper_inventory),
        'a18f_source_review_sha256': SOURCE_REVIEW_SHA,
        'a18f_core_manifest_sha256': CORE_MANIFEST_SHA,
        'a18f_bounded_cpu_sha256': BOUNDED_CPU_SHA,
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
        source_interface_review_scope='Current a18f420 source plus unchanged zero15 and common-only COMMIT-pinned V12 helper3 are accepted only by the separately bound current independent source/CPU receipts. Historical53082e covers only19 unchanged interfaces; actual zero/GPU/mu/READY remain pending.',
        historical_wrapper_receipt_scope={
            'file': 'SOURCE_CPU_RECEIPT.json', 'sha256': sha(package / 'SOURCE_CPU_RECEIPT.json'),
            'scope': 'Immutable inherited wrapper-owner historical SOURCE/CPU metadata only. New source/zero/helper combination acceptance is separate; no old suite relabelled.',
        },
        historical_source_review_scope={
            'path': STAGE + '/HISTORICAL_53082E_CONTEXT_RECEIPT.json', 'sha256': HISTORICAL_REVIEW_SHA,
            'unchanged_core_interface_count': 19,
            'scope': 'Historical53082e evidence only for19 byte-identical core interfaces verified by current20 manifest and independent audit; no inherited new-helper5120, READY, native24, fullTree or behavior-mu acceptance.',
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
        'scope': 'Only new same-h1009 code/CPU config preparation. New actual3317 row96 report has rows1 and limited historical replay scope; no5120 reuse, GPU, Slurm, READY or training release.',
        'code_commit': COMMIT, 'manifest_sha256': manifest_sha, 'archive_sha256': archive_sha,
        'stage': 'fresh_mu_calibration', 'target': TARGET, 'archive': STAGE + '/CODE_PACKAGE.tar.gz',
        'max_uncompressed_bytes': 32 * 1024 ** 2,
        'source_interface_receipt': {'path': STAGE + '/SOURCE_INTERFACE_RECEIPT.json', 'sha256': sha(OUT / 'SOURCE_INTERFACE_RECEIPT.json')},
        'context_free_review': {'path': STAGE + '/SOURCE_CONTEXT_FREE_RECEIPT.json', 'sha256': sha(OUT / 'SOURCE_CONTEXT_FREE_RECEIPT.json')},
        'context_byte_audit': {'path': STAGE + '/SOURCE_BYTE_AUDIT.json', 'sha256': sha(OUT / 'SOURCE_BYTE_AUDIT.json')},
        'previous_terminal_evidence': previous_terminal_evidence,
        'tokenizer_preflight': {'path': NATIVE_REPORT, 'sha256': sha(refs['native3317_preflight'])},
        'actual_assets_transferred': False, 'actual_gpu_accepted': False, 'actual_slurm_submitted': False,
    }
    prep_sha = write('PREPARATION_INPUTS.json', prep_inputs)
    names = {'CODE_PACKAGE.tar.gz', 'PREPARATION_INPUTS.json', 'A18F_BOUNDED_CPU_RECEIPT.json',
             'A18F_AFFECTED_SOURCE_MANIFEST.json', 'A18F_SOURCE_ONLY_REVIEW.json',
             'HISTORICAL_53082E_CONTEXT_RECEIPT.json',
             'fresh200_dispatch_common.py', 'prepare_tree_fresh200_package.py', 'submit_tree_fresh200_once.py',
             'SOURCE_INTERFACE_RECEIPT.json', 'SOURCE_CONTEXT_FREE_RECEIPT.json', 'SOURCE_BYTE_AUDIT.json',
             'HELPER_SOURCE_REVIEW.json', 'RETAINED_TERMINAL3314_PROOF_V2.json',
             'RETAINED_TERMINAL3317_PROOF.json'}
    envelope = {'status': 'REVIEWED_CODE_METADATA_CPU_STAGE_ONLY', 'files': {
        name: {'sha256': sha(OUT / name), 'bytes': (OUT / name).stat().st_size,
               'base64': base64.b64encode((OUT / name).read_bytes()).decode('ascii')}
        for name in sorted(names)}}
    envelope_sha = write('CODE_METADATA_ENVELOPE.json', envelope)
    stage_code = (BASE / 'stage_code_metadata_cpu.py').read_text()
    old_stage = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-53082e-v12-20261006'
    assert stage_code.count(old_stage) == 1
    assert stage_code.count("'HISTORICAL_RAW_CPU_PLAN.json'") == 1
    assert stage_code.count("'RETAINED_TERMINAL3314_PROOF_V2.json'") == 1
    stage_code = stage_code.replace(old_stage, STAGE).replace(
        "'HISTORICAL_RAW_CPU_PLAN.json'",
        "'A18F_BOUNDED_CPU_RECEIPT.json', 'A18F_AFFECTED_SOURCE_MANIFEST.json', "
        "'A18F_SOURCE_ONLY_REVIEW.json', 'HISTORICAL_53082E_CONTEXT_RECEIPT.json'").replace(
        "'RETAINED_TERMINAL3314_PROOF_V2.json'",
        "'RETAINED_TERMINAL3314_PROOF_V2.json', 'RETAINED_TERMINAL3317_PROOF.json'")
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
    preflight_binding = {
        'path': NATIVE_REPORT, 'sha256': sha(refs['native3317_preflight']), 'rows': 1,
        'scope': preflight['scope'], 'limitations': preflight['limitations'],
        'requires_independent_zero_release_limitations_assessment': True,
        'historical5120_helper_byte_identity_reuse': False,
    }
    preflight_binding_sha = write('NATIVE3317_PREFLIGHT_BINDING.json', preflight_binding)
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
        'source_review_path': str(refs['context_free_review']), 'source_review_sha256': sha(refs['context_free_review']),
        'native3317_preflight_binding_sha256': preflight_binding_sha,
        'tokenizer_preflight': preflight_binding, 'previous_terminal_evidence': previous_terminal_evidence,
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
        'stage_local_argv': ['python3', '-B', str(route), stage_command,
                             '--stdin', str(OUT / 'CODE_METADATA_ENVELOPE.json')],
        'prepare_local_argv': ['python3', '-B', str(route), prepare_command],
        'stage_stdin_path': str(OUT / 'CODE_METADATA_ENVELOPE.json'), 'stage_stdin_sha256': envelope_sha,
        'actual_remote_executed': False, 'actual_route_entry_or_route_mock_executed': False,
        'asset_transfer': False, 'gpu_or_slurm_execution': False, 'training_release': False,
    }
    route_sha = write('CPU_MANAGEMENT_ROUTE_ARGV.json', route_record)
    write('LOCAL_PACKET_SEAL.json', {
        'status': 'LOCAL_A18F420_CODE_AUDIT_PACKET_SEALED_NO_REMOTE_RELEASE', 'code_commit': COMMIT,
        'packet_sha256': packet_sha, 'route_sha256': route_sha, 'envelope_sha256': envelope_sha,
        'files': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'actual_gpu_or_slurm_executed': False, 'actual_remote_executed': False, 'training_release': False,
    })
    print(json.dumps({'code_commit': COMMIT, 'manifest_sha256': manifest_sha, 'archive_sha256': archive_sha,
                      'packet_sha256': packet_sha, 'route_sha256': route_sha, 'stage_file_count': len(names),
                      'local_builder_returncode': result.returncode, 'actual_remote_or_gpu_execution': False}))


if __name__ == '__main__':
    main()
