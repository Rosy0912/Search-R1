"""Local proof adapter for an accepted generic closed-raw gzip; no payload scan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
SOURCE_MANIFEST_SHA = '5ffb33951d3d9a8863c4a3fa097d9c2f3328c488ad18443e027a2ae338949228'
DEPENDENCY_SHA = 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
SCHEMA = 'opsd-ablation-closed-raw-reader-adapter-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def meta(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == str(path.absolute()) and '..' not in path.parts,
            'Canonical absolute input required')
    for current in (path, *path.parents):
        require(not current.is_symlink(), 'Symlink input/ancestor refused')
    s = path.stat(); require(stat.S_ISREG(s.st_mode), 'Regular input required')
    return dict(path=str(path), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mode=s.st_mode, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read(ref):
    path = Path(ref['path']); before = meta(path)
    keys = ('device', 'inode', 'bytes', 'mode', 'mtime_ns', 'ctime_ns')
    def handle_meta(handle):
        s = os.fstat(handle.fileno())
        return (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)
    with path.open('rb') as handle:
        require(handle_meta(handle) == tuple(before[k] for k in keys), 'Opened input identity differs')
        raw = handle.read()
        require(handle_meta(handle) == tuple(before[k] for k in keys), 'Input handle changed during read')
    require(meta(path) == before, 'Input changed while read')
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == ref['sha256'] and ('bytes' not in ref or ref['bytes'] == len(raw)), 'Input SHA/bytes differs')
    return raw, dict(before, sha256=digest)


def json_ref(ref):
    raw, proof = read(ref)
    return json.loads(raw), proof


def source_identity(manifest_ref, dependency_ref):
    require(manifest_ref['sha256'] == SOURCE_MANIFEST_SHA and dependency_ref['sha256'] == DEPENDENCY_SHA,
            'Only the published generic source and exact V2 transport accepted')
    manifest, mp = json_ref(manifest_ref); root = Path(mp['path']).parent
    sources = {}; payloads = {}
    for name, expected in manifest['files'].items():
        require(name == Path(name).name and name not in sources, 'Unexpected source member')
        raw, fp = read(dict(path=str(root / name), **expected)); sources[name] = fp; payloads[name] = raw
    require(set(sources) == {'source-operation.py', 'transfer-gzip.py', 'verify-archive.py', 'test_scope.py', 'README.md'},
            'Published source inventory differs')
    dep_raw, dp = read(dependency_ref)
    b = dict(__name__='_sha_bound_raw_transport', __file__=dp['path'])
    exec(compile(dep_raw, dp['path'], 'exec'), b)
    operation = dict(__name__='_sha_bound_raw_operation', __file__=sources['source-operation.py']['path'])
    exec(compile(payloads['source-operation.py'], operation['__file__'], 'exec'), operation)
    require('torch' not in sys.modules, 'Adapter must remain stdlib CPU-only')
    return b, operation, dict(source_manifest=mp, payloads=sources, transport_dependency=dp)


def same_ref(left, right):
    return all(left[k] == right[k] for k in ('path', 'sha256'))


def execution_identity(value, proof, refs, sources):
    require(value['schema'] == 'opsd-ablation-closed-raw-verifier-execution-v1'
            and value['actual_returncode'] == 0, 'Verifier actual return code is absent/nonzero')
    require(same_ref(value['program'], sources['payloads']['verify-archive.py'])
            and value['program']['bytes'] == sources['payloads']['verify-archive.py']['bytes']
            and same_ref(value['complete'], proof) and value['complete']['bytes'] == proof['bytes'],
            'Actual verifier source or original completion differs')
    argv = value['argv']; names = ('scope', 'scope-sha256', 'inventory', 'after-metadata', 'destination',
                                 'source-helper', 'source-operation', 'operation-sha256')
    expected = dict(zip(names, (refs['scope']['path'], refs['scope']['sha256'], refs['inventory']['path'],
        refs['after_metadata']['path'], str(Path(proof['path']).parent), sources['transport_dependency']['path'],
        sources['payloads']['source-operation.py']['path'], sources['payloads']['source-operation.py']['sha256'])))
    require(isinstance(argv, list) and len(argv) == 3 + 2 * len(names) and Path(argv[0]).is_absolute()
            and argv[1:3] == ['-B', value['program']['path']], 'Exact original verifier CLI required')
    flags = dict(zip(argv[3::2], argv[4::2]))
    require(flags == {'--' + key: val for key, val in expected.items()}, 'Verifier invocation does not bind inputs')


def adapt(refs):
    b, operation, sources = source_identity(refs['source_manifest'], refs['transport_dependency'])
    inputs = {}; proofs = {}
    for key in ('scope', 'inventory', 'complete', 'after_metadata', 'transfer_launch', 'transfer_terminal', 'verifier_execution'):
        inputs[key], proofs[key] = json_ref(refs[key])
    scope = inputs['scope']; inv = operation['validate_inventory'](b, scope, refs['scope']['sha256'], inputs['inventory'])
    require(inv['scope'] == dict(delivery='inline-base64', bytes=proofs['scope']['bytes'], sha256=proofs['scope']['sha256'])
            and inv['source_helper_sha256'] == DEPENDENCY_SHA, 'Original inline scope or fixed transport differs')
    complete = inputs['complete']; after = inputs['after_metadata']; launch = inputs['transfer_launch']; terminal = inputs['transfer_terminal']
    destination = Path(proofs['complete']['path']).parent
    require(Path(proofs['complete']['path']).name == 'complete-sha-verified.json'
            and Path(proofs['transfer_launch']['path']) == destination / 'transfer-launch.json'
            and Path(proofs['transfer_terminal']['path']) == destination / 'transfer-terminal.json', 'Original transfer scope differs')
    require(complete['schema'] == 'opsd-ablation-closed-raw-compressed-local-backup-v1' and complete['accepted'] is True
            and complete['scope_sha256'] == refs['scope']['sha256']
            and complete['run_id'] == inv['run_id'] and complete['variant'] == inv['variant']
            and complete['closed_outers'] == list(range(10)) and complete['closures'] == inv['closures']
            and complete['config'] == inv['config'] and complete['excluded_subtrees'] == inv['excluded_subtrees']
            and complete['inventory_sha256'] == proofs['inventory']['sha256']
            and complete['after_metadata_sha256'] == proofs['after_metadata']['sha256']
            and complete['source_operation_sha256'] == sources['payloads']['source-operation.py']['sha256']
            and complete['actual_ssh_returncode'] == 0
            and complete['all_local_decompressed_member_SHA_match'] is True
            and complete['gzip_footer_and_terminal_padding_verified'] is True, 'Original complete acceptance differs')
    rows = {r['relative']: r for r in inv['files']}; members = {r['relative']: r for r in complete['members']}
    require(len(rows) == len(inv['files']) == inv['expected_file_count'] == complete['files']
            == len(members) == len(complete['members']) and rows.keys() == members.keys()
            and sum(r['bytes'] for r in rows.values()) == inv['bytes'] == complete['member_bytes'], 'Original exact member set differs')
    require(all(all(members[name][k] == row[k] for k in ('sha256', 'bytes')) for name, row in rows.items()), 'Original member SHA differs')
    require(after['schema'] == 'opsd-ablation-closed-raw-source-after-v1' and after['accepted'] is True
            and after['scope_sha256'] == proofs['scope']['sha256'] and after['inventory_sha256'] == proofs['inventory']['sha256']
            and after['closures'] == inv['closures'] and after['source'] == inv['source'] and after['run_id'] == inv['run_id']
            and after['files'] == len(inv['files']) and after['all_source_metadata_stable'] is True
            and after['all_noncache_file_set_unchanged'] is True, 'Source-after cannot prove the exact stable set')
    require(launch['inventory_sha256'] == proofs['inventory']['sha256']
            and same_ref(launch['scope'], proofs['scope']) and launch['scope']['bytes'] == proofs['scope']['bytes']
            and launch['scope_transport'] == 'inline-base64'
            and launch['operation_sha256'] == sources['payloads']['source-operation.py']['sha256']
            and terminal['actual_returncode'] == 0 and terminal['local_error'] is None
            and terminal['partial_bytes'] == complete['archive']['bytes'], 'Actual fixed-source transport is unproven')
    execution_identity(inputs['verifier_execution'], proofs['complete'], refs, sources)
    archive = complete['archive']; current = meta(archive['path'])
    require(archive['path'] == str(destination / 'closed-raw.tar.gz') and re.fullmatch('[0-9a-f]{64}', archive['sha256'])
            and all(current[k] == archive[k] for k in current), 'Accepted archive current metadata differs')
    for proof in (*proofs.values(), sources['source_manifest'], sources['transport_dependency'], *sources['payloads'].values()):
        require(meta(proof['path']) == {k: proof[k] for k in current}, 'Input/source changed after read')
    require(meta(archive['path']) == current, 'Archive changed during proof adaptation')
    result = dict(complete)
    result.update(schema=SCHEMA, contiguous_regular_tar_members_verified=True,
        source_noncache_file_set_stable_after=True, source_stream_fixed_handle_SHA=True,
        original_complete=proofs['complete'], adapter_inputs=proofs, adapter_source_dependencies=sources,
        fact_derivations=dict(
            contiguous_regular_tar_members_verified='Exact published verifier execution rc0; verify_members requires regular exact set, offsets=end and offset_data=end+512, full member SHA, gzip/footer.',
            source_stream_fixed_handle_SHA='Exact source-operation from actual transfer; pinned V2 stream_archive verifies opened handle fstat, streamed SHA, post-handle fstat and final path metadata; actual SSH rc0.',
            source_noncache_file_set_stable_after='Original after matches exact scope/inventory digest and closures; published source after compares all paths/exclusions and checks every pinned stat.'),
        adapter_scope='Proof-schema compatibility only; original acceptance is historical with current archive metadata continuity.',
        fresh_archive_SHA_or_member_scan_performed=False, original_schema_retained_in_provenance=complete['schema'],
        no_node_or_model_or_GPU_IO=True, numerical_or_training_or_effect_acceptance=False)
    return result, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'manifest-sha256', 'output'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    refs, manifest = json_ref(dict(path=args.manifest, sha256=args.manifest_sha256))
    require(refs['schema'] == 'opsd-ablation-closed-raw-reader-adapter-inputs-v1', 'Explicit adapter input required')
    result, b = adapt(refs)
    require(meta(manifest['path']) == {k: manifest[k] for k in meta(manifest['path'])}, 'Input manifest changed')
    result['adapter_input_manifest'] = manifest
    result['adapter_source'] = b['fileproof'](Path(__file__).absolute())
    proof = b['write_json'](Path(args.output), result)
    print(json.dumps(dict(accepted=True, schema=SCHEMA, files=result['files'], original_complete=result['original_complete'], output=proof)))


if __name__ == '__main__': main()
