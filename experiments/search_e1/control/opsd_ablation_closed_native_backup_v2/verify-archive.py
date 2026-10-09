"""Full compressed disk/member readback, no bare extraction or numerical audit."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time


def operation(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha: raise ValueError('Published operation SHA differs')
    namespace = dict(__name__='_closed_raw_operation', __file__=str(path))
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def verify_members(b, archive, value):
    expected = {r['relative']: r for r in value['files']}
    require = b['require']; require(len(expected) == len(value['files']), 'Duplicate expected member')
    observed = set(); members = []; end = 0
    with gzip.open(archive, 'rb') as src:
        with tarfile.open(fileobj=src, mode='r:') as tar:
            for member in tar:
                name = b['safe_name'](member.name)
                require(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    and not member.pax_headers and name in expected and name not in observed,
                    'Special/extra/duplicate/escaping member refused')
                require(member.offset == end and member.offset_data == end + 512, 'Noncontiguous USTAR')
                row = expected[name]; require(member.size == row['bytes'], 'Member byte count differs')
                digest = hashlib.sha256(); count = 0
                with tar.extractfile(member) as payload:
                    for block in iter(lambda: payload.read(4 << 20), b''):
                        digest.update(block); count += len(block)
                require(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Member SHA differs')
                observed.add(name); members.append(dict(relative=name, bytes=count, sha256=digest.hexdigest()))
                end = member.offset_data + ((member.size + 511) // 512) * 512
        require(src.tell() >= end + 512, 'Missing first zero USTAR header')
        for block in iter(lambda: src.read(4 << 20), b''):
            require(not any(block), 'Nonzero trailing USTAR data')
        require(src.tell() >= end + 1024 and src.tell() % 512 == 0, 'Missing complete terminal padding')
    require(observed == set(expected), 'Missing archive member')
    return members


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scope', 'scope-sha256', 'inventory', 'after-metadata', 'destination',
                'source-helper', 'source-operation', 'operation-sha256'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args(); m = operation(a.source_operation, a.operation_sha256); b = m['primitives'](a.source_helper)
    scope, sp = b['checked_json'](Path(a.scope), a.scope_sha256)
    raw, ip = b['fileproof'](Path(a.inventory), retain=True)
    value = json.loads(raw)
    m['validate_inventory'](b, scope, a.scope_sha256, value)
    after_raw, ap = b['fileproof'](Path(a.after_metadata), retain=True); after = json.loads(after_raw)
    require = b['require']
    require(after['schema'] == 'opsd-ablation-closed-native-source-after-v1' and after['accepted'] is True
        and after['scope_sha256'] == a.scope_sha256 and after['inventory_sha256'] == ip['sha256']
        and after['all_source_metadata_stable'] is True and after['all_closed_file_set_unchanged'] is True
        and after['closures'] == value['closures'] and after['run_id'] == value['run_id']
        and after['source'] == value['source'] and after['versions'] == value['versions']
        and after['config'] == value['config'] and after['files'] == len(value['files']), 'Missing fresh matching source-after proof')
    destination = b['checked_path'](str(Path(a.destination)), directory=True)
    launch, lp = b['fileproof'](destination / 'transfer-launch.json', retain=True); launch = json.loads(launch)
    terminal, tp = b['fileproof'](destination / 'transfer-terminal.json', retain=True); terminal = json.loads(terminal)
    require(launch['inventory_sha256'] == ip['sha256'] and launch['scope']['sha256'] == a.scope_sha256
        and launch['operation_sha256'] == a.operation_sha256 and launch['scope_transport'] == 'inline-base64'
        and terminal['actual_returncode'] == 0 and terminal['local_error'] is None, 'Actual transport receipt differs')
    archive = destination / 'closed-native512.tar.gz.partial'; fp = b['fileproof'](archive)
    members = verify_members(b, archive, value)
    for proof in (fp, sp, ip, ap, lp, tp): b['stable'](proof)
    final = destination / 'closed-native512.tar.gz'; require(not final.exists(), 'Existing final never overwritten')
    os.rename(archive, final); b['fsync_dir'](destination)
    meta = b['metadata'](final)
    require(all(meta[k] == fp[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'mode')), 'Rename continuity differs')
    complete = dict(schema='opsd-ablation-closed-native-compressed-local-backup-v1', accepted=True,
        scope_sha256=a.scope_sha256, run_id=value['run_id'], variant=value['variant'], versions=value['versions'], primary_slots=value['primary_slots'],
        closures=value['closures'], config=value['config'], archive=dict(meta, sha256=fp['sha256'],
            sha_origin='Full pre-rename disk read plus atomic rename metadata continuity'),
        members=members, files=len(members), member_bytes=value['bytes'], excluded_subtrees=value['excluded_subtrees'],
        inventory_sha256=ip['sha256'], after_metadata_sha256=ap['sha256'], source_operation_sha256=a.operation_sha256,
        actual_ssh_returncode=0, all_local_decompressed_member_SHA_match=True,
        gzip_footer_and_terminal_padding_verified=True, internal_member_padding_independently_checked=False,
        source_noncache_file_set_stable_after=True, source_stream_fixed_handle_SHA=True,
        source_fresh_SHA_before=True, study_mutable_ledger_excluded=True,
        node_originals_retained=True, bare_local_copy=False, restore_requires_space=True,
        no_model_or_GPU=True, numerical_or_training_or_effect_acceptance=False,
        backup_is_training_dependency=False, observed_unix=time.time())
    b['write_json'](destination / 'complete-sha-verified.json', complete)
    print(json.dumps(dict(accepted=True, files=len(members), member_bytes=value['bytes'], archive_bytes=meta['bytes'])))


if __name__ == '__main__':
    main()
