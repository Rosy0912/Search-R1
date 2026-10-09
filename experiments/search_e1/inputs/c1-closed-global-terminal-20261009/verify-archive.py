# Full independent compressed-disk/member readback, without a second bare copy.
import argparse, gzip, hashlib, json, os, pathlib, runpy, tarfile, time

p = argparse.ArgumentParser()
for k in ('inventory', 'after-metadata', 'destination', 'source-helper'):
    p.add_argument('--' + k, required=True)
a = p.parse_args()
H = pathlib.Path(a.source_helper)
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
D = pathlib.Path(a.destination)
I = pathlib.Path(a.inventory)
raw = I.read_bytes()
x = json.loads(raw)
assert x['schema'] == 'c1-closed-global-terminal-fixed-inventory-v1' and x['closed_outers'] == list(range(10))
assert x['source'] == '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008'
assert x['run_id'] == 'a1a56efd32e5456d8c6813454975e38f'
assert x['config']['sha256'] == '13fdbd51cc290bc6d23b0577c67bf8160a17292fab6f315fbfb07ea3f6ed3310'
assert x['expected_file_count'] == len(x['files']) and x['bytes'] == sum(f['bytes'] for f in x['files'])
assert len({f['relative'] for f in x['files']}) == len(x['files'])
assert all(str(pathlib.Path(f['path'])) == str(pathlib.Path(x['source'])/f['relative']) for f in x['files'])
A = pathlib.Path(a.after_metadata)
after = json.loads(A.read_bytes())
assert after['schema'] == 'c1-closed-global-terminal-source-metadata-after-v1'
assert after['accepted'] is True and after['all_source_metadata_stable'] is True
assert after['inventory_sha256'] == hashlib.sha256(raw).hexdigest() and after['source'] == x['source']
assert after['all_noncache_file_set_unchanged'] and after['run_id']==x['run_id']
assert after['config']==x['config'] and after['closures']==x['closures'] and after['files']==len(x['files'])
launch = json.loads((D / 'transfer-launch.json').read_bytes())
assert launch['inventory_sha256'] == hashlib.sha256(raw).hexdigest()
assert launch['operation_sha256'] == hashlib.sha256(pathlib.Path(__file__).with_name('source-operation.stdin.py').read_bytes()).hexdigest()
terminal = json.loads((D / 'transfer-terminal.json').read_bytes())
assert terminal['actual_returncode'] == 0 and terminal['local_error'] is None
archive = D / 'closed-global.tar.gz.partial'
ap = b['fileproof'](archive)
expected = {r['relative']: r for r in x['files']}
assert len(expected) == x['expected_file_count'] and sum(r['bytes'] for r in expected.values()) == x['bytes']
observed, members, end = set(), [], 0
with gzip.open(archive, 'rb') as src:
    with tarfile.open(fileobj=src, mode='r:') as tar:
        for member in tar:
            name = b['safe_name'](member.name)
            assert member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
            assert not member.pax_headers and name in expected and name not in observed
            assert member.offset == end and member.offset_data == end + 512
            ref = expected[name]
            assert member.size == ref['bytes']
            h, count = hashlib.sha256(), 0
            with tar.extractfile(member) as f:
                for block in iter(lambda: f.read(4 << 20), b''):
                    h.update(block)
                    count += len(block)
            assert count == ref['bytes'] and h.hexdigest() == ref['sha256']
            observed.add(name)
            members.append(dict(relative=name, bytes=count, sha256=h.hexdigest()))
            end = member.offset_data + ((member.size + 511) // 512) * 512
    assert src.tell() >= end + 512
    # Tar stopped at the first zero header; drain all remaining padding/footer.
    for block in iter(lambda: src.read(4 << 20), b''):
        assert not any(block)
    decompressed_bytes = src.tell()
    assert decompressed_bytes >= end + 1024 and decompressed_bytes % 512 == 0
assert observed == set(expected)
b['stable'](ap)
final = D / 'closed-global.tar.gz'
assert not final.exists()
os.rename(archive, final)
b['fsync_dir'](D)
finalmeta = b['metadata'](final)
assert all(finalmeta[k] == ap[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'mode'))
finalproof = dict(finalmeta, sha256=ap['sha256'], sha_origin='Full pre-rename read plus atomic rename metadata continuity; no repeated post-rename full read')
proof = dict(schema='c1-closed-global-terminal-compressed-local-backup-v1', accepted=True, run_id=x['run_id'], variant='C1', closed_outers=x['closed_outers'], source=x['source'], config=x['config'], closures=x['closures'], archive=finalproof, pre_rename_archive=ap, members=members, files=len(expected), member_bytes=x['bytes'], excluded_scope=x['excluded_scope'], gzip_footer_and_terminal_padding_verified=True, contiguous_regular_tar_members_verified=True, producer_format='USTAR via pinned published stream_archive', internal_member_padding_independently_checked=False, actual_ssh_returncode=0, all_local_decompressed_member_SHA_match=True, source_fresh_SHA_before=True, source_noncache_file_set_stable_after=True, source_metadata_stable_after=True, source_stream_fixed_handle_SHA=True, node_originals_retained=True, bare_local_copy=False, restore_requires_space=True, inventory_sha256=hashlib.sha256(raw).hexdigest(), after_metadata_sha256=hashlib.sha256(A.read_bytes()).hexdigest(), no_model_or_GPU=True, backup_is_training_dependency=False, observed_unix=time.time())
b['write_json'](D / 'complete-sha-verified.json', proof)
print(json.dumps(dict(accepted=True, files=len(expected), member_bytes=x['bytes'], archive_bytes=finalproof['bytes'])))
