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
assert x['schema'] == 'source100-twelve-fixed-assets-inventory-v1' and len(x['files']) == 12
SOURCE = '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/assets/archive100'
MANIFEST = '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/receipts/source100-model-files.json'
MANIFEST_SHA = '2ef6070a328357b0f9d3a026010aaed0aa3f8840a241bc1021ed9a183e8d75f7'
assert x['source'] == SOURCE and x['source_grpo_step'] == 100
assert x['source_manifest']['path'] == MANIFEST and x['source_manifest']['sha256'] == MANIFEST_SHA
declared = json.loads((D / 'source-model-manifest.exact.json').read_bytes())
assert hashlib.sha256((D / 'source-model-manifest.exact.json').read_bytes()).hexdigest() == MANIFEST_SHA
assert {r['path']: (r['bytes'], r['sha256']) for r in x['files']} == {r['path']: (r['bytes'], r['sha256']) for r in declared['files']}
assert all(r['relative'] == pathlib.Path(r['path']).name and str(pathlib.Path(r['path']).parent) == SOURCE for r in x['files'])
A = pathlib.Path(a.after_metadata)
after = json.loads(A.read_bytes())
assert after['accepted'] is True and after['all_source_metadata_stable'] is True
assert after['inventory_sha256'] == hashlib.sha256(raw).hexdigest() and after['source'] == x['source']
assert after['source_manifest']['path'] == MANIFEST and after['source_manifest']['sha256'] == MANIFEST_SHA
launch = json.loads((D / 'transfer-launch.json').read_bytes())
assert launch['inventory_sha256'] == hashlib.sha256(raw).hexdigest()
assert launch['operation_sha256'] == hashlib.sha256(pathlib.Path(__file__).with_name('source-operation.stdin.py').read_bytes()).hexdigest()
terminal = json.loads((D / 'transfer-terminal.json').read_bytes())
assert terminal['actual_returncode'] == 0 and terminal['local_error'] is None
archive = D / 'source100.tar.gz.partial'
ap = b['fileproof'](archive)
expected = {r['relative']: r for r in x['files']}
assert len(expected) == 12 and sum(r['bytes'] for r in expected.values()) == 13604379671
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
final = D / 'source100.tar.gz'
assert not final.exists()
os.rename(archive, final)
b['fsync_dir'](D)
finalmeta = b['metadata'](final)
assert all(finalmeta[k] == ap[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'mode'))
finalproof = dict(finalmeta, sha256=ap['sha256'], sha_origin='Full pre-rename read plus atomic rename metadata continuity; no repeated post-rename full read')
proof = dict(schema='source100-twelve-assets-compressed-local-backup-v1', accepted=True, source_grpo_step=100, source=x['source'], source_manifest=x['source_manifest'], archive=finalproof, pre_rename_archive=ap, members=members, files=12, member_bytes=13604379671, gzip_footer_and_terminal_padding_verified=True, contiguous_regular_tar_members_verified=True, producer_format='USTAR via pinned published stream_archive', internal_member_padding_independently_checked=False, actual_ssh_returncode=0, all_local_decompressed_member_SHA_match=True, source_fresh_SHA_before=True, source_metadata_stable_after=True, source_stream_fixed_handle_SHA=True, node_originals_retained=True, bare_local_copy=False, restore_requires_space=True, inventory_sha256=hashlib.sha256(raw).hexdigest(), after_metadata_sha256=hashlib.sha256(A.read_bytes()).hexdigest(), no_model_or_GPU=True, backup_is_training_dependency=False, observed_unix=time.time())
b['write_json'](D / 'complete-sha-verified.json', proof)
print(json.dumps(dict(accepted=True, files=12, member_bytes=13604379671, archive_bytes=finalproof['bytes'])))
