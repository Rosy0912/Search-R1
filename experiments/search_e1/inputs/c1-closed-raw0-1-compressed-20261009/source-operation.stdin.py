# Fixed C1 closed-outer raw bytes, reusing published read-only SHA/USTAR primitives.
import hashlib, json, os, pathlib, runpy, sys, tarfile, time
H = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
B = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-C1')
S = B / 'runs/outers'
CFG = B / 'receipts/authorized-launch-20261009/config.json'
CFG_SHA = '13fdbd51cc290bc6d23b0577c67bf8160a17292fab6f315fbfb07ea3f6ed3310'
RUN = 'a1a56efd32e5456d8c6813454975e38f'
PINS = {0:(11,'09a5e605e6afadc8d096cbd048a9ba427268626790ef73713175287013778be3'), 1:(21,'681e246b62807b661cf85457f5f1a25ac4c354dd690371d8b00e06ef2d0089e7')}
config, config_ref = b['checked_json'](CFG, CFG_SHA)
def closure():
    proofs = []
    for index, (updates, cp_sha) in sorted(PINS.items()):
        O = S / ('outer-%05d' % index)
        b['checked_path'](str(O), directory=True)
        raw, complete_ref = b['fileproof'](O/'training/outer-complete.json', retain=True)
        complete = json.loads(raw)
        assert complete['accepted'] and complete['run_id']==RUN
        assert complete['outer_index']==index and complete['optimizer_updates']==updates
        cp_path = B/'runs/checkpoints'/('update-%08d-outer-%05d'%(updates,index))/'manifest.json'
        assert complete['checkpoint']['path']==str(cp_path) and complete['checkpoint']['sha256']==cp_sha
        cp, cp_ref = b['checked_json'](cp_path, cp_sha)
        assert cp['accepted'] and cp['run_id']==RUN and cp['config_sha256']==CFG_SHA
        assert cp['optimizer_updates']==updates and cp['cursor']['outer_index']==index
        proofs.append(dict(outer_index=index, optimizer_updates=updates, complete=complete_ref, checkpoint=cp_ref))
    return proofs

def enumerate_files():
    paths=[]; excluded=[]
    for index in sorted(PINS):
        O=S/('outer-%05d'%index)
        for root, dirs, names in os.walk(O, topdown=True, followlinks=False):
            root=pathlib.Path(root); b['checked_path'](str(root), directory=True)
            if root==O and 'merged-model-cache' in dirs:
                cache=root/'merged-model-cache'
                b['checked_path'](str(cache), directory=True)
                dirs.remove('merged-model-cache')
                excluded.append(dict(path=str(cache), reason='rebuildable inference weight cache only; merge identity/exit proofs remain included'))
            for name in dirs: b['checked_path'](str(root/name), directory=True)
            for name in names:
                p=b['checked_path'](str(root/name)); rel=b['safe_name'](p.relative_to(S).as_posix())
                item=tarfile.TarInfo(rel); item.tobuf(format=tarfile.USTAR_FORMAT)
                paths.append(p)
    paths=sorted(paths, key=lambda p:p.relative_to(S).as_posix())
    assert paths and len(paths)==len(set(paths))
    return paths, excluded
mode=sys.argv[1]
closures=closure()
paths,excluded=enumerate_files()
if mode=='inventory':
    files=[dict(relative=p.relative_to(S).as_posix(), **b['fileproof'](p)) for p in paths]
    for f in files: b['stable'](f)
    assert closures==closure() and paths==enumerate_files()[0]
    print(json.dumps(dict(schema='c1-closed0-1-raw-fixed-inventory-v1', source=str(S), run_id=RUN, variant='C1', closed_outers=sorted(PINS), config=config_ref, closures=closures, files=files, excluded_subtrees=excluded, bytes=sum(f['bytes'] for f in files), expected_file_count=len(files), uniform_original_byte_archive=True, original_NPY_bytes_not_reserialized=True, node_originals_retained=True, backup_is_training_dependency=False, observed_unix=time.time()), sort_keys=True))
else:
    raw=sys.stdin.buffer.read(); x=json.loads(raw)
    assert x['schema']=='c1-closed0-1-raw-fixed-inventory-v1' and x['source']==str(S)
    assert x['run_id']==RUN and x['closed_outers']==sorted(PINS)
    assert x['config']==config_ref and x['closures']==closures
    assert x['expected_file_count']==len(x['files']) and x['bytes']==sum(f['bytes'] for f in x['files'])
    assert [f['path'] for f in x['files']]==[str(p) for p in paths]
    assert all(f['relative']==pathlib.Path(f['path']).relative_to(S).as_posix() for f in x['files'])
    for f in x['files']: b['stable'](f)
    if mode=='stream': b['stream_archive'](x, sys.stdout.buffer)
    else:
        assert mode=='after'
        assert [str(p) for p in enumerate_files()[0]]==[f['path'] for f in x['files']]
        assert closures==closure()
        print(json.dumps(dict(schema='c1-closed0-1-raw-source-metadata-after-v1', accepted=True, source=str(S), run_id=RUN, closed_outers=sorted(PINS), config=config_ref, closures=closures, files=len(paths), inventory_sha256=hashlib.sha256(raw).hexdigest(), all_source_metadata_stable=True, all_noncache_file_set_unchanged=True, observed_unix=time.time()), sort_keys=True))
