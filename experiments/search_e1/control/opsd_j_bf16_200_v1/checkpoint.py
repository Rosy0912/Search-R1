import re
"""CPU-only immutable artifact and complete resumable-checkpoint contracts."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil


def require(ok, message):
    if not ok:
        raise ValueError(message)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def fileproof(path):
    path = Path(path).resolve()
    before = stat_identity(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    require(before == stat_identity(path), 'Artifact changed while hashing: ' + str(path))
    return dict(path=str(path), sha256=h.hexdigest(), **before)


def checked_file(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'File reference requires path/sha256 dict')
    require(isinstance(ref.get('sha256'), str) and len(ref['sha256']) == 64, 'Missing SHA256')
    actual = fileproof(ref['path'])
    require(actual['sha256'] == ref['sha256'], 'Artifact SHA mismatch: ' + ref['path'])
    if 'bytes' in ref:
        require(actual['bytes'] == ref['bytes'], 'Artifact size mismatch')
    return actual


def checked_json(ref):
    actual = checked_file(ref)
    value = json.loads(Path(actual['path']).read_text())
    require(stat_identity(actual['path']) == {k: actual[k] for k in stat_identity(actual['path'])}, 'JSON changed while reading')
    return value, actual


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return fileproof(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    exclusive_json(temporary, value)
    os.replace(temporary, path)
    return fileproof(path)


def check_checkpoint(ref, config, run_id, expected_updates=None, expected_outer=None):
    """Hash actual rank state bytes; never infer complete Adam from adapter export."""
    checkpoint, proof = checked_json(ref)
    require(checkpoint['schema'] == 'opsd-online-checkpoint-v1' and checkpoint['accepted'] is True, 'Incomplete checkpoint')
    prefix=config.get('continuation',{})
    inherited=(ref.get('path')==prefix.get('prefix_checkpoint',{}).get('path') and ref.get('sha256')==prefix.get('prefix_checkpoint',{}).get('sha256') and checkpoint['config_sha256']==prefix.get('prefix_config_sha256') and checkpoint['optimizer_updates']==50 and checkpoint['cursor']['outer_index']==4)
    require(checkpoint['run_id']==run_id and (checkpoint['config_sha256']==config['_config_sha256'] or inherited),'Checkpoint run/config mismatch; only exact accepted CP50 may cross budget revision')
    require(checkpoint['runtime_identity'] == config['runtime_manifest']['sha256'], 'Checkpoint runtime mismatch')
    require(checkpoint.get('contract') == science_contract(config), 'Reverse/variant/precision checkpoint contract differs')
    require(checkpoint.get('common_initial_adapter') == config['common_initial_adapter_manifest'], 'Common initial adapter changed')
    checked_json(checkpoint['outer_old_adapter'])
    require(checkpoint.get('all_rank_scaler_state') and len(checkpoint['all_rank_scaler_state']) == 8,
            'Eight scaler states missing')
    require(all(x == checkpoint['all_rank_scaler_state'][0] for x in checkpoint['all_rank_scaler_state']),
            'Eight scaler states differ')
    require(checkpoint['source_model_identity'] == config['source_model_manifest']['sha256'], 'Teacher/source model mismatch')
    updates = checkpoint['optimizer_updates']
    require(type(updates) is int and updates >= 0, 'Invalid actual optimizer count')
    require(type(checkpoint['cumulative_pair_exposure']) is int and checkpoint['cumulative_pair_exposure'] >= 0, 'Invalid exposure')
    if expected_updates is not None:
        require(updates == expected_updates, 'Checkpoint update counter differs')
    require(checkpoint['complete_rank_count'] == 8 and checkpoint['equal_optimizer_steps'] is True, 'Eight-rank optimizer equality absent')
    ranks = checkpoint['ranks']
    require(len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8)), 'Missing/duplicate rank state')
    rank_proofs = [dict(rank=r['rank'], file=checked_file(r['file'])) for r in ranks]
    require(len({p['file']['path'] for p in rank_proofs}) == 8, 'Rank state files aliased')
    require(len({(p['file']['device'],p['file']['inode']) for p in rank_proofs}) == 8, 'Rank states share same inode')
    require(checkpoint.get('optimizer_storage')=='shared_exact_eight_rank_optimizer_v1','New seven-group optimizer storage contract differs')
    shared=checked_file(checkpoint['shared_optimizer'])
    require(Path(shared['path']).name=='optimizer-shared.pt'
            and Path(shared['path']).parent==Path(proof['path']).parent,'Shared optimizer outside this exact checkpoint')
    optimizer_digests=checkpoint['all_rank_optimizer_tensor_sha256']
    require(len(optimizer_digests)==8 and len(set(optimizer_digests))==1 and len(optimizer_digests[0])==64,'Eight actual full optimizer identities differ')
    adapter = checkpoint['adapter']
    require(adapter['files'] and Path(adapter['path']).is_dir(), 'Missing adapter directory')
    adapter_proofs = [checked_file(f) for f in adapter['files']]
    root = Path(adapter['path']).resolve()
    require(all(Path(f['path']).is_relative_to(root) for f in adapter_proofs), 'Adapter file outside adapter root')
    require({'adapter_config.json', 'adapter_model.safetensors'} <= {Path(f['path']).name for f in adapter_proofs}, 'Incomplete adapter')
    digests = checkpoint['all_rank_adapter_sha256']
    require(len(digests) == 8 and len(set(digests)) == 1 and len(digests[0]) == 64, 'Rank adapters differ')
    require(checkpoint['all_rng'] == ['python', 'numpy', 'torch_cpu', 'cuda_all'], 'Full RNG state declaration missing')
    require(checkpoint['scheduler'] == dict(kind='none', warmup_updates=0), 'Unapproved scheduler/warmup')
    cursor = checkpoint['cursor']
    require(type(cursor['outer_index']) is int and type(cursor['next_pair_cursor']) is int and cursor['next_pair_cursor'] >= 0, 'Invalid resume cursor')
    require(type(cursor['usable_pair_count']) is int and cursor['next_pair_cursor'] <= cursor['usable_pair_count'] <= 512, 'Invalid usable pair cursor')
    if expected_outer is not None:
        require(cursor['outer_index'] == expected_outer, 'Wrong outer checkpoint')
    # A trainer writes these actual identities, including zero-usable-pair rounds.
    for name in ('outer_manifest', 'pairs', 'order', 'encoding'):
        checked_file(checkpoint['training_input'][name])
    require(checkpoint['sampling_model_identity'], 'No fixed sampling model provenance')
    return checkpoint, dict(manifest=proof, ranks=rank_proofs, shared_optimizer=shared,adapter_files=adapter_proofs)


def science_contract(config):
    """New experiment identity cannot be mistaken for historical forward KL."""
    return dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
        variant=config['variant'], precision=config['precision']['mode'],
        mixture_alpha=.1 if config['variant'] in ('M', 'J') else 0.)


def check_outer_complete(ref, config, run_id, outer_index, previous_updates, pair_count):
    value, proof = checked_json(ref)
    require(value['schema'] == 'opsd-online-outer-complete-v1' and value['accepted'] is True, 'Outer completion rejected')
    require(value['run_id'] == run_id and value['outer_index'] == outer_index, 'Wrong outer completion identity')
    require(value['P'] == pair_count and type(value['K']) is int and 0 <= value['K'] <= pair_count <= 512, 'Invalid P/K')
    expected = math.ceil(value['K'] / config['scientific']['global_batch'])
    require(value['optimizer_updates_delta'] == expected, 'Dropped tail / fabricated optimizer count')
    require(value['optimizer_updates'] == previous_updates + expected, 'Optimizer not continuous across outer rounds')
    require(value['scheduler_advanced_updates'] == 0, 'Original no-scheduler policy changed')
    ckpt, cp = check_checkpoint(value['checkpoint'], config, run_id, value['optimizer_updates'], outer_index)
    require(ckpt['cursor']['next_pair_cursor'] == value['K'], 'Outer checkpoint cursor incomplete')
    require(ckpt['cursor']['usable_pair_count'] == value['K'], 'K/checkpoint mismatch')
    return value, dict(completion=proof, checkpoint=cp)


def prune_checkpoints(root,run_id,config_sha,latest_manifest):
    """New run only: retain key/end/last2 state payloads and every immutable manifest."""
    root=Path(root)
    require(root.parent.name.startswith('search-e1-opsd-j-minus-p-bf16-') and root.name=='runs'
            and not root.is_symlink() and not root.parent.is_symlink(),'Pruning refuses any old namespace')
    root=root.resolve()
    identity=json.loads((root/'run-identity.json').read_text())
    require(identity['run_id']==run_id and identity['config_sha256']==config_sha,'Retention run/config changed')
    config,_=checked_json(identity['config']);policy=config['retention']
    require(config['schema']=='opsd-ablation-config-v1' and config['variant']=='J'
            and policy==dict(mode='keep_key_and_resume',authorized=True,merged_cache_cleanup_authorized=True,
                checkpoint_keep_latest=2,protect_updates=[0,1,5],keep_all_outer_end=True,merged_cache_keep=2)
            and identity['retention']==policy,'Explicit new-run key/resume retention missing')
    latest,_=checked_json(latest_manifest)
    require(latest['run_id']==run_id and latest['config_sha256']==config_sha,'Latest full state is not this run')
    candidates=[]
    for path in (root/'checkpoints').glob('*/manifest.json'):
        if re.fullmatch(r'update-\d{8}-outer-\d{5}',path.parent.name) is None:continue
        value,ref=checked_json(fileproof(path))
        require(value['run_id']==run_id and value['config_sha256']==config_sha and value['accepted'] is True,
                'Unexpected foreign/incomplete committed checkpoint')
        if (path.parent/'payload-pruned.json').exists():continue
        candidates.append((value['optimizer_updates'],value['cursor']['outer_index'],path,value,ref))
    candidates.sort(key=lambda item:(item[0],item[1]))
    keep={str(item[2].resolve()) for item in candidates[-2:]}
    keep.add(str(Path(latest_manifest['path']).resolve()))
    removed=[]
    for updates,outer,path,value,ref in candidates:
        if (updates in policy['protect_updates'] or 'outer_end' in value['protection_reason']
                or str(path.resolve()) in keep):continue
        directory=path.parent;require(not directory.is_symlink(),'Checkpoint directory symlink refused')
        payload=value['adapter']['files']+[item['file'] for item in value['ranks']]+[value['shared_optimizer']]
        paths={Path(item['path']) for item in payload}
        require(len(paths)==len(payload) and all(p.is_relative_to(directory) for p in paths),
                'Checkpoint payload references escape the exact committed directory')
        intent_path=directory/'payload-prune-intent.json';intent=None
        if intent_path.exists():
            intent,_=checked_json(fileproof(intent_path))
            require(intent['manifest']==ref and intent['run_id']==run_id and intent['config_sha256']==config_sha,
                    'Existing prune intent belongs to another state')
        else:
            actual={p for p in directory.rglob('*') if p.is_file()}
            require(actual==paths|{path} and not any(p.is_symlink() for p in directory.rglob('*')),
                    'Unbound checkpoint files or symlinks prohibit reclamation')
            for item in payload:
                require(stat_identity(item['path'])=={k:item[k] for k in stat_identity(item['path'])},
                        'State payload changed since its original full-SHA commit')
            intent=dict(schema='opsd-pr100-state-payload-prune-intent-v1',run_id=run_id,config_sha256=config_sha,
                manifest=ref,payload=payload,bytes=sum(item['bytes'] for item in payload),
                keep_latest_manifest=latest_manifest,scope='Non-key/non-outer-end state payload only; immutable manifest preserved')
            exclusive_json(intent_path,intent)
        require(not directory.is_symlink() and all(not p.is_symlink() for p in paths),
                'Pruning refuses symlinked payload after intent')
        for item in payload:
            p=Path(item['path'])
            if p.exists():
                require(stat_identity(p)=={k:item[k] for k in stat_identity(p)},'Payload changed during reclamation')
                p.unlink()
        adapter_dir=directory/'adapter'
        if adapter_dir.exists():adapter_dir.rmdir()
        proof=exclusive_json(directory/'payload-pruned.json',dict(schema='opsd-pr100-state-payload-pruned-v1',
            accepted=True,run_id=run_id,config_sha256=config_sha,original_manifest=ref,intent=fileproof(intent_path),
            bytes_reclaimed=intent['bytes'],scope='Manifest retained for provenance; explicitly not a resumable state',
            all_outer_end_and_key_states_retained=True))
        removed.append(proof)
    return dict(accepted=True,removed=removed,policy=policy,latest_complete_state=latest_manifest)


def prune_merged_caches(root,run_id,config_sha):
    """Prune at most completed CPU merge caches after explicit policy, never source/adapter."""
    root=Path(root).resolve();identity=json.loads((root/'run-identity.json').read_text())
    require(identity['run_id']==run_id and identity['config_sha256']==config_sha,'Merge retention ownership mismatch')
    config,_=checked_json(identity['config'])
    require(config['retention']==identity['retention'] and config['retention']['authorized'] is True and config['retention']['merged_cache_keep']==2,'Merge cache pruning unapproved')
    completed=[]
    for outer in (root/'outers').glob('outer-*'):
        receipt=outer/'merged-model.json';cache=outer/'merged-model-cache'
        if receipt.exists() and cache.exists() and (outer/'training'/'outer-complete.json').exists():completed.append((outer,receipt,cache))
    completed.sort(key=lambda x:x[0].name);removed=[]
    for outer,receipt,cache in completed[:-2]:
        require(not cache.is_symlink() and cache.resolve().parent==outer.resolve(),'Unsafe cache root')
        value,proof=checked_json(fileproof(receipt))
        require(value['kind']=='original-PEFT-CPU-BF16-merge-latest' and Path(value['path']).resolve()==cache.resolve(),'Wrong merge cache identity')
        require(value['source_model_identity']==config['source_model_manifest']['sha256'],'Cache belongs to wrong source')
        checked_files=[checked_file(f) for f in value['files']]
        require({str(p.resolve()) for p in cache.rglob('*') if p.is_file()}=={f['path'] for f in checked_files},'Cache file inventory changed')
        require(not any(p.is_symlink() for p in cache.rglob('*')),'Cache symlink refused')
        require(all(Path(f['path']).is_relative_to(cache.resolve()) for f in checked_files),'Cache proof outside cache')
        exclusive_json(outer/'merged-cache-pruned.json',dict(schema='opsd-online-rebuildable-merge-cache-pruned-v1',
            run_id=run_id,config_sha256=config_sha,merge_receipt=proof,original_files=checked_files,
            reconstruction='Original source100 BF16 + this retained latest adapter/checkpoint; original output byte SHAs preserved',
            raw_trajectory_and_token_loss_preserved=True))
        shutil.rmtree(cache);removed.append(str(cache))
    return dict(accepted=True,removed=removed,cache_keep=2)
