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
    require(checkpoint['run_id'] == run_id and checkpoint['config_sha256'] == config['_config_sha256'], 'Checkpoint run/config mismatch')
    require(checkpoint['runtime_identity'] == config['runtime_manifest']['sha256'], 'Checkpoint runtime mismatch')
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
    return checkpoint, dict(manifest=proof, ranks=rank_proofs, adapter_files=adapter_proofs)


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
    """Explicit reviewed rolling policy; protected/end checkpoints are permanent.

    Only this new run's fully verified checkpoints may be removed. Keep SHA tombstones.
    No pending directory, raw rollout, token trace, legacy source or foreign run is touched.
    """
    root=Path(root).resolve()
    identity=json.loads((root/'run-identity.json').read_text())
    require(identity['run_id']==run_id and identity['config_sha256']==config_sha,'Retention ownership mismatch')
    config,_=checked_json(identity['config']);config['_config_sha256']=config_sha
    policy=config['retention']
    require(policy==identity['retention'] and policy['authorized'] is True and policy['mode']=='rolling','Retention not explicitly authorized')
    require(policy['latest_complete']==2 and policy['protected_updates']==[0,1,5] and policy['preserve_outer_end'] is True,'Retention protections changed')
    require(identity['initial_gate']['accepted'] is True,'Storage/resource gate never accepted')
    latest=latest_manifest if isinstance(latest_manifest,dict) else fileproof(latest_manifest)
    latest_value,_=check_checkpoint(latest,config,run_id)
    checkpoint_root=root/'checkpoints';require(not checkpoint_root.is_symlink(),'Checkpoint root must not be a symlink')
    entries=[]
    for manifest in checkpoint_root.glob('*/manifest.json'):
        # A durable manifest inside a pending directory is not a committed
        # checkpoint. Preserve every pending artifact without inspecting it.
        if re.fullmatch(r'update-\d{8}-outer-\d{5}', manifest.parent.name) is None:
            continue
        directory=manifest.parent
        require(not directory.is_symlink() and directory.resolve().parent==checkpoint_root.resolve(),'Unsafe checkpoint directory')
        value=json.loads(manifest.read_text())
        require(value.get('run_id')==run_id and value.get('config_sha256')==config_sha,'Foreign checkpoint in owned output root')
        if value.get('accepted') is not True:continue
        entries.append((value['optimizer_updates'],value['cursor']['outer_index'],manifest,value))
    entries.sort(key=lambda e:(e[0],e[1]))
    keep={str(e[2].resolve()) for e in entries[-2:]}|{str(Path(latest['path']).resolve())}
    removed=[]
    for _,_,manifest,value in entries:
        reasons=value.get('protection_reason',[])
        protected=value['optimizer_updates'] in (0,1,5) or bool(reasons) or value['cursor']['next_pair_cursor']==value['cursor']['usable_pair_count']
        if str(manifest.resolve()) in keep or protected:continue
        checked,evidence=check_checkpoint(fileproof(manifest),config,run_id)
        # Every actual file is declared; refuse unexpected artifacts, symlinks and races.
        declared={evidence['manifest']['path']}|{r['file']['path'] for r in evidence['ranks']}|{f['path'] for f in evidence['adapter_files']}
        actual={str(p.resolve()) for p in manifest.parent.rglob('*') if p.is_file()}
        require(actual==declared and not any(p.is_symlink() for p in manifest.parent.rglob('*')),'Undeclared/symlink checkpoint contents; refusing pruning')
        tombstone=dict(schema='opsd-online-checkpoint-pruned-v1',run_id=run_id,config_sha256=config_sha,
            original_manifest=evidence['manifest'],original_manifest_value=checked,original_file_proofs=evidence,
            latest_checkpoint=latest,retention_policy=policy,
            reason='Explicit rolling retention; unprotected intermediate complete checkpoint only')
        dest=root/'retention-receipts'/(manifest.parent.name+'.json')
        exclusive_json(dest,tombstone)
        # Recheck SHA immediately before deleting already completed owned files.
        for file in declared:checked_file(dict(path=file,sha256=next(
            ref['sha256'] for ref in [evidence['manifest']]+[r['file'] for r in evidence['ranks']]+evidence['adapter_files'] if ref['path']==file)))
        shutil.rmtree(manifest.parent);removed.append(str(manifest.parent))
    return dict(accepted=True,removed=removed,protected_updates=[0,1,5],keep_latest=2,preserve_outer_end=True,
                latest_optimizer_updates=latest_value['optimizer_updates'])


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
