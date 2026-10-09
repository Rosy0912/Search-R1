"""Read-only HKUST1 protected checkpoint pin/inventory/USTAR/after entry.

Only stdlib; JSON inputs are exact inline bytes, never files written on Node.
The shared optimizer is an opaque original file, including empty Adam at u0.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
DEPENDENCY_SHA = 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
NODE_DEPENDENCY = '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py'
CP_NAME = re.compile(r'update-(\d{8})-outer-(\d{5})\Z')
UPDATES = (0, 1, 5)
VARIANTS = ('R', 'W', 'P', 'J')


def require(value, message):
    if not value:
        raise ValueError(message)


def primitives(path):
    # Execute the already published dependency's SHA-checked source bytes;
    # runpy/importlib must not admit an older timestamp-valid pyc.
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Canonical absolute dependency required')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        require(not current.is_symlink(), 'Dependency symlink refused')
    fields = lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before = fields(path.stat())
    raw = path.read_bytes()
    require(fields(path.stat()) == before and hashlib.sha256(raw).hexdigest() == DEPENDENCY_SHA,
            'Published V2 dependency changed')
    ns = dict(__name__='_protected_backup_primitives', __file__=str(path))
    exec(compile(raw, str(path), 'exec'), ns)
    return ns


def decode_inline(value, sha):
    require(re.fullmatch('[0-9a-f]{64}', sha or ''), 'Exact inline SHA required')
    raw = base64.b64decode(value, validate=True)
    require(hashlib.sha256(raw).hexdigest() == sha, 'Inline JSON bytes differ')
    return json.loads(raw), raw


def identity_shape(identity):
    require(identity.get('schema') == 'opsd-ablation-protected-identity-v1'
            and identity.get('variant') in VARIANTS, 'Only R/W/P/J protected scope')
    require(re.fullmatch('[0-9a-f]{32}', identity.get('run_id', '')), 'Exact run required')
    for key in ('goal', 'config'):
        ref = identity[key]
        require(Path(ref['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
                'Explicit absolute path/SHA required')
    require(Path(identity['output_root']).is_absolute(), 'Exact output root required')
    return identity


def bind_config(b, identity):
    identity_shape(identity)
    goal, gp = b['checked_json'](identity['goal']['path'], identity['goal']['sha256'])
    groups = b['validate_goal'](goal)
    found = [g for g in groups if g['variant'] == identity['variant']]
    require(len(found) == 1, 'Authorized group missing')
    group = found[0]
    require(group['run_id'] == identity['run_id'] and group['config'] == identity['config']
            and group['output_root'] == identity['output_root'], 'Identity outside exact authorized goal')
    cfg, cp = b['checked_json'](group['config']['path'], group['config']['sha256'])
    require(cfg.get('schema') == 'opsd-ablation-config-v1' and cfg['run_id'] == identity['run_id']
            and cfg['variant'] == identity['variant'] and cfg['study_id'] == goal['study_id']
            and cfg['scientific']['outer_iterations'] == 10
            and cfg['paths']['output_root'] == group['output_root']
            and Path(group['output_root']) == Path(cfg['paths']['base']) / 'runs',
            'Actual config differs from authorized group')
    b['stable'](gp); b['stable'](cp)
    return cfg, cp, gp


def checkpoint_shape(cp, ref, cfg, identity, updates, outer):
    directory = Path(identity['output_root']) / 'checkpoints' / ('update-%08d-outer-%05d' % (updates, outer))
    require(type(outer) is int and 0 <= outer <= 9 and updates in UPDATES,
            'Only actual current-schema protected updates/outer0..9')
    require(ref['path'] == str(directory / 'manifest.json'), 'Committed checkpoint path differs')
    expected_contract = dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
        variant=identity['variant'], precision=cfg['precision']['mode'], mixture_alpha=.1 if identity['variant']=='J' else 0.)
    reason = {0:'initial', 1:'update1', 5:'update5'}[updates]
    require(cp.get('schema') == 'opsd-online-checkpoint-v1' and cp.get('accepted') is True
            and cp['run_id'] == identity['run_id'] and cp['config_sha256'] == identity['config']['sha256']
            and cp['optimizer_updates'] == updates and cp['cursor']['outer_index'] == outer
            and reason in cp['protection_reason'], 'Actual protected checkpoint identity differs')
    require(cp['contract'] == expected_contract and cp['runtime_identity'] == cfg['runtime_manifest']['sha256']
            and cp['source_model_identity'] == cfg['source_model_manifest']['sha256']
            and cp['common_initial_adapter'] == cfg['common_initial_adapter_manifest'], 'Science contract differs')
    require(cp['complete_rank_count'] == 8 and cp['equal_optimizer_steps'] is True
            and len(cp['ranks']) == 8 and sorted(x['rank'] for x in cp['ranks']) == list(range(8)), 'Eight rank files required')
    require(cp['optimizer_storage'] == 'shared_exact_eight_rank_optimizer_v1'
            and cp['shared_optimizer']['path'] == str(directory/'optimizer-shared.pt'), 'Shared Adam omitted/wrong')
    require(cp['all_rng'] == ['python','numpy','torch_cpu','cuda_all'], 'Full RNG declaration required')
    for key in ('all_rank_adapter_sha256', 'all_rank_optimizer_tensor_sha256'):
        require(len(cp[key]) == 8 and len(set(cp[key])) == 1
                and re.fullmatch('[0-9a-f]{64}', cp[key][0]), 'Declared eight-rank tensor identity differs')
    require(len(cp['all_rank_scaler_state']) == 8
            and all(x == cp['all_rank_scaler_state'][0] for x in cp['all_rank_scaler_state']), 'Scaler declaration differs')
    require(cp['adapter']['path'] == str(directory/'adapter') and len(cp['adapter']['files']) == 3
            and {Path(x['path']).name for x in cp['adapter']['files']} == {'README.md','adapter_config.json','adapter_model.safetensors'},
            'Exact three actual PEFT files required')
    refs = [ref, cp['shared_optimizer']]
    for row in cp['ranks']:
        require(row['file']['path'] == str(directory/('rank-%d-state.pt' % row['rank'])), 'Rank file path differs')
        refs.append(row['file'])
    for row in cp['adapter']['files']:
        require(Path(row['path']).parent == directory/'adapter', 'Adapter path escapes checkpoint')
        refs.append(row)
    require(len(refs) == 13 and len({x['path'] for x in refs}) == 13, 'Exact thirteen core files required')
    # No optimizer tensor/step reconstruction is attempted; u0's empty Adam
    # state file is mandatory and copied exactly, without a nonempty-state test.
    require(updates != 0 or (outer == 0 and cp['cursor']['next_pair_cursor'] == 0),
            'Current seven-group initial state must be outer0/cursor0')
    return directory, refs


def pin(b, identity):
    cfg, cref, gp = bind_config(b, identity)
    root = Path(identity['output_root']) / 'checkpoints'
    require(not root.is_symlink(), 'Checkpoint root symlink refused')
    candidates = {u: [] for u in UPDATES}
    if root.exists():
        b['checked_path'](root, directory=True)
        for directory in sorted(root.iterdir()):
            match = CP_NAME.fullmatch(directory.name)
            if not match or int(match[1]) not in UPDATES:
                continue  # Pending directories are never committed states.
            u, outer = map(int, match.groups())
            b['checked_path'](directory, directory=True)
            raw, proof = b['fileproof'](directory/'manifest.json', retain=True)
            cp = json.loads(raw)
            # A later K0 checkpoint can repeat the same update count. Pin the
            # earliest actual committed outer for each original protected state.
            if u == 0 and outer != 0:
                continue
            checkpoint_shape(cp, proof, cfg, identity, u, outer)
            candidates[u].append((outer, proof))
    missing = [u for u in UPDATES if not candidates[u]]
    if missing:
        return dict(schema='opsd-ablation-protected-pin-v1', accepted=False, status='PENDING',
                    identity=identity, missing_updates=missing, no_training_dependency=True)
    refs = [dict(optimizer_updates=u, outer_index=min(candidates[u])[0],
                 manifest={k:min(candidates[u])[1][k] for k in ('path','sha256','bytes')}) for u in UPDATES]
    b['stable'](cref); b['stable'](gp)
    return dict(schema='opsd-ablation-protected-scope-v1', accepted=True, identity=identity,
                protected=refs, protected_updates=list(UPDATES), no_outer_end_requirement=True,
                numeric_state_acceptance=False, pinned_unix=time.time())


def scope_shape(scope):
    require(scope.get('schema') == 'opsd-ablation-protected-scope-v1' and scope.get('accepted') is True,
            'Accepted immutable three-manifest scope required')
    identity_shape(scope['identity'])
    require(scope['protected_updates'] == list(UPDATES)
            and [x['optimizer_updates'] for x in scope['protected']] == list(UPDATES), 'Exact u0/u1/u5 required')
    require(len({x['manifest']['path'] for x in scope['protected']}) == 3, 'Duplicate pinned checkpoint')
    for row in scope['protected']:
        require(type(row['outer_index']) is int and 0 <= row['outer_index'] <= 9
                and re.fullmatch('[0-9a-f]{64}', row['manifest']['sha256']), 'Invalid actual outer/SHA')
    return scope


def core_paths(b, directory):
    files = set()
    for parent, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs:
            b['checked_path'](Path(parent)/name, directory=True)
        for name in names:
            files.add(str(b['checked_path'](Path(parent)/name)))
    return files


def inventory(b, scope, scope_sha):
    scope_shape(scope); identity = scope['identity']
    cfg, cref, gp = bind_config(b, identity)
    files, checkpoints = [], []
    root = Path(identity['output_root'])/'checkpoints'
    for row in scope['protected']:
        cp, proof = b['checked_json'](row['manifest']['path'], row['manifest']['sha256'])
        directory, refs = checkpoint_shape(cp, proof, cfg, identity, row['optimizer_updates'], row['outer_index'])
        require(core_paths(b, directory) == {x['path'] for x in refs}, 'Actual checkpoint core set changed')
        for ref in refs:
            fresh = b['fileproof'](ref['path'], ref)
            files.append(dict(relative='checkpoints/'+Path(ref['path']).relative_to(root).as_posix(), **fresh))
        checkpoints.append(dict(optimizer_updates=row['optimizer_updates'], outer_index=row['outer_index'],
                                manifest=proof, core_file_count=len(refs), empty_adam_initial=row['optimizer_updates']==0))
    runpath = Path(identity['output_root'])/'run-identity.json'
    raw, rip = b['fileproof'](runpath, retain=True); run = json.loads(raw)
    require(run['run_id'] == identity['run_id'] and run['config_sha256'] == identity['config']['sha256'], 'Run identity differs')
    for name, ref in [('config.json',cref),('run-identity.json',rip),('goal.json',gp)]:
        files.append(dict(relative='identity/'+name, **ref))
    require(len(files) == 42 and len({r['path'] for r in files}) == 42
            and len({(r['device'],r['inode']) for r in files}) == 42, 'Source aliases or incomplete closure')
    for row in files:
        b['safe_name'](row['relative']); b['stable'](row)
    return dict(schema='opsd-ablation-protected-inventory-v1', accepted=True, scope_sha256=scope_sha,
        identity=identity, protected=scope['protected'], checkpoints=checkpoints, config=cref, run_identity=rip, goal=gp,
        files=sorted(files,key=lambda r:r['relative']), expected_file_count=42,
        core_file_count=39, total_bytes=sum(x['bytes'] for x in files),
        all_source_full_SHA=True, node_originals_retained=True, numeric_state_acceptance=False,
        backup_is_training_dependency=False, observed_unix=time.time())


def validate_inventory(b, scope, sha, value):
    scope_shape(scope)
    require(value.get('schema') == 'opsd-ablation-protected-inventory-v1' and value.get('accepted') is True
            and value['scope_sha256'] == sha and value['identity'] == scope['identity']
            and value['protected'] == scope['protected'] and value['core_file_count'] == 39
            and value['expected_file_count'] == 42 and len(value['files']) == 42, 'Pinned inventory differs')
    files = value['files']
    require(len({r['relative'] for r in files}) == 42 and len({r['path'] for r in files}) == 42
            and len({(r['device'],r['inode']) for r in files}) == 42, 'Inventory aliases')
    root = Path(scope['identity']['output_root'])/'checkpoints'
    expected_core = set()
    for pinrow in scope['protected']:
        directory = Path(pinrow['manifest']['path']).parent
        names = ['manifest.json','optimizer-shared.pt'] + ['rank-%d-state.pt'%r for r in range(8)]
        names += ['adapter/README.md','adapter/adapter_config.json','adapter/adapter_model.safetensors']
        expected_core.update('checkpoints/'+(directory/part).relative_to(root).as_posix() for part in names)
    require({r['relative'] for r in files} == expected_core | {'identity/config.json','identity/run-identity.json','identity/goal.json'},
            'Inventory omitted/extra actual members')
    identities = {'identity/config.json':scope['identity']['config'], 'identity/goal.json':scope['identity']['goal'],
                  'identity/run-identity.json':dict(path=str(Path(scope['identity']['output_root'])/'run-identity.json'))}
    for row in files:
        name = b['safe_name'](row['relative'])
        require(all(k in row for k in (*b['META'],'sha256')) and type(row['bytes']) is int and row['bytes']>=0
                and b['SHA'].fullmatch(row['sha256']), 'Incomplete file identity')
        if name.startswith('checkpoints/'):
            require(row['path'] == str(root/name[len('checkpoints/'):]), 'Core path differs')
        else:
            ref = identities[name]
            require(row['path'] == ref['path'] and ('sha256' not in ref or row['sha256']==ref['sha256']), 'Closure identity differs')
    mapped = {r['path']:r for r in files}
    for row in scope['protected']:
        ref = row['manifest']; actual = mapped[ref['path']]
        require(actual['sha256']==ref['sha256'] and ('bytes' not in ref or actual['bytes']==ref['bytes']), 'Pinned manifest differs')
    require(sum(r['bytes'] for r in files) == value['total_bytes'], 'Inventory byte total differs')
    return {r['relative']:r for r in files}


def after(b, scope, sha, value, inventory_sha):
    validate_inventory(b,scope,sha,value)
    cfg, cp, gp = bind_config(b,scope['identity'])
    for row in scope['protected']:
        content, ref = b['checked_json'](row['manifest']['path'],row['manifest']['sha256'])
        directory, refs = checkpoint_shape(content,ref,cfg,scope['identity'],row['optimizer_updates'],row['outer_index'])
        require(core_paths(b,directory)=={r['path'] for r in refs}, 'Core set changed during transfer')
    for row in value['files']:
        b['stable'](row)
    b['stable'](cp); b['stable'](gp)
    return dict(schema='opsd-ablation-protected-source-after-v1', accepted=True,
                scope_sha256=sha, inventory_sha256=inventory_sha, identity=value['identity'],
                protected=value['protected'], all_source_metadata_stable=True,
                all_core_file_sets_unchanged=True, files=42, observed_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('pin','inventory','stream','after'))
    for k in ('identity-base64','identity-sha256','scope-base64','scope-sha256'):
        p.add_argument('--'+k)
    a=p.parse_args()
    require(os.getuid()==os.geteuid()==1005 and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'HKUST1 UID1005/CVD empty required')
    b=primitives(NODE_DEPENDENCY)
    if a.mode=='pin':
        identity,_=decode_inline(a.identity_base64,a.identity_sha256)
        value=pin(b,identity)
    else:
        scope,_=decode_inline(a.scope_base64,a.scope_sha256)
        if a.mode=='inventory':
            value=inventory(b,scope,a.scope_sha256)
        else:
            raw=sys.stdin.buffer.read(); value=json.loads(raw)
            validate_inventory(b,scope,a.scope_sha256,value)
            if a.mode=='stream':
                # Revalidate pinned metadata before the original fixed-handle
                # SHA USTAR reader; no inference from subprocess absence.
                for row in value['files']: b['stable'](row)
                b['stream_archive'](value,sys.stdout.buffer)
                return
            value=after(b,scope,a.scope_sha256,value,hashlib.sha256(raw).hexdigest())
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
