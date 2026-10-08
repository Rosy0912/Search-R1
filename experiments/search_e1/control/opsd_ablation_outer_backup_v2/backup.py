"""Seven ablations: pinned accepted outer-end checkpoint, read-only verified backup.

This entry uses only stdlib and imports no other experimental Python module.
Inventory/stream write only stdout/stderr; backup writes only a fresh local dir.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
REPO_PREFIX = 'experiments/search_e1/control/opsd_ablation_outer_backup_v2/'
VARIANTS = ('C0','C1','M','R','W','P','J')
META = ('path', 'device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns', 'mode')
SHA = re.compile(r'[0-9a-f]{64}\Z')


class Rejected(ValueError):
    pass


def require(value, message):
    if not value:
        raise Rejected(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Invalid archive/source member name')
    p = PurePosixPath(name)
    require(not p.is_absolute() and all(x not in ('', '.', '..') for x in name.split('/'))
            and str(p) == name, 'Noncanonical or escaping member name')
    return name


def checked_path(path, directory=False):
    """No resolve(): reject a symlink in *every* existing path component."""
    p = Path(path)
    require(p.is_absolute() and str(p) == str(path) and '..' not in p.parts,
            'Source path must be canonical absolute')
    current = Path(p.anchor)
    for index, component in enumerate(p.parts[1:]):
        current /= component
        s = current.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'Symlink path component refused')
        last = index == len(p.parts) - 2
        require(stat.S_ISDIR(s.st_mode) if not last or directory else stat.S_ISREG(s.st_mode),
                'Nonregular file/path component refused')
    return p


def meta_from_stat(path, s):
    return dict(path=str(path), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)


def metadata(path):
    p = checked_path(path)
    return meta_from_stat(p, p.lstat())


def stable(proof):
    require(metadata(proof['path']) == {k: proof[k] for k in META},
            'Source metadata changed after verified read')


def fileproof(path, expected=None, retain=False):
    before = metadata(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(str(path), flags)
    digest = hashlib.sha256()
    raw = bytearray() if retain else None
    with os.fdopen(fd, 'rb') as handle:
        require(meta_from_stat(path, os.fstat(handle.fileno())) == before,
                'Source file changed before fixed-handle read')
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
            if raw is not None:
                raw.extend(block)
        require(meta_from_stat(path, os.fstat(handle.fileno())) == before,
                'Source file changed during fixed-handle read')
    require(metadata(path) == before, 'Source path changed during read')
    proof = dict(before, sha256=digest.hexdigest())
    if expected is not None:
        require(isinstance(expected, dict) and expected.get('path') == str(path)
                and SHA.fullmatch(expected.get('sha256', ''))
                and expected['sha256'] == proof['sha256'], 'Declared file SHA/path differs')
        for key in ('bytes', 'device', 'inode', 'mtime_ns', 'ctime_ns', 'mode'):
            if key in expected:
                require(expected[key] == proof[key], 'Declared file metadata differs: ' + key)
    return (bytes(raw), proof) if retain else proof


def checked_json(path, sha):
    raw, proof = fileproof(path, dict(path=str(path), sha256=sha), retain=True)
    return json.loads(raw), proof


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    # Exclusive outputs never replace an earlier success/failure record.
    with Path(path).open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(Path(path).parent)
    return fileproof(path)


def source_publication(manifest_path, manifest_sha, publication_path, publication_sha, commit, run=None):
    root = Path(__file__).absolute().parent
    require(Path(manifest_path) == root / 'source-manifest.json', 'Wrong current source root')
    manifest, mp = checked_json(manifest_path, manifest_sha)
    expected = manifest['files']
    actual = {p.name for p in root.iterdir() if p.name != 'source-manifest.json'}
    require(actual == set(expected), 'Helper source inventory differs; no pycache is loaded')
    proofs = []
    for name, row in expected.items():
        safe_name(name)
        require('/' not in name, 'Unexpected nested source payload')
        fp = fileproof(root / name, dict(path=str(root / name), **row))
        proofs.append(dict(relative_path=name, **fp))
    pub, pp = checked_json(publication_path, publication_sha)
    require(re.fullmatch(r'[0-9a-f]{40}', commit or '') and pub.get('accepted') is True
            and pub.get('branch') == 'search-e1' and (pub.get('run_id') == run or run in pub.get('run_ids', {}).values())
            and pub.get('source_commit') == pub.get('remote_readback_sha') == commit
            and pub.get('source_manifest_sha256') == mp['sha256'],
            'Published source/remote commit readback binding missing')
    for fp in proofs + [dict(relative_path='source-manifest.json', **mp)]:
        matches = [x for x in pub['files'] if x['path'] == REPO_PREFIX + fp['relative_path']]
        require(len(matches) == 1 and matches[0].get('equal') is True
                and matches[0]['sha256'] == fp['sha256'] and matches[0]['bytes'] == fp['bytes'],
                'Source payload not equal to published remote readback')
    for fp in proofs + [mp, pp]:
        stable(fp)
    return dict(manifest=mp, publication=pp, source_commit=commit, files=proofs)


def validate_goal(goal):
    require(goal.get('schema') == 'opsd-ablation-outer-backup-goal-v1'
            and goal.get('accepted') is True and goal.get('goal_scope') == 'seven-group-outer-end10'
            and goal.get('cluster') == 'HKUST' and goal.get('explicit_node_local_with_mac_backup') is True,
            'Missing explicit new seven-group local/Mac backup authorization')
    require(isinstance(goal.get('study_id'), str) and goal['study_id'], 'Fixed study identity required')
    groups = goal['groups']
    require(len(groups) == 7 and [g['variant'] for g in groups] == list(VARIANTS), 'Exact ordered seven groups required')
    require(len({g['run_id'] for g in groups}) == 7 and len({g['config']['path'] for g in groups}) == 7
            and len({g['output_root'] for g in groups}) == 7,
            'Duplicate group identity')
    for g in groups:
        require(re.fullmatch(r'[0-9a-f]{32}', g['run_id']) and SHA.fullmatch(g['config']['sha256']), 'Invalid bound group identity')
        require(Path(g['config']['path']).is_absolute() and Path(g['output_root']).is_absolute(), 'Absolute group paths required')
    return groups


def target_scope(run, updates, outer, checkpoint_sha, goal):
    groups = validate_goal(goal)
    require(type(updates) is int and updates >= 0 and type(outer) is int and 0 <= outer <= 9,
            'Actual nonnegative update count and outer0..9 required')
    require(SHA.fullmatch(checkpoint_sha or ''), 'Pinned exact manifest SHA required')
    found = [g for g in groups if g['run_id'] == run]
    require(len(found) == 1, 'Run outside this authorized study')
    return found[0]


def bind_goal(ref, config_ref, cp_ref, run, updates, outer, verify_evidence=True):
    goal, proof = checked_json(ref['path'], ref['sha256'])
    group = target_scope(run, updates, outer, cp_ref['sha256'], goal)
    require(group['config'] == dict(path=config_ref['path'], sha256=config_ref['sha256']), 'Pinned config differs from goal')
    return dict(value=goal, file=proof, group=group)


def within(path, root):
    p = checked_path(path)
    require(p.is_relative_to(root), 'Declared source escapes its permitted root')
    return p


def declared_file(ref):
    require(isinstance(ref, dict) and all(key in ref for key in
            ('path', 'sha256', 'bytes', 'device', 'inode', 'mtime_ns', 'ctime_ns')),
            'Declared artifact lacks exact SHA/bytes/stat identity')
    require(isinstance(ref['bytes'], int) and ref['bytes'] >= 0, 'Invalid declared byte count')
    return fileproof(ref['path'], ref)


def closure_file(ref):
    # Historical common-initial payloads declare path/SHA/bytes, without stat.
    # Capture fresh fixed-handle stat; compare every metadata field if declared.
    require(isinstance(ref, dict) and all(k in ref for k in ('path', 'sha256', 'bytes')),
            'Closure artifact requires exact path/SHA/bytes')
    require(type(ref['bytes']) is int and ref['bytes'] >= 0, 'Invalid closure byte count')
    return fileproof(ref['path'], ref)


def inventory(config_path, config_sha, checkpoint_path, checkpoint_sha, run, updates, outer,
              publication=None, goal_ref=None, completion_ref=None):
    require(goal_ref is not None, 'Explicit immutable study goal required')
    config, cref = checked_json(config_path, config_sha)
    cp, cpref = checked_json(checkpoint_path, checkpoint_sha)
    goal = bind_goal(goal_ref, cref, cpref, run, updates, outer)
    group = goal['group']; variant = group['variant']
    require(config['variant'] == variant and config['scientific']['outer_iterations'] == 10
            and config['run_id'] == run and config['study_id'] == goal['value']['study_id'], 'Seven-group/ten-outer config differs')
    expected_contract = dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
        variant=variant, precision=config['precision']['mode'], mixture_alpha=.1 if variant in ('M','J') else 0.)
    require(cp['contract'] == expected_contract and cp['common_initial_adapter'] == config['common_initial_adapter_manifest'],
            'Reverse/variant/precision/common-init contract differs')
    base = checked_path(config['paths']['base'], directory=True)
    output = checked_path(config['paths']['output_root'], directory=True)
    require(output == base/'runs' and str(output) == group['output_root'] and Path(config_path).is_relative_to(base), 'Group paths differ')
    cpdir = output/'checkpoints'/('update-%08d-outer-%05d' % (updates,outer))
    require(Path(checkpoint_path) == cpdir/'manifest.json', 'Pinned committed checkpoint path differs')
    require(cp.get('schema') == 'opsd-online-checkpoint-v1' and cp.get('accepted') is True
            and cp['run_id'] == run and cp['config_sha256'] == config_sha and cp['optimizer_updates'] == updates
            and cp['cursor']['outer_index'] == outer, 'Actual checkpoint identity differs')
    require(cp['runtime_identity'] == config['runtime_manifest']['sha256']
            and cp['source_model_identity'] == config['source_model_manifest']['sha256'], 'Science identities differ')
    require('outer_end' in cp['protection_reason']
            and cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'], 'Not a complete immutable outer end')
    outerdir = output/'outers'/('outer-%05d' % outer)
    completion_path = outerdir/'training/outer-complete.json'
    require(completion_ref and completion_ref['path'] == str(completion_path)
            and SHA.fullmatch(completion_ref['sha256']), 'Explicit pinned completion path/SHA required')
    completion, completion_proof = checked_json(completion_path, completion_ref['sha256'])
    require(completion.get('accepted') is True and completion['schema'] == 'opsd-online-outer-complete-v1'
            and completion['run_id'] == run and completion['outer_index'] == outer
            and completion['optimizer_updates'] == updates
            and completion['checkpoint']['path'] == cpref['path'] and completion['checkpoint']['sha256'] == cpref['sha256']
            and completion['K'] == cp['cursor']['usable_pair_count']
            and completion['training_input'] == cp['training_input'], 'Actual completion does not bind pinned checkpoint')
    require(type(completion['K']) is int and 0 <= completion['K'] <= 512
            and completion['optimizer_updates_delta'] == (completion['K']+31)//32,
            'Actual K/Adam count including valid K0 differs')
    require(cp.get('complete_rank_count') == 8 and cp.get('equal_optimizer_steps') is True
            and len(cp['ranks']) == 8 and sorted(r['rank'] for r in cp['ranks']) == list(range(8)), 'Eight actual rank states required')
    require(cp['optimizer_storage'] == 'shared_exact_eight_rank_optimizer_v1'
            and Path(cp['shared_optimizer']['path']) == cpdir/'optimizer-shared.pt', 'Shared optimizer file omitted/wrong')
    require(len(cp['all_rank_optimizer_tensor_sha256']) == 8 and len(set(cp['all_rank_optimizer_tensor_sha256'])) == 1
            and SHA.fullmatch(cp['all_rank_optimizer_tensor_sha256'][0]), 'Eight declared Adam identities differ')
    require(len(cp['all_rank_scaler_state']) == 8 and all(s == cp['all_rank_scaler_state'][0] for s in cp['all_rank_scaler_state']),
            'Eight scaler declarations differ')
    require(cp['all_rng'] == ['python','numpy','torch_cpu','cuda_all'], 'Full rank RNG declarations required')
    require(Path(cp['adapter']['path']) == cpdir/'adapter' and len(cp['adapter']['files']) == 3
            and {Path(r['path']).name for r in cp['adapter']['files']} == {'README.md','adapter_config.json','adapter_model.safetensors'},
            'Exact three adapter files required')
    require(set(cp['training_input']) == {'outer_manifest','pairs','order','encoding'}, 'Four original inputs required')
    core = [cpref, declared_file(cp['shared_optimizer'])]
    for r in cp['ranks']:
        require(Path(r['file']['path']) == cpdir/('rank-%d-state.pt' % r['rank']), 'Rank path differs')
        core.append(declared_file(r['file']))
    for r in cp['adapter']['files']:
        require(Path(r['path']).parent == cpdir/'adapter', 'Adapter escapes exact checkpoint')
        core.append(declared_file(r))
    actual_cp = set()
    for directory, dirs, files in os.walk(cpdir, followlinks=False):
        for name in dirs: checked_path(Path(directory)/name,directory=True)
        for name in files: actual_cp.add(str(checked_path(Path(directory)/name)))
    require(actual_cp == {r['path'] for r in core}, 'Checkpoint has extra/missing/special files')
    for r in cp['training_input'].values():
        within(r['path'],outerdir); core.append(declared_file(r))
    om,_ = checked_json(cp['training_input']['outer_manifest']['path'], cp['training_input']['outer_manifest']['sha256'])
    require(om['run_id'] == run and om['config_sha256'] == config_sha and om['outer_index'] == outer
            and om['pairs'] == cp['training_input']['pairs'], 'Original outer input differs')
    ri,rip = checked_json(output/'run-identity.json',fileproof(output/'run-identity.json')['sha256'])
    require(ri['schema'] == 'opsd-online-run-identity-v1' and ri['run_id'] == run
            and ri['config_sha256'] == config_sha and ri['config']['path'] == str(config_path)
            and ri['config']['sha256'] == config_sha, 'Run identity differs')
    core += [cref,rip]
    require(len(core) == 19 and len({r['path'] for r in core}) == 19, 'Exact nineteen core artifacts required')
    files = [dict(relative=safe_name('group/'+str(Path(r['path']).relative_to(base))),**r) for r in core]
    identities = {}; closure = []
    def add(ref, name):
        fp = closure_file(ref)
        closure.append(dict(relative=safe_name(name),**fp)); return fp
    for name in ('runtime_manifest','source_model_manifest','helper_source_manifest'):
        _,identities[name] = checked_json(config[name]['path'],config[name]['sha256'])
        add(identities[name], 'closure/science/'+name+'.json')
    initial,ip = checked_json(config['common_initial_adapter_manifest']['path'],config['common_initial_adapter_manifest']['sha256'])
    add(ip,'closure/common/manifest.json')
    require(initial['adapter_tensor_sha256'] == config['common_initial_adapter_tensor_sha256'], 'Actual common identity differs')
    for n,r in enumerate(initial['payload_files']): add(r,'closure/common/payload-%03d/'%n+Path(r['path']).name)
    old,op = checked_json(cp['outer_old_adapter']['path'],cp['outer_old_adapter']['sha256'])
    add(op,'closure/outer-old/manifest.json')
    old_files = old['adapter']['files'] if 'adapter' in old else old['payload_files']
    if 'adapter' in old:
        require(old['run_id'] == run and old['config_sha256'] == config_sha
                and old['cursor']['outer_index'] == outer-1, 'Outer-old must be exact prior group checkpoint')
    else: require(cp['outer_old_adapter']['sha256'] == ip['sha256'] and outer == 0, 'Initial outer-old differs')
    for n,r in enumerate(old_files): add(r,'closure/outer-old/payload-%03d/'%n+Path(r['path']).name)
    add(completion_proof,'closure/outer-complete.json'); add(goal['file'],'closure/goal.json')
    # The same common file may serve outer-old at outer0: deduplicate by exact source path.
    seen = {r['path']:r for r in files}; names = {r['relative'] for r in files}
    closure_paths = []
    for r in closure:
        if r['path'] in seen:
            require(all(r[k] == seen[r['path']][k] for k in (*META,'sha256')), 'Conflicting shared closure file proof')
        else:
            require(r['relative'] not in names,'Closure archive name aliases');files.append(r);seen[r['path']]=r;names.add(r['relative'])
        closure_paths.append(dict(path=r['path'],relative=seen[r['path']]['relative'],sha256=r['sha256']))
    require(len({(r['device'],r['inode']) for r in files}) == len(files), 'Source inode aliases refused')
    for r in files: stable(r)
    return dict(schema='opsd-ablation-outer-backup-inventory-v1',accepted=True,node='hkust1',run_id=run,
        variant=variant,optimizer_updates=updates,outer_index=outer,config=cref,checkpoint=cpref,run_identity=rip,
        base=str(base),goal_identity=goal,source_publication=publication,scientific_identity_manifests=identities,
        shared_optimizer=cp['shared_optimizer'],core_file_count=19,closure_files=closure_paths,files=sorted(files,key=lambda r:r['relative']),
        outer_complete=completion_proof,
        expected_file_count=len(files),total_bytes=sum(r['bytes'] for r in files),checked_unix=time.time(),
        empty_outer_no_new_adam=completion['K']==0,
        scope='File transport identity, not tensor/Adam numerical equality or training/effect acceptance',
        external_assets_not_transferred=dict(source100_model=config['paths'].get('model'),runtime=config['paths'].get('runtime')))


def validate_inventory(value, config_path, config_sha, checkpoint_path, checkpoint_sha, run, updates, outer, goal_ref, completion_ref):
    target_scope(run,updates,outer,checkpoint_sha,value['goal_identity']['value'])
    require(value['goal_identity']['file']['path'] == goal_ref['path'] and value['goal_identity']['file']['sha256'] == goal_ref['sha256'], 'Goal differs')
    require(value.get('schema') == 'opsd-ablation-outer-backup-inventory-v1' and value.get('accepted') is True
            and value['node'] == 'hkust1' and value['run_id'] == run and value['optimizer_updates'] == updates
            and value['outer_index'] == outer and value['core_file_count'] == 19, 'Pinned target differs')
    require(value['config']['path'] == config_path and value['config']['sha256'] == config_sha
            and value['checkpoint']['path'] == checkpoint_path and value['checkpoint']['sha256'] == checkpoint_sha,'Pinned file identity differs')
    files = value['files']; count = value['expected_file_count']
    require(count == len(files) and len({r['relative'] for r in files}) == count
            and len({r['path'] for r in files}) == count and len({(r['device'],r['inode']) for r in files}) == count,'Inventory aliases')
    for r in files:
        require(safe_name(r['relative']).startswith(('group/','closure/')) and Path(r['path']).is_absolute()
                and SHA.fullmatch(r['sha256']) and type(r['bytes']) is int and r['bytes'] >= 0
                and all(k in r for k in META),'Unsafe inventory file')
        if r['relative'].startswith('group/'):
            require(str(Path(value['base'])/r['relative'][6:]) == r['path'],'Core archive path differs')
    require(sum(r['bytes'] for r in files) == value['total_bytes'],'Inventory byte count differs')
    mapped = {r['path']:r for r in files}
    for ref in [value['config'],value['checkpoint'],value['run_identity'],value['shared_optimizer']]:
        require(ref['path'] in mapped and all(ref[k] == mapped[ref['path']][k] for k in (*META,'sha256') if k in ref),'Required core identity omitted')
    for r in value['closure_files']:
        require(r['path'] in mapped and r['relative'] == mapped[r['path']]['relative'] and r['sha256'] == mapped[r['path']]['sha256'],'Closure omitted')
    complete = value['outer_complete']
    require(complete['path'] == completion_ref['path'] and complete['sha256'] == completion_ref['sha256']
            and complete['path'] in mapped
            and mapped[complete['path']]['relative'] == 'closure/outer-complete.json'
            and all(complete[k] == mapped[complete['path']][k] for k in (*META,'sha256')),
            'Saved closure completion differs from exact queued path/SHA')
    core = {r['path'] for r in files if r['relative'].startswith('group/')}
    require(len(core) == 19 and core | {r['path'] for r in value['closure_files']} == set(mapped),
            'Extra undeclared payload or omitted core/closure member')
    return {r['relative']:r for r in files}


def stream_archive(value, output):
    """Fixed handles + hash while streaming; any mismatch exits nonzero remotely."""
    for row in value['files']:
        stable(row)
    with tarfile.open(fileobj=output, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
        for row in value['files']:
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            with os.fdopen(os.open(row['path'], flags), 'rb') as handle:
                require(meta_from_stat(row['path'], os.fstat(handle.fileno())) == {k: row[k] for k in META},
                        'Archive source fixed-handle identity differs')
                h = hashlib.sha256()

                class HashedReader:
                    def read(self, size=-1):
                        data = handle.read(size)
                        h.update(data)
                        return data

                item = tarfile.TarInfo(safe_name(row['relative']))
                item.size = row['bytes']; item.mode = 0o600; item.mtime = 0
                tar.addfile(item, HashedReader())
                require(h.hexdigest() == row['sha256']
                        and meta_from_stat(row['path'], os.fstat(handle.fileno())) == {k: row[k] for k in META},
                        'Archive source content/stat changed')
            stable(row)
    for row in value['files']:
        stable(row)


def required_space(value, margin):
    require(margin >= 5 << 30, 'At least five GiB space margin required')
    archive_upper = sum(512 + ((x['bytes'] + 511) // 512) * 512 for x in value['files']) + 10240
    return archive_upper + value['total_bytes'] + margin


def extract_verified(archive, extraction, expected):
    require(not Path(extraction).exists(), 'Existing extraction never reused')
    extraction = Path(extraction); extraction.mkdir(mode=0o700)
    checked_path(extraction, directory=True)
    archive_before = fileproof(archive)
    observed = set(); end = 0
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            name = safe_name(member.name)
            require(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    and not member.pax_headers and name in expected and name not in observed,
                    'Special/duplicate/unexpected archive member refused')
            # tarfile hides GNU/PAX pseudo-members while interpreting them.
            # Our stream is USTAR: each real header must be exactly one block,
            # at the preceding data end; no hidden extension/header is allowed.
            require(member.offset == end and member.offset_data == end + 512,
                    'Hidden extension/special archive header refused')
            proof = expected[name]
            require(member.size == proof['bytes'], 'Archive declared size differs')
            target = extraction / name
            target.parent.mkdir(parents=True, exist_ok=True)
            checked_path(target.parent, directory=True)
            h = hashlib.sha256(); copied = 0
            with tar.extractfile(member) as src, target.open('xb') as out:
                for block in iter(lambda: src.read(4 << 20), b''):
                    h.update(block); copied += len(block); out.write(block)
                out.flush(); os.fsync(out.fileno())
            require(copied == proof['bytes'] and h.hexdigest() == proof['sha256'], 'Extracted stream SHA/bytes differs')
            observed.add(name)
            end = member.offset_data + ((member.size + 511) // 512) * 512
    require(observed == set(expected), 'Archive missing expected members')
    # Reject nonzero data/second archive after the first end marker as well.
    require(archive_before['bytes'] % 512 == 0 and archive_before['bytes'] >= end + 1024,
            'Truncated archive end marker')
    with Path(archive).open('rb') as handle:
        handle.seek(end)
        for block in iter(lambda: handle.read(4 << 20), b''):
            require(not any(block), 'Trailing nonzero/second archive refused')
    stable(archive_before)
    require(fileproof(archive)['sha256'] == archive_before['sha256'], 'Archive changed during extraction')
    return readback(extraction, expected), archive_before


def readback(extraction, expected):
    """Independent second full local disk read, not the extraction-stream digest."""
    extraction = checked_path(extraction, directory=True)
    actual = set()
    for directory, dirs, files in os.walk(extraction, followlinks=False):
        for name in dirs:
            checked_path(Path(directory) / name, directory=True)
        for name in files:
            p = checked_path(Path(directory) / name)
            actual.add(str(p.relative_to(extraction)))
    require(actual == set(expected), 'Local extracted inventory differs')
    proofs = {}
    for name, row in expected.items():
        fp = fileproof(extraction / name, dict(path=str(extraction / name), sha256=row['sha256'], bytes=row['bytes']))
        proofs[name] = fp
    for fp in proofs.values():
        stable(fp)
    for directory, _, _ in os.walk(extraction, topdown=False):
        fsync_dir(directory)
    return proofs


def remote_command(args, mode):
    require(args.host == 'hkust1', 'Only the authorized HKUST1 SSH alias is allowed')
    root = Path(args.remote_helper_root)
    require(root.is_absolute() and '..' not in root.parts, 'Unsafe remote helper root')
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'E1_RUN_ID=backup-' + args.goal_identity_sha256[:16], 'python3', '-B',
               str(root / 'backup.py'), mode, '--config', args.config, '--config-sha256', args.config_sha256,
               '--checkpoint', args.checkpoint, '--checkpoint-sha256', args.checkpoint_sha256,
               '--expected-run', args.expected_run, '--expected-updates', str(args.expected_updates),
               '--expected-outer', str(args.expected_outer), '--goal-identity', args.remote_goal_identity,
               '--expected-completion', args.expected_completion, '--expected-completion-sha256', args.expected_completion_sha256,
               '--goal-identity-sha256', args.goal_identity_sha256, '--source-manifest', str(root / 'source-manifest.json'),
               '--source-manifest-sha256', args.source_manifest_sha256, '--publication', args.remote_publication,
               '--publication-sha256', args.publication_sha256, '--source-commit', args.source_commit]
    return ['ssh', '-oConnectTimeout=15', args.host, shlex.join(command)]


def backup(args, pub):
    destination = Path(args.destination).absolute()
    checked_path(destination.parent, directory=True)
    require(not destination.exists() and not destination.is_symlink(), 'Destination must be new and independent')
    destination.mkdir(mode=0o700); fsync_dir(destination.parent)
    archive = destination / 'checkpoint.tar.partial'
    try:
        result = subprocess.run(remote_command(args, 'inventory'), capture_output=True, check=False)
        write_json(destination / 'inventory-execution.json', dict(actual_returncode=result.returncode,
                   stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
        require(result.returncode == 0, 'Remote inventory failed; no archive requested')
        value = json.loads(result.stdout)
        expected = validate_inventory(value, args.config, args.config_sha256, args.checkpoint,
                                      args.checkpoint_sha256, args.expected_run, args.expected_updates, args.expected_outer,
                                      dict(path=args.remote_goal_identity, sha256=args.goal_identity_sha256),
                                      dict(path=args.expected_completion, sha256=args.expected_completion_sha256))
        local_goal,_ = checked_json(args.goal_identity, args.goal_identity_sha256)
        require(value['goal_identity']['value'] == local_goal, 'Remote/local goal content differs')
        require(value['source_publication']['manifest']['sha256'] == pub['manifest']['sha256']
                and value['source_publication']['source_commit'] == pub['source_commit']
                and value['source_publication']['publication']['sha256'] == pub['publication']['sha256'],
                'Remote helper publication differs from Mac audited helper')
        ip = write_json(destination / 'node-inventory.json', value)
        need = required_space(value, args.margin_gib << 30)
        require(shutil.disk_usage(destination).free >= need, 'Insufficient local archive/extraction/margin space')
        write_json(destination / 'transfer-start.json', dict(accepted=False,
                   run_id=args.expected_run, started_unix=time.time(), required_free_bytes=need,
                   source_publication=pub, node_inventory=ip, no_node_writes=True))
        with archive.open('xb') as out, (destination / 'transfer.stderr').open('xb') as err:
            process = subprocess.Popen(remote_command(args, 'stream'), stdin=subprocess.PIPE, stdout=out, stderr=err)
            try:
                write_json(destination / 'transfer-child.json', dict(pid=process.pid, started_unix=time.time(), accepted=False))
                process.communicate(canonical(value))
            finally:
                # Closing stdin causes an incomplete remote data request to fail;
                # no signal is sent. Collect the real SSH rc even on local I/O failure.
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                process.wait()
                # Persist the real wait result even if communication/logging raised.
                write_json(destination / 'transfer-wait.json', dict(actual_returncode=process.returncode,
                           ended_unix=time.time(), source_job_status_not_inferred=True))
            out.flush(); os.fsync(out.fileno()); err.flush(); os.fsync(err.fileno())
            rc = process.returncode
        write_json(destination / 'transfer-exit.json', dict(actual_returncode=rc, ended_unix=time.time(),
                   archive=fileproof(archive), stderr=fileproof(destination / 'transfer.stderr')))
        require(rc == 0, 'Transfer failed; partial and node originals retained')
        local, ap = extract_verified(archive, destination / 'verified-files', expected)
        final = destination / 'checkpoint.tar'
        os.rename(archive, final); fsync_dir(destination)
        finalproof = fileproof(final, dict(path=str(final), sha256=ap['sha256'], bytes=ap['bytes']))
        proof = dict(schema='opsd-ablation-outer-backup-complete-v1', accepted=True,
                     scope='Transfer identity only; no numeric/Adam/effect/seven-group acceptance',
                     run_id=args.expected_run, optimizer_updates=args.expected_updates, outer_index=args.expected_outer,
                     checkpoint=value['checkpoint'], config=value['config'], node_inventory=ip,
                     outer_complete=value['outer_complete'],
                     source_publication=pub, goal_identity=value['goal_identity'], archive=finalproof, files=local, file_count=len(local),
                     actual_transfer_returncode=rc, independent_local_disk_readback=True,
                     node_outputs_retained=True, node_files_changed=False, completed_unix=time.time())
        write_json(destination / 'complete-sha-verified.json', proof)
        return dict(accepted=True, destination=str(destination), file_count=len(local), scope=proof['scope'])
    except Exception as error:
        failure = dict(schema='opsd-ablation-outer-backup-failed-v1', accepted=False,
                       reason=type(error).__name__ + ': ' + str(error), run_id=args.expected_run,
                       checkpoint_sha256=args.checkpoint_sha256, failed_unix=time.time(),
                       destination=str(destination), partial_retained=archive.exists(), node_outputs_retained=True)
        write_json(destination / 'failed.json', failure)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inventory', 'stream', 'backup'))
    for name in ('config', 'config-sha256', 'checkpoint', 'checkpoint-sha256', 'expected-run',
                 'expected-completion', 'expected-completion-sha256',
                 'goal-identity', 'goal-identity-sha256', 'source-manifest', 'source-manifest-sha256', 'publication', 'publication-sha256', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-updates', type=int, required=True)
    parser.add_argument('--expected-outer', type=int, required=True)
    parser.add_argument('--host', default='hkust1')
    parser.add_argument('--remote-helper-root')
    parser.add_argument('--remote-publication')
    parser.add_argument('--remote-goal-identity')
    parser.add_argument('--destination')
    parser.add_argument('--margin-gib', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    require(args.host == 'hkust1', 'Only HKUST1 is in scope')
    if args.mode != 'backup':
        require(os.getuid() == os.geteuid() == 1005, 'Remote entry requires actual HKUST1 owner UID1005')
        require(Path(args.config).is_absolute() and Path(args.checkpoint).is_absolute(),
                'Remote config/checkpoint must be explicit absolute paths')
    pub = source_publication(args.source_manifest, args.source_manifest_sha256, args.publication,
                             args.publication_sha256, args.source_commit, args.expected_run)
    bind_goal(dict(path=args.goal_identity,sha256=args.goal_identity_sha256),
              dict(path=args.config,sha256=args.config_sha256),
              dict(path=args.checkpoint,sha256=args.checkpoint_sha256),
              args.expected_run,args.expected_updates,args.expected_outer, verify_evidence=args.mode != 'backup')
    if args.mode == 'backup':
        require(args.destination and args.remote_helper_root and args.remote_publication and args.remote_goal_identity, 'Missing explicit Mac/remote paths')
        result = backup(args, pub)
    elif args.mode == 'inventory':
        result = inventory(args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256,
                           args.expected_run, args.expected_updates, args.expected_outer, pub,
                           dict(path=args.goal_identity,sha256=args.goal_identity_sha256),
                           dict(path=args.expected_completion,sha256=args.expected_completion_sha256))
        source_publication(args.source_manifest, args.source_manifest_sha256, args.publication,
                           args.publication_sha256, args.source_commit, args.expected_run)
    else:
        old = json.loads(sys.stdin.buffer.read())
        validate_inventory(old, args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256,
                           args.expected_run, args.expected_updates, args.expected_outer,
                                      dict(path=args.goal_identity, sha256=args.goal_identity_sha256),
                                      dict(path=args.expected_completion, sha256=args.expected_completion_sha256))
        require(old['source_publication']['manifest']['sha256'] == pub['manifest']['sha256']
                and old['source_publication']['publication']['sha256'] == pub['publication']['sha256']
                and old['source_publication']['source_commit'] == pub['source_commit'], 'Inventory source publication changed')
        fresh = inventory(args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256,
                          args.expected_run, args.expected_updates, args.expected_outer, pub,
                           dict(path=args.goal_identity,sha256=args.goal_identity_sha256),
                           dict(path=args.expected_completion,sha256=args.expected_completion_sha256))
        require(fresh['files'] == old['files'] and fresh['config'] == old['config']
                and fresh['checkpoint'] == old['checkpoint'] and fresh['run_identity'] == old['run_identity']
                and fresh['scientific_identity_manifests'] == old['scientific_identity_manifests']
                and fresh['goal_identity'] == old['goal_identity'],
                'Source inventory changed between inventory and transfer')
        stream_archive(fresh, sys.stdout.buffer)
        source_publication(args.source_manifest, args.source_manifest_sha256, args.publication,
                           args.publication_sha256, args.source_commit, args.expected_run)
        return
    print(canonical(result).decode())


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # stream stdout is reserved exclusively for tar. Details go to stderr.
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        raise SystemExit(1)
