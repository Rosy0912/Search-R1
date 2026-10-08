"""Explicit old CP374 or seven-ablation final checkpoint: read-only verified backup.

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
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
CP89_SHA = '325c9f41f5a44d5e16a32578c5fa03dfaecec92bbdbd87a4f5e289c81c9bf75f'
NODE_BASE = '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008'
REPO_PREFIX = 'experiments/search_e1/control/opsd_ablation_backup_v1/'
OLD_CP374_SHA = '14278230c81774f271679a5345828213b2d57929887341ffcca9e41e2af28b40'
OLD_TERMINAL_SHA = '8fd4eb76e9fbbd1bddf69278fccd512a2dc38415322683c90d40bfbdbf37eba9'
GOAL_CREATED_UNIX = 1791472699
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


def source_publication(manifest_path, manifest_sha, publication_path, publication_sha, commit, run=RUN):
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
            and pub.get('branch') == 'search-e1' and pub.get('run_id') == run
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


def target_scope(run, updates, outer, checkpoint_sha, goal):
    require(type(updates) is int and updates >= 0 and type(outer) is int and outer >= 0,
            'Explicit nonnegative checkpoint counters required')
    require(SHA.fullmatch(checkpoint_sha or ''), 'Missing exact checkpoint manifest SHA')
    require(goal and goal.get('schema') == 'opsd-ablation-backup-goal-v1'
            and goal.get('accepted') is True and goal.get('goal_created_unix') == GOAL_CREATED_UNIX,
            'Explicit current human goal identity required')
    require(goal['run_id'] == run and goal['optimizer_updates'] == updates
            and goal['outer_index'] == outer and goal['checkpoint']['sha256'] == checkpoint_sha,
            'Goal-scoped target differs')
    scope = goal['goal_scope']
    if scope == 'old-online374':
        require(run == RUN and updates == 374 and outer == 37 and checkpoint_sha == OLD_CP374_SHA,
                'Only exact preserved old CP374 is authorized')
        require(goal['source_job_exit']['classification'] == 'authorized_signal_stop'
                and goal['source_job_exit']['actual_tmux_status'] == 143
                and goal['source_job_exit']['receipt']['sha256'] == OLD_TERMINAL_SHA,
                'Old current controller is an authorized signal stop, not UNKNOWN/rc0')
        require(goal['historical_wait_exit_status'] == 'UNKNOWN',
                'Historical first wait must remain independently UNKNOWN')
    else:
        require(scope == 'ablation-final10' and goal['variant'] in VARIANTS and outer == 9,
                'Only seven explicitly selected final10 variants are authorized')
        require(goal['source_job_exit']['classification'] == 'NOT_AUDITED_BY_BACKUP',
                'Backup must not claim training terminal success')
    return scope


def bind_goal(ref, config_ref, cp_ref, run, updates, outer, verify_evidence=True):
    goal, proof = checked_json(ref['path'], ref['sha256'])
    target_scope(run, updates, outer, cp_ref['sha256'], goal)
    require(goal['config'] == dict(path=config_ref['path'], sha256=config_ref['sha256'])
            and goal['checkpoint'] == dict(path=cp_ref['path'], sha256=cp_ref['sha256']),
            'Goal config/checkpoint exact references differ')
    evidence = {}
    if verify_evidence and goal['goal_scope'] == 'old-online374':
        terminal = goal['source_job_exit']['receipt']
        value, evidence['source_stop'] = checked_json(terminal['path'], terminal['sha256'])
        require(value.get('accepted') is True and value['actual_tmux_status'] == 143
                and value['config_sha256'] == config_ref['sha256'], 'Actual authorized stop evidence differs')
    return dict(value=goal, file=proof, evidence=evidence)


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


def inventory(config_path, config_sha, checkpoint_path, checkpoint_sha, run, updates, outer,
              publication=None, goal_ref=None):
    require(goal_ref is not None, 'Missing explicit goal receipt path/SHA')
    config, cref = checked_json(config_path, config_sha)
    cp, cpref = checked_json(checkpoint_path, checkpoint_sha)
    goal = bind_goal(goal_ref, cref, cpref, run, updates, outer)
    scope = goal['value']['goal_scope']
    if scope == 'ablation-final10':
        variant = goal['value']['variant']
        expected_contract = dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
            variant=variant, precision=config['precision']['mode'], mixture_alpha=.1 if variant in ('M','J') else 0.)
        require(config['variant'] == variant and config['scientific']['outer_iterations'] == 10
                and cp['contract'] == expected_contract
                and cp['common_initial_adapter'] == config['common_initial_adapter_manifest'],
                'New variant/10outer/reverse/common-init checkpoint contract differs')
    base = checked_path(config['paths']['base'], directory=True)
    output = checked_path(config['paths']['output_root'], directory=True)
    if scope == 'old-online374':
        require(base == Path(NODE_BASE), 'Old target base differs')
    if scope == 'ablation-final10':
        final = goal['value']['final_completion']
        completion, goal['evidence']['final_completion'] = checked_json(final['path'], final['sha256'])
        require(completion.get('accepted') is True and completion['schema'] == 'opsd-online-outer-complete-v1'
                and completion['run_id'] == run and completion['outer_index'] == 9
                and completion['optimizer_updates'] == updates
                and completion['checkpoint']['path'] == cpref['path']
                and completion['checkpoint']['sha256'] == cpref['sha256'],
                'Actual final completion does not bind this checkpoint (K0 prior-outer HOLD)')
    require(output == base / 'runs' and Path(config_path).is_relative_to(base), 'Wrong base/output/config paths')
    cpdir = output / 'checkpoints' / ('update-%08d-outer-%05d' % (updates, outer))
    require(Path(checkpoint_path) == cpdir / 'manifest.json', 'Pinned manifest path/count differs')
    require(config['run_id'] == run and cp.get('schema') == 'opsd-online-checkpoint-v1'
            and cp.get('accepted') is True and cp['run_id'] == run
            and cp['config_sha256'] == config_sha and cp['optimizer_updates'] == updates
            and cp['cursor']['outer_index'] == outer, 'Checkpoint/config/run identity mismatch')
    require(cp['runtime_identity'] == config['runtime_manifest']['sha256']
            and cp['source_model_identity'] == config['source_model_manifest']['sha256'],
            'Checkpoint scientific identity differs from config')
    identities = {}
    for name in ('runtime_manifest', 'source_model_manifest', 'helper_source_manifest'):
        ref = config[name]
        _, identities[name] = checked_json(ref['path'], ref['sha256'])
    require('outer_end' in cp['protection_reason'] and cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'],
            'Target is not the retained immutable outer-end checkpoint')
    require(cp.get('complete_rank_count') == 8 and cp.get('equal_optimizer_steps') is True
            and sorted(x['rank'] for x in cp['ranks']) == list(range(8)), 'Eight declared rank files required')
    require(Path(cp['adapter']['path']) == cpdir / 'adapter'
            and {Path(x['path']).name for x in cp['adapter']['files']} ==
            {'README.md', 'adapter_config.json', 'adapter_model.safetensors'}
            and len(cp['adapter']['files']) == 3, 'Exact three-file PEFT adapter inventory required')
    require(set(cp['training_input']) == {'outer_manifest', 'pairs', 'order', 'encoding'},
            'Four original training inputs required')
    outerdir = output / 'outers' / ('outer-%05d' % outer)
    declared = [cpref]
    for row in cp['ranks']:
        ref = row['file']
        require(Path(ref['path']) == cpdir / ('rank-%d-state.pt' % row['rank']), 'Rank file path differs')
        within(ref['path'], cpdir)
        declared.append(declared_file(ref))
    for ref in cp['adapter']['files']:
        require(Path(ref['path']).parent == cpdir / 'adapter', 'Adapter file escapes exact directory')
        within(ref['path'], cpdir)
        declared.append(declared_file(ref))
    shared_optimizer = cp.get('shared_optimizer')
    if shared_optimizer is not None:
        require(scope == 'ablation-final10', 'Old CP374 must retain original inline Adam format')
        require(isinstance(shared_optimizer, dict) and Path(shared_optimizer['path']) == cpdir/'optimizer-shared.pt'
                and cp.get('optimizer_storage') == 'shared_exact_eight_rank_optimizer_v1', 'Shared optimizer must be one explicitly referenced checkpoint PT')
        digests = cp.get('all_rank_optimizer_tensor_sha256', [])
        require(len(digests) == 8 and all(SHA.fullmatch(x or '') for x in digests)
                and len(set(digests)) == 1, 'Shared Adam eight declared exact digests differ')
        within(shared_optimizer['path'], cpdir)
        declared.append(declared_file(shared_optimizer))
    expected_file_count = 19 if shared_optimizer is not None else 18
    actual_cp = set()
    for directory, dirs, files in os.walk(cpdir, followlinks=False):
        for name in dirs:
            checked_path(Path(directory) / name, directory=True)
        for name in files:
            actual_cp.add(str(checked_path(Path(directory) / name)))
    require(actual_cp == {x['path'] for x in declared}, 'Checkpoint has extra/missing/special files')
    for ref in cp['training_input'].values():
        within(ref['path'], outerdir)
        declared.append(declared_file(ref))
    require(Path(cp['training_input']['outer_manifest']['path']) == outerdir / 'outer-manifest.json',
            'Wrong original outer manifest path')
    om, _ = checked_json(cp['training_input']['outer_manifest']['path'], cp['training_input']['outer_manifest']['sha256'])
    require(om['run_id'] == run and om['config_sha256'] == config_sha and om['outer_index'] == outer
            and om['runtime_identity'] == cp['runtime_identity'] and om['source_model_identity'] == cp['source_model_identity']
            and om['pairs'] == cp['training_input']['pairs'], 'Original training input identity mismatch')
    ri, rip = checked_json(output / 'run-identity.json', fileproof(output / 'run-identity.json')['sha256'])
    require(ri.get('schema') == 'opsd-online-run-identity-v1' and ri['run_id'] == run
            and ri['config_sha256'] == config_sha and ri['config']['path'] == str(config_path)
            and ri['config']['sha256'] == config_sha, 'Original run identity differs')
    fileproof(config_path, ri['config'])
    declared += [cref, rip]
    names = [safe_name(str(Path(x['path']).relative_to(base))) for x in declared]
    require(len(names) == len(set(names)) == expected_file_count, 'Expected exact backup members including optional shared optimizer')
    require(len({(x['device'], x['inode']) for x in declared}) == expected_file_count, 'Aliased/hardlinked source files refused')
    for fp in declared:
        stable(fp)
    for fp in identities.values():
        stable(fp)
    stable(goal['file'])
    for fp in goal['evidence'].values(): stable(fp)
    return dict(schema='opsd-ablation-backup-inventory-v1', accepted=True,
                scope='Transfer identity only; no numerical/Adam/effect acceptance', node='hkust1',
                run_id=run, optimizer_updates=updates, outer_index=outer, base=str(base),
                config=cref, checkpoint=cpref, run_identity=rip, source_publication=publication, goal_identity=goal,
                scientific_identity_manifests=identities, expected_file_count=expected_file_count,
                shared_optimizer=shared_optimizer,
                scientific_scope='Fresh three identity-manifest JSON SHAs; no model/runtime tree or tensor audit',
                files=[dict(relative=name, **fp) for name, fp in sorted(zip(names, declared))],
                total_bytes=sum(x['bytes'] for x in declared), checked_unix=time.time())


def validate_inventory(value, config_path, config_sha, checkpoint_path, checkpoint_sha, run, updates, outer, goal_ref):
    goal = value['goal_identity']['value']
    target_scope(run, updates, outer, checkpoint_sha, goal)
    require(value['goal_identity']['file']['path'] == goal_ref['path']
            and value['goal_identity']['file']['sha256'] == goal_ref['sha256'], 'Remote goal binding differs')
    require(value.get('schema') == 'opsd-ablation-backup-inventory-v1' and value.get('accepted') is True
            and value['node'] == 'hkust1' and value['run_id'] == run
            and value['optimizer_updates'] == updates and value['outer_index'] == outer,
            'Remote inventory target identity mismatch')
    require(value['config']['path'] == str(config_path) and value['config']['sha256'] == config_sha
            and value['checkpoint']['path'] == str(checkpoint_path) and value['checkpoint']['sha256'] == checkpoint_sha,
            'Remote inventory exact config/checkpoint differs')
    files = value['files']
    shared = value.get('shared_optimizer')
    count = 19 if shared is not None else 18
    require(shared is None or goal['goal_scope'] == 'ablation-final10', 'Old inline Adam inventory cannot add shared payload')
    require(value['expected_file_count'] == count and len(files) == count
            and len({x['relative'] for x in files}) == count, 'Wrong inventory member count')
    for row in files:
        name = safe_name(row['relative'])
        require(str(Path(value['base']) / name) == row['path'] and SHA.fullmatch(row.get('sha256', ''))
                and isinstance(row['bytes'], int) and row['bytes'] >= 0
                and all(key in row for key in META), 'Unsafe inventory member')
    require(len({(x['device'], x['inode']) for x in files}) == count, 'Remote inventory aliases source files')
    mapped = {x['path']: x for x in files}
    if shared is not None:
        require(shared['path'] in mapped and all(mapped[shared['path']][field] == shared[field]
                for field in ('sha256','bytes','device','inode','mtime_ns','ctime_ns')),
                'Shared optimizer inventory omitted/differs')
    for key in ('config', 'checkpoint', 'run_identity'):
        require(value[key]['path'] in mapped
                and all(mapped[value[key]['path']][field] == value[key][field] for field in (*META, 'sha256')),
                'Inventory primary identity omitted/differs')
    require(value['total_bytes'] == sum(x['bytes'] for x in files), 'Inventory byte count differs')
    return {x['relative']: x for x in files}


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
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'python3', '-B',
               str(root / 'backup.py'), mode, '--config', args.config, '--config-sha256', args.config_sha256,
               '--checkpoint', args.checkpoint, '--checkpoint-sha256', args.checkpoint_sha256,
               '--expected-run', args.expected_run, '--expected-updates', str(args.expected_updates),
               '--expected-outer', str(args.expected_outer), '--goal-identity', args.remote_goal_identity,
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
                                      dict(path=args.remote_goal_identity, sha256=args.goal_identity_sha256))
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
        proof = dict(schema='opsd-ablation-backup-complete-v1', accepted=True,
                     scope='Transfer identity only; no numeric/Adam/effect/final200 acceptance',
                     run_id=args.expected_run, optimizer_updates=args.expected_updates, outer_index=args.expected_outer,
                     checkpoint=value['checkpoint'], config=value['config'], node_inventory=ip,
                     source_publication=pub, goal_identity=value['goal_identity'], archive=finalproof, files=local, file_count=len(local),
                     actual_transfer_returncode=rc, independent_local_disk_readback=True,
                     node_outputs_retained=True, node_files_changed=False, completed_unix=time.time())
        write_json(destination / 'complete-sha-verified.json', proof)
        return dict(accepted=True, destination=str(destination), file_count=len(local), scope=proof['scope'])
    except Exception as error:
        failure = dict(schema='opsd-ablation-backup-failed-v1', accepted=False,
                       reason=type(error).__name__ + ': ' + str(error), run_id=args.expected_run,
                       checkpoint_sha256=args.checkpoint_sha256, failed_unix=time.time(),
                       destination=str(destination), partial_retained=archive.exists(), node_outputs_retained=True)
        write_json(destination / 'failed.json', failure)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inventory', 'stream', 'backup'))
    for name in ('config', 'config-sha256', 'checkpoint', 'checkpoint-sha256', 'expected-run',
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
                           dict(path=args.goal_identity,sha256=args.goal_identity_sha256))
        source_publication(args.source_manifest, args.source_manifest_sha256, args.publication,
                           args.publication_sha256, args.source_commit, args.expected_run)
    else:
        old = json.loads(sys.stdin.buffer.read())
        validate_inventory(old, args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256,
                           args.expected_run, args.expected_updates, args.expected_outer,
                                      dict(path=args.goal_identity, sha256=args.goal_identity_sha256))
        require(old['source_publication']['manifest']['sha256'] == pub['manifest']['sha256']
                and old['source_publication']['publication']['sha256'] == pub['publication']['sha256']
                and old['source_publication']['source_commit'] == pub['source_commit'], 'Inventory source publication changed')
        fresh = inventory(args.config, args.config_sha256, args.checkpoint, args.checkpoint_sha256,
                          args.expected_run, args.expected_updates, args.expected_outer, pub,
                           dict(path=args.goal_identity,sha256=args.goal_identity_sha256))
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
