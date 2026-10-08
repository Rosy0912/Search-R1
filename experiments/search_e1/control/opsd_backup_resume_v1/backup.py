"""Exact CP89 byte-prefix USTAR resume, read-only node, fresh private Mac copy.

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
REPO_PREFIX = 'experiments/search_e1/control/opsd_backup_resume_v1/'
ORIGINAL_PARTIAL_BYTES = 1741258752
ORIGINAL_PARTIAL_SHA = '44859dd3b9e4bfbe61dccfbaa3986e63f0489cdd1bc181b768a5ef1e9f9dd58f'
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


def source_publication(manifest_path, manifest_sha, publication_path, publication_sha, commit):
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
            and pub.get('branch') == 'search-e1' and pub.get('run_id') == RUN
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


def target_scope(run, updates, outer, checkpoint_sha):
    require(run == RUN, 'Wrong authorized online run')
    require(isinstance(updates, int) and not isinstance(updates, bool) and updates > 0,
            'Expected real optimizer update count required')
    require(outer == 8 and updates == 89 and checkpoint_sha == CP89_SHA,
            'Only exact original CP89 interrupted backup is in scope')
    require(SHA.fullmatch(checkpoint_sha or ''), 'Missing exact checkpoint manifest SHA')


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
              publication=None):
    target_scope(run, updates, outer, checkpoint_sha)
    config, cref = checked_json(config_path, config_sha)
    cp, cpref = checked_json(checkpoint_path, checkpoint_sha)
    base = checked_path(config['paths']['base'], directory=True)
    output = checked_path(config['paths']['output_root'], directory=True)
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
    require(len(names) == len(set(names)) == 18, 'Expected eighteen unique backup members')
    require(len({(x['device'], x['inode']) for x in declared}) == 18, 'Aliased/hardlinked source files refused')
    for fp in declared:
        stable(fp)
    for fp in identities.values():
        stable(fp)
    return dict(schema='opsd-online-pinned-backup-inventory-v1', accepted=True,
                scope='Transfer identity only; no numerical/Adam/effect acceptance', node='hkust1',
                run_id=run, optimizer_updates=updates, outer_index=outer, base=str(base),
                config=cref, checkpoint=cpref, run_identity=rip, source_publication=publication,
                scientific_identity_manifests=identities,
                scientific_scope='Fresh three identity-manifest JSON SHAs; no model/runtime tree or tensor audit',
                files=[dict(relative=name, **fp) for name, fp in sorted(zip(names, declared))],
                total_bytes=sum(x['bytes'] for x in declared), checked_unix=time.time())


def validate_inventory(value, config_path, config_sha, checkpoint_path, checkpoint_sha, run, updates, outer):
    target_scope(run, updates, outer, checkpoint_sha)
    require(value.get('schema') == 'opsd-online-pinned-backup-inventory-v1' and value.get('accepted') is True
            and value['node'] == 'hkust1' and value['run_id'] == run
            and value['optimizer_updates'] == updates and value['outer_index'] == outer,
            'Remote inventory target identity mismatch')
    require(value['config']['path'] == str(config_path) and value['config']['sha256'] == config_sha
            and value['checkpoint']['path'] == str(checkpoint_path) and value['checkpoint']['sha256'] == checkpoint_sha,
            'Remote inventory exact config/checkpoint differs')
    files = value['files']
    require(len(files) == 18 and len({x['relative'] for x in files}) == 18, 'Wrong inventory member count')
    for row in files:
        name = safe_name(row['relative'])
        require(str(Path(value['base']) / name) == row['path'] and SHA.fullmatch(row.get('sha256', ''))
                and isinstance(row['bytes'], int) and row['bytes'] >= 0
                and all(key in row for key in META), 'Unsafe inventory member')
    require(len({(x['device'], x['inode']) for x in files}) == 18, 'Remote inventory aliases source files')
    mapped = {x['path']: x for x in files}
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



class PrefixWriter:
    """Suppress precisely offset bytes; never emit before prefix digest matches."""
    def __init__(self, output, offset, expected_sha):
        require(type(offset) is int and offset > 0 and SHA.fullmatch(expected_sha or ''), 'Explicit positive byte prefix/SHA required')
        self.output=output;self.offset=offset;self.expected_sha=expected_sha
        self.position=0;self.digest=hashlib.sha256();self.verified=False;self.emitted=0
    def write(self,data):
        count=len(data);remaining=max(0,self.offset-self.position);prefix=data[:remaining]
        self.digest.update(prefix);self.position+=len(prefix)
        if self.position==self.offset and not self.verified:
            require(self.digest.hexdigest()==self.expected_sha,'Generated USTAR prefix SHA mismatch; no suffix emitted')
            self.verified=True
        suffix=data[len(prefix):]
        if suffix:
            require(self.verified,'Unverified prefix cannot emit data')
            self.output.write(suffix);self.emitted+=len(suffix);self.position+=len(suffix)
        return count
    def finish(self):
        require(self.verified and self.position>=self.offset,'Prefix offset exceeds deterministic archive length')
        if hasattr(self.output,'flush'):self.output.flush()
        return dict(offset=self.offset,prefix_sha256=self.digest.hexdigest(),total_archive_bytes=self.position,suffix_bytes=self.emitted)


def match_inventory(fresh, original):
    for key in ('schema','accepted','node','run_id','optimizer_updates','outer_index','base','config','checkpoint',
                'run_identity','scientific_identity_manifests','files','total_bytes'):
        require(fresh[key]==original[key],'Fresh source inventory differs from original: '+key)
    return True


def original_input(raw,args):
    require(hashlib.sha256(raw).hexdigest()==args.original_inventory_sha256,'Original inventory stdin SHA differs')
    value=json.loads(raw)
    validate_inventory(value,args.config,args.config_sha256,args.checkpoint,args.checkpoint_sha256,RUN,89,8)
    publication=value.get('source_publication',{})
    require(SHA.fullmatch(publication.get('manifest',{}).get('sha256',''))
            and re.fullmatch(r'[0-9a-f]{40}',publication.get('source_commit','')),
            'Original inventory lacks its actual published helper identity')
    return value


def command(args,mode):
    require(args.host=='hkust1','Only HKUST1 is authorized')
    root=Path(args.remote_helper_root)
    require(root.is_absolute() and '..' not in root.parts,'Explicit canonical remote helper root required')
    remote=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','python3','-B',str(root/'backup.py'),mode,
        '--config',args.config,'--config-sha256',args.config_sha256,'--checkpoint',args.checkpoint,
        '--checkpoint-sha256',args.checkpoint_sha256,'--original-inventory-sha256',args.original_inventory_sha256,
        '--prefix-bytes',str(args.prefix_bytes),'--prefix-sha256',args.prefix_sha256,
        '--source-manifest',str(root/'source-manifest.json'),'--source-manifest-sha256',args.source_manifest_sha256,
        '--publication',args.remote_publication,'--publication-sha256',args.publication_sha256,'--source-commit',args.source_commit]
    return ['ssh','-oConnectTimeout=15',args.host,shlex.join(remote)]


def copy_prefix(path,expected,destination):
    source=fileproof(path,expected);before=metadata(path);h=hashlib.sha256();count=0
    with os.fdopen(os.open(str(path),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)),'rb') as src,destination.open('xb') as out:
        require(meta_from_stat(path,os.fstat(src.fileno()))==before,'Original partial changed before copy')
        for block in iter(lambda:src.read(4<<20),b''):
            h.update(block);count+=len(block);out.write(block)
        out.flush();os.fsync(out.fileno())
        require(meta_from_stat(path,os.fstat(src.fileno()))==before,'Original partial changed during copy')
    stable(source)
    require(count==source['bytes'] and h.hexdigest()==source['sha256'],'Copied prefix differs')
    copied=fileproof(destination,dict(path=str(destination),sha256=source['sha256'],bytes=source['bytes']))
    return source,copied


def resume(args,pub):
    original_raw,original_proof=fileproof(args.original_inventory,dict(path=args.original_inventory,sha256=args.original_inventory_sha256),retain=True)
    original=original_input(original_raw,args)
    partial=fileproof(args.original_partial,dict(path=args.original_partial,sha256=args.prefix_sha256,bytes=args.prefix_bytes))
    require(partial['bytes']==ORIGINAL_PARTIAL_BYTES and partial['sha256']==ORIGINAL_PARTIAL_SHA,
            'Only exact already failed CP89 prefix is authorized')
    dest=Path(args.destination).absolute();checked_path(dest.parent,directory=True)
    require(not dest.exists() and not dest.is_symlink(),'Fresh independent local destination required')
    dest.mkdir(mode=0o700);fsync_dir(dest.parent);archive=dest/'checkpoint.tar.partial'
    try:
        result=subprocess.run(command(args,'inventory'),input=original_raw,capture_output=True,check=False)
        write_json(dest/'inventory-execution.json',dict(actual_returncode=result.returncode,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
        require(result.returncode==0,'Fresh node inventory failed; no suffix requested')
        fresh=json.loads(result.stdout);match_inventory(fresh,original)
        require(fresh['source_publication']['manifest']['sha256']==pub['manifest']['sha256']
            and fresh['source_publication']['publication']['sha256']==pub['publication']['sha256']
            and fresh['source_publication']['source_commit']==pub['source_commit'],'Node new helper publication differs')
        expected=validate_inventory(fresh,args.config,args.config_sha256,args.checkpoint,args.checkpoint_sha256,RUN,89,8)
        require(shutil.disk_usage(dest).free>=required_space(fresh,args.margin_gib<<30),'Local full archive/extraction/margin unavailable')
        _,copied=copy_prefix(args.original_partial,partial,archive)
        ip=write_json(dest/'fresh-inventory.json',fresh)
        write_json(dest/'resume-start.json',dict(original_inventory=original_proof,original_partial=partial,copied_prefix=copied,
            original_failed_backup_retained=True,source_publication=pub,offset=args.prefix_bytes,scope='Backup transport only; no training rc/effect acceptance'))
        with archive.open('ab') as out,(dest/'suffix.stderr').open('xb') as err:
            process=subprocess.Popen(command(args,'suffix'),stdin=subprocess.PIPE,stdout=out,stderr=err)
            try:
                write_json(dest/'suffix-child.json',dict(pid=process.pid,accepted=False,started_unix=time.time()))
                process.communicate(original_raw)
            finally:
                if process.stdin is not None and not process.stdin.closed:process.stdin.close()
                process.wait()
                write_json(dest/'suffix-wait.json',dict(actual_returncode=process.returncode,ended_unix=time.time(),no_signal_sent=True))
            out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
        rc=process.returncode
        write_json(dest/'suffix-exit.json',dict(actual_returncode=rc,archive=fileproof(archive),stderr=fileproof(dest/'suffix.stderr')))
        require(rc==0,'Suffix transfer nonzero; original and new partial remain')
        local,ap=extract_verified(archive,dest/'verified-files',expected)
        stable(partial);stable(original_proof)
        final=dest/'checkpoint.tar';os.rename(archive,final);fsync_dir(dest)
        finalproof=fileproof(final,dict(path=str(final),sha256=ap['sha256'],bytes=ap['bytes']))
        proof=dict(schema='opsd-cp89-resume-complete-v1',accepted=True,run_id=RUN,optimizer_updates=89,outer_index=8,
            original_inventory=original_proof,original_partial=partial,source_publication=pub,fresh_inventory=ip,
            archive=finalproof,files=local,actual_transfer_returncode=rc,independent_local_disk_readback=True,
            scope='Transport only; source job returncode not inferred; no numeric Adam/effect acceptance',
            original_failed_backup_retained=True,node_files_changed=False,completed_unix=time.time())
        write_json(dest/'complete-sha-verified.json',proof)
        return dict(accepted=True,destination=str(dest),file_count=len(local),scope=proof['scope'])
    except Exception as error:
        write_json(dest/'failed.json',dict(schema='opsd-cp89-resume-failed-v1',accepted=False,reason=type(error).__name__+': '+str(error),
            original_failed_backup_retained=True,new_partial_retained=archive.exists(),node_files_changed=False,failed_unix=time.time()))
        raise


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('inventory','suffix','resume'))
    for n in ('config','config-sha256','checkpoint','checkpoint-sha256','original-inventory-sha256','prefix-sha256',
              'source-manifest','source-manifest-sha256','publication','publication-sha256','source-commit'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--prefix-bytes',type=int,required=True);p.add_argument('--host',default='hkust1')
    for n in ('original-inventory','original-partial','remote-helper-root','remote-publication','destination'):p.add_argument('--'+n)
    p.add_argument('--margin-gib',type=int,default=5);return p.parse_args()


def main():
    args=parse_args();require(args.host=='hkust1','Only HKUST1 is authorized')
    target_scope(RUN,89,8,args.checkpoint_sha256)
    require(args.prefix_bytes==ORIGINAL_PARTIAL_BYTES and args.prefix_sha256==ORIGINAL_PARTIAL_SHA,'Exact failed prefix identity required')
    pub=source_publication(args.source_manifest,args.source_manifest_sha256,args.publication,args.publication_sha256,args.source_commit)
    if args.mode=='resume':
        require(all(getattr(args,k) for k in ('original_inventory','original_partial','remote_helper_root','remote_publication','destination')),'Missing explicit local/remote paths')
        result=resume(args,pub);print(canonical(result).decode());return
    require(os.getuid()==os.geteuid()==1005,'Actual remote UID1005 required')
    require(args.config==NODE_BASE+'/receipts/config.json' and str(Path(args.checkpoint).parent.parent.parent)==NODE_BASE+'/runs','Exact old node source required')
    original=original_input(sys.stdin.buffer.read(),args)
    fresh=inventory(args.config,args.config_sha256,args.checkpoint,args.checkpoint_sha256,RUN,89,8,pub)
    match_inventory(fresh,original)
    if args.mode=='inventory':
        source_publication(args.source_manifest,args.source_manifest_sha256,args.publication,args.publication_sha256,args.source_commit)
        print(canonical(fresh).decode());return
    writer=PrefixWriter(sys.stdout.buffer,args.prefix_bytes,args.prefix_sha256)
    stream_archive(fresh,writer);writer.finish()
    source_publication(args.source_manifest,args.source_manifest_sha256,args.publication,args.publication_sha256,args.source_commit)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(1)
