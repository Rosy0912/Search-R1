"""CPU-only, fixed-run remote archiving. Never reads a tensor or controls a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import socket
import sys
import tarfile
import time

SOURCE = '3da904591125f117b1eae1ba09546ebe793cb2d4'
RUN = 'ef09290e69454f20b6a9594ad8e485b6'
CONFIG = '95d9f8f2195a35f189ff06af8e449d419040ae96c4cde4fc50f9d7062db414f7'
DATA = '7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
T = Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
DEST = Path('/weka_3/shaoshitong/recursive-bran-b-hkust5-archive-20261012')
PROTECTED = {5, 50, 100, 150, 200}
PROTOCOL = 'recursive-bran-b-closed-archive-v1'


def raw(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name:
        raise ValueError('Unsafe archive member')
    return p


def own(path, directory=False):
    p = Path(path)
    s = p.lstat()
    if s.st_uid != os.getuid() or p.is_symlink():
        raise ValueError('Owner/symlink mismatch')
    if directory and not stat.S_ISDIR(s.st_mode):
        raise ValueError('Expected directory')
    if not directory and not stat.S_ISREG(s.st_mode):
        raise ValueError('Expected regular file')
    return s


def files_under(root, dirs):
    result = []
    own(root, True)
    for rel in dirs:
        safe_name(rel)
        d = root / rel
        for parent in d.parents:
            if parent == root:
                break
            own(parent, True)
        own(d, True)
        for p in sorted(d.rglob('*')):
            if p.is_dir() and not p.is_symlink():
                own(p, True)
                continue
            s = own(p)
            result.append((str(p.relative_to(root)), s))
    if len({n for n, _ in result}) != len(result):
        raise ValueError('Duplicate source members')
    return result


def fixed_source(root):
    cfg = root / 'receipts/configuration-d79908be70f74dd2a0ce38b5ef880f1e/resolved.yaml'
    if sha(cfg) != CONFIG:
        raise ValueError('Resolved config changed')
    start = json.loads((root / 'receipts/start.json').read_text())
    if start.get('run_id') != RUN or start.get('source_commit') != SOURCE:
        raise ValueError('Different training run')


def accepted_manifest(m, step):
    i = m['identity']
    if (m['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED' or i['completed_step'] != step
            or i['code_commit'] != SOURCE or i['config_sha256'] != CONFIG
            or i['data_identity'] != DATA or i['job_id'] != 'hkust-direct-' + RUN):
        raise ValueError('Checkpoint identity/status mismatch')
    if {w['identity']['rank'] for w in m['worker_states']} != set(range(4)):
        raise ValueError('Incomplete checkpoint ranks')
    return m


def boundary(root, step):
    p = root / f'checkpoints/training_state/global_step_{step}/manifest.json'
    if p.exists():
        return accepted_manifest(json.loads(p.read_text()), step)
    p = root / f'checkpoints/rolling-pruned-receipts/global_step_{step}.json'
    j = json.loads(p.read_text())
    if j['status'] != 'PRUNE_AUTHORIZED_ROLLING_ONLY' or j['replaced_by'] < step + 2:
        raise ValueError('Missing completed native boundary')
    return accepted_manifest(j['original_manifest'], step)


def latest(root):
    steps = []
    for p in (root / 'checkpoints/training_state').glob('global_step_*/manifest.json'):
        n = int(p.parent.name.split('_')[-1])
        accepted_manifest(json.loads(p.read_text()), n)
        steps.append(n)
    return max(steps, default=0)


def closed_event_dirs(root, step):
    result = []
    for d in sorted((root / 'output/recursive-events').iterdir()):
        if not d.is_dir():
            continue
        paths = list(d.glob('*/events.jsonl'))
        if len(paths) != 512:
            continue
        with paths[0].open() as f:
            first = json.loads(next(f))
        # Directory hashes/order do not identify an outer. Read actual event identity.
        if first.get('outer') != step or first.get('split') != 'train':
            continue
        for p in paths:
            starts = ends = 0
            for line in p.open():
                e = json.loads(line)
                if (e.get('outer') != step or e.get('run_id') != RUN
                        or e.get('source_commit') != SOURCE or e.get('config_sha256') != CONFIG):
                    raise ValueError('Event identity changed')
                starts += e.get('event') == 'question_start'
                ends += e.get('event') == 'question_end'
            if starts != 1 or ends != 1:
                raise ValueError('Event shard not closed')
        result.append(str(d.relative_to(root)))
    if not result:
        raise ValueError('No closed 512-question event directory')
    return result


def select_dirs(root, kind, step):
    fixed_source(root)
    completed = latest(root)
    if step < 1 or completed < step + 2:
        raise ValueError('Keep current and previous boundaries on K5')
    m = boundary(root, step)
    if kind == 'checkpoint':
        if step not in PROTECTED or step == 200:
            raise ValueError('Only old protected checkpoints; final200 stays on K5')
        dirs = [f'checkpoints/{n}/global_step_{step}' for n in ('training_state', 'actor', 'algorithm')]
        return [d for d in dirs if (root / d).exists()], m
    if kind != 'output':
        raise ValueError('Unknown kind')
    dirs = []
    for d in sorted((root / 'output/trajectories').glob(f'outer-{step:06d}-attempt-*')):
        j = json.loads((d / 'manifest.json').read_text())
        i = j['identity']
        if (j['status'] != 'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED' or j['rows'] != 2560
                or i['source'] != SOURCE or i['config'] != CONFIG or i['data'] != DATA
                or i['outer_step'] != step or i['job_id'] != 'hkust-direct-' + RUN):
            raise ValueError('Trajectory manifest changed')
        for n, h in j['payloads'].items():
            safe_name(n)
            if sha(d / n) != h:
                raise ValueError('Native trajectory payload mismatch')
        dirs.append(str(d.relative_to(root)))
    if not dirs:
        raise ValueError('No native published trajectories')
    return dirs + closed_event_dirs(root, step), m


def inventory(root, kind, step):
    dirs, m = select_dirs(root, kind, step)
    members = []
    for name, s in files_under(root, dirs):
        p = root / name
        h = sha(p)
        if p.stat().st_size != s.st_size or p.stat().st_mtime_ns != s.st_mtime_ns:
            raise ValueError('Source changed while hashing')
        members.append(dict(name=name, bytes=s.st_size, sha256=h))
    # Native checkpoint hashes are independent of this transport inventory.
    if kind == 'checkpoint':
        expected = {f'checkpoints/training_state/global_step_{step}/driver.pt': m['driver']['sha256']}
        expected.update({f'checkpoints/training_state/global_step_{step}/rank-{w["identity"]["rank"]:02d}.pt': w['sha256'] for w in m['worker_states']})
        expected.update({f'checkpoints/actor/global_step_{step}/{n}': h for n, h in m['hf_identity']['files'].items()})
        actual = {x['name']: x['sha256'] for x in members}
        if any(actual.get(n) != h for n, h in expected.items()):
            raise ValueError('Native checkpoint SHA mismatch')
    return dict(protocol=PROTOCOL, run_id=RUN, source_commit=SOURCE, config_sha256=CONFIG,
                data_identity=DATA, kind=kind, step=step, directories=dirs, members=members)


def validate_inventory(j):
    if (j['protocol'] != PROTOCOL or j['run_id'] != RUN or j['source_commit'] != SOURCE
            or j['config_sha256'] != CONFIG or j['data_identity'] != DATA):
        raise ValueError('Foreign inventory')
    if j['kind'] not in ('checkpoint', 'output') or not 1 <= j['step'] < 200:
        raise ValueError('Invalid archived boundary')
    dirs = [safe_name(n) for n in j['directories']]
    if not dirs or len(set(dirs)) != len(dirs):
        raise ValueError('Invalid directories')
    for d in dirs:
        allowed = ('checkpoints/training_state/', 'checkpoints/actor/', 'checkpoints/algorithm/') if j['kind'] == 'checkpoint' else ('output/trajectories/', 'output/recursive-events/')
        if not any(str(d).startswith(prefix) for prefix in allowed):
            raise ValueError('Unexpected archive directory')
        if j['kind'] == 'checkpoint' and (j['step'] not in PROTECTED or j['step'] == 200
                or d.name != f'global_step_{j["step"]}' or len(d.parts) != 3):
            raise ValueError('Unexpected protected checkpoint directory')
        if j['kind'] == 'output':
            if len(d.parts) != 3:
                raise ValueError('Unexpected output directory depth')
            if d.parts[1] == 'trajectories' and not d.name.startswith(f'outer-{j["step"]:06d}-attempt-'):
                raise ValueError('Different trajectory outer')
    names = set()
    for x in j['members']:
        n = safe_name(x['name'])
        if str(n) in names or not any(d in n.parents for d in dirs):
            raise ValueError('Duplicate/out-of-directory member')
        if not isinstance(x['bytes'], int) or x['bytes'] < 0 or len(x['sha256']) != 64:
            raise ValueError('Invalid member size/hash')
        names.add(str(n))
    if not names:
        raise ValueError('Empty inventory')


def archive_id(j):
    validate_inventory(j)
    return f'{j["kind"]}-{j["step"]:06d}-' + hashlib.sha256(raw(j)).hexdigest()


def pack(root, j, stream):
    import io
    validate_inventory(j)
    if inventory(root, j['kind'], j['step']) != j:
        raise ValueError('Source no longer exactly matches inventory')
    with tarfile.open(fileobj=stream, mode='w|', format=tarfile.PAX_FORMAT) as t:
        b = raw(j)
        info = tarfile.TarInfo('inventory.json'); info.size = len(b)
        t.addfile(info, io.BytesIO(b))
        for x in j['members']:
            p = root / x['name']
            own(p)
            info = tarfile.TarInfo(x['name']); info.size = x['bytes']; info.mode = 0o600
            with p.open('rb') as f:
                t.addfile(info, f)


def receive(dest, stream):
    dest.mkdir(parents=True, exist_ok=True)
    own(dest, True)
    stage = None
    with tarfile.open(fileobj=stream, mode='r|') as t:
        iterator = iter(t)
        first = next(iterator)
        if first.name != 'inventory.json' or not first.isreg() or first.size > 16 * 1024 * 1024:
            raise ValueError('Inventory must be the first regular member')
        j = json.loads(t.extractfile(first).read())
        ident = archive_id(j)
        stage = dest / (ident + '.partial-' + str(os.getpid()))
        stage.mkdir()  # Failure evidence is retained; never overwrite/reuse partials.
        (stage / 'inventory.json').write_bytes(raw(j))
        if shutil.disk_usage(dest).free < sum(x['bytes'] for x in j['members']) + 2**30:
            raise ValueError('Insufficient archive disk')
        expected = {x['name']: x for x in j['members']}
        seen = set()
        for info in iterator:
            safe_name(info.name)
            if (not info.isreg() or info.name not in expected or info.name in seen
                    or info.size != expected[info.name]['bytes']):
                raise ValueError('Unexpected/duplicate/nonregular/size-mismatched member')
            p = stage / 'files' / info.name
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f:
                shutil.copyfileobj(t.extractfile(info), f, length=8 * 1024 * 1024)
                f.flush(); os.fsync(f.fileno())
            seen.add(info.name)
        if seen != set(expected):
            raise ValueError('Missing archive members')
    # Fresh SHA after extraction, not merely checksums of network chunks.
    for n, x in expected.items():
        p = stage / 'files' / n
        if p.stat().st_size != x['bytes'] or sha(p) != x['sha256']:
            raise ValueError('Extracted member SHA mismatch')
    receipt = dict(protocol=PROTOCOL, archive_id=ident, inventory_sha256=hashlib.sha256(raw(j)).hexdigest(),
                   destination=str(dest / ident), status='ALL_MEMBERS_FRESH_SHA_ACCEPTED',
                   members=len(expected), bytes=sum(x['bytes'] for x in expected.values()), time=time.time())
    (stage / 'complete.json').write_bytes(raw(receipt))
    final = dest / ident
    if final.exists():
        raise ValueError('Archive destination exists; retain verified staging, no overwrite')
    os.rename(stage, final)
    return receipt


def verify_destination(dest, j):
    ident = archive_id(j)
    final = dest / ident
    if not final.exists():
        return dict(exists=False)
    own(final, True)
    if (final/'inventory.json').read_bytes() != raw(j):
        raise ValueError('Existing destination inventory differs')
    actual = {n: s.st_size for n, s in files_under(final/'files',j['directories'])}
    expected = {x['name']: x['bytes'] for x in j['members']}
    if actual != expected or any(sha(final/'files'/x['name']) != x['sha256'] for x in j['members']):
        raise ValueError('Existing destination member changed')
    r = json.loads((final/'complete.json').read_text())
    if r['inventory_sha256'] != hashlib.sha256(raw(j)).hexdigest() or r['archive_id'] != ident:
        raise ValueError('Existing receipt identity differs')
    r.update(time=time.time(),status='ALL_MEMBERS_FRESH_SHA_ACCEPTED',destination=str(final))
    return dict(exists=True,receipt=r)


def prune(root, j, receipt):
    ident = archive_id(j)
    if (receipt['protocol'] != PROTOCOL or receipt['archive_id'] != ident
            or receipt['inventory_sha256'] != hashlib.sha256(raw(j)).hexdigest()
            or receipt['status'] != 'ALL_MEMBERS_FRESH_SHA_ACCEPTED'
            or receipt['destination'] != str(DEST / ident)
            or receipt['members'] != len(j['members'])
            or receipt['bytes'] != sum(x['bytes'] for x in j['members'])):
        raise ValueError('No matching complete K4 receipt')
    # Revalidate actual new boundary and every original member immediately before removal.
    if inventory(root, j['kind'], j['step']) != j:
        raise ValueError('Source changed after transfer; do not remove')
    rd = root / 'receipts/remote-archive'
    rd.mkdir(exist_ok=True)
    proof = rd / (ident + '.json')
    with proof.open('x') as f:
        json.dump(dict(time=time.time(),inventory=j, destination_receipt=receipt,
                       status='VERIFIED_ARCHIVE_REDUNDANT_BIG_FILES_REMOVAL_AUTHORIZED'), f)
        f.flush(); os.fsync(f.fileno())
    # Keep all small metadata and native manifests at their original paths.
    # Full archived directories are restored from K4 before a cold replay/recovery.
    for x in j['members']:
        if (x['name'].startswith('output/recursive-events/') or x['bytes'] > 1024 * 1024
                or Path(x['name']).suffix in ('.pt', '.safetensors', '.gz')):
            (root / x['name']).unlink()
    return dict(archive_id=ident,status='ARCHIVED_BIG_FILES_REMOVED_SMALL_METADATA_RETAINED',
                proof=str(proof),destination=receipt['destination'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['inventory','pack','receive','verify','prune','scan'])
    p.add_argument('--kind', choices=['output','checkpoint']); p.add_argument('--step', type=int)
    a = p.parse_args()
    if os.getuid() != 1001:
        raise ValueError('Expected task UID1001')
    if a.mode not in ('receive','verify') and socket.gethostname().split('.')[0] != 'xie-lab-4':
        raise ValueError('Source must be actual HKUST5')
    if a.mode in ('receive','verify') and socket.gethostname().split('.')[0] != 'xielab5':
        raise ValueError('Archive disk must be actual HKUST4')
    if a.mode == 'receive':
        result = receive(DEST, sys.stdin.buffer)
    elif a.mode == 'verify':
        result = verify_destination(DEST,json.load(sys.stdin))
    elif a.mode in ('pack','prune'):
        data = json.load(sys.stdin)
        if a.mode == 'pack':
            pack(T, data, sys.stdout.buffer); return
        result = prune(T, data['inventory'], data['receipt'])
    elif a.mode == 'scan':
        fixed_source(T); n = latest(T); candidates = []
        for step in range(1, n-1):
            for kind in ('output','checkpoint'):
                if kind == 'checkpoint' and (step not in PROTECTED or step == 200):
                    continue
                rd = T / 'receipts/remote-archive'
                if list(rd.glob(f'{kind}-{step:06d}-*.json')):
                    continue
                if kind == 'output' and not list((T/'output/trajectories').glob(f'outer-{step:06d}-attempt-*')):
                    continue
                candidates.append(dict(kind=kind,step=step))
        result = dict(completed=n,candidates=candidates,available=shutil.disk_usage(T).free)
    else:
        result = inventory(T, a.kind, a.step)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
