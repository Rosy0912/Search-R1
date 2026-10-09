"""Collect real hop receipts and independently reread original HPC files.

This is a metadata-only Mac controller. The existing reservation guard remains
responsible for the supervisor handoff. It never uploads or starts training.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path(__file__).resolve().parents[1]
E = BASE / 'outputs/evidence'
SHA = 'edc951c6a9cade2134ca0badc58b03852aae800b73d55d498d1a4c3a95331787'
SIZE = 3483281305
REMOTE = '/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/migration-hf-20261009'
HK2 = '/data/shaoshitong/cosearch-hf-migration-20261009'
HK3 = '/backup/shaoshitong/cosearch-hf-migration-20261009'
spec = importlib.util.spec_from_file_location('control', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctl)


def save(name, obj):
    p = E / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, p)


def query(host, code, timeout=30):
    command = 'python3 -c ' + shlex.quote(code)
    if host == 'hpc3':
        args = ctl.route(host, command)
    else:
        args = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
                '-oConnectTimeout=8', host, command]
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode and host in ('hkust2', 'hkust3'):
        # Only this connection: documented alias failures, no global changes.
        args = ['ssh', '-T', '-B', 'en0', '-oProxyCommand=none', '-oProxyJump=none',
                '-oBatchMode=yes', '-oStrictHostKeyChecking=yes', '-oConnectTimeout=8', host, command]
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        if 'AssertionError' in result.stderr:
            raise AssertionError('Remote SHA acceptance failed')
        raise RuntimeError('Control read failed: ' + host)
    return json.loads(result.stdout)


def receipts(host, root, names):
    code = ('import pathlib,json;r=pathlib.Path(' + repr(root) + ');ns=' + repr(names) +
            ';print(json.dumps({n:json.loads((r/n).read_text()) for n in ns if (r/n).exists()}))')
    return query(host, code)


def main():
    lock = (E / 'native-x-hpc-hf-acceptance.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save('native-x-hpc-hf-acceptance-controller.json', {'pid': os.getpid(), 'epoch': time.time()})
    attempt = 0
    while True:
        attempt += 1
        try:
            guard = json.loads((E / 'native-x-hpc-retriever-supplement-progress.json').read_text())
            assert guard['writer_reservation_only'] and guard['status'] == 'waiting_hf_route_destination_verification'
            os.kill(guard['pid'], 0)
            with (E / 'native-x-hpc-retriever-supplement.lock').open('a') as reservation:
                try:
                    fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError('Supplement reservation is no longer held')
            target = receipts('hpc3', REMOTE, ['target-fresh-verified.json', 'archive-verified.json'])
            if 'target-fresh-verified.json' not in target:
                save('native-x-hpc-hf-acceptance-progress.json', {'status': 'WAITING_TARGET_FRESH_RECEIPT', 'epoch': time.time()})
                time.sleep(30)
                continue
            two = receipts('hkust2', HK2, ['retriever-supplement-download-verified.json'])
            three = receipts('hkust3', HK3, ['forward-verified.json'])
            source = json.loads((E / 'native-x-hpc-hf-route-plan-20261009.json').read_text())['source_preparation']['archive-verified.json']
            hf = json.loads((E / 'native-x-hpc-hf-upload-verified-20261009.json').read_text())
            assert source['status'] == 'SOURCE_ARCHIVE_AND_FILES_SHA_VERIFIED'
            assert hf['status'] == 'HF_UPLOAD_COMMITTED' and hf['commit'] == '220f71043c64b4e5663769be6f2b96a89075a036'
            for obj in (source, hf, target['archive-verified.json']):
                assert obj['archive_sha256'] == SHA and obj['archive_bytes'] == SIZE
            download = two['retriever-supplement-download-verified.json']
            forwarded = three['forward-verified.json']
            assert download['status'] == 'SOURCE_HF_DOWNLOAD_ARCHIVE_SHA256_VERIFIED'
            assert forwarded['status'] == 'HKUST2_HKUST3_ARCHIVE_SHA256_VERIFIED'
            for obj in (download, forwarded):
                assert obj['sha256'] == SHA and obj['bytes'] == SIZE
            proof = target['target-fresh-verified.json']
            assert proof['status'] == 'SUPPLEMENT_ALL_ORIGINAL_FILES_FRESH_SHA_VERIFIED'
            assert proof['fresh_original_files_verified'] and proof['archive_sha256'] == SHA
            assert (proof['verified_files'], proof['verified_bytes'], proof['symlinks']) == (39714, 7529241082, 4)
            save('native-x-hpc-hf-acceptance-progress.json', {'status': 'INDEPENDENT_FRESH_SHARED_SHA', 'epoch': time.time()})
            # Independent full reread, not receipt-only acceptance or another extraction.
            code = '''import pathlib,json,hashlib,os,time
r=pathlib.Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source')
m=pathlib.Path(MANIFEST);assert hashlib.sha256(m.read_bytes()).hexdigest()=='968611a77c53c4782c6b1715650d49434bd2ea51ddc65adbd98405a44de383ca'
rows=[json.loads(s) for s in m.read_text().splitlines()];files=total=links=0
for v in rows:
 if v['kind'] not in ('file','symlink'):continue
 rel=pathlib.PurePosixPath(v['path']);assert not rel.is_absolute() and '..' not in rel.parts
 p=r/v['path'];parent=p.parent
 while parent!=r:
  assert not parent.is_symlink();parent=parent.parent
 if v['kind']=='symlink':assert p.is_symlink() and os.readlink(p)==v['target'];links+=1;continue
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==v['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 assert h.hexdigest()==v['sha256'],v['path'];files+=1;total+=v['bytes']
assert (files,total,links)==(39714,7529241082,4)
print(json.dumps({'verified_files':files,'verified_bytes':total,'symlinks':links,'epoch':time.time()}))
'''.replace('MANIFEST', repr(REMOTE + '/manifest.jsonl'))
            fresh = query('hpc3', code, timeout=3600)
            result = dict(proof, independent_fresh_shared_read=fresh,
                          hop_archive_sha_verified={h: True for h in ['source', 'hf', 'hkust2', 'hkust3', 'hpc3']},
                          hop_receipts={'source': source, 'hf': hf, 'hkust2': download,
                                        'hkust3': forwarded, 'hpc3': target['archive-verified.json']},
                          accepted_epoch=time.time(), training_started=False)
            save('native-x-hpc-hf-supplement-target-fresh-verified.json', result)
            return
        except AssertionError:
            save('native-x-hpc-hf-acceptance-failure.json', {'status': 'INTEGRITY_HARDSTOP', 'epoch': time.time()})
            raise
        except (OSError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            save('native-x-hpc-hf-acceptance-failure.json', {'status': 'RETRY_METADATA_READ',
                 'epoch': time.time(), 'attempt': attempt, 'error_type': type(error).__name__})
            time.sleep(min(120, attempt * 15))


if __name__ == '__main__':
    main()
