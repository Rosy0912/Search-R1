"""Resumable byte relay. Model/data bytes never become local files."""
import fcntl
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import time

WORK = pathlib.Path(__file__).resolve().parent
LOCAL = WORK.parent / 'outputs/hpc3-migration'
CONFIG = '/Users/shaoshitong/Documents/remote_codex_control/access/hpc3-20261008/ssh_config'
DEST = '/data/user/jxu957/experiments/topd-20261008'
SOURCE = '/dataset/shaoshitong/search-r1-rl/topd/migration-hpc3-20261008'
STREAMS = int(os.environ.get('TOPD_MIGRATION_STREAMS', '4'))
assert 1 <= STREAMS <= 4


def source(command, timeout=120):
    r = subprocess.run(['python3', str(WORK / 'remote_topd_retained18.py')],
                       input=command, text=True, capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError('Source SSH: ' + r.stderr[-1500:])
    return json.loads(r.stdout)


def ssh_args(command):
    return ['ssh', '-T', '-F', CONFIG, '-o', 'BatchMode=yes', 'hpc3', command]


def target(code, timeout=120):
    r = subprocess.run(ssh_args('env -u LD_LIBRARY_PATH python3 -'), input=code,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError('Destination SSH: ' + r.stderr[-1500:])
    return json.loads(r.stdout)


def transfer(record, source_reader=None):
    name = record['group']
    sha = record['archive_sha256']
    folder = DEST + '/migration/archives'
    part = folder + '/' + name + '.' + sha + '.part'
    final = folder + '/' + name + '.tar.zst'
    expected = record['archive_bytes']
    receipt = DEST + '/migration/' + name + '-verified.json'
    state = target("import pathlib,json; p=pathlib.Path(" + repr(part) + ");r=pathlib.Path(" + repr(receipt) + ");f=pathlib.Path("+repr(final)+");print(json.dumps({'offset':p.stat().st_size if p.exists() else 0,'archive_complete':f.exists() and f.stat().st_size=="+str(expected)+",'verified':json.loads(r.read_text()) if r.exists() else None}))")
    if state['verified'] and state['verified'].get('archive_sha256') == sha:
        return state['verified']
    offset = state['offset']
    if offset > expected:
        raise ValueError('Destination partial exceeds source package')
    src_code = "import sys,pathlib; p=pathlib.Path(" + repr(record['archive']) + ");assert p.stat().st_size==" + str(expected) + ";f=p.open('rb');f.seek(" + str(offset) + ")\nwhile True:\n b=f.read(1048576)\n if not b:break\n sys.stdout.buffer.write(b)\nsys.stdout.buffer.flush()\n"
    src_command = 'python3 -c ' + shlex.quote(src_code)
    dst_code = "import pathlib,sys,os; p=pathlib.Path(" + repr(part) + ");p.parent.mkdir(parents=True,exist_ok=True);assert (p.stat().st_size if p.exists() else 0)==" + str(offset) + ";f=p.open('ab')\nwhile True:\n b=sys.stdin.buffer.read(1048576)\n if not b:break\n f.write(b)\nf.flush();os.fsync(f.fileno());f.close()\nassert p.stat().st_size==" + str(expected) + "\n"
    start = time.time()
    copied = offset
    last_report = start
    with (LOCAL / (name + '-source-stderr.log')).open('ab') as srcerr, (LOCAL / (name + '-destination-stderr.log')).open('ab') as dsterr:
      if not state['archive_complete']:
        src = subprocess.Popen(source_reader or ['python3', str(WORK / 'remote_topd_retained18.py'), '--timeout', '43200'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=srcerr)
        dst = subprocess.Popen(ssh_args('python3 -c ' + shlex.quote(dst_code)),
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=dsterr)
        src.stdin.write(src_command.encode())
        src.stdin.close()
        try:
            while True:
                chunk = src.stdout.read(1024 * 1024)
                if not chunk:
                    break
                dst.stdin.write(chunk)
                copied += len(chunk)
                now = time.time()
                if now - last_report >= 30:
                    progress = {'group': name, 'status': 'STREAMING', 'epoch': now,
                                'bytes_at_destination_sent': copied, 'total_bytes': expected,
                                'resumed_from': offset, 'mean_mib_per_second': (copied-offset)/max(now-start,1)/1024**2,
                                'intermediate_model_files': False}
                    (LOCAL / (name + '-progress.json')).write_text(json.dumps(progress, indent=2) + '\n')
                    print(json.dumps(progress), flush=True)
                    last_report = now
            dst.stdin.close()
            a, b = src.wait(timeout=40), dst.wait(timeout=60)
            if a or b:
                raise RuntimeError('Transfer exit codes source=%s destination=%s' % (a, b))
        except BaseException:
            src.terminate()
            dst.terminate()
            raise
    # Destination archive SHA and every unpacked member SHA are independently
    # checked there. The original source hash is captured while packing.
    code = r'''import pathlib,json,hashlib,subprocess,tarfile,os,time
part=pathlib.Path(PART);final=pathlib.Path(FINAL);root=pathlib.Path(ROOT)
record=RECORD
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
input_archive=part if part.exists() else final
assert input_archive.stat().st_size==record['archive_bytes']
assert digest(input_archive)==record['archive_sha256'],'Archive SHA256 mismatch'
if input_archive==part:os.replace(part,final)
expected={r['path']:r for r in record['files']};seen=set();root.mkdir(parents=True,exist_ok=True)
z=subprocess.Popen(['zstd','-dq','-c',str(final)],stdout=subprocess.PIPE)
with tarfile.open(fileobj=z.stdout,mode='r|') as tar:
 for m in tar:
  assert m.isfile() and m.name in expected and m.name not in seen
  rel=pathlib.PurePosixPath(m.name);assert not rel.is_absolute() and '..' not in rel.parts
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True);assert root.resolve() in p.resolve().parents
  tmp=p.with_name(p.name+'.migration-part');h=hashlib.sha256();count=0
  with tar.extractfile(m) as f,tmp.open('wb') as out:
   while True:
    b=f.read(1048576)
    if not b:break
    out.write(b);h.update(b);count+=len(b)
   out.flush();os.fsync(out.fileno())
  e=expected[m.name];assert count==e['size'] and h.hexdigest()==e['sha256'],m.name
  os.chmod(tmp,m.mode & 0o777);os.replace(tmp,p);seen.add(m.name)
assert z.wait()==0 and seen==set(expected)
result={'status':'ARCHIVE_AND_ALL_FILES_SHA256_VERIFIED','group':record['group'],'archive_sha256':record['archive_sha256'],'archive_bytes':record['archive_bytes'],'file_count':len(seen),'files':record['files'],'verified_epoch':time.time()}
p=pathlib.Path(RECEIPT);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,p)
print(json.dumps(result))
'''.replace('PART',repr(part)).replace('FINAL',repr(final)).replace('ROOT',repr(DEST+'/bundle')).replace('RECORD',repr(record)).replace('RECEIPT',repr(receipt))
    result = target(code, timeout=3600)
    (LOCAL / (name + '-verified.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'group': name, 'status': result['status'], 'archive_bytes': expected,
                      'file_count': result['file_count']}), flush=True)
    return result


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / 'relay.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (LOCAL / 'controller.json').write_text(json.dumps({'pid':os.getpid(),'started_epoch':time.time(),'source':SOURCE,'destination':DEST,'local_disk_contains_models':False,'parallel_streams':STREAMS,'retry_policy':'unlimited_with_30_to_180_second_backoff'},indent=2)+'\n')
    completed = {}
    # Retain verified receipts across controller restarts. Final acceptance still
    # rereads every actual destination file and checks its original SHA256.
    for receipt in LOCAL.glob('*-verified.json'):
        item = json.loads(receipt.read_text())
        if item.get('status') == 'ARCHIVE_AND_ALL_FILES_SHA256_VERIFIED':
            completed[item['group']] = item
    failures = 0
    while True:
        try:
            scan = source("python3 - <<'SCAN'\nimport pathlib,json,os\np=pathlib.Path("+repr(SOURCE)+");print(json.dumps({'plan':json.loads((p/'plan.json').read_text()) if (p/'plan.json').exists() else None,'ready':[json.loads(x.read_text()) for x in sorted(p.glob('*.json')) if x.name not in ['producer.json','producer-launch.json','plan.json','complete.json','migration-protocol.json']],'complete':(p/'complete.json').exists(),'failed':json.loads((p/'failed.json').read_text()) if (p/'failed.json').exists() else None}))\nSCAN\n")
            if scan.get('failed'):raise RuntimeError('Source packer failed: '+str(scan['failed']))
            (LOCAL / 'source-scan.json').write_text(json.dumps(scan, indent=2) + '\n')
            if scan['plan']:
                order = [r['name'] for r in scan['plan']['groups']]
                ready = {r['group']:r for r in scan['ready'] if r.get('status')=='PACKED_SOURCE_SHA256'}
                for name in list(completed):
                    if name not in ready or completed[name]['archive_sha256'] != ready[name]['archive_sha256']:
                        del completed[name]
                pending = [name for name in order if name in ready and name not in completed][:STREAMS]
                # Bounded independent package streams share no destination files.
                # On restart each stream resumes its own physical partial.
                with concurrent.futures.ThreadPoolExecutor(max_workers=STREAMS) as pool:
                    futures = {pool.submit(transfer, ready[name]): name for name in pending}
                    errors = []
                    for future in concurrent.futures.as_completed(futures):
                        name = futures[future]
                        try:
                            completed[name] = future.result()
                        except Exception as error:
                            errors.append((name, str(error)))
                    if errors:
                        raise RuntimeError('Package transfer failures: ' + str(errors))
                if scan['complete'] and set(completed)==set(order):
                    final_check = r'''import pathlib,json,hashlib,time
r=pathlib.Path(ROOT);files={}
for p in sorted((r/'migration').glob('*-verified.json')):
 for f in json.loads(p.read_text())['files']:
  assert f['path'] not in files
  files[f['path']]=f
total=0
for name,f in files.items():
 p=r/'bundle'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==f['size'],name
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 assert h.hexdigest()==f['sha256'],name
 total+=f['size']
print(json.dumps({'status':'FINAL_SHARED_FILES_FRESH_SHA256_PASS','files':len(files),'bytes':total,'epoch':time.time()}))
'''.replace('ROOT',repr(DEST))
                    physical = target(final_check, timeout=7200)
                    (LOCAL/'final-shared-files-sha256.json').write_text(json.dumps(physical,indent=2)+'\n')
                    result={'status':'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED','destination':DEST,'groups':list(completed),'archive_bytes':sum(r['archive_bytes'] for r in completed.values()),'verified_epoch':time.time(),'source_originals_retained':True,'student_initial_only':True,'final_physical_sha256':physical}
                    (LOCAL/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
                    target("import pathlib,json; p=pathlib.Path("+repr(DEST+'/migration/complete.json')+");p.write_text("+repr(json.dumps(result,indent=2)+'\n')+");print(json.dumps({'saved':str(p)}))")
                    print(json.dumps(result),flush=True)
                    return
            failures = 0
            time.sleep(15)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            failures += 1
            with (LOCAL/'failures.jsonl').open('a') as f:
                f.write(json.dumps({'epoch':time.time(),'error':str(error),'consecutive':failures})+'\n')
            print(json.dumps({'status':'RETRY_PRESERVING_SOURCE_AND_DESTINATION_PARTIAL','error':str(error),'consecutive':failures}),flush=True)
            # Keep retrying transient SSH/storage failures until all hashes pass.
            time.sleep(min(180, 30 * failures))


if __name__ == '__main__':
    main()
