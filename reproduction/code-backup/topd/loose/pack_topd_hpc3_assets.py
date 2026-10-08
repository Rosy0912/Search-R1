"""Source-side immutable, lossless asset packages; no GPU or credentials."""
import concurrent.futures
import fcntl
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import tarfile
import threading
import time

ROOT = pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
OUT = ROOT / 'migration-hpc3-20261008'


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(temp, path)


class Reader:
    def __init__(self, file):
        self.file = file
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, length=-1):
        data = self.file.read(length)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def pack(group):
    name, files = group
    receipt = OUT / (name + '.json')
    if receipt.exists():
        old = json.loads(receipt.read_text())
        assert old['status'] == 'PACKED_SOURCE_SHA256'
        assert pathlib.Path(old['archive']).stat().st_size == old['archive_bytes']
        return old
    archive = OUT / (name + '.tar.zst')
    part = archive.with_suffix('.part')
    digest = hashlib.sha256()
    compressor = subprocess.Popen(['zstd', '-q', '-1', '-T2', '-c'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    drain_errors = []

    def drain():
        try:
            with part.open('wb') as output:
                while True:
                    chunk = compressor.stdout.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except BaseException as error:
            drain_errors.append(repr(error))
            compressor.kill()

    thread = threading.Thread(target=drain)
    thread.start()
    records = []
    started = time.time()
    try:
        with tarfile.open(fileobj=compressor.stdin, mode='w|', bufsize=1024 * 1024) as tar:
            for source, logical in files:
                before = source.stat()
                info = tar.gettarinfo(str(source), arcname=logical)
                if not info.isfile():
                    # A trusted source symlink is copied as content, never as a
                    # remote link into another filesystem.
                    info.type = tarfile.REGTYPE
                    info.linkname = ''
                    info.size = before.st_size
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with source.open('rb') as f:
                    reader = Reader(f)
                    tar.addfile(info, reader)
                after = source.stat()
                assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
                    after.st_size, after.st_mtime_ns, after.st_ino), str(source)
                assert reader.bytes == before.st_size
                records.append({'source': str(source), 'path': logical,
                                'size': reader.bytes, 'sha256': reader.digest.hexdigest()})
        compressor.stdin.close()
        thread.join()
        assert compressor.wait() == 0 and not drain_errors, drain_errors
        os.replace(part, archive)
    except BaseException:
        compressor.kill()
        thread.join(timeout=10)
        raise
    record = {'status': 'PACKED_SOURCE_SHA256', 'group': name,
              'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest.hexdigest(), 'files': records,
              'elapsed_seconds': time.time() - started, 'completed_epoch': time.time()}
    atomic(receipt, record)
    print(json.dumps({k: record[k] for k in ['group','archive_bytes','elapsed_seconds']}), flush=True)
    return record


def tree(source, logical, exclude=()):
    return [(p, logical + '/' + str(p.relative_to(source)))
            for p in sorted(source.rglob('*')) if p.is_file()
            and '.git' not in p.relative_to(source).parts
            and '__pycache__' not in p.relative_to(source).parts
            and p.name not in exclude]


def main():
    OUT.mkdir(mode=0o700, exist_ok=True)
    lock = (OUT / 'producer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert pathlib.Path('/proc/self/cmdline').exists()
    atomic(OUT / 'producer.json', {'pid': os.getpid(), 'started_epoch': time.time(),
                                  'gpu_work': False, 'source_root': str(ROOT)})
    teacher_run = ROOT / 'logs/r1-standard32b-grpo100-common-v5-r1'
    teacher = teacher_run / 'checkpoints/actor/global_step_85'
    student = ROOT / 'assets/qwen2.5-3b-instruct-aa8e725'
    data = ROOT / 'assets/common-instruct-0b002b'
    assert hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest() == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    groups = []
    for role, source in [('teacher', teacher), ('student_initial', student)]:
        files = tree(source, role)
        metadata = [row for row in files if not row[0].name.endswith('.safetensors')]
        groups.append((role + '_metadata', metadata))
        for p, logical in files:
            if p.name.endswith('.safetensors'):
                groups.append((role + '_' + p.stem, [(p, logical)]))
    groups.insert(0, ('common_data', tree(data, 'datasets/common-instruct-0b002b')))
    groups.insert(1, ('e5_model', tree(ROOT / 'assets-root/assets/e5-base-v2', 'retrieval/e5-base-v2', exclude=('model.safetensors.partial',))))
    code = []
    for relative in ['tools/topd-instruct300-source-v1', 'tools/topd-instruct300-launch-v6',
                     'tools/topd-vllm-teacher-v3', 'tools/qwen25-32b-window-safety-direct-own-sm-v2',
                     'tools/qwen25-32b-terminal-fix-refreeze', 'tools/search-r1-32b-grpo100-source-v6',
                     'tools/search-r1-32b-grpo100-tools-direct-v5', 'repo-terminal-fixed413',
                     'retriever-gpu-lib', 'test-tools',
                     'assets/compiler-headers/python3.10-3.10.12-22.04.18-amd64']:
        code += tree(ROOT / relative, 'code/h100-layout/' + relative, exclude=('.lock',))
    for name in ['bind_topd_instruct300_v8.py', 'start_topd_teacher85_approved.py',
                 'pack_topd_hpc3_assets.py', 'migrate_topd_to_hpc3_stream.py',
                 'remote_topd_retained18.py', 'watch_topd3224_teacher85_readonly.py',
                 'deploy_topd_teacher85_existing3224.py']:
        code.append((ROOT / 'tools' / name, 'code/h100-layout/tools/' + name))
    groups.insert(2, ('implementation', code))
    metadata = []
    for rel in ['actual-outer.json', 'static-config.json', 'evidence/effective-runtime-config.json',
                'evidence/train-result.json', 'evidence/trainer-first-exception.json']:
        p = teacher_run / rel
        if p.is_file():
            metadata.append((p, 'provenance/teacher3228/' + rel))
    for rel in ['logs/qwen25-32b-terminal-fixed413-binding.json',
                'logs/topd-instruct300-teacher85-approved-v1/teacher85-user-approval.json',
                'logs/topd-instruct300-teacher85-approved-v1/static-config.json']:
        metadata.append((ROOT / rel, 'provenance/h100-layout/' + rel))
    migration_note = OUT / 'migration-protocol.json'
    atomic(migration_note, {
        'teacher_saved_step': 85, 'teacher_actual_historical_updates': 89,
        'teacher_dtype': 'original FP32 checkpoint unchanged',
        'student_actual_updates_at_selection': 0,
        'student_checkpoint': 'official initial Qwen2.5-3B-Instruct actor and fixed reference',
        'student_revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
        'teacher_is_frozen_for_TOPD': True, 'teacher_optimizer_not_required_for_scoring': True,
        'exact_training_resume': False,
        'offline_migration_human_authorized': True,
        'online_cross_cluster_retrieval_or_scoring': False,
        'later_HPC3_training_requires_local_assets_and_environment_path_adaptation': True,
        'package_excludes_credentials_weights_are_separate_from_Git': True,
        'installed_dependencies': sorted([{'name': d.metadata['Name'], 'version': d.version}
                                         for d in importlib.metadata.distributions()], key=lambda x: x['name'].lower()),
    })
    metadata.append((migration_note, 'provenance/migration-protocol.json'))
    groups.insert(3, ('provenance', metadata))
    groups += [('wiki_corpus', [(ROOT / 'assets-root/assets/wiki/wiki-18.jsonl', 'retrieval/wiki/wiki-18.jsonl')]),
               ('wiki_index', [(ROOT / 'assets-root/assets/wiki/e5_Flat.index', 'retrieval/wiki/e5_Flat.index')])]
    atomic(OUT / 'plan.json', {'groups': [{'name': n, 'files': len(fs), 'source_bytes': sum(p.stat().st_size for p, _ in fs)} for n, fs in groups],
                             'source_originals_retained': True, 'no_training_cancelled': True})
    # One pack at a time; compression uses only two CPU threads.
    results = [pack(group) for group in groups]
    atomic(OUT / 'complete.json', {'status': 'ALL_SOURCE_PACKAGES_READY', 'groups': results,
                                 'archive_bytes': sum(r['archive_bytes'] for r in results),
                                 'source_bytes': sum(f['size'] for r in results for f in r['files'])})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        OUT.mkdir(mode=0o700, exist_ok=True)
        atomic(OUT / 'failed.json', {'status': 'SOURCE_PACK_FAILED', 'epoch': time.time(),
                                   'pid': os.getpid(), 'error': repr(error)})
        raise
