"""Local orchestration of the published, read-only HKUST first-five extractor.

Runs no model or training. Source goes to remote stdin; private output stays on
Mac. This wrapper does not modify SSH configuration, source inputs or Node files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def check(ok, why):
    if not ok:
        raise ValueError(why)


def raw_bound(path, sha):
    p = Path(path)
    check(p.is_absolute() and p.is_file() and not p.is_symlink(), 'Absolute regular input required')
    before = p.stat()
    raw = p.read_bytes()
    after = p.stat()
    check((before.st_ino, before.st_size, before.st_mtime_ns) ==
          (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed')
    check(hashlib.sha256(raw).hexdigest() == sha, 'Input SHA differs')
    return raw


def file_ref(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=h.hexdigest())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode() + b'\n'


def main():
    p = argparse.ArgumentParser()
    for key in ('source', 'request', 'publication'):
        p.add_argument('--' + key, required=True)
        p.add_argument('--' + key + '-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source = raw_bound(a.source, a.source_sha256)
    request = raw_bound(a.request, a.request_sha256)
    publication = json.loads(raw_bound(a.publication, a.publication_sha256))
    check(publication['accepted'] is True and publication['local_commit'] == publication['remote_commit'],
          'Published remote readback required before extraction')
    check(any(r['sha256'] == a.source_sha256 for r in publication['files']),
          'Extractor not in published exact file inventory')
    check(len(request) < 80000, 'Bounded CLI request required')
    check(hashlib.sha256(request.decode().encode()).hexdigest() == a.request_sha256,
          'Request UTF8 bytes differ')
    out = Path(a.output)
    check(out.is_absolute() and not out.exists(), 'New absolute output required')
    out.mkdir(parents=True)
    remote = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
                        'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                        'nice', '-n', '19',
                        '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python',
                        '-B', '-', '--request-json', request.decode(),
                        '--request-sha256', a.request_sha256])
    argv = ['/usr/bin/ssh', '-B', 'en0', '-oProxyCommand=none', '-oProxyJump=none',
            '-oControlMaster=no', '-S', 'none', '-oBatchMode=yes', '-oConnectTimeout=10',
            'hkust1', remote]
    stdout = out / 'extraction.stdout.private.jsonl'
    stderr = out / 'extraction.stderr.private.txt'
    started = time.time()
    with stdout.open('xb') as f, stderr.open('xb') as e:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=f, stderr=e)
        (out / 'started.private.json').write_bytes(canonical(dict(
            ssh_pid=proc.pid, started_unix=started, request_sha256=a.request_sha256,
            extractor_source_sha256=a.source_sha256, source_publication_commit=publication['remote_commit'],
            source_publication_sha256=a.publication_sha256)))
        proc.communicate(source)
    proof = dict(schema='opsd-first5-em-loss-extraction-execution-v1', returncode=proc.returncode,
                 started_unix=started, ended_unix=time.time(), ssh_pid=proc.pid,
                 extractor_source_sha256=a.source_sha256, request_sha256=a.request_sha256,
                 publication_sha256=a.publication_sha256,
                 source_publication_commit=publication['remote_commit'],
                 command_without_private_request=argv[:-1] + ['<SHA-bound private request and source stdin>'],
                 stdout_sha256=file_ref(stdout)['sha256'], stderr_sha256=file_ref(stderr)['sha256'],
                 stdout=file_ref(stdout), stderr=file_ref(stderr),
                 no_gpu_or_training_changes=True)
    (out / 'execution.private.json').write_bytes(canonical(proof))
    check(proc.returncode == 0, 'Remote extraction failed; actual stderr/output retained')
    raw_bound(a.source, a.source_sha256)
    targets = {'pair_feature': out / 'pair-features.private.jsonl',
               'cohort_feature': out / 'cohort-features.private.jsonl',
               'outer_proof': out / 'outer-proofs.private.jsonl'}
    streams = {k: v.open('xb') for k, v in targets.items()}
    counts = {k: 0 for k in streams}
    summary = None
    stream_hash = hashlib.sha256()
    try:
        with stdout.open('rb') as f:
            for line in f:
                check(line.endswith(b'\n') and summary is None, 'Partial or post-summary extractor output')
                r = json.loads(line)
                kind = r['record_kind']
                if kind == 'summary':
                    summary = r
                    check(r['output_stream_sha256_before_summary'] == stream_hash.hexdigest(),
                          'Extractor stream digest differs')
                else:
                    check(kind in streams, 'Unexpected extractor record')
                    stream_hash.update(line)
                    streams[kind].write(line)
                    counts[kind] += 1
    finally:
        for f in streams.values():
            f.close()
    check(summary is not None and summary['accepted'] is True, 'No actual accepted summary')
    check(summary['counts'] == dict(pairs=4529, valid_tokens=940295, cohort_rows=7680,
                                   outers=15, optimizer_updates=151), 'Actual native first-five totals differ')
    check(counts == dict(pair_feature=4529, cohort_feature=7680, outer_proof=15), 'Actual output coverage differs')
    summary_path = out / 'extraction-summary.private.json'
    summary_path.write_bytes(canonical(summary))
    manifest = dict(schema='opsd-first5-em-loss-stats-inputs-v1',
                    pair_features=file_ref(targets['pair_feature']),
                    cohort_features=file_ref(targets['cohort_feature']),
                    extraction_summary=file_ref(summary_path),
                    root_execution=file_ref(out / 'execution.private.json'),
                    seed=42, cv_folds=5, bootstrap_replicates=1000)
    (out / 'stats-inputs.private.json').write_bytes(canonical(manifest))
    print(json.dumps(dict(accepted=True, counts=summary['counts'],
                          stats_manifest=file_ref(out / 'stats-inputs.private.json'))))


if __name__ == '__main__':
    main()
