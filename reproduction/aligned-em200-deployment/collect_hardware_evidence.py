"""Read-only job3345 observations and terminal raw archive; never an acceptance."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE = '080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
JOB = '3345'
TOKEN = '13e07285c97044fd97974d94f53fea1b'
COMMENT = 'BAPO_ALIGNED_EM200:hardware:' + SOURCE + ':' + TOKEN
RUN_NAME = 'hardware-aligned-em200-' + TOKEN + '-' + JOB
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}
REQUIRED_RUN = [
    'source-commit.txt', 'hardware.log', 'numeric-fixture.log', 'resource-watch.jsonl',
    *['numeric-fixture/rank-%d.json' % r for r in range(8)],
    'retrieval-golden.log', 'retrieval-golden.json', 'GPU-GOLDEN-CAPTURE-CLAIM.json',
    'capture-completed-before-live.json', 'own-retriever-birth.json', 'retriever.log',
    'live-retrieval-acceptance.json', 'own-process-cleanup.json', 'own-process-cleanup.err',
    'gpu-exit.csv', 'gpu-exit.err', 'original-exit-code.txt', 'cleanup-exit-code.txt']


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def execute(argv):
    """Only caller-supplied fixed read commands; raw nonzero/timeout is retained."""
    record = dict(command=argv, start=now())
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
        record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        record.update(exit_code=None, stdout=decoded(getattr(error, 'stdout', None)),
                      stderr=decoded(getattr(error, 'stderr', None)),
                      observation_error=type(error).__name__, observation_error_detail=str(error))
    record['end'] = now()
    return record


def observe(query=execute):
    records = dict(
        scontrol=query(['scontrol', 'show', 'job', '-o', JOB]),
        sacct=query(['sacct', '-j', JOB, '-X', '--noheader', '--parsable2',
                     '--format=JobIDRaw,JobName,State,ExitCode,NodeList,Comment%200,Start,End,Submit']))
    control_primary, accounting_primary, issues, unavailable = None, None, [], []
    control = records['scontrol']
    if control['exit_code'] == 0:
        raw = control['stdout']
        values = dict(re.findall(r'(\w+)=([^\s]*)', raw))
        command = re.search(r'(?:^|\s)Command=(.*?)(?=\s+\w+=|$)', raw)
        expected_command = str(ROOT / 'tools/deployment/hardware.sbatch') + ' ' + TOKEN
        required = {'JobId', 'JobName', 'Comment', 'WorkDir', 'JobState', 'ExitCode', 'EndTime', 'SubmitTime', 'StartTime'}
        node = values.get('NodeList', values.get('BatchHost'))
        if (not required <= values.keys() or node is None or values.get('JobId') != JOB
                or values.get('JobName') != 'bapo-aligned-em200-hardware'
                or values.get('Comment') != COMMENT or values.get('WorkDir') != str(ROOT)
                or command is None or command.group(1).strip() != expected_command
                or values.get('ReqNodeList') != 'h100-5'
                or node not in ('h100-5', '(null)', 'None')):
            issues.append('scontrol strong original job/source/token/command/root/node binding differs')
        else:
            control_primary = dict(job=JOB, name=values['JobName'], state=values['JobState'],
                state_raw=values['JobState'], exit_code=values['ExitCode'], node=node,
                comment=values['Comment'], start=values['StartTime'], end=values['EndTime'],
                submit=values['SubmitTime'], command=expected_command, workdir=str(ROOT),
                required_node='h100-5', authority='scontrol')
    else:
        unavailable.append('scontrol unavailable; raw error retained')
    accounting = records['sacct']
    if accounting['exit_code'] == 0:
        for line in [line for line in accounting['stdout'].splitlines() if line.strip()]:
            fields = line.split('|')
            if len(fields) == 10 and fields[-1] == '':
                fields.pop()
            if len(fields) != 9 or not fields[2].strip():
                issues.append('unexpected sacct field shape/state')
                continue
            job, name, state, exit_code, node, comment, start, end, submit = fields
            if job == JOB:
                if accounting_primary is not None:
                    issues.append('duplicate primary sacct record')
                accounting_primary = dict(job=job, name=name, state=state.split()[0].rstrip('+'),
                    state_raw=state, exit_code=exit_code, node=node, comment=comment,
                    start=start, end=end, submit=submit, authority='sacct')
        if accounting_primary and (accounting_primary['name'] != 'bapo-aligned-em200-hardware'
                                   or accounting_primary['comment'] != COMMENT
                                   or accounting_primary['node'] not in ('h100-5', '(null)', 'None')):
            issues.append('sacct original job/source/token identity differs')
    else:
        # Disabled accounting is absence of a secondary observation, not job failure.
        unavailable.append('sacct unavailable/disabled; raw error retained')
    if control_primary and accounting_primary and any(control_primary[k] != accounting_primary[k]
            for k in ('job', 'name', 'state', 'exit_code', 'comment', 'start', 'end', 'submit')):
        issues.append('Slurm observations disagree; recollect after transition')
    parsed = control_primary or accounting_primary
    if parsed is None:
        issues.append('no strongly bound original scheduler record')
    terminal = bool(parsed and parsed['state'] in TERMINAL and not issues)
    if terminal:
        try:
            def timestamp(value):
                require(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value), 'invalid scheduler timestamp')
                return datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
            end, submit = timestamp(parsed['end']), timestamp(parsed['submit'])
            require(end >= submit and re.fullmatch(r'\d+:\d+', parsed['exit_code']), 'invalid terminal timeline/exit code')
            if parsed['state'] == 'COMPLETED':
                require(submit <= timestamp(parsed['start']) <= end, 'completed job start/end timeline differs')
        except (ValueError, TypeError):
            issues.append('terminal timestamp/timeline/exit code unavailable or invalid')
            terminal = False
    return dict(observed_utc=now(), records=records, primary=parsed, issues=issues,
                unavailable_secondary=unavailable, terminal_verified=terminal, acceptance=False,
                status='TERMINAL_OBSERVED_NOT_ACCEPTED' if terminal else 'NOT_TERMINAL_OR_UNVERIFIED_NOT_ACCEPTED')


def copy_regular(source, destination):
    """Copy bytes once; refuse symlinks/special files and changing originals."""
    source = Path(source)
    require(source.resolve() == source, 'noncanonical evidence source')
    with source.open('rb') as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode), 'not a regular evidence file')
        require((before.st_dev, before.st_ino) == (source.lstat().st_dev, source.lstat().st_ino), 'source replaced')
        destination.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with destination.open('xb') as out:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        after = os.fstat(handle.fileno())
        final = source.lstat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(fields(before) == fields(after) == fields(final), 'evidence changed while copying')
    return dict(source=str(source), archive=str(destination), bytes=before.st_size,
                sha256=h.hexdigest(), device=before.st_dev, inode=before.st_ino,
                mtime_ns=before.st_mtime_ns)


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def archive_terminal(root, output, observation, query=execute):
    """Factual archive only. Internal roots enable CPU fixtures, not CLI retargeting."""
    require(observation['terminal_verified'], 'nonterminal collection has no terminal archive')
    run = root / 'runs' / RUN_NAME
    require(root.resolve() == root and output.parent == root / 'evidence'
            and output.parent.resolve() == output.parent and not output.is_symlink(), 'own canonical archive destination required')
    output.mkdir()  # Exclusive; no existing snapshot or original is overwritten.
    files, problems = [], []
    write_json(output / 'slurm-before.json', observation)
    candidates = [(root / 'scheduler-head' / n, Path('scheduler-head') / n) for n in
                  ('hardware-submit-intent.json', 'hardware-submit-receipt.json', 'hardware-sbatch-response.json')]
    candidates += [(root / 'logs' / ('slurm-hardware-' + JOB + suffix), Path('logs') / ('slurm-hardware-' + JOB + suffix))
                   for suffix in ('.out', '.err')]
    if run.is_dir() and run.resolve() == run:
        for path in sorted(run.rglob('*')):
            if path.is_dir():
                continue
            relative = path.relative_to(run)
            if str(relative) not in set(REQUIRED_RUN) | {'identity-ready.tmp'}:
                problems.append('excluded unexpected artifact (no asset hashing): ' + str(path))
                continue
            candidates.append((path, Path('run') / relative))
    else:
        problems.append('original own RUN absent or noncanonical (possible pre-start terminal failure)')
    # Original tiny launcher/validator code is retained; no asset corpus/index/model is followed or rehashed.
    deployment = root / 'tools' / 'deployment'
    names = ['hardware.sbatch', 'hardware_worker.sh', 'hardware_inner.sh', 'submit_hardware.py',
             'capture_retrieval_golden.py', 'verify_live_retrieval.py', 'bind_retrieval_proof.py',
             'numerics_gpu_fixture.py', 'reference/retrieval_server.py', 'reference/retrieval-provenance.json']
    candidates += [(deployment / n, Path('tools') / n) for n in names]
    candidates += [(root / 'tools' / n, Path('tools') / n) for n in ('resource_watch.py', 'owned_processes.py')]
    candidates.append((Path(__file__).resolve(), Path('tools/collect_hardware_evidence.py')))
    for source, relative in candidates:
        try:
            if not source.exists() and not source.is_symlink():
                problems.append('missing: ' + str(source))
                continue
            files.append(copy_regular(source, output / relative))
        except (OSError, ValueError) as error:
            problems.append('not immutable/copied: ' + str(source) + ': ' + str(error))
    archived = {record['archive'] for record in files}
    missing = [n for n in REQUIRED_RUN if str(output / 'run' / n) not in archived]
    # Scheduler receipt is the original fixed submission, not a new job/receipt claim.
    receipt = output / 'scheduler-head/hardware-submit-receipt.json'
    try:
        value = json.loads(receipt.read_text())
        require(value['job'] == JOB and value['token'] == TOKEN and value['source'] == SOURCE
                and value['root'] == str(root) and value['comment'] == COMMENT and value['run'] == str(run), 'original submission receipt differs')
    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append('submission binding unavailable: ' + str(error))
    after = observe(query)
    write_json(output / 'slurm-after.json', after)
    if not after['terminal_verified'] or after['primary'] != observation['primary']:
        problems.append('terminal Slurm identity/state changed during archive')
    inventory = []
    valid = {record['archive']: record for record in files}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            record = valid.get(str(path))
            if record is None:
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        h.update(block)
                record = dict(sha256=h.hexdigest(), bytes=path.stat().st_size)
            inventory.append(dict(relative_path=str(path.relative_to(output)), bytes=record['bytes'],
                                  sha256=record['sha256'], validated_original_copy=str(path) in valid))
    manifest = dict(schema='bapo-hardware-terminal-raw-archive-v1', source_commit=SOURCE,
                    root=str(root), job=JOB, token=TOKEN, run=str(run), observed_utc=now(),
                    collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    slurm_primary=observation['primary'], files=files, archive_inventory=inventory,
                    manifest_self_hash='printed separately; file-manifest.json excluded from its own inventory',
                    missing_run_evidence=missing,
                    archival_issues=problems, expected_numeric_ranks=list(range(8)),
                    acceptance=False, independent_review_receipt=False,
                    status='TERMINAL_RAW_ARCHIVED_NOT_ACCEPTED',
                    scope='ORIGINAL_JOB3345_HARDWARE_EVIDENCE_ARCHIVE_NOT_GPU_OR_TRAINING_ACCEPTANCE',
                    asset_policy='retain original golden asset SHA pins; never rehash corpus/index/encoder')
    write_json(output / 'file-manifest.json', manifest)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New exclusive directory directly under fixed ROOT/evidence; required only at terminal')
    args = parser.parse_args()
    require(ROOT.resolve() == ROOT and sys.prefix == str(ROOT / 'env')
            and os.uname().nodename == 'h100-5', 'actual fixed target/root/environment required')
    require(subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'fixed source080 differs')
    observation = observe()
    if not observation['terminal_verified']:
        print(json.dumps(observation, sort_keys=True))
        return
    require(args.output is not None, 'terminal archive needs an explicit new own output directory')
    result = archive_terminal(ROOT, args.output, observation)
    print(json.dumps(dict(status=result['status'], acceptance=False, job=JOB, state=result['slurm_primary']['state'],
                          path=str(args.output), manifest_sha256=hashlib.sha256((args.output / 'file-manifest.json').read_bytes()).hexdigest(),
                          missing_run_evidence=result['missing_run_evidence'], archival_issues=result['archival_issues']), sort_keys=True))


if __name__ == '__main__':
    main()
