"""Preserve the exact current attempt before a future native resume replaces it.

Archiving is safe while the allocation runs. It does not admit a restart or
prove completion; those need terminal scheduler and full-state lineage checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from .prepare_ablation import condition_for


def validate(record, entry, run, config_bytes, log_text, scheduler_text, expected_job):
    run = Path(run).resolve()
    if (hashlib.sha256(config_bytes).hexdigest() != record['config_sha256']
            or record['condition'] != condition_for(record['group'])
            or Path(record['base']) != run or record['execution_backend'] != 'slurm_lumia'):
        raise ValueError('Current immutable configuration does not identify this study run')
    if (str(entry['allocation']) != str(expected_job) or entry['uid'] != 1066
            or any(entry[k] != record[k] for k in ('run_id', 'group', 'source_commit', 'config_sha256'))
            or Path(entry['output']).parent != run / 'output'
            or re.fullmatch(r'attempt-[0-9a-f]{32}', Path(entry['output']).name) is None):
        raise ValueError('Current entry is not the exact owned attempt')
    matched = []
    for line in log_text.splitlines():
        if 'LUMIA_ABLATION_ENTRY=' in line:
            value = json.JSONDecoder().raw_decode(line.split('LUMIA_ABLATION_ENTRY=', 1)[1])[0]
            if str(value['allocation']) == str(expected_job):
                matched.append(value)
    if matched != [entry]:
        raise ValueError('Actual launch log does not preserve this exact entry')
    rows = [r.split('|') for r in scheduler_text.splitlines() if r.strip()]
    matching = [r for r in rows if r[0] == str(expected_job)]
    if len(matching) != 1 or matching[0][1] != 'jmzhang':
        raise ValueError('Scheduler does not identify the exact owned allocation')
    return matching[0][2]


def archive(run, log, expected_job, query=None):
    run = Path(run).resolve()
    configuration = run / 'receipts/configuration.json'
    entry_path = run / 'receipts/current-entry.json'
    config_record_bytes = configuration.read_bytes()
    entry_bytes = entry_path.read_bytes()
    record = json.loads(config_record_bytes)
    entry = json.loads(entry_bytes)
    config_bytes = Path(record['config']).read_bytes()
    log = Path(log).resolve()
    log_text = log.read_text(errors='replace')
    if query is None:
        query = lambda job: subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job),
            '--format=JobIDRaw,User,State,ExitCode'], text=True)
    scheduler = query(expected_job)
    state = validate(record, entry, run, config_bytes, log_text, scheduler, expected_job)
    # Recheck after reading the launch log: never archive across a replaced entry.
    if entry_path.read_bytes() != entry_bytes or configuration.read_bytes() != config_record_bytes:
        raise ValueError('Entry/configuration changed during the read-only capture')
    entry_sha = hashlib.sha256(entry_bytes).hexdigest()
    payload = dict(version=1, job=str(expected_job), entry=entry, entry_sha256=entry_sha,
        configuration=record, configuration_record_sha256=hashlib.sha256(config_record_bytes).hexdigest(),
        resolved_configuration_sha256=record['config_sha256'],
        entry_original_text=entry_bytes.decode(), configuration_original_text=config_record_bytes.decode(),
        log=str(log), log_entry_exactly_matched=True,
        scope='Immutable attempt identity only. No terminal state, training completion or permission to restart is implied.')
    directory = run / 'receipts/entry-history'
    directory.mkdir(exist_ok=True)
    path = directory / (str(expected_job) + '-' + entry_sha + '.json')
    try:
        with path.open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    except FileExistsError:
        if json.loads(path.read_text()) != payload:
            raise ValueError('Existing immutable entry archive disagrees')
    return dict(at=time.time(), archive=str(path), archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job=str(expected_job), observed_scheduler_state=state, entry_sha256=entry_sha,
        restart_authorized_by_this_receipt=False, scope=payload['scope'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.run, args.log, args.job)), flush=True)


if __name__ == '__main__':
    main()
