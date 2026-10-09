#!/usr/bin/env python3
"""Mac wrapper: preserve actual SSH wait/stdout and exact collected ledger bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from collect import META, REQUEST_LIMIT, STUDY_ROOT, canonical, json_value, reference, require


def fileproof(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def capture(argv, request_raw, output, expected_source_sha):
    output = canonical(str(output))
    require(not output.exists() and all(not p.is_symlink() for p in output.parents), 'New nonsymlink local output required')
    output.mkdir(parents=True)
    started = time.time()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(request_raw)
    sp, ep = output / 'physical-inventory.stdout.json', output / 'ssh.stderr'
    sp.write_bytes(stdout); ep.write_bytes(stderr)
    actual = dict(actual_returncode=process.returncode, argv=argv, popen_pid=process.pid,
                  started_unix=started, ended_unix=time.time(), stdout=fileproof(sp), stderr=fileproof(ep),
                  actual_wait=True, no_exit_code_inferred_from_absence=True, no_signals_sent=True)
    execution = output / 'ssh-execution.private.json'; write_json(execution, actual)
    require(process.returncode == 0, 'Actual SSH wait was nonzero; preserve HOLD evidence only')
    result = json_value(stdout)
    require(result['schema'] == 'opsd-ablation-final-vocab-physical-inventory-v1' and result['accepted'] is True and
            result['actual_read_returncode'] == 0 and result['source_metadata_stable'] is True and
            result['request_sha256'] == hashlib.sha256(request_raw).hexdigest() and
            result['collector_source']['sha256'] == expected_source_sha, 'Collector did not return the pinned accepted read')
    ledger = reference(result['ledger']); raw = base64.b64decode(result['ledger_content_base64'], validate=True)
    require(len(raw) == ledger['bytes'] and hashlib.sha256(raw).hexdigest() == ledger['sha256'] and
            result['ledger_source_before'] == result['ledger_source_after'] and
            all(result['ledger_source_before'][k] == ledger[k] for k in META), 'Exact stable ledger copy differs')
    documents = []
    for index, item in enumerate(result['collection_documents']):
        ref = reference(item['original']); content = base64.b64decode(item['content_base64'], validate=True)
        require(len(content) == ref['bytes'] and hashlib.sha256(content).hexdigest() == ref['sha256'], 'Closure copy bytes differ')
        documents.append((index, ref, content))
    lp = output / 'ledger.original.json'; lp.write_bytes(raw); local = fileproof(lp)
    dp = output / 'documents'; dp.mkdir()
    copies = []
    for index, ref, content in documents:
        dest = dp / ('%03d.json' % index); dest.write_bytes(content)
        copies.append(dict(original=ref, local=fileproof(dest)))
    # These accepted execution receipts are produced only after actual wait0 and
    # verified exact local byte copies. ssh-execution exists even on failure.
    write_json(output / 'physical_inventory_execution.private.json', actual)
    write_json(output / 'ledger-copy-execution.private.json', dict(actual_returncode=process.returncode,
               operation='Extract exact ledger bytes from actual SSH stdout', argv=argv,
               ssh_execution=fileproof(execution), source=ledger, local=local,
               source_before=result['ledger_source_before'], source_after=result['ledger_source_after'],
               started_unix=result['started_unix'], actual_wait=True, no_exit_code_inferred_from_absence=True))
    write_json(output / 'consumer-copy-refs.private.json', dict(documents=copies,
               ledger=dict(original=ledger, local=local, copy_execution=fileproof(output / 'ledger-copy-execution.private.json')),
               physical_inventory=fileproof(sp), physical_inventory_execution=fileproof(output / 'physical_inventory_execution.private.json')))
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('request', 'request-sha256', 'remote-collector', 'source-sha256', 'output'):
        p.add_argument('--' + option, required=True)
    args = p.parse_args()
    try:
        request = canonical(args.request)
        require(all(not q.is_symlink() for q in (request, *request.parents)), 'Nonsymlink local request required')
        before = request.stat(); require(before.st_size <= REQUEST_LIMIT, 'Request too large')
        raw = request.read_bytes()
        require(request.stat() == before and hashlib.sha256(raw).hexdigest() == args.request_sha256, 'Request changed/SHA differs')
        remote = canonical(args.remote_collector)
        require(STUDY_ROOT / 'source' in remote.parents and remote.name == 'collect.py', 'Published fixed-study collector path required')
        command = shlex.join(['python3', '-B', str(remote), '--request-sha256', args.request_sha256, '--source-sha256', args.source_sha256])
        capture(['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', 'hkust1', command], raw, Path(args.output), args.source_sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('HOLD: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
