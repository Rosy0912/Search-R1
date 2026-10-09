#!/usr/bin/env python3
"""Run frozen shared-vocabulary readers on HKUST files; never run a writer."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
ORDER = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
PINS = dict(
    collector='4dac0a6d607e4e62e31018ae63c85e7e9c8285cd238657c5edc60d66e8a7bff0',
    consumer='0123b06b13c6abb0554d0a825bb1a28565f192e0b046ef878feda4af35f8a2a0',
    aggregate='b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a',
    producer='f42fee77297897b1e58c066d12510422ea639eaabf1b2d359ddd0c692b923e7b',
    producer_manifest='8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41')
META = ('device', 'inode', 'bytes', 'mode', 'nlink', 'mtime_ns', 'ctime_ns')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def path(value):
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts and str(p) == value and
            BASE in p.parents and all(not q.is_symlink() for q in (p, *p.parents)),
            'Canonical nonsymlink HKUST study path required')
    return p


def metadata(s):
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                nlink=s.st_nlink, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read(ref, limit=64 << 20):
    p = path(ref['path']); before = metadata(p.stat())
    require(stat.S_ISREG(before['mode']) and before['bytes'] <= limit, 'Regular bounded input required')
    with p.open('rb') as handle:
        require(metadata(os.fstat(handle.fileno())) == before, 'Opened input changed')
        raw = handle.read(limit + 1)
        require(metadata(os.fstat(handle.fileno())) == before, 'Input changed during read')
    require(metadata(p.stat()) == before and len(raw) == ref['bytes'] and
            hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Input SHA/bytes/stat differ')
    require(all(ref[k] == before[k] for k in META if k in ref), 'Declared input metadata differs')
    return raw, before


def reference(p):
    p = path(str(p)); before = metadata(p.stat())
    require(stat.S_ISREG(before['mode']) and before['bytes'] <= 64 << 20, 'Small reference required')
    raw = p.read_bytes()
    require(before == metadata(p.stat()), 'Reference changed while reading')
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), **before)


def value(ref):
    return json.loads(read(ref)[0])


def same_ref(a, b):
    return all(a[k] == b[k] for k in ('path', 'sha256', 'bytes'))


def write_raw(p, raw):
    path(str(p)); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as handle:
        handle.write(raw)
    return reference(p)


def save(p, obj):
    return write_raw(p, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def closed_inputs(inputs):
    require(inputs['schema'] == 'opsd-ablation-remote-final-vocab-inputs-v1' and inputs['ready'] is True,
            'Explicit ready final inputs required')
    facts = value(inputs['facts'])
    require(facts['schema'] == 'opsd-ablation-outer-facts-v1' and facts['accepted'] is True and
            facts['coverage']['all70_complete'] is True and facts['coverage']['completed_outers'] == 70 and
            facts['coverage']['by_variant'] == {v: 10 for v in ORDER}, 'Actual complete all70 facts required')
    require(tuple(g['variant'] for g in inputs['groups']) == ORDER, 'Seven ordered explicit groups required')
    copied_groups = {}
    require(inputs['copy_batches'], 'Actual independent-copy executions required')
    for batch in inputs['copy_batches']:
        copies = value(batch['complete']); waited = value(batch['actual_wait'])
        require(copies['schema'] == 'opsd-ablation-remote-raw-copy-v1' and copies['accepted'] is True and
                waited['actual_ssh_returncode'] == 0 and
                waited['output'] == str(Path(batch['complete']['path']).parent),
                'Actual independent-copy complete and captured wait0 differ')
        for variant, ref in copies['groups'].items():
            closed = value(ref)
            require(variant in ORDER and variant not in copied_groups and closed['accepted'] is True and
                    closed['variant'] == variant and closed['independent_inode_copy'] is True,
                    'Repeated/unaccepted independent-copy group')
            copied_groups[variant] = closed
    require(set(copied_groups) == set(ORDER), 'Actual complete-copy batches do not cover all seven groups')
    groups = []
    for item in inputs['groups']:
        variant = item['variant']; config = value(item['config']); root = BASE / ('group-' + variant) / 'runs'
        require(config['variant'] == variant and config['schema'] == 'opsd-ablation-config-v1' and
                config['scientific']['outer_iterations'] == 10 and config['study_id'] == inputs['study_id'] and
                config['paths']['output_root'] == str(root), 'Original ten-outer config differs')
        require(any(same_ref(item['config'], ref) for ref in facts['configs']), 'Config absent from all70 facts')
        evaluation = value(item['evaluation_config'])
        require(same_ref(evaluation['training']['config'], item['config']) and
                evaluation['training']['run_id'] == config['run_id'], 'Bound evaluation training identity differs')
        # C0 reuses its original writer; the bound ref is the only terminal route.
        terminal = evaluation['training']['terminal']; read(terminal)
        rows = sorted((r for r in facts['outers'] if r['variant'] == variant), key=lambda r: r['outer'])
        require([r['outer'] for r in rows] == list(range(10)) and
                all(r['status'] == 'COMPLETE_FACTS' and r['run_id'] == config['run_id'] for r in rows),
                'Ten ordered complete original fact rows required')
        outers = [r['sources']['completion'] for r in rows]
        require(all(ref['path'] == str(root / 'outers' / ('outer-%05d' % i) / 'training/outer-complete.json')
                    for i, ref in enumerate(outers)), 'Original fact completion paths differ')
        coverage = item['coverage']
        require(coverage['kind'] == 'recorded' and coverage['config_sha256'] == item['config']['sha256'],
                'Actual recorded seven-group coverage required')
        for field in ('publication', 'details', 'derived_manifest'):
            read(coverage[field])
        require(coverage['backups'] and all(b['kind'] == 'direct' for b in coverage['backups']),
                'Actual accepted independent HKUST direct copies required')
        copied = copied_groups[variant]
        require(same_ref(copied['config'], item['config']) and copied['run_id'] == config['run_id'] and
                copied['backups'] == coverage['backups'], 'Coverage persistence is not the actual completed group copy')
        for backup in coverage['backups']:
            for field in ('source', 'local'):
                read(backup[field])
        groups.append(dict(variant=variant, run_id=config['run_id'], config=item['config'],
                           status=reference(root / 'status.json'), terminal=terminal,
                           outers=outers, coverage=coverage))
    return groups


def execute(role, command, output, stdin=None):
    stdout = output / (role + '.stdout.private.json'); stderr = output / (role + '.stderr')
    started = time.time()
    with stdout.open('xb') as out_handle, stderr.open('xb') as error_handle:
        child = subprocess.Popen(command, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=out_handle, stderr=error_handle,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        if stdin is not None:
            child.communicate(input=stdin)
            rc = child.returncode
        else:
            rc = child.wait()
    receipt = dict(argv=command, actual_child_pid=child.pid, actual_returncode=rc,
        actual_wait_returncode=rc, started_unix=started, ended_unix=time.time(), stdout=reference(stdout),
        stderr=reference(stderr), processing_host='HKUST1', reads_remote_local_filesystem=True,
        no_network=True, no_model_or_GPU=True, no_training_locks=True, no_signals=True)
    return receipt, save(output / (role + '-execution.json'), receipt)


def copy_bytes(original, raw, destination, expected_metadata):
    started = time.time(); observed, before = read(original)
    require(observed == raw and before == expected_metadata, 'Collected original bytes/metadata differ before copy')
    local = write_raw(destination, raw)
    require((local['device'], local['inode']) != (before['device'], before['inode']), 'Independent inode copy required')
    after_raw, after = read(original)
    require(after_raw == raw and after == before and local['sha256'] == original['sha256'] and
            local['bytes'] == original['bytes'], 'Source/local byte identity changed')
    return local, dict(actual_returncode=0, argv=list(sys.argv) + ['operation=independent-byte-copy'],
        source=original, local=local, source_before=before, source_after=after, started_unix=started,
        ended_unix=time.time(), processing_host='HKUST1', independent_inode_copy=True,
        no_external_exit_code_inferred=True, no_network=True, no_GPU=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'inputs-sha256', 'source-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(os.getuid() == os.geteuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            'torch' not in sys.modules and 'numpy' not in sys.modules, 'HKUST original UID1005 CPU-only execution required')
    own = reference(Path(__file__).absolute()); require(own['sha256'] == args.source_sha256, 'Wrapper source pin differs')
    input_ref = reference(args.inputs); require(input_ref['sha256'] == args.inputs_sha256, 'Input manifest pin differs')
    inputs = value(input_ref); sources = inputs['sources']
    for role, expected in PINS.items():
        require(sources[role]['sha256'] == expected, 'Frozen source pin differs: ' + role); read(sources[role])
    groups = closed_inputs(inputs)
    output = path(args.output)
    require(BASE / 'analysis' in output.parents and not output.exists(), 'New independent HKUST analysis directory required')
    output.mkdir(parents=True, exist_ok=False)
    executions = {}; started = time.time()
    request = dict(schema='opsd-ablation-final-vocab-collection-inputs-v1', study_root=str(BASE),
        study_id=inputs['study_id'], groups=[{k:v for k,v in g.items() if k != 'coverage'} for g in groups],
        accepted_source_inventories=[])
    request_ref = save(output / 'collection-request.private.json', request)
    receipt, execution_ref = execute('collection', [sys.executable, '-B', sources['collector']['path'],
        '--request-sha256', request_ref['sha256'], '--source-sha256', sources['collector']['sha256']],
        output, read(request_ref)[0])
    executions['collection'] = execution_ref
    inventory = value(receipt['stdout'])
    require(receipt['actual_wait_returncode'] == 0 and inventory['accepted'] is True and
            inventory['actual_read_returncode'] == 0 and inventory['fresh_sha_count'] == len(inventory['files']) and
            inventory['sha_reuse_count'] == 0, 'Fresh physical collector did not actually finish accepted')
    documents = []
    for index, doc in enumerate(inventory['collection_documents']):
        raw = base64.b64decode(doc['content_base64'], validate=True)
        original = doc['original']; source_meta = {k:original[k] for k in META}
        local, copied = copy_bytes(original, raw, output / 'documents' / ('%03d.json' % index), source_meta)
        proof = save(output / 'document-copies' / ('%03d-execution.json' % index), copied)
        documents.append(dict(original=original, local=local, copy_execution=proof))
    ledger_raw = base64.b64decode(inventory['ledger_content_base64'], validate=True)
    ledger_local, ledger_copy = copy_bytes(inventory['ledger'], ledger_raw,
        output / 'ledger-copy.private.json', inventory['ledger_source_before'])
    ledger_execution = save(output / 'ledger-copy-execution.json', ledger_copy)
    manifest = dict(schema='opsd-ablation-final-vocab-ledger-inputs-v1', study_id=inputs['study_id'],
        aggregate_source=sources['aggregate'], producer_source=sources['producer'],
        producer_manifest=sources['producer_manifest'], facts=inputs['facts'], documents=documents,
        groups=[{k:v for k,v in g.items() if k not in ('run_id', 'outers')} for g in groups],
        ledger=dict(original=inventory['ledger'], local=ledger_local, copy_execution=ledger_execution),
        physical_inventory=receipt['stdout'], physical_inventory_execution=execution_ref,
        classifications=inputs.get('classifications', []))
    manifest_ref = save(output / 'consumer-inputs.private.json', manifest)
    consumer, consumer_execution = execute('consumer', [sys.executable, '-B', sources['consumer']['path'],
        '--manifest', manifest_ref['path'], '--manifest-sha256', manifest_ref['sha256'],
        '--output', str(output / 'ledger-audit')], output)
    executions['consumer'] = consumer_execution
    audit_ref = reference(output / 'ledger-audit/audit.private.json'); audit = value(audit_ref)
    summary_ref = reference(output / 'ledger-audit/summary.json'); summary = value(summary_ref)
    require(consumer['actual_wait_returncode'] in (0, 2) and
            audit['actual_exit_code'] == consumer['actual_wait_returncode'], 'Actual consumer wait and audit differ')
    complete = save(output / 'complete.private.json', dict(schema='opsd-ablation-remote-final-vocab-execution-v1',
        accepted=consumer['actual_wait_returncode'] == 0 and audit.get('artifact_complete') is True,
        status=audit['status'], input=input_ref, wrapper_source=own, frozen_sources=sources,
        actual_executions=executions, physical_inventory=receipt['stdout'], ledger_copy=ledger_execution,
        actual_copy_batches=inputs['copy_batches'],
        consumer_inputs=manifest_ref, audit=audit_ref, consumer_summary=summary_ref,
        started_unix=started, ended_unix=time.time(), processing_host='HKUST1',
        all_private_payloads_remain_HKUST=True, Mac_raw_payloads=False, reads_remote_local_filesystem=True,
        no_network=True, no_model_or_GPU=True, no_training_locks=True, no_signals=True,
        original_scientific_sources_unchanged=True,
        semantics='Original consumer local-file I/O runs on HKUST1; no external Node tools/network calls.'))
    public = dict(summary, wrapper_schema='opsd-ablation-remote-final-vocab-summary-v1',
        actual_collection_wait=receipt['actual_wait_returncode'], actual_consumer_wait=consumer['actual_wait_returncode'],
        processing_host='HKUST1', Mac_raw_payloads=False, reads_remote_local_filesystem=True,
        original_sources_unchanged=True, wrapper_source_sha256=own['sha256'], execution_sha256=complete['sha256'],
        execution_semantics='Frozen readers use the HKUST1 local filesystem. No model/GPU, network, locks or signals.')
    public_ref = save(output / 'summary.json', public)
    print(json.dumps(dict(status=audit['status'], accepted=public.get('artifact_complete', False),
                         actual_returncode=consumer['actual_wait_returncode'], complete=complete, summary=public_ref)))
    return consumer['actual_wait_returncode']


if __name__ == '__main__':
    raise SystemExit(main())
