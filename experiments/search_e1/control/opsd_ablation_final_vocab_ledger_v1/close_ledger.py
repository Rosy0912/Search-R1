#!/usr/bin/env python3
"""Local final-ledger joins only; never import or execute the production writer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import types

VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
CAP = 24 * (1 << 30)
DISTRIBUTIONS = {'student_logp', 'teacher_logp', 'outer_old_logp', 'target_logp'}
AGGREGATE_SHA = 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a'
COVERAGE_SHA = '99e97c08e8eb1378f5b90b2437a826715d6568abb7d82c954a3923c5bc778fb9'
PRODUCER_SHA = 'f42fee77297897b1e58c066d12510422ea639eaabf1b2d359ddd0c692b923e7b'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value):
    require(type(value) is int and value >= 0, 'Expected nonnegative integer')
    return value


def path_name(value):
    p = PurePosixPath(value)
    require(p.is_absolute() and '..' not in p.parts and str(p) == value, 'Noncanonical original path')
    return p


def key(ref):
    path_name(ref['path'])
    require(re.fullmatch('[0-9a-f]{64}', ref['sha256']) and integer(ref['bytes']) > 0, 'Invalid fileproof')
    return (ref['path'], ref['sha256'], ref['bytes'])


def same_ref(left, right):
    return key(left) == key(right)


def identity(value):
    require(all(type(value.get(k)) is int for k in ('pid', 'uid', 'starttick')) and
            value['pid'] > 0 and value['starttick'] > 0 and isinstance(value.get('argv'), list), 'Opaque process identity')
    return tuple(value[k] for k in ('pid', 'uid', 'starttick', 'run_id')) + (tuple(value['argv']),)


def metadata(p):
    s = p.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read_bytes(ref):
    key(ref); p = Path(ref['path'])
    require(all(not q.is_symlink() for q in (p, *p.parents)) and stat.S_ISREG(p.stat().st_mode), 'Regular nonsymlink local input required')
    before = metadata(p); raw = p.read_bytes()
    require(before == metadata(p) and len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Input SHA/stat differs')
    return raw


def read_json(ref):
    return json.loads(read_bytes(ref))


def load_source(ref, expected, name):
    require(ref['sha256'] == expected, 'Unexpected reusable source')
    raw = read_bytes(ref); module = types.ModuleType(name); module.__file__ = ref['path']
    exec(compile(raw, ref['path'], 'exec'), module.__dict__)
    require('torch' not in sys.modules and 'numpy' not in sys.modules, 'Reader must remain stdlib only')
    return module


class Documents:
    """Original Node fileproof -> explicit immutable local copy, never original-path I/O."""
    def __init__(self, copies):
        self.copies = {}; self.cache = {}
        for item in copies:
            original, local = item['original'], item['local']
            require(key(original)[1:] == key(local)[1:] and original['path'] not in self.copies, 'Duplicate/different document copy')
            self.copies[original['path']] = item

    def read(self, original):
        item = self.copies[original['path']]
        require(same_ref(item['original'], original), 'Original document ref differs')
        if original['path'] not in self.cache:
            self.cache[original['path']] = read_json(item['local'])
        return self.cache[original['path']]


def audit_release(value, run, known=None):
    require(value['accepted'] is True and value['no_signals_sent'] is True, 'Unaccepted release')
    a = value['audit']
    require(a['schema'] == 'same-uid-owned-process-audit-v2' and a['run_id'] == run and
            a['accepted_no_owned_or_opaque'] is True and a['blockers'] == [] and
            a['no_exit_code_inferred_from_absence'] is True, 'Release audit not this clean run')
    if known is not None:
        require(a['uid']==known['uid'] and identity(known) in {identity(x) for x in a['known_owned']}, 'Controller absent from actual release anchors')
    return a['ended_unix']


def writer_closure(group, facts, docs):
    config = docs.read(group['config']); run = config['run_id']; variant = group['variant']
    require(config['schema'] == 'opsd-ablation-config-v1' and config['variant'] == variant and
            config['scientific']['outer_iterations'] == 10, 'Wrong original ten-outer config')
    output = path_name(config['paths']['output_root'])
    require(any(same_ref(group['config'], r) for r in facts['configs']), 'Config absent from accepted collector facts')
    contract = dict(direction='student_to_target_reverse', target_version='opsd-ablation-reverse-v1',
                    variant=variant, precision=config['precision']['mode'], mixture_alpha=.1 if variant in ('M', 'J') else 0.)
    rows = sorted((r for r in facts['outers'] if r['variant'] == variant), key=lambda r: r['outer'])
    require([r['outer'] for r in rows] == list(range(10)), 'Ten unique original completion facts required')
    updates = exposure = 0; previous = None
    for index, row in enumerate(rows):
        c = docs.read(row['sources']['completion']); n = row['counts']
        require(row['status'] == 'COMPLETE_FACTS' and row['run_id'] == run and row['contract'] == contract and
                c['schema'] == 'opsd-online-outer-complete-v1' and c['accepted'] is True and
                c['run_id'] == run and c['outer_index'] == index and c['contract'] == contract, 'Original completion identity differs')
        require(row['sources']['completion']['path'] == str(output / 'outers' / ('outer-%05d' % index) / 'training/outer-complete.json'), 'Completion path differs')
        K = integer(c['K']); require(K <= integer(c['P']) <= 512, 'Invalid K/P')
        delta = (K + 31) // 32
        require(n['optimizer_updates_before'] == updates and n['optimizer_updates_delta'] == c['optimizer_updates_delta'] == delta and
                n['optimizer_updates'] == c['optimizer_updates'] == updates + delta and n['K'] == K and n['P'] == c['P'] and
                n['valid_tokens'] == c['global_valid_tokens'] and c['cumulative_pair_exposure'] == exposure + K and
                n['cumulative_pair_exposure'] == exposure + K, 'Native completion counters/chain differ')
        require([u['update'] for u in row['updates']] == list(range(updates + 1, updates + delta + 1)) and
                sum(u['global_actual_pairs'] for u in row['updates']) == K and
                sum(u['global_valid_tokens'] for u in row['updates']) == c['global_valid_tokens'], 'Committed fact update coverage differs')
        require(same_ref(c['checkpoint'], row['sources']['checkpoint']), 'Completion checkpoint differs from accepted facts')
        if previous is not None:
            require(same_ref(c['outer_old_adapter'], previous), 'Outer-old does not bind previous canonical checkpoint')
        previous = c['checkpoint']; updates += delta; exposure += K
    status = docs.read(group['status']); terminal = docs.read(group['terminal'])
    require(status['schema'] == 'opsd-ablation-controller-status-v1' and status['phase'] == 'TRAIN_COMPLETE' and
            status['outer_iterations'] == 10 and status['variant'] == variant and status['run_id'] == run and
            status['config_sha256'] == group['config']['sha256'] and status['contract'] == contract and
            status['actual_optimizer_updates'] == updates and same_ref(status['latest_checkpoint'], previous), 'Native TRAIN_COMPLETE10 differs')
    require(terminal['schema'] == 'opsd-ablation-group-foreground-terminal-v1' and terminal['accepted'] is True and
            terminal['actual_exit_code'] == 0 and terminal['no_signals'] is True and terminal['signal_termination_not_exit0'] is False and
            terminal['run_id'] == run and terminal['config_sha256'] == group['config']['sha256'] and
            same_ref(terminal['status'], group['status']) and identity(terminal['controller']) == identity(status['owner']), 'Real foreground terminal differs')
    child = docs.read(terminal['actual_child_exit']); release = docs.read(terminal['actual_release'])
    require(child['actual_wait_returncode'] == 0 and child['run_id'] == run and child['no_signals'] is True and
            child['signal_termination_not_exit0'] is False and identity(child['identity']) == identity(status['owner']), 'Actual parent wait0/identity missing')
    own = status['resource_release']; audit_release(own['all_owned_release'], run)
    require(own['all_owned_release']['audit']['uid']==status['owner']['uid'] and
            own['all_owned_release']['audit']['controller_starttick']==status['owner']['starttick'], 'Native release owner UID/tick differs')
    require(own['resource_admission']['owned_gpu_processes'] == [], 'Native training still owns GPU processes')
    require(release['accepted'] is True and release['after']['accepted'] is True and release['after']['phase'] == 'cpu-merge' and
            release['after']['run_id'] == run and release['after']['owned_gpu_processes'] == [], 'Actual external full release absent')
    ended = max(audit_release(release['owned'], run, status['owner']), release['after']['checked_unix'])
    return config, dict(variant=variant, run_id=run, outers=10, optimizer_updates=updates, pair_exposure=exposure,
                        final_checkpoint=previous, controller=status['owner'], released_unix=ended)


def coverage_group(spec, config, helper):
    """Consume accepted coverage, not token logs. C0 has an earlier honest schema."""
    backups = helper.FixedBackups(spec['backups'], config['run_id'])
    if spec['kind'] == 'recorded':
        public = read_json(spec['publication']); details = read_json(spec['details']); derived = read_json(spec['derived_manifest'])
        require(public['accepted'] is True and details['accepted'] is True and derived['accepted'] is True and
                derived['source_sha256'] == COVERAGE_SHA and public['provenance']['source_sha256'] == COVERAGE_SHA and
                derived['artifacts']['details.private.json']['sha256'] == spec['details']['sha256'] and
                derived['artifacts']['debug-coverage-publication.json']['sha256'] == spec['publication']['sha256'], 'Coverage source/artifact chain differs')
        v = config['variant']; p = public['groups'][v]; d = details['groups'][v]
        require(p['status'] == 'COMPLETE' and p['run_id'] == config['run_id'] and d['config']['sha256'] == spec['config_sha256'] and
                p['full_vocabulary']['deterministic_preselection_verified'] is True and
                p['full_vocabulary']['all_committed_payload_values_finite'] is True, 'Incomplete/wrong recorded coverage')
        records = d['full_vocabulary_records']; checks = d['NPY_checks']
        require(len(records)==p['full_vocabulary']['requested_positions']<=2, 'Accepted preselection request count differs')
        for r in records:
            require(r['record']['identity']['run_id']==config['run_id'] and r['record']['identity']['variant']==v,
                    'Committed snapshot record belongs to another group')
    else:
        require(spec['kind'] == 'C0_legacy' and config['variant'] == 'C0', 'Unknown coverage schema')
        p = read_json(spec['publication']); finite = read_json(spec['finite_reader'])
        require(p['accepted'] is True and p['run_id'] == config['run_id'] and
                p['full_vocabulary']['actual_preselection_verified_from_committed_headers'] is True and
                finite['accepted'] is True and finite['run_id'] == config['run_id'] and
                finite['source']['finite_reader']['sha256'] == COVERAGE_SHA and
                finite['prior_coverage']['sha256'] == spec['publication']['sha256'], 'C0 selection/finite provenance differs')
        records = []
        for r in p['full_vocabulary']['records']:
            files = [dict(path=f['source_path'], sha256=f['sha256'], bytes=f['bytes'], distribution=f['distribution'],
                          dtype='float32', shape=f['shape']) for f in r['files']]
            records.append(dict(outer=r['outer'], pair_index=r['pair_index'], optimizer_update=r['optimizer_update'], token_index=r['token_index'],
                                record=dict(full_vocab_dump='complete', files=files)))
        checks = finite['files']
        require(len(records)==p['full_vocabulary']['requested_tokens']<=2, 'C0 request count differs')
    finite_refs = set()
    for c in checks:
        require(c['finite'] is True and c['fresh_sha256'] is True and c['shape'] == [151936] and c['dtype'] == '<f4' and
                c['values_checked'] == 151936 and c['normalized_or_KL_recomputed'] is False, 'Incomplete finite proof')
        finite_refs.add(key(c['ref']))
    return backups, records, finite_refs


def reconcile(ledger, physical, committed, recoverable, finite_refs, classifications, output_roots):
    """Pure small-record accounting; the tests exercise this join, never real arrays."""
    used = integer(ledger['bytes']); require(used <= CAP, 'Ledger exceeds cap')
    reservations = {}; ledger_files = []; actual = reserved = 0; classifications = dict(classifications)
    for r in ledger['reservations']:
        output = str(path_name(r['output'])); token = integer(r['token_index']); k = (output, token)
        require(k not in reservations and any(path_name(output).is_relative_to(path_name(root)) for root in output_roots), 'Duplicate/out-of-scope reservation')
        require(re.search(r'/training/attempt-[^/]+/rank-[0-7]/vocabulary-update-[0-9]{8}-retry-[0-9]{4}-pair-[0-9]{3}$', output), 'Unauthorized vocabulary output')
        require(r['status'] in ('complete', 'reserved') and integer(r['reserved_bytes']) > 0, 'Invalid reservation state')
        if r['status'] == 'complete':
            fs = r['files']; require(len(fs) == 4 and {f['distribution'] for f in fs} == DISTRIBUTIONS, 'Four distributions required')
            for f in fs:
                key(f); require(f['dtype'] == 'float32' and f['shape'] == [151936] and
                    f['path'] == output + '/token-%05d-%s.npy' % (token, f['distribution']), 'Complete file shape/name differs')
            n = sum(f['bytes'] for f in fs); require(n == integer(r['actual_bytes']) <= r['reserved_bytes'], 'Complete byte accounting differs')
            actual += n; ledger_files.extend(fs)
        else:
            require('actual_bytes' not in r and 'files' not in r, 'Reserved entry incorrectly claims complete files')
            reserved += r['reserved_bytes']
        reservations[k] = r
    refs = [key(f) for f in ledger_files]; require(len({x[0] for x in refs}) == len(refs), 'Duplicate file path')
    require(Counter(key(f) for f in ledger['files']) == Counter(refs), 'Global files/reservation union differs')
    require(used == actual + reserved, 'Ledger total does not equal conservative accounting')
    pm = {}; physical_ids = set(); physical_bytes = 0; per_reservation = Counter()
    for f in physical:
        path = str(path_name(f['path'])); require(path not in pm and stat.S_ISREG(f['mode']) and f['nlink'] == 1, 'Duplicate/link/special physical NPY')
        ik = (f['device'], f['inode']); require(ik not in physical_ids, 'Aliased original physical NPY identity')
        physical_ids.add(ik); pm[path] = f; physical_bytes += integer(f['bytes'])
        match = re.fullmatch(r'token-([0-9]{5})-(student_logp|teacher_logp|outer_old_logp|target_logp)\.npy', path_name(path).name)
        require(match is not None and (str(path_name(path).parent), int(match[1])) in reservations, 'Unregistered physical NPY')
        per_reservation[(str(path_name(path).parent), int(match[1]))] += f['bytes']
    require(physical_bytes <= used <= CAP, 'Physical bytes exceed conservative accounted bytes')
    for k, r in reservations.items():
        require(per_reservation[k] <= r['reserved_bytes'], 'Reservation physical payload exceeds budget')
    pending = []; committed_keys = set(); skips = 0
    for d in committed:
        require(d['outer'] == 0, 'Snapshot requested outside outer0')
        rec = d['record']; token = d['token_index']
        if rec['full_vocab_dump'] == 'cap_skip':
            require(rec['study_cap_bytes'] == CAP and 'files' not in rec, 'Invalid cap skip'); skips += 1; continue
        require(rec['full_vocab_dump'] == 'complete', 'Unknown committed dump state')
        fs = rec['files']; k = (str(path_name(fs[0]['path']).parent), token)
        require(k not in committed_keys and k in reservations and reservations[k]['status'] == 'complete', 'Committed reservation missing/duplicated')
        require(Counter(key(f) for f in fs) == Counter(key(f) for f in reservations[k]['files']), 'Wrong committed file join')
        committed_keys.add(k)
    for f in ledger_files:
        k = key(f)
        if f['path'] not in pm or pm[f['path']]['bytes'] != f['bytes'] or pm[f['path']].get('sha256') != f['sha256']:
            pending.append(dict(reason='complete_physical_identity_or_SHA_unresolved', ref=f))
        if k not in recoverable: pending.append(dict(reason='complete_array_not_persisted', ref=f))
        if k not in finite_refs: pending.append(dict(reason='complete_array_finite_proof_unresolved', ref=f))
    joined = []
    for k, r in reservations.items():
        kind = 'committed_complete' if k in committed_keys else 'uncommitted_complete' if r['status'] == 'complete' else 'residual_reserved'
        evidence = classifications.get(k)
        if kind != 'committed_complete' and evidence is None: pending.append(dict(reason='reservation_provenance_unexplained', output=k[0], token_index=k[1]))
        if kind == 'uncommitted_complete':
            pending.append(dict(reason='ledger_complete_not_committed_training', output=k[0], token_index=k[1]))
        if r['status'] == 'reserved':
            pending.append(dict(reason='residual_reserved_not_producer_complete', output=k[0], token_index=k[1]))
        if r['status'] == 'reserved' and per_reservation[k]:
            pending.append(dict(reason='preserved_partial_reserved_payload_not_complete', output=k[0], token_index=k[1]))
        joined.append(dict(output=k[0], token_index=k[1], classification=kind, physical_bytes=per_reservation[k], provenance=evidence))
    require(set(classifications) <= set(reservations), 'Classification for unknown reservation')
    return dict(accounting_complete=True, committed_snapshot_coverage_complete=True,
                recoverable_complete_arrays=not any(x['reason']=='complete_array_not_persisted' for x in pending),
                artifact_complete=not pending, ledger_accounted_bytes=used, complete_NPY_bytes=actual,
                reserved_capacity_bytes=reserved, physical_unique_NPY_bytes=physical_bytes,
                complete_reservations=sum(r['status']=='complete' for r in reservations.values()),
                residual_reserved=sum(r['status']=='reserved' for r in reservations.values()),
                complete_array_files=len(refs), physical_unique_array_files=len(pm), committed_saved_positions=len(committed_keys),
                committed_cap_skip_positions=skips, pending=pending, reservation_index=joined)


def run(manifest):
    require(manifest['schema'] == 'opsd-ablation-final-vocab-ledger-inputs-v1', 'Explicit final-only inputs required')
    helper = load_source(manifest['aggregate_source'], AGGREGATE_SHA, '_ledger_backup_reader')
    require(manifest['producer_source']['sha256'] == PRODUCER_SHA, 'Wrong immutable producer'); read_bytes(manifest['producer_source'])
    facts = read_json(manifest['facts']); require(facts['schema']=='opsd-ablation-outer-facts-v1' and facts['accepted'] is True and
        facts['coverage']['completed_outers']==70 and facts['coverage']['all70_complete'] is True and
        facts['coverage']['by_variant']=={v:10 for v in VARIANTS}, 'Accepted original all70 facts required')
    docs = Documents(manifest['documents']); groups = manifest['groups']
    require(tuple(g['variant'] for g in groups) == VARIANTS, 'Exactly seven ordered groups required')
    configs = {}; closures = []; roots = []; dumps = []; finite_refs = set(); recoverable = set(); source_files = {}
    for group in groups:
        config, close = writer_closure(group, facts, docs); configs[group['variant']] = config; closures.append(close)
        roots.append(config['paths']['output_root']); backs, records, finite = coverage_group(group['coverage'], config, helper)
        require(group['coverage']['config_sha256'] == group['config']['sha256'], 'Coverage config differs')
        dumps.extend(records); finite_refs |= finite
        for source, local in backs.files.values():
            if not source['path'].endswith('.npy'): continue
            require(source['path'] not in source_files, 'Repeated original array across group inputs')
            source_files[source['path']] = source
            if source['bytes'] > 0 and (local['kind'] != 'direct' or helper.metadata(local['path'].stat()) == local['metadata']):
                recoverable.add(key(source))
    require(len({c['run_id'] for c in closures}) == 7, 'Distinct original group runs required')
    require({c['study_id'] for c in configs.values()} == {manifest['study_id']}, 'Different actual study IDs')
    for field in ('source_model_manifest','runtime_manifest','common_initial_adapter_manifest','common_cohort_manifest'):
        require(len({c[field]['sha256'] for c in configs.values()})==1, 'Different common source/runtime/initial/cohort')
    producer_manifest = read_json(manifest['producer_manifest'])
    require(manifest['producer_manifest']['sha256']=='8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41' and
        {c['helper_source_manifest']['sha256'] for c in configs.values()}=={manifest['producer_manifest']['sha256']} and
        producer_manifest['files']['vocabulary_dump.py']['sha256']==PRODUCER_SHA, 'Original frozen helper manifest differs')
    shared = {c['logging']['full_vocab_study_root'] for c in configs.values()}; require(len(shared)==1, 'Different shared ledger roots')
    shared = shared.pop(); ledger_ref = manifest['ledger']['original']; require(ledger_ref['path'] == shared+'/ledger.json', 'Wrong ledger path')
    copy = read_json(manifest['ledger']['copy_execution']); local = manifest['ledger']['local']
    require(copy['actual_returncode']==0 and same_ref(copy['source'], ledger_ref) and same_ref(copy['local'], local) and
            copy['source_before']==copy['source_after'] and copy['started_unix'] >= max(c['released_unix'] for c in closures) and
            key(ledger_ref)[1:]==key(local)[1:], 'Ledger copy not stable after all writers closed')
    require(all(copy['source_before'][k]==ledger_ref[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns','mode')),
            'Copy source metadata not bound to final original ledger')
    ledger = read_json(local); inventory = read_json(manifest['physical_inventory'])
    ie = read_json(manifest['physical_inventory_execution'])
    require(ie['actual_returncode']==0 and isinstance(ie['argv'],list) and ie['argv'] and
            same_ref(ie['stdout'], manifest['physical_inventory']), 'Actual physical-inventory read execution missing')
    require(inventory['schema']=='opsd-ablation-final-vocab-physical-inventory-v1' and inventory['accepted'] is True and
            inventory['actual_read_returncode']==0 and inventory['started_unix'] >= max(c['released_unix'] for c in closures) and
            sorted(inventory['scanned_roots'])==sorted(r+'/outers' for r in roots) and
            sorted(inventory['run_ids'])==sorted(c['run_id'] for c in closures) and inventory['source_metadata_stable'] is True,
            'Final physical inventory not exact closed study scope')
    require(same_ref(inventory['ledger'],ledger_ref) and inventory['ledger_source_before']==copy['source_before'] and
            inventory['ledger_source_after']==copy['source_after'], 'Physical scan and ledger are not one stable closed snapshot')
    for f in inventory['files']:
        old = source_files.get(f['path'])
        if old is not None:
            require(all(f[k]==old[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns','mode','sha256')),
                    'Final original array metadata changed from accepted backup inventory')
    classified = {}
    for item in manifest.get('classifications', []):
        k = (item['output'], item['token_index']); require(k not in classified, 'Duplicate classification')
        evidence = [dict(ref=r, value=read_json(r)) for r in item['evidence']]
        require(evidence and item['kind'] in ('failed_attempt', 'retry_abandoned', 'committed_record_write_interrupted', 'inert_reserved'), 'Actual attempt evidence required')
        # This is a declared interpretation, not a new parser of every historical
        # failure schema. Uncommitted complete/partial entries remain PARTIAL.
        classified[k] = dict(kind=item['kind'], reason=item['reason'], evidence=[e['ref'] for e in evidence], inferred_completion=False)
    result = reconcile(ledger, inventory['files'], dumps, recoverable, finite_refs, classified, roots)
    result.update(writer_quiescent=True, closed_outers=70, writers=closures, actual_final_ledger=ledger_ref,
                  status='COMPLETE' if result['artifact_complete'] else 'PARTIAL')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('manifest', 'manifest-sha256', 'output'): p.add_argument('--'+option, required=True)
    args = p.parse_args(); out = Path(args.output)
    require(out.is_absolute() and '..' not in out.parts and not out.exists() and all(not q.is_symlink() for q in out.parents), 'New local output only')
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__); raw = source.read_bytes(); source_sha = hashlib.sha256(raw).hexdigest()
    try:
        mref = dict(path=args.manifest, sha256=args.manifest_sha256, bytes=Path(args.manifest).stat().st_size)
        value = run(read_json(mref)); rc = 0 if value['artifact_complete'] else 2
    except (ValueError, KeyError, TypeError, OSError) as error:
        value = dict(status='PARTIAL', artifact_complete=False, writer_quiescent=False,
                     error_type=type(error).__name__, reason=str(error)); rc = 2
    private = dict(schema='opsd-ablation-final-vocab-ledger-audit-v1', CLI_argv=sys.argv, source_sha256=source_sha,
                   input_sha256=args.manifest_sha256, actual_exit_code=rc, **value)
    (out/'audit.private.json').write_text(json.dumps(private, indent=2, allow_nan=False)+'\n')
    public = {k:v for k,v in private.items() if k not in ('CLI_argv','writers','actual_final_ledger','reservation_index','pending','reason')}
    public.update(pending_reasons=dict(Counter(x['reason'] for x in private.get('pending', []))),
                  limitations=['Final local references and historical finite/member-SHA proofs only; no NPY payload reread in this CLI.',
                               'No distribution normalization, KL, gradients, generation scores or GPU resume verified.',
                               'No producer, Node/network, locks, reservation refund or repair; formal5632 is separate.'])
    (out/'summary.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    artifacts = {n:dict(bytes=len((out/n).read_bytes()), sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in ('audit.private.json','summary.json')}
    (out/'derivation-manifest.json').write_text(json.dumps(dict(source_sha256=source_sha,input_sha256=args.manifest_sha256,artifacts=artifacts,actual_exit_code=rc),indent=2)+'\n')
    print(json.dumps(dict(status=private['status'],artifact_complete=private['artifact_complete'],actual_exit_code=rc)))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
