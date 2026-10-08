#!/usr/bin/env python3
"""Local, immutable-backup-only aggregation of explicitly committed C0 token logs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

FACT_SHA = 'b519d7d3a540e3fbea27f93a7e9d6e0149b268714af1fe7f33ec614574bcb982'
RUN = 'bfc195746b6d41aaa74db5ac4d7a8be7'
MASS_KEYS = {k: f'clipped_{k}_probability_mass' for k in ('student', 'teacher', 'mixture')}
LOSS_KEYS = {'raw': 'pre_clip_full_vocabulary_kl', 'post': 'post_clip_full_vocabulary_loss'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metadata(s):
    return dict(bytes=s.st_size, device=s.st_dev, inode=s.st_ino, mode=s.st_mode,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def stable_json(path, expected_sha=None):
    path = Path(path)
    before = metadata(path.stat())
    with path.open('rb') as f:
        require(metadata(os.fstat(f.fileno())) == before, 'JSON opened identity changed')
        raw = f.read()
        require(metadata(os.fstat(f.fileno())) == before, 'JSON changed during read')
    require(metadata(path.stat()) == before, 'JSON path changed during read')
    digest = hashlib.sha256(raw).hexdigest()
    require(expected_sha is None or digest == expected_sha, 'JSON SHA mismatch: ' + str(path))
    return json.loads(raw), dict(path=str(path.resolve()), sha256=digest, **before)


def finite(value, field):
    require(type(value) in (int, float) and math.isfinite(value), 'non-finite/non-numeric: ' + field)
    return float(value)


def equal_fp32_log(a, b):
    # Logged token scalars originate from FP32 tensors. Native recorded scalar
    # summaries sum Python lists chunk by chunk; this local sum is sequential.
    # Neither is a reconstruction of the differentiable loss/GEMM/reduction.
    return math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6)


class FixedBackups:
    def __init__(self, execution_root):
        self.files = {}
        self.inputs = []
        self.reads = []
        self.seen = set()
        for suffix in ('0-7', '8-9'):
            root = execution_root / f'raw-c0-closed{suffix}-direct-20261009'
            source, source_proof = stable_json(root / 'source-full-sha-private.json')
            local, local_proof = stable_json(root / 'local-full-sha-private.json')
            require(source['accepted'] is True and source['all_source_pinned_metadata_stable'] is True,
                    'source inventory unaccepted')
            require(source['run_id'] == RUN, 'source inventory run differs')
            require(local['accepted'] is True and local['exact_inventory'] is True and
                    local['local_metadata_stable'] is True and local['no_symlinks_or_special_files'] is True
                    and local['source_sha_matched_all_files'] is True, 'local inventory unaccepted')
            source_map = {r['relative']: r for r in source['files']}
            local_map = {r['relative']: r for r in local['files']}
            require(len(source_map) == len(source['files']) == source['file_count'], 'duplicate source inventory')
            require(len(local_map) == len(local['files']) == local['file_count'], 'duplicate local inventory')
            require(source_map.keys() == local_map.keys(), 'inventory members differ')
            require(sum(r['bytes'] for r in source_map.values()) == source['total_bytes'] == local['total_bytes'],
                    'inventory total bytes differ')
            destination = Path(local['destination']).resolve()
            require(destination == (root / 'files').resolve(), 'destination differs from fixed local backup')
            for relative, s in source_map.items():
                l = local_map[relative]
                require(s['sha256'] == l['sha256'] and s['bytes'] == l['bytes'], 'source/local digest differs')
                path = Path(l['path'])
                require(path.resolve() == destination / relative and path.is_relative_to(destination), 'local path escape')
                require(s['path'] == str(Path(source['source_root']) / relative), 'source path escape')
                require(s['path'] not in self.files, 'duplicate source across backups')
                self.files[s['path']] = (s, l, path)
            self.inputs.append(dict(source_inventory=source_proof, local_inventory=local_proof,
                                    inventory_files=source['file_count'], inventory_bytes=source['total_bytes']))

    def opened(self, ref):
        require(ref['path'] in self.files, 'committed reference missing in fixed backup')
        s, l, path = self.files[ref['path']]
        require(ref['path'] not in self.seen, 'duplicate committed file reference')
        require(ref['sha256'] == s['sha256'] and ref['bytes'] == s['bytes'], 'receipt/inventory SHA or bytes differ')
        for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns', 'mode'):
            if k in ref:
                require(ref[k] == s['pinned_metadata'][k], 'source metadata differs: ' + k)
        require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), 'nonregular local member')
        require(metadata(path.stat()) == l['metadata'], 'local member no longer has accepted fixed metadata')
        f = path.open('rb')
        require(metadata(os.fstat(f.fileno())) == l['metadata'], 'opened member identity differs')
        return f, s, l, path

    def finish(self, ref, f, s, l, path, digest, bytes_read, category):
        require(digest == ref['sha256'] and bytes_read == ref['bytes'], 'fresh content digest or bytes mismatch')
        require(metadata(os.fstat(f.fileno())) == l['metadata'] == metadata(path.stat()), 'member changed during read')
        self.seen.add(ref['path'])
        self.reads.append(dict(source_path=ref['path'], local_path=str(path), sha256=digest,
                               bytes=bytes_read, category=category, local_fixed_metadata=l['metadata']))

    def receipt(self, ref):
        f, s, l, p = self.opened(ref)
        with f:
            raw = f.read()
            self.finish(ref, f, s, l, p, hashlib.sha256(raw).hexdigest(), len(raw), 'committed_update_receipt')
        return json.loads(raw)


def fresh_stats():
    return dict(valid_token_rows=0, clipped_count=0, hit_tokens=0, vocabulary_slots=0,
                auxiliary_vocabulary_dump_rows=0,
                loss_sum={k: 0. for k in LOSS_KEYS}, mass_sum={k: 0. for k in MASS_KEYS},
                mass_max={k: 0. for k in MASS_KEYS}, vocabulary_sizes=set())


def token_log(backups, ref, pair, receipt, rank):
    f, s, l, path = backups.opened(ref)
    digest = hashlib.sha256()
    size = 0
    stats = fresh_stats()
    with f:
        line = f.readline()
        digest.update(line); size += len(line)
        require(line.endswith(b'\n'), 'incomplete token header')
        header = json.loads(line)
        require(header['schema'] == 'opsd-online-training-token-header-v1', 'token header schema differs')
        require(header['run_id'] == RUN and header['outer_index'] == receipt['outer_index'] and
                header['optimizer_update'] == receipt['optimizer_updates'] and header['rank'] == rank and
                header['pair_index'] == pair['pair_index'], 'header identity differs')
        require(header['contract'] == receipt['contract'] and header['component_clip'] == 10 and
                header['pair_weight'] == 1 and pair['pair_weight'] == 1 and header['retry'] == 0,
                'C0 contract or accepted retry differs')
        require(header['loss_scope'] == 'actual_training_forward_before_optimizer', 'header scope differs')
        ids, mask = header['token_ids'], header['policy_mask']
        require(len(ids) == len(mask) == len(header['token_offsets']), 'header response arrays differ')
        require(all(v in (0, 1) for v in mask), 'invalid policy mask')
        positions = [i for i, value in enumerate(mask) if value]
        require(len(positions) == pair['valid_tokens'] > 0, 'policy positions differ from pair denominator')
        for line in f:
            digest.update(line); size += len(line)
            require(line.endswith(b'\n'), 'partial token line')
            row = json.loads(line)
            if 'full_vocab_dump' in row:
                require(row['full_vocab_dump'] in ('complete', 'cap_skip') and
                        type(row['token_index']) is int and row['token_index'] in positions,
                        'invalid auxiliary full-vocabulary dump row')
                stats['auxiliary_vocabulary_dump_rows'] += 1
                continue
            count = stats['valid_token_rows']
            require(count < len(positions), 'extra token metric row')
            index = positions[count]
            require(row['token_index'] == index and row['token_id'] == ids[index] and
                    row['token_offsets'] == header['token_offsets'][index] and row['policy_mask'] is True,
                    'policy token identity/order differs')
            require(row['valid_trajectory_tokens'] == len(positions) and
                    row['global_actual_pairs'] == receipt['global_actual_pairs'] and row['pair_weight'] == 1,
                    'token normalization denominator differs')
            require(row['direction'] == 'student_to_fixed_target_reverse' and row['target_kind'] == 'teacher'
                    and row['loss_scope'] == 'actual_training_forward_before_optimizer'
                    and row['teacher_forced_fixed_student_prefix'] is True
                    and row['actual_generation_scores'] is False, 'token loss scope differs')
            hit, vocab = row['clipped_vocabulary_count'], row['vocabulary_size']
            require(type(hit) is int and type(vocab) is int and 0 <= hit <= vocab and vocab > 0, 'invalid hit count')
            fraction = finite(row['clipped_vocabulary_fraction'], 'vocabulary fraction')
            require(math.isclose(fraction, hit / vocab, rel_tol=2e-7, abs_tol=2e-10), 'vocab fraction differs')
            stats['clipped_count'] += hit
            stats['hit_tokens'] += bool(hit)
            stats['vocabulary_slots'] += vocab
            stats['vocabulary_sizes'].add(vocab)
            values = {k: finite(row[field], field) for k, field in LOSS_KEYS.items()}
            require(values['raw'] + 2e-6 >= values['post'], 'upper clip increased scalar loss')
            for key, value in values.items():
                stats['loss_sum'][key] += value
            for key, field in MASS_KEYS.items():
                mass = finite(row[field], field)
                require(0 <= mass <= 1 + 2e-6 and (hit > 0 or mass == 0), 'invalid hit probability mass')
                stats['mass_sum'][key] += mass
                stats['mass_max'][key] = max(stats['mass_max'][key], mass)
            require(row[MASS_KEYS['teacher']] == row[MASS_KEYS['mixture']], 'C0 teacher/target mass differs')
            for field in ('student_to_teacher_raw_kl', 'student_to_outer_old_raw_kl',
                          'student_to_mixture_raw_kl', 'trajectory_contribution', 'global_batch_contribution',
                          'student_observed_token_logp', 'student_entropy', 'teacher_observed_token_logp',
                          'teacher_entropy', 'outer_old_observed_token_logp', 'outer_old_entropy',
                          'mixture_observed_token_logp', 'mixture_entropy'):
                finite(row[field], field)
            require(equal_fp32_log(row['trajectory_contribution'], values['post'] / len(positions)), 'trajectory contribution differs')
            require(equal_fp32_log(row['global_batch_contribution'], values['post'] / len(positions) / receipt['global_actual_pairs']),
                    'global contribution differs')
            stats['valid_token_rows'] += 1
        require(stats['valid_token_rows'] == len(positions), 'token metric rows incomplete')
        backups.finish(ref, f, s, l, path, digest.hexdigest(), size, 'committed_token_log')
    require(stats['clipped_count'] == pair['clipped_vocabulary_count'], 'pair clip count differs')
    errors = {}
    for key in LOSS_KEYS:
        mean = stats['loss_sum'][key] / len(positions)
        native = finite(pair[key + '_loss'], 'native pair ' + key)
        errors[key] = abs(mean - native)
        require(equal_fp32_log(mean, native), 'pooled token/native chunked recorded mean differs beyond stated rounding tolerance')
    stats['native_mean_rounding_errors'] = errors
    return stats


def output_stats(stats, trajectory_count, trajectory_mass_sum, trajectory_loss_sum):
    n, hit = stats['valid_token_rows'], stats['hit_tokens']
    return dict(valid_token_rows=n, clipped_vocabulary_count=stats['clipped_count'],
                auxiliary_vocabulary_dump_rows=stats['auxiliary_vocabulary_dump_rows'],
                hit_token_count=hit, hit_token_fraction=hit / n,
                vocabulary_slots=stats['vocabulary_slots'], vocabulary_sizes=sorted(stats['vocabulary_sizes']),
                pooled_vocabulary_fraction=stats['clipped_count'] / stats['vocabulary_slots'],
                masses={key: dict(sum_over_valid_tokens=stats['mass_sum'][key],
                                  pooled_valid_token_mean=stats['mass_sum'][key] / n,
                                  hit_token_mean=stats['mass_sum'][key] / hit if hit else None,
                                  equal_trajectory_token_mean=trajectory_mass_sum[key] / trajectory_count,
                                  maximum_token_mass=stats['mass_max'][key]) for key in MASS_KEYS},
                loss={key: dict(pooled_valid_token_mean=stats['loss_sum'][key] / n,
                                equal_trajectory_token_mean=trajectory_loss_sum[key] / trajectory_count) for key in LOSS_KEYS},
                all_inspected_metric_fields_finite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    facts, facts_proof = stable_json(args.execution_root / 'goal-continuation-20261009-7/ten-outer-committed-facts.stdout.json', FACT_SHA)
    require(facts['accepted'] is True, 'facts unaccepted')
    outers = sorted([r for r in facts['outers'] if r['variant'] == 'C0'], key=lambda r: r['outer'])
    require([r['outer'] for r in outers] == list(range(10)), 'C0 ten outer coverage differs')
    backups = FixedBackups(args.execution_root.resolve())
    results, all_update_numbers, global_pairs = [], [], set()
    aggregate = fresh_stats()
    aggregate_pair_count = 0
    aggregate_mass_means = {k: 0. for k in MASS_KEYS}
    aggregate_loss_means = {k: 0. for k in LOSS_KEYS}
    max_rounding = {k: 0. for k in LOSS_KEYS}
    for outer in outers:
        require(outer['status'] == 'COMPLETE_FACTS' and outer['run_id'] == RUN, 'outer facts not closed C0')
        stats = fresh_stats(); pair_count = 0; pair_indices = set()
        mass_means = {k: 0. for k in MASS_KEYS}; loss_means = {k: 0. for k in LOSS_KEYS}
        for update in outer['updates']:
            receipt = backups.receipt(update['source'])
            require(receipt['accepted'] is True and receipt['run_id'] == RUN and
                    receipt['outer_index'] == outer['outer'] and receipt['optimizer_updates'] == update['update']
                    and receipt['contract'] == outer['contract'], 'committed update identity differs')
            require(receipt['loss_reduction'] == 'trajectory_mean' and receipt['micro_batch'] == 1
                    and receipt['learning_rate'] == 5e-6, 'actual update scientific contract differs')
            require(receipt['scaler']['accepted'] is True and receipt['scaler']['adam_advanced'] is True
                    and receipt['scaler']['overflow'] is False and receipt['retries'] == 0, 'accepted update overflow/retry differs')
            require(receipt['global_actual_pairs'] == update['global_actual_pairs']
                    and receipt['global_valid_tokens'] == update['global_valid_tokens'], 'update totals differ')
            ranks = receipt['all_rank_actual_token_metrics']
            require(sorted(r['rank'] for r in ranks) == list(range(8)), 'actual rank coverage differs')
            update_pairs = update_tokens = update_clips = 0
            for rank in sorted(ranks, key=lambda r: r['rank']):
                require(rank['actual']['rank'] == rank['rank'], 'rank actual identity differs')
                for pair in rank['pairs']:
                    identity = (outer['outer'], pair['pair_index'])
                    require(identity not in global_pairs, 'duplicate pair exposure')
                    global_pairs.add(identity); pair_indices.add(pair['pair_index'])
                    s = token_log(backups, pair['token_log'], pair, receipt, rank['rank'])
                    update_pairs += 1; pair_count += 1
                    update_tokens += s['valid_token_rows']; update_clips += s['clipped_count']
                    for destination in (stats, aggregate):
                        for key in ('valid_token_rows', 'clipped_count', 'hit_tokens', 'vocabulary_slots', 'auxiliary_vocabulary_dump_rows'):
                            destination[key] += s[key]
                        destination['vocabulary_sizes'].update(s['vocabulary_sizes'])
                        for key in MASS_KEYS:
                            destination['mass_sum'][key] += s['mass_sum'][key]
                            destination['mass_max'][key] = max(destination['mass_max'][key], s['mass_max'][key])
                        for key in LOSS_KEYS:
                            destination['loss_sum'][key] += s['loss_sum'][key]
                    for key in MASS_KEYS:
                        mass_means[key] += s['mass_sum'][key] / s['valid_token_rows']
                    for key in LOSS_KEYS:
                        loss_means[key] += s['loss_sum'][key] / s['valid_token_rows']
                        max_rounding[key] = max(max_rounding[key], s['native_mean_rounding_errors'][key])
            require(update_pairs == update['global_actual_pairs'] and update_tokens == update['global_valid_tokens'], 'actual update row/pair totals differ')
            require(update_clips == update['metrics']['clipped_vocabulary_count'], 'update clip totals differ from facts')
            all_update_numbers.append(update['update'])
        require(pair_count == outer['counts']['K'] and pair_indices == set(range(outer['counts']['K'])), 'outer pair exposure differs')
        require(stats['valid_token_rows'] == outer['counts']['valid_tokens'] and
                stats['clipped_count'] == outer['metrics']['clipped_vocabulary_count'], 'outer token/clip totals differ from facts')
        result = dict(outer_index=outer['outer'], pair_exposures=pair_count, committed_updates=len(outer['updates']),
                      cumulative_optimizer_updates=outer['counts']['optimizer_updates'],
                      facts_valid_tokens=outer['counts']['valid_tokens'], valid_tokens_exact_match=True,
                      **output_stats(stats, pair_count, mass_means, loss_means))
        for key in LOSS_KEYS:
            native_mean = outer['metrics']['equal_trajectory_means'][key + '_loss']['value']
            require(equal_fp32_log(result['loss'][key]['equal_trajectory_token_mean'], native_mean), 'outer scalar means disagree')
            result['loss'][key]['native_recorded_equal_trajectory_mean'] = native_mean
        results.append(result)
        aggregate_pair_count += pair_count
        for key in MASS_KEYS: aggregate_mass_means[key] += mass_means[key]
        for key in LOSS_KEYS: aggregate_loss_means[key] += loss_means[key]
        print(json.dumps(dict(outer=outer['outer'], valid_tokens=stats['valid_token_rows'], clipped_count=stats['clipped_count'], hit_tokens=stats['hit_tokens'])), flush=True)
    require(all_update_numbers == list(range(1, 101)) and aggregate_pair_count == 3012
            and aggregate['valid_token_rows'] == 583074 and aggregate['clipped_count'] == 1021,
            'closed ten-chain totals differ')
    result = dict(schema='c0-committed-token-clipmass-full-local-aggregation-v1', accepted=True,
                  started_unix=started, ended_unix=time.time(), run_id=RUN, variant='C0',
                  actual_updates=100, pair_exposures=aggregate_pair_count, outers=results,
                  totals=output_stats(aggregate, aggregate_pair_count, aggregate_mass_means, aggregate_loss_means),
                  scope='Only explicitly referenced token logs of 100 committed updates, no failed attempt logs, no model/GPU/Node access.',
                  definitions={'hit_token_fraction': 'Number of policy-valid token positions with clipped_vocabulary_count > 0 divided by all valid token positions; not a vocabulary fraction.',
                               'pooled_vocabulary_fraction': 'Sum of clipped vocabulary contributions divided by sum of vocabulary_size over policy-valid positions.',
                               'mass': 'Sum of student/teacher/fixed-target probabilities on the same component > 10 hit set, at a single token. Sums across tokens are exposure sums, not one probability.',
                               'means': 'Pooled token mean uses all valid tokens; equal-trajectory mean averages each trajectory token mean with weight 1; hit-token mean uses only hit positions.',
                               'C0_mixture_alias': 'C0 use_mixture=false: recorded mixture is fixed teacher target, and teacher and mixture masses were exactly equal on all inspected rows.',
                               'raw_and_post': 'Logged vocabulary sums include positive and negative components. Source clips only upper components at 10. Negative component sum/count was not logged separately and cannot be reconstructed.',
                               'limitations': 'Recorded forward metrics and hashes verified, not reconstruction of logits, gradients, full vocabulary probabilities, generation quality, causal effects, or heldout512 accuracy.'},
                  finite_scope='All explicitly inspected scalar metric fields; top32 arrays and auxiliary full-vocabulary dump payload files not revalidated.',
                  native_chunked_vs_local_sequential_recorded_mean_max_absolute_error=max_rounding,
                  native_mean_comparison_tolerance=dict(relative=2e-6, absolute=2e-6),
                  inputs=dict(facts=facts_proof, backups=backups.inputs),
                  referenced_files=dict(update_receipts=100, token_logs=aggregate_pair_count,
                                        total_files=len(backups.reads), total_bytes=sum(r['bytes'] for r in backups.reads)))
    (out/'C0-ten-outer-clipmass.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (out/'referenced-files-proof.private.json').write_text(json.dumps(dict(accepted=True, files=backups.reads), indent=2, allow_nan=False)+'\n')
    columns = ['outer_index','pair_exposures','committed_updates','cumulative_optimizer_updates','valid_token_rows','facts_valid_tokens','clipped_vocabulary_count','hit_token_count','hit_token_fraction','pooled_vocabulary_fraction','raw_token_mean','post_token_mean','raw_trajectory_mean','post_trajectory_mean']
    for key in MASS_KEYS:
        columns += [key+'_'+s for s in ('mass_sum','valid_token_mean','hit_token_mean','trajectory_mean','mass_max')]
    with (out/'C0-ten-outer-clipmass.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in results:
            flat={k:row[k] for k in columns if k in row}
            flat.update(raw_token_mean=row['loss']['raw']['pooled_valid_token_mean'],post_token_mean=row['loss']['post']['pooled_valid_token_mean'],raw_trajectory_mean=row['loss']['raw']['equal_trajectory_token_mean'],post_trajectory_mean=row['loss']['post']['equal_trajectory_token_mean'])
            for key in MASS_KEYS:
                mass=row['masses'][key]
                for suffix, field in [('mass_sum','sum_over_valid_tokens'),('valid_token_mean','pooled_valid_token_mean'),('hit_token_mean','hit_token_mean'),('trajectory_mean','equal_trajectory_token_mean'),('mass_max','maximum_token_mass')]:
                    flat[key+'_'+suffix]=mass[field]
            writer.writerow(flat)
    inventory=[]
    for name in ('aggregate_clipmass.py','C0-ten-outer-clipmass.json','C0-ten-outer-clipmass.csv','referenced-files-proof.private.json'):
        _, proof = stable_json(out/name) if name.endswith('.json') else (None, None)
        if proof is None:
            p=out/name; proof=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        inventory.append(proof)
    manifest=dict(schema='c0-clipmass-derived-manifest-v1',accepted=True,generated_unix=time.time(),files=inventory,
                  coverage=result['referenced_files'], totals={k:result[k] for k in ('actual_updates','pair_exposures')},
                  valid_token_rows=aggregate['valid_token_rows'],all_referenced_files_fresh_sha_verified=True)
    (out/'derived-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(accepted=True, output=str(out), valid_token_rows=aggregate['valid_token_rows'],
                         hit_tokens=aggregate['hit_tokens'],clipped_count=aggregate['clipped_count'],
                         manifest_sha256=hashlib.sha256((out/'derived-manifest.json').read_bytes()).hexdigest())),flush=True)


if __name__ == '__main__':
    main()
