"""Strict accounting across already-audited, same-configuration native attempts.

These functions validate a restoration link and combine round evidence. They
never launch/stop training. Physical file checks and per-rank audits must precede
them; no fresh-only completion gate is relaxed by importing this module.
"""


def validate_link(parent_entry, child_entry, manifest, restored, manifest_path, manifest_sha256):
    stable = ('run_id', 'group', 'source_commit', 'config_sha256', 'host')
    if (any(parent_entry[k] != child_entry[k] for k in stable)
            or str(parent_entry['allocation']) == str(child_entry['allocation'])
            or child_entry['resume_manifest'] != manifest_path
            or parent_entry['output'] == child_entry['output']):
        raise ValueError('Native continuation must link distinct attempts of the same frozen run/host')
    identity = manifest['identity']
    step = identity['completed_step']
    if (manifest['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or type(step) is not int or not 0 < step < 150 or identity['actual_update_limit'] != 150
            or str(identity['job_id']) != str(parent_entry['allocation'])
            or identity['code_commit'] != parent_entry['source_commit']
            or identity['config_sha256'] != parent_entry['config_sha256']
            or manifest['driver']['next_step'] != step + 1):
        raise ValueError('Parent manifest is not this attempt’s complete nonterminal boundary')
    expected = manifest['worker_states']
    actual = restored['workers']
    if (len(expected) != 8 or {w['identity']['rank'] for w in expected} != set(range(8))
            or len(actual) != 8 or {w['rank'] for w in actual} != set(range(8))
            or restored['status'] != 'ABLATION_CURRENT_CONDITION_NATIVE_STATE_RESTORED'
            or restored['completed_outer'] != step or restored['next_outer'] != step + 1
            or restored['parent_manifest'] != manifest_path
            or restored['parent_manifest_sha256'] != manifest_sha256):
        raise ValueError('Actual native restoration does not cover the exact eight-rank parent boundary')
    writes = {}
    for old in expected:
        rank = old['identity']['rank']
        new = next(w for w in actual if w['rank'] == rank)
        total = old['actual_total_optimizer_calls']
        if (type(total) is not int or not 0 <= total <= step * 20
                or old['identity']['world_size'] != 8
                or any(old['identity'][k] != v for k, v in identity.items())
                or old['scheduler_last_epoch'] != step
                or old['full_worker_state_finite'] is not True
                or old['observed_adam_steps'] != ([total] if total else [])
                or new['actual_state_equal'] is not True or new['scheduler_step'] != step
                or new['actual_total_optimizer_calls'] != total
                or new['parent_state_sha256'] != old['sha256']):
            raise ValueError('Restored rank model/Adam/scheduler identity or counters differ')
        writes[str(rank)] = total
    if len(set(writes.values())) != 1:
        raise ValueError('Parent actual Adam totals disagree across ranks')
    return dict(parent_job=str(parent_entry['allocation']), child_job=str(child_entry['allocation']),
        completed_outer=step, next_outer=step + 1, actual_adam_per_rank=writes,
        manifest_path=manifest_path, manifest_sha256=manifest_sha256,
        limits='Requires separately verified manifest bytes, frozen runtime source and physical attempt records; no cross-host or vLLM bitwise equivalence claim.')


def merge_rounds(segments, links):
    """Merge per-rank-verified attempts, clipping a failed attempt at saved state.

    A parent may have made updates after its last durable checkpoint. Those
    updates are retained as discarded work, never credited twice to the model
    restored by the next attempt. Normal guard rounds consume a round while
    adding only their actual Adam writes.
    """
    if not segments or len(links) != len(segments) - 1:
        raise ValueError('Need one verified restoration link between each consecutive attempt')
    merged = []
    discarded = []
    total = 0
    previous_step = 0
    for index, segment in enumerate(segments):
        job = str(segment['job'])
        if index:
            link = links[index - 1]
            if (link['parent_job'] != str(segments[index - 1]['job']) or link['child_job'] != job
                    or link['completed_outer'] != previous_step
                    or link['actual_adam_per_rank'] != {str(i): total for i in range(8)}):
                raise ValueError('Restoration link disagrees with the audited parent prefix')
        stop = links[index]['completed_outer'] if index < len(links) else None
        if stop is not None and (links[index]['parent_job'] != job or stop < previous_step):
            raise ValueError('Parent checkpoint cannot precede this attempt’s restored boundary')
        rows = segment['attempts']
        if [r['outer'] for r in rows] != list(range(previous_step + 1, previous_step + 1 + len(rows))):
            raise ValueError('Attempt rounds skip or repeat the expected restored cursor')
        if stop is not None and (not rows or rows[-1]['outer'] < stop):
            raise ValueError('Parent checkpoint credits rounds absent from physical attempt evidence')
        for row in rows:
            writes = row['actual_adam_writes_per_rank']
            guard = row['normal_guard']
            if type(writes) is not int or not 0 <= writes <= 20:
                raise ValueError('Invalid actual Adam count')
            if writes < 20 and (not guard or guard['status'] != 'NORMAL_SAFETY_GUARD_ROUND_CONSUMED'
                    or guard['optimizer_writes'] != writes or guard['opportunities_processed'] != 20
                    or guard['opportunities_skipped'] != 20 - writes):
                raise ValueError('Partial/zero updates require explicit normal-guard evidence')
            if stop is not None and row['outer'] > stop:
                discarded.append(dict(job=job, outer=row['outer'], actual_adam_per_rank=writes,
                    reason='Not present in the full native boundary restored by the next attempt'))
                continue
            total += writes
            if row['cumulative_adam_writes_per_rank'] != {str(i): total for i in range(8)}:
                raise ValueError('Cumulative Adam counts reset or double-counted across continuation')
            merged.append(dict(row, allocation=job))
            previous_step = row['outer']
    return dict(attempts=merged, completed_rounds=previous_step,
        actual_adam_per_rank={str(i): total for i in range(8)}, discarded_after_restored_boundary=discarded,
        scope='Accounting over separately verified physical attempt records, not a standalone completion verdict.')
