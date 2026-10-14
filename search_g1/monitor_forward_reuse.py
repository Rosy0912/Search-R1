"""Bind reused pre-update diagnostics to the preceding real post-update check."""
import hashlib
import json


def receipt_digest(row):
    # Ledger wrappers are added after the actor computes the receipt.
    value = {key: value for key, value in row.items() if key not in ('kind', 'run_id')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def source_identity(row):
    return dict(context=row['context'], rank=row['rank'],
                optimizer_writes=row['optimizer_writes'],
                monitor_model_version=row['monitor_model_version'],
                monitor_input_sha256=row['monitor_input_sha256'],
                report_sha256=receipt_digest(row))


def summarize_monitor_execution(rows):
    """Validate new reuse receipts; historical rows remain real-forward rows."""
    actual = reused = local_actual = local_reused = 0
    previous = None
    scope = None
    metadata = {'kind', 'run_id', 'context', 'stage', 'rejection_count',
                'forward_reused', 'actual_monitor_forward', 'forward_source',
                'actual_monitor_forward_count', 'reused_monitor_forward_count'}
    for row in rows:
        context = row['context']
        current_scope = (context['logical_update'], context.get('attempt_id'), context.get('attempt_index'))
        if current_scope != scope:
            local_actual = local_reused = 0
            previous = None
            scope = current_scope
        if 'forward_reused' not in row:
            actual += 1
            local_actual += 1
            previous = row
            continue
        assert type(row['forward_reused']) is bool
        assert row['actual_monitor_forward'] is (not row['forward_reused'])
        assert type(row['monitor_model_version']) is int and row['monitor_model_version'] >= 0
        assert isinstance(row['monitor_input_sha256'], str) and len(row['monitor_input_sha256']) == 64
        if row['forward_reused']:
            assert row['stage'] == 'before_minibatch' and row['allowed'] is True
            assert previous is not None and previous['stage'] == 'after_minibatch'
            assert previous.get('actual_monitor_forward') is True and previous['allowed'] is True
            assert row['forward_source'] == source_identity(previous)
            before, after = row['context'], previous['context']
            assert before['minibatch_index'] == after['minibatch_index'] + 1
            assert {k:v for k,v in before.items() if k not in ('minibatch_index','prior_optimizer_writes')} == \
                   {k:v for k,v in after.items() if k not in ('minibatch_index','prior_optimizer_writes')}
            assert before['prior_optimizer_writes'] == row['optimizer_writes'] == previous['optimizer_writes']
            assert {k:v for k,v in row.items() if k not in metadata} == \
                   {k:v for k,v in previous.items() if k not in metadata}
            reused += 1
            local_reused += 1
        else:
            assert row['forward_source'] is None
            actual += 1
            local_actual += 1
        assert row['actual_monitor_forward_count'] == local_actual
        assert row['reused_monitor_forward_count'] == local_reused
        previous = row
    return dict(actual_monitor_forward_count=actual, reused_monitor_forward_count=reused,
                logical_monitor_count=len(rows))
