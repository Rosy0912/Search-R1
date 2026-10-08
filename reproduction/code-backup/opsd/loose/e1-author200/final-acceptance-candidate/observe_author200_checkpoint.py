"""Observe an immutable later author200 checkpoint. Never writes a training manifest."""
import argparse
import datetime
import os
from pathlib import Path

import author200_observation as shared


def observe(args):
    shared.need(type(args.checkpoint_step) is int and args.checkpoint_step in shared.STEPS,
                'Only reviewed later save steps are accepted')
    shared.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only observer environment required')
    context = shared.load_accepted_context(args)
    driver, aggregate, drec, arec = shared.observed_prefixes(args, context)
    joined = shared.join_log_prefixes(driver, aggregate, context, args.checkpoint_step)
    evaluations = shared.check_evaluation_sequence(driver, joined['authoritative_last_outer'])
    # Aggregate evaluation transport may lag, but any observed records must be an
    # exact prefix of the authoritative records, never invented or silently fixed.
    shared.need(len(aggregate['evaluations']) <= len(driver['evaluations'])
                and all(shared.canonical(row) == shared.canonical(driver['evaluations'][i])
                        for i, row in enumerate(aggregate['evaluations'])),
                'Aggregate evaluation forwarding conflicts with own driver')
    dataset = shared.verify_dataset_files(context)
    expected_hf = {role for role in context['anchor_files']
                   if role.startswith('actor/') and not role.startswith('actor/training_state/')}
    files = shared.checkpoint_files(context['run'], args.checkpoint_step, expected_hf)
    ranks, driver_report = shared.checkpoint_reports(files, args.checkpoint_step, context)
    observed_console = []
    seen = set()
    for record in aggregate['native_console']:
        if record.get('step') != args.checkpoint_step:
            continue
        rank = record.get('rank')
        shared.need(type(rank) is int and 0 <= rank < 8 and rank not in seen,
                    'Duplicate/invalid observed native writer rank')
        shared.need(shared.canonical(record) == shared.canonical(ranks[rank]),
                    'Observed native writer differs from original saved report')
        seen.add(rank)
        observed_console.append(record)
    # Recheck every member's pathname/inode/size; full SHA was streamed with
    # member mtime/ctime stability above. No old anchor inode is counted here.
    for record in files.values():
        shared.recheck_member(record, context['run'])
    out = Path(args.output_directory)
    shared.need(out.is_absolute() and out.parent.resolve(strict=True) == out.parent,
                'New physical observer directory required')
    out.relative_to(context['run'] / 'evidence')
    out.mkdir(mode=0o700, exist_ok=False)
    inventory = dict(version=1, kind=shared.KIND,
        observation_scope='FILES_AND_IDENTITY_OBSERVED_NOT_LIVE_RESTORE_OR_EFFECT_ACCEPTANCE',
        observation_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observer_source_sha256=shared.stable_file(Path(__file__).resolve())[0]['sha256'],
        common_helper_sha256=shared.stable_file(Path(shared.__file__).resolve())[0]['sha256'],
        inspection_execution_identity=shared.execution_identity(),
        source_commit=shared.SOURCE, source_root=str(Path(args.source_root)),
        native_import_closure_path=context['closure_path'], native_import_closure_sha256=shared.CLOSURE_SHA,
        explicit_source_files=context['source_files'], formal_manifest_path=context['formal_path'],
        formal_manifest_sha256=shared.FORMAL_SHA, job_id=shared.JOB,
        run_root=str(context['run']), checkpoint_step=args.checkpoint_step,
        config_sha256=shared.CONFIG_SHA, profile='author200', horizon=200, world_size=8,
        accepted5=context['accepted5'], anchor_files=context['anchor_files'],
        training_prefix=dict(arec, last_outer=joined['aggregate_last_outer']),
        driver_prefix=dict(drec, last_outer=joined['authoritative_last_outer']),
        log_join=joined, evaluation_coverage=evaluations, dataset_bindings=dataset,
        files=files, rank_reports=ranks, driver_report=driver_report,
        raw_statuses=dict(ranks=[r['status'] for r in ranks], driver=driver_report['status']),
        observed_native_console=dict(records=observed_console, observed_count=len(observed_console),
                                     transport_records=aggregate.get('native_transport', []),
                                     scope='Observed subset only; repeat tags never infer missing ranks'),
        limitations=['No live same-object reload or new-process recovery',
                     'No native PT content or full HF model load is performed by observer',
                     'No final evaluation/terminal/allocation-release/effect acceptance',
                     'Accepted first5 large weights/PT are signed references, not re-inspected here',
                     'Fixed log prefixes do not prove their entire future lifetime'])
    target = out / 'observed-checkpoint-inventory.json'
    record = shared.exclusive_json(target, inventory)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared.parser_anchors(parser)
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    print(shared.canonical(observe(args)).decode(), flush=True)


if __name__ == '__main__':
    main()
