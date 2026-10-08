"""Independent fixed-data aggregate evaluation parser; never fabricates per-question evidence."""
import argparse
import datetime
import os
from pathlib import Path

import author200_observation as shared


def inspect(args):
    shared.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only parser environment required')
    context = shared.load_accepted_context(args)
    driver, aggregate, drec, arec = shared.observed_prefixes(args, context)
    saved_step = 200 if args.require_final else 5
    join = shared.join_log_prefixes(driver, aggregate, context, saved_step)
    if args.require_final:
        shared.need(join['authoritative_last_outer'] == join['aggregate_last_outer'] == 200,
                    'Formal/driver actual200 completion differs')
        shared.need(all(r['bytes'] == r['observed_file_size_at_open'] == r['observed_file_size_at_close']
                        for r in (drec, arec)), 'Final evaluation needs whole observed logs, not a short prefix')
    report = shared.check_evaluation_sequence(driver, join['authoritative_last_outer'], args.require_final)
    shared.need(len(aggregate['evaluations']) <= len(driver['evaluations'])
                and all(shared.canonical(row) == shared.canonical(driver['evaluations'][i])
                        for i, row in enumerate(aggregate['evaluations'])),
                'Aggregate evaluation records differ from authoritative driver')
    if args.require_final:
        shared.need(shared.canonical(aggregate['final']) == shared.canonical(driver['final'])
                    and shared.canonical(aggregate['evaluations']) == shared.canonical(driver['evaluations']),
                    'Whole final aggregate/driver evaluations differ')
    dataset = shared.verify_dataset_files(context)
    result = dict(version=1, kind='AUTHOR200_FIXED_DATA_AGGREGATE_EVALUATION_REVIEW',
        scope='FIXED_SOURCE_COUNTS_METRICS_AND_OBSERVED_LOGS_NOT_EFFECT_IMPROVEMENT_OR_TERMINAL_ACCEPTANCE',
        inspected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parser_source_sha256=shared.stable_file(Path(__file__).resolve())[0]['sha256'],
        common_helper_sha256=shared.stable_file(Path(shared.__file__).resolve())[0]['sha256'],
        source_commit=shared.SOURCE, config_sha256=shared.CONFIG_SHA, job_id=shared.JOB,
        run_root=str(context['run']), driver_pid=shared.DRIVER_PID,
        inspection_execution_identity=shared.execution_identity(), accepted5=context['accepted5'],
        training_prefix=arec, driver_prefix=drec, log_join=join,
        evaluation_coverage=report, dataset_bindings=dataset, requested_final_scope=args.require_final,
        passed=True, per_question_raw_rescoring_performed=False,
        limitations=['No per-question response/mask/span archive exists in ordinary native validation',
                     'Observed full log bytes alone do not prove a closed writer or scheduler exit',
                     'No relative final effect claim without a matched base final evaluation',
                     'No GPU inference/state restore/signature or training control'])
    return shared.exclusive_json(Path(args.output), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared.parser_anchors(parser)
    parser.add_argument('--require-final', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.relative_to(Path(args.run_root) / 'evidence')
    print(shared.canonical(inspect(args)).decode(), flush=True)


if __name__ == '__main__':
    main()
