"""Exact-moment Pearson for complete, identity-bound historical TRAIN pairs.

Inputs are independently authenticated per-trajectory C_raw/reward records, not
iteration means or calibration rows.  The production CLI reads only frozen
remote JSONL/coverage files and prints aggregates.  It does not reconstruct a
reward, load a model, access a service, or modify a scientific run.

Each finite binary64 reward is accumulated as an exact integer multiple of
2**-1074; squared rewards and cross products are exact too.  This avoids the
catastrophic cancellation of floating sum(R**2) - sum(R)**2/N.  Reported decimal
moments and correlations are rounded only at the final conversion to binary64.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys


SCALE = 1 << 1074
HASH_MODE = 'sha256_canonical_sorted_unique_ids_v1'
SEMANTICS = {'reconstructed_frozen_historical_input', 'original_online_record'}
MAX_PAIR_JSONL_BYTES = 1024 * 1024


class Hold(Exception):
    """Only fixed, non-private reason codes may be placed in this exception."""


def require(value, reason):
    if not value:
        raise Hold(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def trajectory_ids_sha256(ids):
    require(len(ids) == len(set(ids)), 'duplicate_trajectory_for_hash')
    return hashlib.sha256(canonical(sorted(ids))).hexdigest()


def _json_object(items):
    result = {}
    for name, value in items:
        require(name not in result, 'duplicate_JSON_object_key')
        result[name] = value
    return result


def parse_json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_json_object)
    except Hold:
        raise
    except (ValueError, UnicodeError):
        raise Hold('invalid_JSON_record') from None
    require(isinstance(value, dict), 'non_object_JSON_record')
    return value


def reward_units(value):
    require(type(value) in (int, float), 'non_numeric_or_boolean_reward')
    try:
        value = float(value)
    except OverflowError:
        raise Hold('nonfinite_reward') from None
    require(math.isfinite(value), 'nonfinite_reward')
    numerator, denominator = value.as_integer_ratio()
    # A finite IEEE binary64 denominator is a power of two, at most 2**1074.
    exponent = denominator.bit_length() - 1
    require(denominator == 1 << exponent and exponent <= 1074,
            'unsupported_reward_number_representation')
    return numerator << (1074 - exponent)


def finite_fraction(numerator, denominator):
    try:
        result = float(Fraction(numerator, denominator))
    except OverflowError:
        raise Hold('aggregate_moment_exceeds_binary64_reporting_range') from None
    require(math.isfinite(result), 'aggregate_moment_exceeds_binary64_reporting_range')
    return result


def correlation(covariance_numerator, x_variance_numerator, y_variance_numerator):
    require(x_variance_numerator >= 0 and y_variance_numerator >= 0,
            'negative_exact_variance')
    if x_variance_numerator == 0 or y_variance_numerator == 0:
        return dict(value=None, defined=False,
                    reason='zero_both_variances' if x_variance_numerator == y_variance_numerator == 0
                    else 'zero_C_variance' if x_variance_numerator == 0 else 'zero_R_variance')
    require(covariance_numerator * covariance_numerator <=
            x_variance_numerator * y_variance_numerator, 'exact_Cauchy_Schwarz_violation')
    with localcontext() as context:
        context.prec = 80
        value = Decimal(covariance_numerator) / (
            Decimal(x_variance_numerator) * Decimal(y_variance_numerator)).sqrt()
    return dict(value=float(value), defined=True, reason=None)


class ExactBinaryMoments:
    """Six sufficient statistics; never retains per-row C or reward values."""
    def __init__(self):
        self.N = self.sumC = self.sumC2 = 0
        self._sumR = self._sumR2 = self._sumCR = 0

    def add(self, C, R):
        require(type(C) is int and C in (0, 1), 'C_raw_is_not_strict_binary_integer')
        units = reward_units(R)
        self.N += 1
        self.sumC += C
        self.sumC2 += C * C
        self._sumR += units
        self._sumR2 += units * units
        self._sumCR += C * units

    def merge(self, other):
        self.N += other.N
        self.sumC += other.sumC
        self.sumC2 += other.sumC2
        self._sumR += other._sumR
        self._sumR2 += other._sumR2
        self._sumCR += other._sumCR

    def centered_numerators(self):
        return (self.N * self._sumCR - self.sumC * self._sumR,
                self.N * self.sumC2 - self.sumC * self.sumC,
                self.N * self._sumR2 - self._sumR * self._sumR)

    def report(self):
        require(self.N > 0, 'empty_pair_moments')
        cov, varC, varR = self.centered_numerators()
        result = dict(N=self.N, sumC=self.sumC, sumR=finite_fraction(self._sumR, SCALE),
                      sumC2=self.sumC2, sumR2=finite_fraction(self._sumR2, SCALE * SCALE),
                      sumCR=finite_fraction(self._sumCR, SCALE),
                      C_mean=finite_fraction(self.sumC, self.N),
                      R_mean=finite_fraction(self._sumR, self.N * SCALE),
                      covariance_population=finite_fraction(cov, self.N * self.N * SCALE),
                      C_variance_population=finite_fraction(varC, self.N * self.N),
                      R_variance_population=finite_fraction(varR, self.N * self.N * SCALE * SCALE),
                      Pearson=correlation(cov, varC, varR))
        correct, wrong = self.sumC, self.N - self.sumC
        result['point_biserial'] = dict(defined=False, value=None, reason='no_both_C_classes')
        if correct and wrong:
            # This independently uses class mean separation and population std:
            # (mu_correct-mu_wrong)*sqrt(p*(1-p))/sigma_R.
            separation = Fraction(self._sumCR, correct) - Fraction(self._sumR - self._sumCR, wrong)
            with localcontext() as context:
                context.prec = 80
                separation_decimal = Decimal(separation.numerator) / Decimal(separation.denominator)
                if varR:
                    value = separation_decimal * Decimal(correct * wrong).sqrt() / Decimal(varR).sqrt()
                    result['point_biserial'] = dict(defined=True, value=float(value), reason=None,
                        correct_count=correct, wrong_count=wrong,
                        mean_R_correct=finite_fraction(self._sumCR, correct * SCALE),
                        mean_R_wrong=finite_fraction(self._sumR - self._sumCR, wrong * SCALE))
                else:
                    result['point_biserial']['reason'] = 'zero_R_variance'
            if result['point_biserial']['defined']:
                require(result['point_biserial']['value'] == result['Pearson']['value'],
                        'point_biserial_crosscheck_mismatch')
        return result


def summarize_steps(moments):
    """Pool rows directly; additionally expose within/between-step diagnostics."""
    require(moments and all(item.N > 0 for item in moments), 'empty_step_collection')
    require(len({item.N for item in moments}) == 1, 'unequal_step_sizes_for_decomposition')
    pooled = ExactBinaryMoments()
    for item in moments:
        pooled.merge(item)
    within_cov = within_C = within_R = 0
    for item in moments:
        cov, varC, varR = item.centered_numerators()
        within_cov += cov; within_C += varC; within_R += varR
    count, size = len(moments), moments[0].N
    sum_step_C = sum(item.sumC for item in moments)
    sum_step_R = sum(item._sumR for item in moments)
    between_cov = count * sum(item.sumC * item._sumR for item in moments) - sum_step_C * sum_step_R
    between_C = count * sum(item.sumC * item.sumC for item in moments) - sum_step_C * sum_step_C
    between_R = count * sum(item._sumR * item._sumR for item in moments) - sum_step_R * sum_step_R
    # Equality is checked in exact integer arithmetic, before reporting floats.
    require(pooled.centered_numerators()[0] == count * within_cov + between_cov,
            'within_between_exact_covariance_identity_failed')
    return dict(pooled=pooled.report(),
        within_step_centered_Pearson=correlation(within_cov, within_C, within_R),
        between_step_means_Pearson=correlation(between_cov, between_C, between_R),
        covariance_decomposition=dict(
            pooled_covariance_population=pooled.report()['covariance_population'],
            within_step_covariance_population=finite_fraction(within_cov, size * pooled.N * SCALE),
            between_step_covariance_population=finite_fraction(between_cov, count * count * size * size * SCALE),
            exact_additive_identity_pass=True),
        pooled_is_all_individual_pairs=True, pooled_is_not_mean_of_step_correlations=True,
        between_step_means_is_not_requested_individual_pair_coefficient=True)


class CompletePairLedger:
    def __init__(self, coverage, reward_field):
        require(reward_field in ('R_reconstructed', 'R_shaped'), 'unsupported_explicit_reward_field')
        required = ('run_id', 'start_step', 'end_step', 'trajectories_per_step',
                    'trajectory_ids_hash_mode', 'reward_semantics', 'updates')
        require(all(name in coverage for name in required), 'incomplete_coverage_document')
        self.run_id = coverage['run_id']
        require(isinstance(self.run_id, str) and re.fullmatch('[0-9a-f]{32}', self.run_id), 'invalid_run_binding')
        start, end = coverage['start_step'], coverage['end_step']
        require(type(start) is type(end) is int and 1 <= start <= end <= 150, 'scope_outside_TRAIN_1_to_150')
        require(type(coverage['trajectories_per_step']) is int and coverage['trajectories_per_step'] == 2560 and coverage['trajectory_ids_hash_mode'] == HASH_MODE,
                'unsupported_complete_coverage_contract')
        require(coverage['reward_semantics'] in SEMANTICS, 'missing_or_unsupported_reward_semantics')
        self.semantics, self.reward_field = coverage['reward_semantics'], reward_field
        require(isinstance(coverage['updates'], list) and
                [item.get('step') for item in coverage['updates']] == list(range(start, end + 1)),
                'coverage_updates_not_exact_consecutive_scope')
        self.bindings, self.steps = {}, {}
        for item in coverage['updates']:
            require(type(item['step']) is int, 'noninteger_coverage_step')
            attempt, sha = item.get('attempt_id'), item.get('trajectory_ids_sha256')
            require(isinstance(attempt, str) and re.fullmatch('[0-9a-f]{32}', attempt), 'invalid_attempt_binding')
            require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'missing_original_complete_ID_set_sha')
            require(attempt not in [binding['attempt_id'] for binding in self.bindings.values()],
                    'reused_attempt_identity_across_steps')
            step = item['step']
            self.bindings[step] = item
            self.steps[step] = dict(moments=ExactBinaryMoments(), ids=set(), parents={})

    def add(self, row):
        required = ('run_id', 'step', 'attempt_id', 'trajectory_id', 'C_raw', self.reward_field)
        require(isinstance(row, dict) and all(name in row for name in required), 'missing_pair_identity_or_value')
        step = row['step']
        require(type(step) is int and step in self.bindings, 'unexpected_or_noninteger_TRAIN_step')
        binding = self.bindings[step]
        require(row['run_id'] == self.run_id and row['attempt_id'] == binding['attempt_id'],
                'pair_run_or_attempt_mismatch')
        if 'reward_semantics' in row:
            require(row['reward_semantics'] == self.semantics, 'mixed_reward_semantics')
        if 'original_online_reward' in row:
            require(row['original_online_reward'] is (self.semantics == 'original_online_record'),
                    'reward_online_provenance_flag_mismatch')
        identity = row['trajectory_id']
        require(isinstance(identity, str), 'invalid_trajectory_identity')
        prefix = self.run_id + '-' + str(step) + '-attempt-' + binding['attempt_id'] + '-'
        match = re.fullmatch(re.escape(prefix) + r'(\d+)-([0-9a-f]{64})-sample-([0-4])', identity)
        require(match is not None, 'trajectory_run_step_attempt_or_sibling_mismatch')
        occurrence, group, sibling = int(match[1]), match[2], int(match[3])
        require(0 <= occurrence < 512, 'parent_occurrence_outside_complete512')
        state = self.steps[step]
        require(identity not in state['ids'], 'duplicate_pair_identity')
        require(state['moments'].N < 2560, 'more_than_2560_pairs_in_step')
        if occurrence in state['parents']:
            require(state['parents'][occurrence][0] == group, 'parent_hash_changed_between_siblings')
            require(sibling not in state['parents'][occurrence][1], 'duplicate_parent_sibling')
        else:
            state['parents'][occurrence] = (group, set())
        state['moments'].add(row['C_raw'], row[self.reward_field])
        state['ids'].add(identity)
        state['parents'][occurrence][1].add(sibling)

    def finish(self):
        reports, moments = [], []
        for step, state in self.steps.items():
            require(state['moments'].N == len(state['ids']) == 2560, 'missing_individual_pairs_in_step')
            require(set(state['parents']) == set(range(512)) and
                    all(siblings == set(range(5)) for _, siblings in state['parents'].values()),
                    'missing_parent_or_sibling_coverage')
            require(trajectory_ids_sha256(state['ids']) == self.bindings[step]['trajectory_ids_sha256'],
                    'complete_original_trajectory_ID_set_sha_mismatch')
            reports.append(dict(step=step, **state['moments'].report()))
            moments.append(state['moments'])
        return dict(status='ACCEPTED_FULL_INDIVIDUAL_PAIR_CORRELATION', run_id=self.run_id,
            reward_field=self.reward_field, reward_semantics=self.semantics,
            historical_original_online_R_claimed=self.semantics == 'original_online_record',
            steps=reports, completed_steps=len(reports), total_pairs=len(reports) * 2560,
            missing_pairs=0, duplicate_pairs=0, nonfinite_pairs=0,
            complete_original_ID_sets_verified=True,
            moments_accumulation='exact_integer_sums_of_binary64_inputs_and_exact_products',
            final_reporting='80_decimal_digit_sqrt_then_binary64; six moments rounded only on report',
            **summarize_steps(moments))


def analyze_jsonl(paths, expected_shas, coverage, reward_field):
    require(len(paths) == len(expected_shas) > 0, 'input_path_hash_count_mismatch')
    accumulator = CompletePairLedger(coverage, reward_field)
    inputs = []
    for path, expected in zip(paths, expected_shas):
        path = Path(path).absolute()
        require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'invalid_frozen_pair_file_path')
        require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'invalid_pair_file_sha_binding')
        before = path.stat()
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while True:
                raw = stream.readline(MAX_PAIR_JSONL_BYTES + 1)
                if not raw:
                    break
                require(len(raw) <= MAX_PAIR_JSONL_BYTES, 'oversized_pair_JSONL_record')
                require(raw.endswith(b'\n') and raw.strip(), 'incomplete_or_blank_pair_JSONL_record')
                digest.update(raw)
                accumulator.add(parse_json(raw)); count += 1
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'pair_file_changed_during_read')
        require(digest.hexdigest() == expected, 'frozen_pair_file_sha_mismatch')
        inputs.append(dict(path=str(path), fresh_sha256=digest.hexdigest(), records=count))
    report = accumulator.finish()
    report['input_pair_files'] = inputs
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', required=True)
    parser.add_argument('--coverage-sha256', required=True)
    parser.add_argument('--pairs-jsonl', action='append', required=True)
    parser.add_argument('--pairs-sha256', action='append', required=True)
    parser.add_argument('--reward-field', choices=('R_reconstructed', 'R_shaped'), required=True)
    parser.add_argument('--expected-node', required=True)
    parser.add_argument('--output-new', required=True)
    args = parser.parse_args()
    try:
        require(sys.platform.startswith('linux') and socket.gethostname() == args.expected_node,
                'production_CLI_not_bound_remote_Linux_node')
        coverage_path = Path(args.coverage).absolute()
        require(coverage_path.resolve() == coverage_path and coverage_path.is_file(), 'invalid_coverage_path')
        raw = coverage_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == args.coverage_sha256, 'coverage_file_sha_mismatch')
        coverage = parse_json(raw)
        report = analyze_jsonl(args.pairs_jsonl, args.pairs_sha256, coverage, args.reward_field)
        report['coverage_fresh_sha256'] = args.coverage_sha256
        report['CPU_only_no_model_or_service'] = True
        target = Path(args.output_new).absolute()
        require(target.resolve() == target and target.parent.is_dir() and not target.exists(),
                'aggregate_output_not_fresh_independent_path')
        require(target not in [Path(path).absolute() for path in args.pairs_jsonl] and target != coverage_path,
                'aggregate_output_overlaps_input')
        with target.open('xb') as stream:
            stream.write(canonical(report) + b'\n')
            stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='HOLD', reason=str(error) if isinstance(error, Hold) else type(error).__name__,
                              coefficient_accepted=False, raw_exception_not_exported=True)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
