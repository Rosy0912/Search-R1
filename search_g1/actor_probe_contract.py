"""Explicit live-actor binary readout variant; legacy runs remain unchanged."""

from search_g1.reward_schedule import ACTOR_PROBE_SCHEDULE

MODE = 'actor_detached_binary_heads_v1'
INTERVAL = 10
LAYER_CHOICES = (21, 24, 27)


def actor_probes(run):
    mode = run.get('probe_representation_source')
    if mode is None:
        if run.get('reward_schedule') == ACTOR_PROBE_SCHEDULE or any(
                k in run for k in ('probe_refresh_interval', 'probe_parameter_ownership')):
            raise ValueError('Partial actor probe contract')
        return False
    if mode != MODE:
        raise ValueError('Unknown probe representation source')
    if run.get('reward_schedule') != ACTOR_PROBE_SCHEDULE:
        raise ValueError('Actor probes require the explicitly selected after100 experiment')
    if type(run.get('probe_refresh_interval')) is not int or run['probe_refresh_interval'] != INTERVAL:
        raise ValueError('Actor probes require ten complete outer iterations')
    if run.get('probe_parameter_ownership') != 'independent_binary_heads_only':
        raise ValueError('Probe optimizer must not own the actor or generation head')
    if run.get('initial_probe_adoption_source_base') is not None:
        raise ValueError('New actor probes require genuine fresh calibration admission')
    return True


def experiment_fields(schedule):
    if schedule != ACTOR_PROBE_SCHEDULE:
        return {}
    return dict(probe_representation_source=MODE, probe_refresh_interval=INTERVAL,
                probe_parameter_ownership='independent_binary_heads_only')


def probe_rounds(run):
    return tuple(range(0, 201, INTERVAL)) if actor_probes(run) else (0, 50, 100, 150)


def head_round(logical_update):
    if type(logical_update) is not int or not 1 <= logical_update <= 200:
        raise ValueError('Formal probe prediction requires logical iteration1..200')
    return ((logical_update - 1) // INTERVAL) * INTERVAL


def checkpoint_head_round(completed_outer):
    """A saved completed boundary includes its newly fitted pair, including200."""
    if type(completed_outer) is not int or not 0 <= completed_outer <= 200:
        raise ValueError('Checkpoint probe binding requires completed iteration0..200')
    return (completed_outer // INTERVAL) * INTERVAL


def validate_fitted_version(run, version):
    """Authenticate a fitted pair without inventing a prediction iteration201."""
    if not actor_probes(run):
        raise ValueError('Live actor prediction requires explicit admission')
    if version.get('run_id') != run['run_id'] or not version.get('version'):
        raise ValueError('Head version belongs to another run')
    if type(version.get('round')) is not int or version['round'] not in probe_rounds(run):
        raise ValueError('Head does not belong to a scheduled fitting boundary')
    if version.get('representation_source') != MODE:
        raise ValueError('Frozen encoder prediction is not permitted in actor-probe mode')
    if version.get('actor_step') != version['round']:
        raise ValueError('Head fitting actor step differs from its version')
    if version.get('calibration_data_sha256') != run['input_sha256']['calibration-300.parquet']:
        raise ValueError('Head calibration data changed')
    return version


def validate_version(run, version, logical_update):
    validate_fitted_version(run, version)
    if version['round'] != head_round(logical_update):
        raise ValueError('Stale or future probe head for this outer iteration')
    return version


def validate_checkpoint_version(run, version, completed_outer):
    validate_fitted_version(run, version)
    if version['round'] != checkpoint_head_round(completed_outer):
        raise ValueError('Saved Head does not match the completed checkpoint boundary')
    return version
