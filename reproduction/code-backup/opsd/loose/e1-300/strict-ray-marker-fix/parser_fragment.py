# Standard-library parser kept in checkpoint_gate.py and shared by the external consumer.
REVIEW_LOG_MARKERS = ('OPSD_E1_OUTER_JSON=', 'OPSD_E1_STEP5_PAUSED_JSON=',
                      'OPSD_E1_STEP5_RELEASED_JSON=', 'FULL_STATE_RANK_JSON=',
                      'BASELINE_EVALUATION_METRICS_JSON=', 'SEARCH_R1_FINAL_METRICS_JSON=')


def parse_log_marker(line, marker, *, driver_pid, job_id, worker_pids=(), checkpoint_warning_path=None):
    """Decode one exact object; accept only the observed typed own-worker warning.

    Ray may interleave its warning record on the same physical stdout line.
    No generated-code, arbitrary suffix, second event or JSON is ignored.
    Runtime callers use actual pinned native-rank PIDs and installed Torch path.
    """
    import importlib.util
    import re
    if marker not in REVIEW_LOG_MARKERS or line.count(marker) != 1:
        raise ValueError('Unknown/duplicate evidence marker')
    if any(other != marker and other in line for other in REVIEW_LOG_MARKERS):
        raise ValueError('Interleaved evidence markers require a clean record')
    if type(driver_pid) is not int or driver_pid <= 0:
        raise ValueError('Actual driver PID required')
    workers = set(worker_pids)
    if any(type(pid) is not int or pid <= 0 for pid in workers):
        raise ValueError('Actual worker PID set required')
    prefix, raw = line.split(marker, 1)
    prefix = prefix.strip()
    if prefix:
        match = re.fullmatch(r'(?P<colour>\x1b\[36m)?\((?P<kind>main_task|WorkerDict) pid=(?P<pid>[1-9][0-9]*)\)(?(colour)\x1b\[0m)', prefix)
        if not match:
            raise ValueError('Unrecognized evidence transport prefix')
        kind, pid = match.group('kind'), int(match.group('pid'))
        if ((kind == 'main_task' and pid != driver_pid) or
                (kind == 'WorkerDict' and (marker != 'FULL_STATE_RANK_JSON=' or pid not in workers))):
            raise ValueError('Evidence transport PID/kind mismatch')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate marker JSON key: ' + key)
            result[key] = value
        return result
    def finite_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError('Overflowed/nonfinite marker JSON float')
        return value
    decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_float=finite_float,
             parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite marker JSON')))
    value, end = decoder.raw_decode(raw.lstrip())
    if not isinstance(value, dict):
        raise ValueError('Evidence marker must contain one JSON object')
    if (('driver_pid' in value and value['driver_pid'] != driver_pid) or
            ('job_id' in value and value['job_id'] != job_id)):
        raise ValueError('Evidence body driver/job mismatch')
    if prefix and kind == 'WorkerDict' and value.get('pid') != pid:
        raise ValueError('Native writer body/transport PID mismatch')
    suffix = raw.lstrip()[end:].strip()
    if not suffix:
        return value
    if checkpoint_warning_path is None:
        spec = importlib.util.find_spec('torch')
        if spec is None or spec.origin is None:
            raise ValueError('Actual Torch warning source unavailable')
        checkpoint_warning_path = Path(spec.origin).resolve().parent / 'utils' / 'checkpoint.py'
    checkpoint_warning_path = Path(checkpoint_warning_path)
    if not checkpoint_warning_path.is_absolute():
        raise ValueError('Actual absolute checkpoint warning source required')
    warning = (':1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. '
               "Please use `torch.amp.autocast('cpu', args...)` instead.")
    pattern = (r'(?P<colour>\x1b\[36m)?\(WorkerDict pid=(?P<pid>[1-9][0-9]*)\)(?(colour)\x1b\[0m) '
               + re.escape(str(checkpoint_warning_path) + warning)
               + r'(?:(?P<dedupcolour>\x1b\[32m)? \[repeated (?P<repeat>[1-8])x across cluster\](?(dedupcolour)\x1b\[0m))?')
    match = re.fullmatch(pattern, suffix)
    if not match or int(match.group('pid')) not in workers:
        raise ValueError('Unrecognized/unowned interleaved warning suffix')
    return value


def manifest_worker_pids(manifest, root):
    """Worker identities are read only from every already-pinned native rank report."""
    world = manifest['world_size']
    if world != 8:
        raise ValueError('Actual eight native rank reports required')
    pids = set()
    for rank in range(world):
        report = load_json(verify_file(manifest['files'][f'actor/training_state/rank_{rank}.json'], root))
        if (report.get('version') != 3 or report.get('algorithm') != 'pure_opsd' or
                report.get('status') != 'EXACT_FULL_STATE_RELOAD_OK' or
                report.get('model_state_dict_type') != 'SHARDED_STATE_DICT' or
                report.get('model_dtensor_entries', 0) <= 0):
            raise ValueError('Successful native full-state report required')
        pt = manifest['files'][f'actor/training_state/rank_{rank}.pt']
        if (report.get('path') != pt['path'] or
                report.get('sha256') != pt['sha256'] or report.get('state_sha256') != pt['sha256']):
            raise ValueError('Native report/raw file binding differs')
        if (report['rank'], report['world_size'], report['step']) != (rank, 8, 5):
            raise ValueError('Native rank identity differs')
        pid = report['pid']
        if type(pid) is not int or pid <= 0 or pid in pids:
            raise ValueError('Native rank PID missing/aliased')
        pids.add(pid)
    return pids
