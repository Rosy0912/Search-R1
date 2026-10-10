"""Only original boundary Hold retries; original evaluation/worker math untouched."""
import json
from pathlib import Path
import time

from recovery_common import checked_json, require, same_ref


def retry_hold(original, hold_class, pause=time.sleep, report=None):
    def boundary(*args, **kwargs):
        while True:
            try:
                return original(*args, **kwargs)
            except hold_class as error:
                if report is None:
                    print(json.dumps(dict(status='HOLD_ORIGINAL_EVALUATION_BOUNDARY',
                                          error=str(error), original_gate_retried=True,
                                          no_gate_bypass=True)), flush=True)
                else:
                    report(error)
                pause(1)
    return boundary


def evaluation_dispatch(parsed, python, configs, original_root):
    reference = dict(path=parsed.config, sha256=parsed.config_sha256)
    require(reference['path'] in configs, 'Evaluator outside four original configs')
    config, expected = configs[reference['path']]
    require(reference == expected and python == config['paths']['python']
            and parsed.completed_outer in (15, 25), 'Evaluator config/python/15-or25 changed')
    output = Path(config['paths']['output_root']) / 'evaluations' / ('outer-%02d' % parsed.completed_outer)
    require(parsed.output_dir == str(output / 'execution')
            and parsed.receipt == str(output / 'child-complete.json'), 'Evaluator canonical scope changed')
    checkpoint_ref = dict(path=parsed.checkpoint, sha256=parsed.checkpoint_sha256)
    checkpoint = checked_json(checkpoint_ref)
    complete = checked_json(dict(path=str(Path(config['paths']['output_root']) / 'outers' /
                        ('outer-%05d' % (parsed.completed_outer-1)) / 'training' / 'outer-complete.json'),
                         sha256=_metadata_sha(config, parsed.completed_outer)))
    require(checkpoint['accepted'] is True and checkpoint['run_id'] == config['run_id']
            and checkpoint['config_sha256'] == expected['sha256']
            and checkpoint['cursor']['outer_index'] == parsed.completed_outer-1
            and checkpoint['cursor']['next_pair_cursor'] == checkpoint['cursor']['usable_pair_count']
            and complete['accepted'] is True and complete['run_id'] == config['run_id']
            and complete['outer_index'] == parsed.completed_outer-1
            and checkpoint['optimizer_updates'] == complete['optimizer_updates']
            and same_ref(complete['checkpoint'], checkpoint_ref),
            'Evaluator must execute exact actual committed raw15/25 end state')
    require(str(Path(original_root) / 'evaluation.py') ==
            str(Path(config['paths']['helper_root']) / 'evaluation.py'), 'Original evaluator source differs')
    return checkpoint_ref


def _metadata_sha(config, completed_outer):
    import hashlib
    path = Path(config['paths']['output_root']) / 'outers' / ('outer-%05d' % (completed_outer-1)) / 'training' / 'outer-complete.json'
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Actual raw completion changed')
    return hashlib.sha256(raw).hexdigest()


def validate_actual_evaluation_entry(original_entry, root, actual_argv, manifest, file_ref):
    path = str(Path(root) / 'evaluation.py')
    require(original_entry == path and actual_argv.count('--original-entry') == 1
            and actual_argv.index('--original-entry')+1 < len(actual_argv)
            and actual_argv[actual_argv.index('--original-entry')+1] == path and path in actual_argv,
            'Real /proc evaluator argv must expose exact original evaluation.py source')
    proof = file_ref(path); expected = manifest['files']['evaluation.py']
    require(proof['sha256'] == expected['sha256'] and proof['bytes'] == expected['bytes'],
            'Actually executed original evaluator bytes differ')
    return proof
