"""Direct eight-GPU memory.free samples; observations never stop processes."""
import argparse
import json
import os
from pathlib import Path
import time

try:
    from .acceptance import inventory, sha, process
    from .checkpoint import exclusive_json
except ImportError:
    from acceptance import inventory, sha, process
    from checkpoint import exclusive_json


def observe(config_path, output, binding_output, interval=2):
    config_path = Path(config_path).resolve()
    config_hash = sha(config_path)
    config = json.loads(config_path.read_text())
    require_uid = config['owner']['uid']
    if os.getuid() != require_uid:
        raise ValueError('Memory observer was started under another UID')
    output = Path(output).resolve()
    root = Path(config['paths']['receipts']).resolve()
    execution_root = Path(config['paths']['output_root']).resolve()
    binding_output = Path(binding_output).resolve()
    if (not output.is_relative_to(root) or not binding_output.is_relative_to(root)
            or output.is_relative_to(execution_root) or binding_output.is_relative_to(execution_root)):
        raise ValueError('Observer receipts must be in this experiment receipts, outside child output scope')
    identity = process(os.getpid())
    if not identity.get('run_id') or identity['run_id'] == config['run_id']:
        raise ValueError('Observer requires an independent nonempty E1_RUN_ID marker')
    identity = {key: identity[key] for key in ('pid', 'uid', 'starttick', 'run_id')}
    identity['source_sha256'] = sha(Path(__file__).resolve())
    manifest = json.loads(Path(config['helper_source_manifest']['path']).read_text())
    if sha(config['helper_source_manifest']['path']) != config['helper_source_manifest']['sha256']:
        raise ValueError('Observer helper manifest changed')
    expected = manifest['files']['memory_observer.py']
    expected = expected['sha256'] if isinstance(expected, dict) else expected
    if identity['source_sha256'] != expected:
        raise ValueError('Observer source differs from published helper identity')
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusive_json(binding_output, dict(schema='opsd-online-memory-observer-binding-v1', accepted=True,
        experiment_run_id=config['run_id'], config_sha256=config_hash,
        live_report_path=str(output), observer_identity=identity))
    rows_path = output.with_suffix('.samples.jsonl')
    expected = config['owner']['gpu_uuids']
    minimum = {index: None for index in range(8)}
    samples, failures = 0, 0
    with rows_path.open('x') as stream:
        while True:
            if sha(config_path) != config_hash:
                raise RuntimeError('Frozen observer configuration changed')
            devices, applications = inventory()
            if [item['uuid'] for item in devices] != expected:
                raise RuntimeError('Observer GPU UUID mapping changed')
            samples += 1
            for device in devices:
                index, free = device['index'], device['actual_free_mib']
                minimum[index] = free if minimum[index] is None else min(minimum[index], free)
            good = all(device['actual_free_mib'] >= config['resources']['training_minimum_free_mib']
                       for device in devices)
            failures += int(not good)
            current = dict(sample_unix=time.time(), devices=devices, compute_applications=applications,
                           actual_free_limit_met=good)
            stream.write(json.dumps(current, sort_keys=True) + '\n')
            stream.flush()
            report = dict(schema='opsd-online-direct-memory-v1', observer_identity=identity,
                          run_id=config['run_id'], config_sha256=config_hash,
                          measurement='nvidia-smi memory.free, not memory.total-memory.used',
                          samples=samples, minimum_actual_free_mib=minimum,
                          samples_below_limit=failures, current=current,
                          updated_unix=time.time(), no_signals_sent=True)
            temporary = output.with_name(output.name + '.tmp-' + str(os.getpid()))
            temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
            os.replace(temporary, output)
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--binding-output', required=True)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    if not 0.2 <= args.interval <= 60:
        raise ValueError('Observer interval must be between .2 and 60 seconds')
    observe(args.config, args.output, args.binding_output, args.interval)


if __name__ == '__main__':
    main()
