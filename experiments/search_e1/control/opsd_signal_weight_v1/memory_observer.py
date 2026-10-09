"""Direct four-owned-GPU memory.free samples; observations never stop processes."""
import argparse
import json
import os
from pathlib import Path
import time

try:
    from .gates import inventory, sha, process
    from .checkpoint import exclusive_json, fileproof, checked_json
except ImportError:
    from gates import inventory, sha, process
    from checkpoint import exclusive_json, fileproof, checked_json


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
    minimum = {index: None for index in config['owner']['physical_gpus']}
    samples, failures = 0, 0
    with rows_path.open('x') as stream:
        while True:
            if sha(config_path) != config_hash:
                raise RuntimeError('Frozen observer configuration changed')
            all_devices, all_applications = inventory()
            devices = [d for d in all_devices if d['index'] in config['owner']['physical_gpus']]
            owned_uuids = set(config['owner']['gpu_uuids'])
            applications = [a for a in all_applications if a['gpu_uuid'] in owned_uuids]
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
            status_path=execution_root/'status.json'
            if status_path.exists():
                try:status,status_proof=checked_json(fileproof(status_path))
                except (FileNotFoundError,ValueError,json.JSONDecodeError):status=None  # Atomic publication read race; observe again.
                if status is not None and (status.get('run_id')==config['run_id'] and status.get('phase')=='TRAIN_COMPLETE'):
                    if (status.get('config_sha256')!=config_hash or status.get('outer_iterations')!=25
                            or status.get('resource_release',{}).get('all_owned_release',{}).get('accepted') is not True):
                        raise ValueError('Observer cannot accept an incomplete/wrong-config terminal status')
                    ack_path=root/'study-driver-release.json'
                    if not ack_path.exists():
                        time.sleep(interval);continue  # Keep the driver live-role gate satisfiable until its independent release audit.
                    try:ack,ack_proof=checked_json(fileproof(ack_path))
                    except (FileNotFoundError,ValueError,json.JSONDecodeError):
                        time.sleep(interval);continue  # ACK durable write may be observed before its fsync completes.
                    registry,_=checked_json(fileproof(config['gates']['scoped_cpu_processes_path']))
                    drivers=[role['identity'] for role in registry['roles'] if role['role']=='study_driver']
                    if len(drivers)!=1:
                        raise ValueError('Observer needs exactly one registered independent study driver')
                    if ack.get('driver_identity')!=drivers[0]:
                        time.sleep(interval);continue  # Preserve the previous owner's ACK; await this driver's actual release.
                    try:driver=process(drivers[0]['pid'])
                    except (FileNotFoundError,ProcessLookupError):
                        time.sleep(interval);continue  # Previous driver has actually exited; a new owner must bind itself.
                    if any(driver.get(k)!=drivers[0].get(k) for k in ('pid','uid','starttick','ppid','pgid','argv','run_id')):
                        time.sleep(interval);continue  # PID reuse does not validate an old driver's ACK.
                    if not (ack.get('schema')=='opsd-signal-study-driver-release-v1' and ack.get('accepted') is True
                            and ack.get('run_id')==config['run_id'] and ack.get('config_sha256')==config_hash
                            and ack.get('actual_wait_returncode')==0 and ack.get('release',{}).get('accepted') is True):
                        raise ValueError('Observer termination requires actual current-driver wait0 and accepted release ACK')
                    terminal,_=checked_json(ack['controller_terminal'])
                    if (terminal.get('actual_wait_returncode')!=0 or
                            any(terminal['controller'].get(k)!=status['owner'].get(k) for k in ('pid','uid','starttick','run_id'))):
                        raise ValueError('Observer ACK controller wait0 is not this completed25 controller')
                    # The group driver will separately wait() this actual process.
                    # No signals or ownership assumptions from PID disappearance.
                    stream.flush();os.fsync(stream.fileno())
                    exclusive_json(root/'memory-terminal.json',dict(schema='opsd-signal-memory-terminal-v1',
                        accepted=True,run_id=config['run_id'],config_sha256=config_hash,observer_identity=identity,
                        terminal_training_status=status_proof,live_report=fileproof(output),samples_file=fileproof(rows_path),
                        study_driver_release_ack=ack_proof,actual_driver_identity=driver,
                        samples=samples,samples_below_limit=failures,current=current,
                        natural_exit_after_complete25=True,no_signals_sent=True,finished_unix=time.time()))
                    return
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
