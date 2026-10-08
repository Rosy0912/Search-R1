"""CPU-only post-job evidence capture; never submits or restarts training."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN = ROOT / 'runs/tree-instruct300-3154'
BASE = ROOT / 'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
CHECKER = ROOT / 'tools/check_real_checkpoint.py'
CHECKER_SHA = 'b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
COMMIT = 'f994af33c6c93b6b9844863be912cab2a4020dc1'
OBSERVER_SHA = '7d17b52594ba78542f4dac2c85b136f6cb806935c9b42bd863c6d8eae41d4eef'


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    assert os.environ.get('SLURMD_NODENAME') == 'h100-9'
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') in ('', 'NoDevFiles')
    output = RUN / 'terminal-cpu-verification'
    output.mkdir(exist_ok=False)
    started = time.time()
    cpu_started = time.process_time()
    report = {'status': 'INCOMPLETE', 'parent_job_id': 3154,
              'cpu_job_id': os.environ.get('SLURM_JOB_ID'),
              'allocated_gpus': None,
              'scope': 'Parent terminal capture, actual counters, final coverage and CPU HF300 only; independent final acceptance still required'}
    try:
        terminal = subprocess.run(['scontrol', 'show', 'job', '3154'], capture_output=True, text=True)
        save(output / 'parent-slurm.json', {'returncode': terminal.returncode,
                                          'stdout': terminal.stdout, 'stderr': terminal.stderr})
        assert socket.gethostname().split('.')[0] == 'h100-9'
        cpu_job = os.environ.get('SLURM_JOB_ID', '')
        assert cpu_job.isdigit() and cpu_job != '3154'
        allocation = subprocess.run(['scontrol', 'show', 'job', cpu_job], capture_output=True, text=True)
        save(output / 'cpu-allocation.json', {'returncode': allocation.returncode,
                                             'stdout': allocation.stdout, 'stderr': allocation.stderr})
        assert allocation.returncode == 0
        fields = dict(re.findall(r'(\S+?)=(\S+)', allocation.stdout))
        assert fields['JobId'] == cpu_job and fields['JobState'] == 'RUNNING'
        assert fields['NodeList'] == fields['BatchHost'] == 'h100-9'
        assert int(fields['NumCPUs']) >= 8 and int(fields['CPUs/Task']) >= 8
        memory = re.fullmatch(r'(\d+)([MG])', fields['MinMemoryNode'])
        assert memory and int(memory[1]) * (1024 if memory[2] == 'G' else 1) >= 32768
        for key in ('TRES', 'TresPerNode'):
            for gpu in re.findall(r'(?:gres[/:])?gpu(?::[^,=\s]+)?[=:](\d+)', fields.get(key, '')):
                assert int(gpu) == 0, 'CPU verification must reserve no GPUs'
        assert not os.environ.get('SLURM_JOB_GPUS'), 'CPU job has reserved GPUs'
        report['allocated_gpus'] = 0
        report['allocated_cpus'] = int(fields['NumCPUs'])
        report['allocated_memory_node'] = fields['MinMemoryNode']
        assert terminal.returncode == 0, 'Parent scheduler record unavailable; preserve limitation'
        parent = dict(re.findall(r'(\S+?)=(\S+)', terminal.stdout))
        assert parent['JobId'] == '3154' and parent['NodeList'] == parent['BatchHost'] == 'h100-9'
        assert re.search(r'\bJobState=COMPLETED\b', terminal.stdout)
        assert re.search(r'\bExitCode=0:0\b', terminal.stdout)
        assert re.search(r'\bRestarts=0\b', terminal.stdout)
        result = json.loads((RUN / 'run-result.json').read_text())
        assert result['status'] == 'ACTUAL_STAGE_COMPLETED_UNREVIEWED'
        assert result['job_id'] == '3154' and result['code_commit'] == COMMIT
        assert result['training_body_exit_code'] == 0 and result['entry_config_verified'] is True
        assert result['final_guard_healthy'] is True
        assert isinstance(result['cleanup_residual'], dict) and result['cleanup_residual'] == {}
        numbers = []
        for line in (RUN / 'stage.log').read_text().splitlines():
            match = re.search(r'(?<![\w/])step:(\d+) - ', line)
            counter = re.search(r'(?:^| - )training/actual_outer_updates:([\d.]+)(?: - |$)', line)
            if match and counter:
                assert int(match[1]) == float(counter[1])
                numbers.append(int(match[1]))
        assert numbers == list(range(1, 301)), 'Actual logged update sequence differs from 1..300'
        reports = [(p, json.loads(p.read_text())) for p in (RUN / 'output/validation-cost').glob('*.json')]
        final_paths = [(p, r) for p, r in reports if r.get('phase') == 'final']
        final = [r for p, r in final_paths]
        assert len(final) == 1
        assert final[0]['status'] == 'VALIDATION_COST_COMPLETE'
        assert final[0]['dataset_rows'] == final[0]['evaluated_rows'] == 51713
        assert final[0]['job_id'] == '3154'
        assert final[0]['global_step'] == 301 and final[0]['observer_source_sha256'] == OBSERVER_SHA
        assert final[0]['pid'] == 4048887, 'Final observer must use the accepted live Ray trainer'
        assert final[0]['run'] == str(RUN / 'output')
        ledger = final_paths[0][0].with_suffix('.jsonl')
        assert hashlib.sha256(ledger.read_bytes()).hexdigest() == final[0]['ledger_sha256']
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert events and [e['event'] for e in events] == list(range(1, len(events) + 1))
        assert all(e['phase'] == 'final' and e['global_step'] == 301
                   and e['pid'] == final[0]['pid']
                   and e['validation_sequence'] == final[0]['validation_sequence'] for e in events)
        assert events[0]['kind'] == 'validation_start' and events[0]['dataset_rows'] == 51713
        assert events[0]['observer_source_sha256'] == OBSERVER_SHA
        assert events[0]['job_id'] == '3154' and events[0]['run'] == str(RUN / 'output')
        assert events[-1]['kind'] == 'validation_completed' and events[-1]['evaluated_rows'] == 51713
        assert sum(e['rows'] for e in events if e['kind'] == 'real_sample_batch') == 51713
        assert not any(e['kind'].endswith('_failed') for e in events)
        assert final[0]['counters']['real_sample_rows'] == 51713
        assert hashlib.sha256(CHECKER.read_bytes()).hexdigest() == CHECKER_SHA
        checkpoint = RUN / 'output/checkpoints/tree-instruct300/actor/global_step_300'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
        with (output / 'hf300-reload.log').open('x') as log:
            completed = subprocess.run([str(ROOT / 'env/bin/python'), str(CHECKER),
                                        '--base', str(BASE), '--checkpoint', str(checkpoint),
                                        '--output', str(output / 'hf300-reload.json')],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
        assert completed.returncode == 0, 'HF300 CPU reload failed'
        hf = json.loads((output / 'hf300-reload.json').read_text())
        assert hf['status'] == 'HF_RELOAD_PARAMETER_DELTA_ACCEPTED'
        report.update(status='TERMINAL_COUNTER_COVERAGE_HF300_CPU_VERIFIED_UNREVIEWED',
                      actual_updates=300, final_evaluated_rows=51713,
                      checkpoint=str(checkpoint), final_report=final[0])
    except Exception as error:
        report.update(status='TERMINAL_VERIFICATION_FAILED_OR_INCOMPLETE',
                      error_type=type(error).__name__, error=str(error))
    finally:
        report['wall_seconds'] = time.time() - started
        report['process_cpu_seconds'] = time.process_time() - cpu_started
        report['child_cpu_seconds'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        save(output / 'result.json', report)
        print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'TERMINAL_COUNTER_COVERAGE_HF300_CPU_VERIFIED_UNREVIEWED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
