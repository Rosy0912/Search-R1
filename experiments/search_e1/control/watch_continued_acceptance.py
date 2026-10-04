"""Read-only acceptance observer; never stops or restarts the training job."""
import argparse
import json
import re
import time
from pathlib import Path

from verify_checkpoint import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--driver-pid', required=True, type=int)
    args = parser.parse_args()
    base = Path(args.base)
    checkpoint = base / 'runs/r1/round-1-grpo/actor/global_step_3'
    source = base / 'runs/r1/smoke-025/actor/global_step_2'
    proof_path = base / 'receipts/continued-step3-weights.json'
    while True:
        receipt = json.loads((base / 'receipts/r1-round-1-grpo.json').read_text())
        if receipt['run_id'] != args.run_id:
            raise RuntimeError('Run identity changed; observer will not follow another run')
        if receipt['status'].startswith('FAILED'):
            raise RuntimeError('Training failed; intervention required')
        text = (base / 'runs/r1/round-1-grpo/train.log').read_text(errors='replace')
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        steps = {int(s) for s in re.findall(r'\bstep:(\d+) -', text)}
        continued = {int(s) for s in re.findall(
            r'\(main_task pid=' + str(args.driver_pid) + r'\) step:(\d+) -', text)}
        if 3 in continued and not proof_path.exists():
            initial = verify(base / 'assets/model', checkpoint)
            previous = verify(source, checkpoint)
            report = dict(run_id=args.run_id, time=time.time(), initial=initial,
                          previous_step2=previous, accepted=initial['accepted'] and previous['accepted'])
            proof_path.write_text(json.dumps(report, indent=2))
            print('CONTINUED_STEP3_WEIGHTS', json.dumps(report), flush=True)
            if not report['accepted']:
                raise RuntimeError('Continued checkpoint proof incomplete; inspect receipt')
        if {3, 4, 5} <= continued and proof_path.exists():
            report = dict(run_id=args.run_id, time=time.time(), completed_steps=sorted(steps),
                          source_steps=[1, 2], continued_steps=[3, 4, 5],
                          checkpoint_proof=str(proof_path), status=receipt['status'],
                          heartbeat_age=time.time() - receipt['updated_unix'],
                          peaks_mib=receipt['peaks_mib'], full_reproduction_complete=False)
            if report['status'] == 'RUNNING' and report['heartbeat_age'] < 120:
                (base / 'receipts/continued-five-iterations.json').write_text(json.dumps(report, indent=2))
                print('CONTINUED_FIVE_ITERATIONS_RECORDED_TRAINING_CONTINUES', flush=True)
                return
        time.sleep(30)


if __name__ == '__main__':
    main()
