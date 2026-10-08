"""Submit the complete two-cycle experiment as five dependent stage allocations."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

root = Path(__file__).resolve().parent.parent
os.chdir(root)
now = datetime.now(timezone.utc)
run = root/'runs'/('staged-3b-'+now.strftime('%Y%m%d-%H%M%S'))
run.mkdir()  # Never reuse an experiment directory.
receipt = {'created_at_utc': now.isoformat(), 'experiment_dir': str(run),
           'gpus': 4, 'qos': 'normal', 'partition': 'L40S', 'jobs': [],
           'configuration': {'cycles': 2, 'updates_per_cycle': 200,
                             'train_batch': 512, 'ppo_minibatch': 128,
                             'actor_micro_per_gpu': 2, 'score_micro_per_gpu': 4,
                             'trajectory_mean_batching': True}}
previous = None
for phase in ('grpo1', 'opsd1', 'grpo2', 'opsd2', 'eval'):
    args = ['sbatch', '--parsable', '--job-name=search-e1-'+phase,
            '--export=ALL,PHASE='+phase+',RUN='+str(run)]
    if previous:
        args += ['--dependency=afterok:'+previous, '--kill-on-invalid-dep=yes']
    args += ['scripts/lumia_phase.sbatch']
    job = subprocess.check_output(args, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected Slurm receipt: '+job)
    receipt['jobs'].append({'phase': phase, 'job_id': int(job), 'afterok': int(previous) if previous else None})
    previous = job
    # Preserve partial receipts if a later submission fails; do not submit duplicates.
    (run/'submission.json').write_text(json.dumps(receipt, indent=2)+'\n')
(root/'runs'/'latest_submission.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
