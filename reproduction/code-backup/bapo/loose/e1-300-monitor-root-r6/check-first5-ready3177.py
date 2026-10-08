import json
import subprocess
from pathlib import Path

run = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177')
lines = (run / 'train.log').read_text(errors='replace').splitlines()
def events(marker):
    return [json.loads(line.split(marker, 1)[1]) for line in lines if marker in line]
paused = events('E1_STEP5_PAUSED_JSON=')
outer = events('BAPO_TRAIN_METRICS_JSON=')
ranks = events('ACTOR_RANK_UPDATE_JSON=')
slurm = subprocess.run(['scontrol', 'show', 'job', '-dd', '3177'], capture_output=True, text=True)
print(json.dumps(dict(job_id=3177, paused=paused, completed_steps=[e['step'] for e in outer],
    rank_records=len(ranks), released=events('E1_STEP5_RELEASED_JSON='),
    release_exists=(run / 'logs/step5-independent-release.json').exists(),
    slurm_returncode=slurm.returncode, slurm=slurm.stdout, slurm_stderr=slurm.stderr,
    train_tail=lines[-5:]), ensure_ascii=False, indent=2))
