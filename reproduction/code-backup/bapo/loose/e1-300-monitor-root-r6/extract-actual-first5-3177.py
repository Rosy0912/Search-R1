import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004')
run = root / 'runs/bapo-instruct300-formal300-3177'
tool = root / 'evidence-tools/step5-extract.py'
assert hashlib.sha256(tool.read_bytes()).hexdigest() == '563adfa7f0d3ffecc0a5cbf1b9c95ecdc90ee23c5e8d7c8f9a68792bf607f9e6'
text = (run / 'train.log').read_text()
def events(marker):
    return [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
paused = events('E1_STEP5_PAUSED_JSON=')
assert len(paused) == 1 and paused[0]['pid'] == 3908944
assert [e['step'] for e in events('BAPO_TRAIN_METRICS_JSON=')] == [1, 2, 3, 4, 5]
assert len(events('ACTOR_RANK_UPDATE_JSON=')) == 40
assert not events('E1_STEP5_RELEASED_JSON=')
assert not (run / 'logs/step5-independent-release.json').exists()
slurm = subprocess.check_output(['scontrol', 'show', 'job', '-dd', '3177'])
assert b'JobState=RUNNING' in slurm and b'Restarts=0' in slurm
capture = run / 'logs/slurm-step5-current-firstactual-20261004.txt'
with capture.open('xb') as handle:
    handle.write(slurm)
out = run / 'logs/independent-step5-extraction-firstactual-20261004'
assert not out.exists()
env = os.environ.copy()
env.update(PYTHONPATH=str(root / 'repo'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8',
    MKL_NUM_THREADS='8', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
args = [str(root / 'env/bin/python'), str(tool), '--run', str(run), '--job-id', '3177',
    '--driver-pid', '3908944', '--output-dir', str(out), '--slurm-evidence', str(capture), '--cpu-threads', '8']
raise SystemExit(subprocess.run(args, cwd=root / 'repo', env=env).returncode)
