"""Read-only actual run snapshot; does not accept final artifacts or mutate a job."""
import json
import math
import subprocess
from pathlib import Path

run = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177')
lines = (run / 'train.log').read_text(errors='replace').splitlines()
def events(marker):
    return [json.loads(line.split(marker, 1)[1]) for line in lines if marker in line]
outer = events('BAPO_TRAIN_METRICS_JSON=')
ranks = events('ACTOR_RANK_UPDATE_JSON=')
released = events('E1_STEP5_RELEASED_JSON=')
completed = max((e['step'] for e in outer), default=0)
issues = []
if [e['step'] for e in outer] != list(range(1, completed + 1)):
    issues.append('Outer summaries not consecutive or unique')
for e in outer:
    step, m = e['step'], e['metrics']
    if m['actor/optimizer_steps'] != 20 or m['actor/scheduler_outer_steps'] != step:
        issues.append('Optimizer/scheduler clock at outer ' + str(step))
    for key, wanted in [('actor/lr_before_outer', min(step-1,57)*1e-6/57), ('actor/lr', min(step,57)*1e-6/57)]:
        if not math.isclose(m[key], wanted, rel_tol=1e-10, abs_tol=1e-15):
            issues.append('LR clock ' + key + ' at outer ' + str(step))
    for key in ('bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/mask_on_padding_tokens','bapo/model_tokens_missing_loss','bapo/new_logprob_nonfinite','bapo/old_logprob_nonfinite'):
        if m[key] != 0: issues.append(key + ' at outer ' + str(step))
    for key in ('actor/pg_loss','actor/kl_loss','actor/grad_norm','bapo/grad_norm_min','bapo/grad_norm_max'):
        if not math.isfinite(m[key]): issues.append(key + ' nonfinite at outer ' + str(step))
    if m['bapo/stage_evaluated_count'] != 64: issues.append('Stage size at outer ' + str(step))
    records = [r for r in ranks if r['scheduler_last_epoch'] == step]
    if len(records) != 8 or {r['rank'] for r in records} != set(range(8)):
        issues.append('Incomplete/duplicate rank records at outer ' + str(step))
    for r in records:
        if r['optimizer_steps'] != 20: issues.append('Rank optimizer count at outer ' + str(step))
        for key in ('actor/pg_loss','actor/entropy_loss','actor/grad_norm'):
            values = r['metrics'].get(key, [])
            if len(values) != 20 or not all(math.isfinite(v) for v in values):
                issues.append('Rank finite mini metrics ' + key + ' at outer ' + str(step))
watch = [json.loads(x) for x in (run/'resource-watch.jsonl').read_text().splitlines() if x]
slurm = subprocess.run(['scontrol','show','job','-dd','3177'], capture_output=True, text=True)
print(json.dumps(dict(job_id=3177, run=str(run), actual_completed_outer=completed,
    source=(run/'source-commit.txt').read_text().strip(), audit_issues=issues,
    resource_samples=len(watch), resource_pauses=sum(bool(r.get('pause')) for r in watch),
    latest_resource_time=watch[-1].get('time') if watch else None,
    released=released, current_step5_pause=bool(events('E1_STEP5_PAUSED_JSON=')) and not released,
    last_outer=outer[-1] if outer else None,
    recent_outer=[dict(step=e['step'], seconds=e['metrics']['timing_s/step'], stage=e['metrics']['bapo/stage_validation']) for e in outer[-5:]],
    rank_records=len(ranks), train_tail=[line[:1200] for line in lines[-3:]],
    checkpoint_dirs=sorted(p.name for p in (run/'checkpoints/actor').glob('global_step_*') if p.is_dir()),
    slurm_returncode=slurm.returncode, slurm=slurm.stdout, slurm_stderr=slurm.stderr), ensure_ascii=False, indent=2))
