"""Final gate for this exact weight-continuation segment and full seven-source EM."""
import json
import re
from pathlib import Path
from verl.utils.checkpoint_integrity import validate_saved_checkpoint

BASE=Path('/data/search-r1')
m=json.loads((BASE/'logs/a800-continuation-manifest.json').read_text())
five=json.loads((BASE/'logs/a800-five-resumed-updates-acceptance.json').read_text())
assert five['run_id']==m['run_id'] and five['accepted_updates']==5
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',Path(m['train_log']).read_text(errors='replace'))
assert text.splitlines()[0]=='SEARCH_R1_SEGMENT '+m['run_id']
steps=[int(match.group(1)) for match in re.finditer(r'\bstep:(\d+) - [^\n]*critic/rewards/mean:',text)]
assert steps==list(range(m['completed_updates']+1,501)), 'Missing or duplicate new actual updates'
checkpoint=Path(m['checkpoint_dir'])/'actor/global_step_500'
count=validate_saved_checkpoint(checkpoint,Path(m['reference_model']),m['run_started'])
metrics=[json.loads(line.split('SEARCH_R1_FINAL_METRICS_JSON=',1)[1]) for line in text.splitlines() if 'SEARCH_R1_FINAL_METRICS_JSON=' in line]
assert len(metrics)==1, 'Missing or ambiguous final validation'
metrics=metrics[0]
expected={'nq':3610,'triviaqa':11313,'popqa':14267,'hotpotqa':7405,'2wikimultihopqa':12576,'musique':2417,'bamboogle':125}
assert json.loads((BASE/'data/paper-grpo/dataset-audit.json').read_text())['eval_counts']==expected
for source,n in expected.items():
    assert metrics['val/evaluated_count/'+source]==n, 'Incomplete evaluation '+source
    assert 0<=metrics['val/test_score/'+source]<=1, 'Invalid EM '+source
result=dict(run_id=m['run_id'],final_checkpoint=str(checkpoint),logical_completed_updates=500,
            actual_updates_this_segment=len(steps),validated_tensor_count=count,
            finite_all_saved_tensors=True,metrics=metrics,
            checkpoint_semantics='Interrupted weight continuation with fresh Adam, not state-exact resume')
(BASE/'logs/full-run-acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
