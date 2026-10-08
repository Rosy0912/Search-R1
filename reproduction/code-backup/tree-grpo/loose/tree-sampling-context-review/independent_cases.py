"""Independent fabricated CPU counterexamples; no remote reads or writes."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import time

import torch

WORK = Path(__file__).resolve().parents[1]
CANDIDATE = Path(__file__).parent/'candidate-1ffa12889d57b27a732ea440e46128429de20b324afbbab589d974833649848d.py'
spec = importlib.util.spec_from_file_location('candidate', CANDIDATE)
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

def snap(ids, response, *, raw=False, positions=None, attention=None, old=None, sampled=None, loss=None, weights=None):
    ids = list(ids); response = list(response)
    value = {'input_ids': torch.tensor([ids], dtype=torch.int64),
             'responses': torch.tensor([response], dtype=torch.int64),
             'position_ids': torch.tensor([positions or list(range(len(ids)))], dtype=torch.int64),
             'attention_mask': torch.tensor([attention or [1]*len(ids)], dtype=torch.int64),
             'old_log_probs': torch.tensor([old if old is not None else [-2.]*len(response)],dtype=torch.float64)}
    if raw:
        value['sampled_log_probs'] = torch.tensor([sampled if sampled is not None else [-2.]*len(response)],dtype=torch.float64)
    else:
        value['loss_mask'] = torch.tensor([loss if loss is not None else [1]*len(response)],dtype=torch.int64)
        value['policy_weights'] = torch.tensor([weights if weights is not None else [1./len(response)]*len(response)],dtype=torch.float64)
    return value

def rejected(fn):
    try: fn()
    except (ValueError,RuntimeError,KeyError): return True
    return False

checks=[]; defects=[]
raw=snap([1,2,3],[2,3],raw=True)
index=A.RawIndex(); index.add_snapshot(raw,'raw')
changed=snap([1,2,3],[2,3],positions=[0,1,99])
r=A.analyze_targets(index,[('changed-current-position',changed)])
assert r['exact_context_matched_token_occurrences']==1 and r['exact_context_unmatched_token_occurrences']==1
checks.append('current_position_key_change_excludes_only_changed_response')

raw_hole=snap([1,99,2,3],[2,3],raw=True,positions=[0,999,1,2],attention=[1,0,1,1])
idx=A.RawIndex();idx.add_snapshot(raw_hole,'attention-hole')
r=A.analyze_targets(idx,[('stripped-padding',snap([1,2,3],[2,3]))])
assert r['exact_context_matched_token_occurrences']==2
checks.append('interior_attention_zero_token_and_position_not_added_to_context')

bad_shape=dict(raw);bad_shape['old_log_probs']=raw['old_log_probs'][:,:1]
assert rejected(lambda:A.RawIndex().add_snapshot(bad_shape,'broadcast-attempt'))
bad_suffix=dict(raw);bad_suffix['responses']=torch.tensor([[99,99]])
assert rejected(lambda:A.RawIndex().add_snapshot(bad_suffix,'wrong-suffix'))
bad_attention=dict(raw);bad_attention['attention_mask']=torch.tensor([[1.,float('nan'),1.]])
assert rejected(lambda:A.RawIndex().add_snapshot(bad_attention,'nan-mask'))
checks.append('exact_shapes_suffix_and_binary_attention_refuse_broadcast_or_invalid_mask')

one=snap([1,2],[2],raw=True,old=[float('nan')],sampled=[float('inf')])
idx=A.RawIndex();idx.add_snapshot(one,'nonfinite-raw')
r=A.analyze_targets(idx,[('finite-final',snap([1,2],[2]))])
assert idx.report()['raw_same_turn_actor_minus_sampled']['invalid_paired_occurrences']==1
assert r['finite_numeric_evidence']['matched_tokens_with_finite_raw_sampled_and_final_old']==0
json.dumps(r,allow_nan=False)
checks.append('nonfinite_raw_probabilities_counted_and_not_used_in_final_numeric_comparison')

final=snap([1,2,3],[2,3],weights=[.25,-1.])
r=A.analyze_targets(index,[('negative-effective-weight',final)])
assert r['tree_weights']['invalid_effective_weights']==1
assert r['final_leaf_old_minus_actual_raw_sampled']['matched_weight_mean_signed_bounds'] is None
json.dumps(r,allow_nan=False)
checks.append('negative_effective_weights_invalidate_all_weighted_totals')

empty=snap([1,2,3],[2,3],loss=[0,0],weights=[float('nan')]*2)
r=A.analyze_targets(index,[('no-supervised-row',empty)])
assert r['leaf_rows_denominator']==1 and r['nonempty_supervised_leaf_rows']==0
assert r['effective_supervised_token_occurrences']==0 and r['tree_weights']['invalid_effective_weights']==0
checks.append('empty_leaf_row_keeps_original_row_denominator_without_masked_nan')

# The nonempty row is one half of the global equal-leaf denominator, even
# though the other row has no supervision at all.
final=snap([1,2,3],[2,3],old=[-3.,-3.])
r=A.analyze_targets(index,[('nonempty',final),('empty',empty)])
assert r['final_leaf_old_minus_actual_raw_sampled']['original_full_leaf_row_mean_signed_bounds']==[-.5,-.5]
checks.append('global_leaf_row_denominator_keeps_empty_rows')

extreme=snap([1,2],[2],raw=True,old=[-1.7e308],sampled=[-1.7e308])
idx=A.RawIndex();idx.add_snapshot(extreme,'negative-finite-extreme')
r=A.analyze_targets(idx,[('opposite-finite-extreme',snap([1,2],[2],old=[1.7e308]))])
assert r['finite_numeric_evidence']['matched_tokens_with_finite_raw_sampled_and_final_old']==1
try:json.dumps(r,allow_nan=False)
except ValueError:
    defects.append(dict(id='finite_subtraction_overflow',severity='diagnostic_robustness',
        detail='Two finite final/raw values can subtract to infinity; helper increments finite comparison count and emits infinite range/numerator, so final strict JSON serialization fails. No false successful JSON is produced.'))
else:raise AssertionError('Expected overflow counterexample')

with tempfile.TemporaryDirectory(prefix='independent-context-json-',dir=Path(__file__).parent) as temp:
    path=Path(temp)/'record.json';path.write_text('{"bound": true}')
    expected=A.file_sha(path);original=A.file_sha
    def mutate_after_hash(p):
        observed=original(p)
        if Path(p)==path:path.write_text('{"bound": false}')
        return observed
    A.file_sha=mutate_after_hash
    try:
        loaded=A.json_bound(path,expected)
        assert loaded=={'bound':False}
    finally:A.file_sha=original
    defects.append(dict(id='json_sha_read_race',severity='evidence_binding_blocker',
        detail='JSON can change after file_sha closes its descriptor and before read_bytes opens another descriptor. json_bound accepts bytes whose SHA differs from the supplied pin. Counterexample mutates only a temporary synthetic file.'))

# Deliberately small synthetic scale sample to estimate dictionary memory.
before_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
started=time.perf_counter();n=200_000
scale_index=A.RawIndex()
scale_index.add_snapshot(snap(range(n+1),range(1,n+1),raw=True),'synthetic-scale')
after_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
factor=1 if sys.platform=='darwin' else 1024
scale=dict(raw_unique_keys=len(scale_index.values),wall_seconds=time.perf_counter()-started,
    process_peak_rss_increase_bytes=max(0,after_rss-before_rss)*factor,
    approximate_incremental_bytes_per_key=max(0,after_rss-before_rss)*factor/n,
    limitation='Process high-water increase includes tensors, lists and per-key dictionary/index objects; extrapolation is approximate and must include actual actor/monitor tensor sizes. No actual model assets read.')

result=dict(status='INDEPENDENT_CONTEXT_REVIEW_BLOCKED',candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
    checks=checks,defects=defects,scale=scale,actual_model_assets_loaded=False,remote_calls=False,
    cuda_initialized=torch.cuda.is_initialized())
print(json.dumps(result,sort_keys=True,allow_nan=False))
