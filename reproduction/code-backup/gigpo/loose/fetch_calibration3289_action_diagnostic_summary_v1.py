"""Fetch only bounded diagnostic metadata; complete tokens/text stay on H100."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004'
version = sys.argv[1] if len(sys.argv) > 1 else 'v1'
assert version in {'v1', 'v2'}, 'Only reviewed 3289 diagnostic versions'
TARGET = ROOT + '/logs/CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-' + version + '.json'
DEST = Path('outputs/gigpo-e1-policy-safety/CALIBRATION3289-ACTION-IDENTITY-ACTUAL-SUMMARY-' + version + '.json')
assert not DEST.exists(), 'Preserve the existing actual summary'
code = r'''import hashlib,json
from pathlib import Path
p=Path(TARGET_VALUE)
assert p.is_file() and not p.is_symlink() and p.resolve()==p
raw=p.read_bytes()
assert len(raw)<=4*1024*1024
d=json.loads(raw)
assert d['job_id']==3289 and d['scope']=='READONLY_CPU_ACTION_DIAGNOSTIC_NOT_PROBABILITY_ACCEPTANCE'
assert not d['cuda_initialized_after'] and not d['model_loaded']
assert d['forward_calls']==d['restore_calls']==d['optimizer_writes']==d['retained_updates_contributed']==0
keys=['status','scope','job_id','source_commit','binding_sha256','actual_run','torch_version',
 'cuda_initialized_before','cuda_initialized_after','model_loaded','forward_calls','restore_calls',
 'optimizer_writes','retained_updates_contributed','exact_frozen_postprocess_ast_sha256',
 'loaded_tokenizer_class','official_revision','all_originals_unchanged','originals_before','originals_after',
 'counters','length_histograms','structural_finding','turn_coverage','unique_source_steps','covered_sources',
 'bad_examples_limit','example_selection','findings_are_not_probability_acceptance',
 'probabilities_or_thresholds_selected','error_class','error']
out={k:d[k] for k in keys if k in d}
out['original_diagnostic']={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
casekeys=['source','step_id','serial','rank','local_row','flags','sampled_raw_width',
 'v9_global_padded_sample_width','actual_engine_row_width','first_pad_index',
 'raw_sampled_nonpad_count','raw_middle_pad_positions','eos_positions',
 'returned_response_width','retained_length','postprocessed_length','expected_retained_length',
 'decode_reencode_full_length','full_decode_reencode_equals_sampled_without_terminal_pad',
 'full_decode_reencode_equals_sampled_after_explicit_special_removal','raw_special_ids_removed_by_decode',
 'first_difference_sampled_retained','first_difference_replayed_retained',
 'generation_prefix_length','scoring_prefix_length','generation_prefix_token_sha256',
 'scoring_prefix_token_sha256','scoring_action_token_sha256','byte_boundary','raw_pt',
 'retained_text_equals_processed']
out['bounded_bad_example_metadata']=[{k:e[k] for k in casekeys if k in e}
 for e in d.get('bounded_bad_examples',[])]
out['full_action_text_and_token_arrays_retained_on_h100_only']=True
encoded=json.dumps(out,ensure_ascii=False,allow_nan=False)
assert len(encoded.encode())<=256*1024
print(encoded)
'''.replace('TARGET_VALUE', repr(TARGET))
script = '/usr/bin/python3 - <<\'REMOTE\'\n' + code + '\nREMOTE\n'
r = subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'], input=script,
                   text=True,capture_output=True,timeout=180)
assert r.returncode==0, r.stderr
out=json.loads(r.stdout)
encoded=(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
with DEST.open('xb') as stream:
    stream.write(encoded)
print(json.dumps({'path':str(DEST.resolve()),'bytes':len(encoded),
                  'sha256':hashlib.sha256(encoded).hexdigest(),
                  'actual':out.get('counters'),'finding':out.get('structural_finding')}))
