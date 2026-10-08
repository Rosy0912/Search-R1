"""Read-only actual byte-level decoder diagnostics; never export raw bytes."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

CODE = r'''
import collections, hashlib, json, math, os, stat, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
p=json.load(sys.stdin);b=Path(p['base']);run=json.loads((b/'logs/run-manifest.json').read_bytes())
assert run['run_id']==p['run_id'] and run['source_commit']==p['source_commit']
assert all((b/'repo'/n).resolve()==b/'repo'/n and hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest()==h for n,h in p['source_sha256'].items())
def owned_bytes(path, limit):
 assert path.resolve()==path and not path.is_symlink()
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  s=os.fstat(fd);assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_size<=limit
  content=b''
  with os.fdopen(os.dup(fd),'rb') as f:content=f.read(limit+1)
  t=os.fstat(fd);assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_dev,t.st_size,t.st_mtime_ns) and len(content)==s.st_size
  return content,dict(size_bytes=s.st_size,sha256=hashlib.sha256(content).hexdigest(),mtime_ns=s.st_mtime_ns,mode=oct(stat.S_IMODE(s.st_mode)))
 finally:os.close(fd)
out=dict(role='failed_native_boundary_scalar_diagnostic',run_id=run['run_id'],job_id=p['job_id'],source_commit=run['source_commit'],source_snapshot_kind='disk_bytes_not_loaded_module_proof',accepted=False,production_admission=False,healthy_checkpoint_accepted=False,checked_unix=time.time())
ledger=[]
for rank in range(8):
 for name in ('optimizer-rank-'+str(rank)+'.jsonl','policy-calibration-optimizer-rank-'+str(rank)+'.json'):
  path=b/'logs'/name;row=dict(path=name,exists=path.exists())
  if path.exists():
   raw,meta=owned_bytes(path,16*1024**2);row.update(meta);row['line_count']=len(raw.splitlines())
  ledger.append(row)
out['optimizer_evidence_census']=ledger
train=Path(run['train_log']);raw,meta=owned_bytes(train,32*1024**2);text=raw.decode('utf-8',errors='replace')
out['training_log']=dict(meta,optimizer_write_marker_count=text.count('SEARCH_G1_OPTIMIZER_WRITE'),outer_budget_marker_count=text.count('SEARCH_G1_OUTER_BUDGET'),closing_boundary_error_count=text.count('closing_tag_inside_token_nonwhitespace_tail'))
out['actual_updates_exists']=(b/'logs/actual-updates.jsonl').exists();out['behavior_gate_files']=[x.name for x in (b/'logs').glob('behavior-probability-outer-*.json')]
ep=b/'logs/policy-calibration-evaluation-before.json'
out['initial512_exists']=ep.exists()
if ep.exists():
 raw,meta=owned_bytes(ep,4*1024**2);ev=json.loads(raw);out['initial512']=dict(meta)
 out['initial512']['fields']=list(ev)
 for k,v in ev.items():
  if k in ('question_ids','rows','question_scores','per_question','question_results','per_question_results'):
   out['initial512'][k+'_count']=len(v);out['initial512'][k+'_sha256']=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  elif not isinstance(v,(list,dict)) or k in ('by_source','per_source','source_counts','metrics'):
   out['initial512'][k]=v
path=b/'artifacts/behavior-traces/outer-1.json';out['private_failed_trace_exists']=path.exists()
if path.exists():
 assert stat.S_IMODE(path.parent.stat().st_mode)==0o700
 raw,meta=owned_bytes(path,256*1024**2);assert meta['mode']=='0o600';trace=json.loads(raw)
 assert trace['role']=='common8_failed_partial_generation' and trace['run_id']==run['run_id'] and trace['source_sha256']==run['source_sha256'] and trace['slurm_job_id']==p['job_id'] and trace['logical_update']==1 and trace['complete_outer'] is False
 out['private_failed_trace']=dict(meta,role=trace['role'],logical_update=1,complete_outer=False,failure_reason_class=trace['failure_reason'])
 records=trace['actual_sampling_records'];native=[r for r in records if 'raw_generated_token_ids' in r];events=[r for r in records if r.get('role')=='actual_environment_event']
 assert len(native)<=5*(2560+7) and len(events)<=5
 out['private_failed_trace'].update(record_count=len(records),sampling_record_count=len(native),environment_event_count=len(events))
 sys.path.insert(0,str(b/'repo'))
 from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag,BehaviorTraceError
 from transformers import AutoTokenizer
 model=Path('/dataset/shaoshitong/Search-G1-common8/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725')
 out['tokenizer_file_sha256']={n:hashlib.sha256((model/n).read_bytes()).hexdigest() for n in ('tokenizer.json','tokenizer_config.json')}
 tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True);decode=lambda v:tok.decode(v,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 ambiguities=[]
 for index,r in enumerate(native):
  tokens=r['raw_generated_token_ids'];eligible=tokens[:-1] if tokens and tokens[-1]==r['eos_token_id'] else tokens;full=decode(eligible)
  if '\ufffd' not in full:continue
  closing='</search>' if '</search>' in full else '</answer>' if '</answer>' in full else None
  row=dict(record_index=index,ordinal=r['rollout_ordinal'],raw_token_count=len(tokens),request_token_count=len(r['actual_request_token_ids']),full_replacement_count=full.count('\ufffd'),full_replacement_positions=[i for i,c in enumerate(full) if c=='\ufffd'],closing_tag=closing,full_character_count=len(full),trailing_eos=bool(tokens and tokens[-1]==r['eos_token_id']))
  from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
  mapping={value:key for key,value in bytes_to_unicode().items()}
  pieces=[tok.convert_ids_to_tokens(value) for value in eligible]
  assert all(isinstance(value,str) and all(c in mapping for c in value) for value in pieces)
  bytepieces=[bytes(mapping[c] for c in value) for value in pieces]
  actualbytes=b''.join(bytepieces)
  try:actualbytes.decode('utf-8',errors='strict');row['raw_byte_stream_valid_utf8']=True
  except UnicodeDecodeError as error:
   row['raw_byte_stream_valid_utf8']=False;row['utf8_error_start_byte']=error.start;row['utf8_error_end_byte']=error.end;row['utf8_error_reason']=error.reason
  row['literal_replacement_utf8_byte_sequence_count']=actualbytes.count(bytes((239,191,189)))
  row['raw_byte_decode_replace_equals_tokenizer_decode']=actualbytes.decode('utf-8',errors='replace')==full
  if closing:
   boundary=full.find(closing)+len(closing);row['full_closing_end_character']=boundary
   for count in range(1,len(eligible)+1):
    candidate=decode(eligible[:count])
    if closing in candidate:
     end=candidate.find(closing)+len(closing);tail=candidate[end:]
     keptbytes=b''.join(bytepieces[:count])
     try:keptbytes.decode('utf-8',errors='strict');row['kept_byte_stream_valid_utf8']=True
     except UnicodeDecodeError as error:row['kept_byte_stream_valid_utf8']=False;row['kept_utf8_error_start_byte']=error.start;row['kept_utf8_error_end_byte']=error.end
     row['kept_byte_decode_replace_equals_candidate']=keptbytes.decode('utf-8',errors='replace')==candidate
     row['kept_byte_stream_ends_on_complete_closing_tag_or_ascii_suffix']=keptbytes.endswith(tail.encode()) if tail else keptbytes.endswith(closing.encode())
     row.update(prefix_token_count=count,candidate_character_count=len(candidate),candidate_has_replacement_character='\ufffd' in candidate,candidate_is_full_decoding_prefix=full.startswith(candidate),internal_special_in_kept=bool(set(eligible[:count]).intersection(tok.all_special_ids)),suffix_utf8_bytes=len(tail.encode()),suffix_sha256=hashlib.sha256(tail.encode()).hexdigest(),suffix_classification='none' if not tail else 'whitespace' if tail.isspace() else 'nonwhitespace',replacement_entirely_in_discarded_raw_suffix=all(i>=len(candidate) for i,c in enumerate(full) if c=='\ufffd'))
     break
  ambiguities.append(row)
 out['unicode_replay']=dict(rows=ambiguities,scope='Actual full raw decoding and first closing token-prefix only; no raw IDs/text exported; no source mutation or healthy admission')
out['finished_unix']=time.time();print(json.dumps(out,allow_nan=False))
'''

def main():
    launch=json.loads((PREP/'policy-discovery-v4-native-launch.json').read_bytes())
    birth=json.loads((PREP/'policy-discovery-v4-observation-1791249060407970.json').read_bytes())
    launch['run_id']=birth['run-manifest.json']['run_id']
    interpreter=str(Path(launch['base'])/'env/bin/python')
    result=subprocess.run(control.route('h100-8',shlex.quote(interpreter)+' -c '+shlex.quote(CODE)),input=json.dumps(launch),capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError('Read-only diagnosis failed: '+result.stderr[-2500:])
    value=json.loads(result.stdout)
    raw=json.dumps(value,indent=2,allow_nan=False).encode()
    target=PREP/('policy-discovery-v4-byte-diagnostic-'+str(int(value['checked_unix']*1e6))+'.json')
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    summary={k:v for k,v in value.items() if k not in ('boundary_replay','optimizer_evidence_census')}
    if 'boundary_replay' in value:
        replay=value['boundary_replay'];summary['boundary_replay']={k:v for k,v in replay.items() if k!='crossing_summaries'};summary['first_crossing']=replay['crossing_summaries'][:1]
    print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest(),summary=summary),indent=2))

if __name__=='__main__':main()
