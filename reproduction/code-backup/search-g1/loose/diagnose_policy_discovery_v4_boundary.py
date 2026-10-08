"""Read-only native CPU scalar/hash diagnostics. Never export tokens or QA."""
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
 reasons=collections.Counter();calls=collections.Counter();cross=[];lp_bad=0;budgets=[]
 for index,r in enumerate(native):
  tokens=r['raw_generated_token_ids'];request=r['actual_request_token_ids'];probs=r['selected_logprobs'];assert len(tokens)<=500 and len(request)<=4096
  calls[str(r['rollout_ordinal'])]+=1;budgets.append((len(request),len(tokens)))
  lp_bad+=int(len(probs)!=len(tokens) or any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in probs))
  try:raw_prefix_to_first_closing_tag(tokens,decode,eos_token_id=r['eos_token_id'],pad_token_id=r['pad_token_id'],special_token_ids=tok.all_special_ids);reasons['passed_old_boundary']+=1
  except BehaviorTraceError as e:
   reasons[str(e)]+=1
   if str(e)=='closing_tag_inside_token_nonwhitespace_tail':
    eligible=tokens[:-1] if tokens and tokens[-1]==r['eos_token_id'] else tokens;full=decode(eligible);closing='</search>' if '</search>' in full else '</answer>' if '</answer>' in full else None
    for count in range(1,len(eligible)+1):
     candidate=decode(eligible[:count])
     if closing in candidate:
      end=candidate.find(closing)+len(closing);tail=candidate[end:]
      row=dict(record_index=index,ordinal=r['rollout_ordinal'],closing_tag=closing,request_token_count=len(request),raw_token_count=len(tokens),prefix_token_count=count,suffix_utf8_bytes=len(tail.encode()),suffix_character_count=len(tail),suffix_nonwhitespace_character_count=sum(not c.isspace() for c in tail),suffix_sha256=hashlib.sha256(tail.encode()).hexdigest(),candidate_has_replacement_character='\ufffd' in candidate,candidate_is_full_decoding_prefix=full.startswith(candidate),internal_special_in_kept=bool(set(eligible[:count]).intersection(tok.all_special_ids)),trailing_eos=bool(tokens and tokens[-1]==r['eos_token_id']))
      cross.append(row);break
 out['boundary_replay']=dict(reason_counts=dict(reasons),call_ordinal_counts=dict(calls),selected_lp_bad_record_count=lp_bad,max_request_token_count=max(v[0] for v in budgets),max_raw_token_count=max(v[1] for v in budgets),crossing_count=len(cross),crossing_summaries=cross,scope='Actual old helper and official tokenizer replay on native CPU; raw text/tokenIDs/LPs stay private; not GPU or healthy admission')
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
    target=PREP/('policy-discovery-v4-boundary-diagnostic-'+str(int(value['checked_unix']*1e6))+'.json')
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    summary={k:v for k,v in value.items() if k not in ('boundary_replay','optimizer_evidence_census')}
    if 'boundary_replay' in value:
        replay=value['boundary_replay'];summary['boundary_replay']={k:v for k,v in replay.items() if k!='crossing_summaries'};summary['first_crossing']=replay['crossing_summaries'][:1]
    print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest(),summary=summary),indent=2))

if __name__=='__main__':main()
