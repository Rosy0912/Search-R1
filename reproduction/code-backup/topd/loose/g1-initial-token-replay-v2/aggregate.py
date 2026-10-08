"""Combine only a complete unique same-input call receipt set; no approval."""
import argparse,json,hashlib,pathlib,os

def combine(paths):
 rows=[json.loads(pathlib.Path(p).read_text()) for p in paths]
 if not rows:raise ValueError('No actual call receipts')
 first=rows[0];total=first['total_original_call_count'];seen=[]
 for r in rows:
  if r['status']!='CALL_REPLAY_MATCH' or r['all_calls_verified'] is not False or len(r['selected_call_indices'])!=1:raise ValueError('Not a completed single-call actual replay')
  for k in ('reader_sha256','raw_refs','source_refs','tokenizer_files','labels','native_defaults','native_versions','total_original_call_count','total_original_output_rows','original_cache_source_sha256'):
   if r[k]!=first[k]:raise ValueError('Mixed source/input/tokenizer/call identity: '+k)
  if r['mismatch_count'] or r['prompt_mismatch_count'] or r['cuda_initialized'] or r['vllm_imported']:raise ValueError('Failed or GPU call receipt')
  seen+=r['selected_call_indices']
 if sorted(seen)!=list(range(total)):raise ValueError('Missing/duplicate actual original call receipts')
 if sum(r['generated_output_rows'] for r in rows)!=first['total_original_output_rows']:raise ValueError('Incomplete original output-row coverage')
 return {'scope':'COMPLETE_ORIGINAL_CALL_REPLAY_SET_NOT_CALIBRATION_APPROVAL','status':'COMPLETE_CALL_SET_MATCHED','rows':sum(r['generated_output_rows'] for r in rows),'output_tokens':sum(r['output_tokens'] for r in rows),'call_indices':sorted(seen),'labels':first['labels'],'raw_refs':first['raw_refs'],'reader_sha256':first['reader_sha256'],'receipt_refs':{str(p):hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},'missing_B_not_accepted':any(r['missing_B_not_accepted'] for r in rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--receipts',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=combine(a.receipts);fd=os.open(a.output,os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':r['status'],'rows':r['rows'],'calls':r['call_indices']}))
if __name__=='__main__':main()
