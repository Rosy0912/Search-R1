import pathlib,json,hashlib,gzip,re,math
p=pathlib.Path('work/tree-safety110-review/actual3268');b=json.loads((p/'RESTORED512_RAW.json').read_text());files=b['files'];reportpath=next(k for k in files if k.endswith('-1.json'));reportfile=dict(files[reportpath],path=reportpath);r=json.loads(reportfile['text']);ledgerfile=files[reportpath+'l'];events=[json.loads(x) for x in ledgerfile['text'].splitlines()]
for f in b['files'].values():assert hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256']
assert r['status']=='VALIDATION_COST_COMPLETE' and r['global_step']==110 and r['job_id']=='3268' and r['dataset_rows']==r['evaluated_rows']==512
assert r['pid']==564819 and r['validation_sequence']==1 and r['observer_source_sha256']=='5372020ada958139405b5d7ef365aeab5a01d5ff3d9d9d7dde9c1a1a55b05d35'
assert ledgerfile['sha256']==r['ledger_sha256'];assert [e['event'] for e in events]==list(range(1,len(events)+1))
assert all(e['pid']==564819 and e['global_step']==110 and e['validation_sequence']==1 and e['phase']=='restored_initial' for e in events)
assert events[0]['kind']=='validation_start' and events[0]['dataset_rows']==512 and events[-1]['kind']=='validation_completed' and events[-1]['evaluated_rows']==512
assert not any(e['kind'].endswith('_failed') for e in events)
from collections import Counter
kinds=Counter(e['kind'] for e in events)
def total(kind,key):return sum(e[key] for e in events if e['kind']==kind)
counters=dict(generation_attempts=kinds['generation_start'],generation_completed=kinds['generation_completed'],generation_failed=0,submitted_rows=total('generation_start','rows'),returned_rows=total('generation_completed','rows'),submitted_prefill_token_events=total('generation_start','prefill_token_events'),returned_nonpad_model_token_events=total('generation_completed','nonpad_model_token_events'),real_sample_rows=total('real_sample_batch','rows'),active_row_events=total('gpu_padding_submission','active_rows'),submitted_padding_row_events=total('gpu_padding_submission','padding_rows'),http_attempts=kinds['http_start'],http_completed=kinds['http_completed'],http_failed=0,http_query_events=total('http_start','queries'),http_returned_query_rows=total('http_completed','query_result_rows'))
assert counters==r['counters'],(counters,r['counters'])
for first,last in [('generation_start','generation_completed'),('http_start','http_completed')]:
 assert [e['call'] for e in events if e['kind']==first]==list(range(1,65))
 assert [e['call'] for e in events if e['kind']==last]==list(range(1,65))

line=next(x for x in b['score_lines'] if 'val/qa_count/all:' in x);metrics={k:float(v) for k,v in re.findall(r'([\w/]+):([-+\deE.]+)',line)}
counts={ds:int(metrics['val/qa_count/'+ds]) for ds in ('nq','hotpotqa')};assert counts=={'nq':256,'hotpotqa':256}
assert all(metrics['val/evaluated_count/'+ds]==counts[ds] for ds in counts)
f1=sum(metrics['val/f1/'+ds]*n for ds,n in counts.items())/512;em=sum(metrics['val/em/'+ds]*n for ds,n in counts.items())/512
assert math.isclose(f1,metrics['val/f1/all'],rel_tol=0,abs_tol=1e-15) and metrics['val/f1/all']==.49859095982142876 and em==metrics['val/em/all']==210/512
assert metrics['val/format_rate/all']==0 and metrics['val/answer_format/all']==497/512

proof=dict(status='PASS_ACTUAL_RESTORED110_FIXED512_ONLY',job_id='3268',code_commit='a632ba91840fb116aacfc0e770681a860647b295',config_sha256='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b',observer_pid=564819,observer_sequence=1,global_step=110,dataset_counts=counts,f1_all=f1,em_all=em,em_correct=210,score_equal_parent110=True,answer_format=metrics['val/answer_format/all'],strict_format=metrics['val/format_rate/all'],report_sha256=reportfile['sha256'],ledger_sha256=ledgerfile['sha256'],ledger_events=len(events),counters=counters,wall_seconds=r['wall_seconds'],trainer_process_cpu_seconds=r['trainer_process_cpu_seconds'],pad_token_id=r['pad_token_id'],eos_token_id=r['eos_token_id'],actual_new_optimizer_writes_accepted=False,limits='Aggregate score and full cost ledger only; not individual-answer independent rescore or final-leaf sampling agreement. PAD151643 differs from EOS151645 in this actual run.')
(p/'RESTORED512_RECEIPT.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
