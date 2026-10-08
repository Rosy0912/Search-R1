import json,hashlib,re,math,collections
from pathlib import Path
p=Path(__file__).parent;b=json.loads((p/'PERIODIC110_RAW.json').read_text());files=b['files']
for f in files.values():assert hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256']
reportpath=next(k for k,v in files.items() if k.endswith('.json') and '/validation-cost/' in k);r=json.loads(files[reportpath]['text']);events=[json.loads(x) for x in files[reportpath+'l']['text'].splitlines()]
assert r['status']=='VALIDATION_COST_COMPLETE' and r['global_step']==110 and r['job_id']=='3234' and r['pid']==358612 and r['validation_sequence']==2 and r['dataset_rows']==r['evaluated_rows']==512
assert r['observer_source_sha256']=='ac8f2783830d12c5db2de3d19e9f2d19b594aa82d240849b4d24220ddd482598' and r['run']=='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output'
assert files[reportpath+'l']['sha256']==r['ledger_sha256'] and [e['event'] for e in events]==list(range(1,len(events)+1))
assert all(e['pid']==358612 and e['global_step']==110 and e['validation_sequence']==2 and e['phase']=='periodic' for e in events)
assert events[0]['kind']=='validation_start' and events[0]['dataset_rows']==512 and events[-1]['kind']=='validation_completed' and events[-1]['evaluated_rows']==512
assert not any(e['kind'].endswith('_failed') for e in events)
kinds=collections.Counter(e['kind'] for e in events)
def total(kind,key):return sum(e[key] for e in events if e['kind']==kind)
counters=dict(generation_attempts=kinds['generation_start'],generation_completed=kinds['generation_completed'],generation_failed=0,submitted_rows=total('generation_start','rows'),returned_rows=total('generation_completed','rows'),submitted_prefill_token_events=total('generation_start','prefill_token_events'),returned_nonpad_model_token_events=total('generation_completed','nonpad_model_token_events'),real_sample_rows=total('real_sample_batch','rows'),active_row_events=total('gpu_padding_submission','active_rows'),submitted_padding_row_events=total('gpu_padding_submission','padding_rows'),http_attempts=kinds['http_start'],http_completed=kinds['http_completed'],http_failed=0,http_query_events=total('http_start','queries'),http_returned_query_rows=total('http_completed','query_result_rows'))
assert counters==r['counters']
for first,last in [('generation_start','generation_completed'),('http_start','http_completed')]:
 n=kinds[first];assert n==kinds[last]
 assert [e['call'] for e in events if e['kind']==first]==list(range(1,n+1)) and [e['call'] for e in events if e['kind']==last]==list(range(1,n+1))
line=next(s for s in b['step110_console_lines'] if 'val/qa_count/all:' in s);metrics={k:float(v) for k,v in re.findall(r'([\w/]+):([-+\deE.]+)',line)}
counts={ds:int(metrics['val/qa_count/'+ds]) for ds in ['nq','hotpotqa']};assert counts==dict(nq=256,hotpotqa=256) and all(metrics['val/evaluated_count/'+d]==256 for d in counts)
f1=sum(metrics['val/f1/'+d]*n for d,n in counts.items())/512;em=sum(metrics['val/em/'+d]*n for d,n in counts.items())/512
assert math.isclose(f1,metrics['val/f1/all'],abs_tol=1e-15,rel_tol=0) and em==metrics['val/em/all']
assert metrics['training/actual_outer_updates']==110 and metrics['actor/optimizer_steps']==10
assert f1>=.34242827316810346
receipt=dict(status='PASS_PERIODIC110_FIXED512_SCORE_LEDGER_ONLY',job_id='3234',code_commit='502e9fafa8e3aba722dbfcb493abc2b9d3ece326',config_sha256='e54d322ba3635beb8d930c4f8282997f1867b4dbf5a0ca727e54a569bb198f35',data_identity='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',global_step=110,parent_completed100=100,new_completed_updates=10,observer_pid=358612,observer_sequence=2,evaluated512=512,dataset_counts=counts,dataset_f1={d:metrics['val/f1/'+d] for d in counts},dataset_em={d:metrics['val/em/'+d] for d in counts},weighted_f1=metrics['val/f1/all'],weighted_recomputed_f1=f1,em_all=em,em_correct=round(em*512),answer_format_all=metrics['val/answer_format/all'],strict_format_rate_all=metrics['val/format_rate/all'],counters=counters,ledger_events=len(events),report_sha256=files[reportpath]['sha256'],ledger_sha256=r['ledger_sha256'],wall_seconds=r['wall_seconds'],trainer_cpu_seconds=r['trainer_process_cpu_seconds'],raw_proof_sha256=hashlib.sha256((p/'PERIODIC110_RAW.json').read_bytes()).hexdigest(),final300_accepted=False,fullHF110_reload_accepted=False,limits='Weighted original evaluation aggregates and complete ledger, not per-answer independent rescore, complete HF CPU reload or final51713. PAD/EOS follow actual ledger IDs. Strictformat and answer_format measure distinct rules.')
(p/'PERIODIC110_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(hashlib.sha256((p/'PERIODIC110_RECEIPT.json').read_bytes()).hexdigest())
