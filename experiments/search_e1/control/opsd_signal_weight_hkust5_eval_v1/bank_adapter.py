"""CPU-only canonical inputs and pre-endpoint source-zero reference bank.

The build_bank body is unchanged from published analysis235e376. Actual
student/teacher probabilities are NOT calculated here: these are posthoc input
records, never training-forward or processed-generation scores.
"""
import hashlib,json,time,sys
from pathlib import Path
from metrics import require

def semantic_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def build_bank(data, records, trace_reader, protocol_checker):
    """One row per fixed question; never drop source-zero failures from EM."""
    require(len(data) == len(records) == 512 and len({q['question_id'] for q in data}) == 512,
        'Bank requires all512 fixed validation questions')
    index = {}
    for row in records:
        i = row['validation_index']
        require(type(i) is int and i in range(512) and i not in index and row['primary'] is True
            and row['version'] == 'source100' and row['completed_outer'] == 0, 'Bank cannot select another endpoint')
        q = data[i]
        require(row['question_id'] == q['question_id'] and row['row_semantic_sha256'] == semantic_sha(q),
            'Bank validation identity differs')
        trace = trace_reader(row['trace']); result = trace['native_result']
        require(trace['validation_index'] == i and trace['question_id'] == q['question_id']
            and trace['question'] == q['question'] and trace['answers'] == q['answers']
            and semantic_sha(result) == row['native_result_sha256'], 'Common-zero native trace changed')
        decoded = protocol_checker(result, q['answers'])
        require(type(decoded['protocol_em']) is int and decoded['protocol_em'] in (0, 1)
            and decoded == trace['protocol'] and decoded['protocol_em'] == row['protocol_em'],
            'Frozen authored-final scorer disagrees with common-zero result')
        success = decoded['protocol_em'] == 1
        index[i] = dict(validation_index=i, question_id=q['question_id'], dataset=q['data_source'],
            validation_row_semantic_sha256=semantic_sha(q), source_zero_em=decoded['protocol_em'],
            source_zero_trace=row['trace'], source_zero_native_result_sha256=row['native_result_sha256'],
            available=success, missing_reason=None if success else 'source_zero_not_strict_correct',
            reference_response=result['response'] if success else None,
            reference_response_sha256=hashlib.sha256(result['response'].encode()).hexdigest() if success else None,
            reference_policy_spans=result['policy_spans'] if success else None,
            reference_gold_used_for_eligibility=True)
    require(set(index) == set(range(512)), 'Bank fixed512 coverage incomplete')
    return dict(schema='opsd-signal-fixed-common-zero-bank-v1', accepted=True,
        entries=[index[i] for i in range(512)], questions=512,
        available=sum(x['available'] for x in index.values()),
        selection='This study common source100 greedy authored-final strict correct responses only',
        gold_used_for_reference_eligibility=True, reference_generation_added=0,
        all_endpoints_share_same_frozen_reference=True,
        original_wrong_to_new_correct_teacher_signals_not_covered=True,
        baseline_bank_subset_em_constant_one=True, predictive_or_causal_claim=False)


def export_endpoint(root,directory,endpoint,data,results,summary_ref,request,protocol_checker,write,proof):
    require(len(data)==len(results)==512,'Canonical replay needs full512')
    qindex={q['question_id']:i for i,q in enumerate(data)}
    byid={r['question_id']:r for r in results}
    require(len(qindex)==len(byid)==512 and set(qindex)==set(byid),'Duplicate/foreign replay question')
    version='source100' if endpoint=='common-zero' else endpoint.rsplit('-',1)[0]
    require(version in ('source100','G_POS','G_NEG','H_POS','H_NEG'),'Foreign endpoint version')
    completed=request['completed_outer'];require(completed==(0 if version=='source100' else int(endpoint.rsplit('-',1)[1])),'Endpoint index differs')
    traces=directory/'replay-traces';traces.mkdir(exist_ok=False);records=[];tracevalues={}
    for i,q in enumerate(data):
        result=byid[q['question_id']]
        require(result['question']==q['question'] and result['answers']==q['answers'],'Replay question/gold changed')
        decoded=protocol_checker(result,q['answers'])
        require(decoded==result['protocol'] and decoded['protocol_em']==result['reward'],'Actual strict endpoint score differs')
        trace=dict(validation_index=i,question_id=q['question_id'],question=q['question'],answers=q['answers'],
            native_result=result,protocol=decoded,generation_protocol='vllm-greedy-rep1p05-bf16-merged-fixed512-v1',
            raw_checkpoint=request['checkpoint'],endpoint_summary=summary_ref,
            probability_scope='No probabilities here; future fixed-HF retokenized posthoc probe, not vLLM processed score')
        trace_ref=write(traces/('%03d.json'%i),trace);tracevalues[trace_ref['path']]=trace
        records.append(dict(validation_index=i,question_id=q['question_id'],row_semantic_sha256=semantic_sha(q),
            primary=True,version=version,completed_outer=completed,protocol_em=decoded['protocol_em'],
            native_result_sha256=semantic_sha(result),trace=trace_ref))
    rows=write(directory/'replay-records.json',records)
    output=dict(schema='opsd-signal-batched-replay-input-v1',accepted=True,endpoint=endpoint,version=version,
        completed_outer=completed,questions=512,records=rows,raw_checkpoint=request['checkpoint'],
        config=request['config'],validation_identity=request['data'],endpoint_summary=summary_ref,
        generation_protocol='vllm-greedy-rep1p05-bf16-merged-fixed512-v1',
        posthoc_probabilities_not_yet_computed=True)
    inputref=write(directory/'replay-input.json',output)
    bankref=None
    if version=='source100':
        require(not any(p.is_dir() and p.name!='common-zero' for p in root.iterdir()),
            'Reference bank would be frozen after a post-training endpoint began')
        bank=build_bank(data,records,lambda ref:tracevalues[ref['path']],protocol_checker)
        bank.update(common_zero_replay_input=inputref,validation_identity=request['data'],
            protocol=output['generation_protocol'],configs=request.get('analysis_contract',{}).get('configs',{}),
            source_model_manifest=request.get('analysis_contract',{}).get('source_model_manifest'),
            runtime_manifest=request.get('analysis_contract',{}).get('runtime_manifest'),
            common_initial_adapter=request.get('analysis_contract',{}).get('common_initial_adapter'),
            generation_protocol=output['generation_protocol'],fixed_before_eight_new_endpoint_outputs=True,
            frozen_before_all_group_15_25_outputs=True,frozen_unix=time.time(),
            legacy_HF239_bank_not_reused=True,source_model='common source100 zero-LoRA',parent_torch_imported=('torch' in sys.modules),
            parent_cuda_initialized=(sys.modules['torch'].cuda.is_initialized() if 'torch' in sys.modules else False))
        bankref=write(root/'fixed-common-zero-bank.json',bank)
    return inputref,bankref
