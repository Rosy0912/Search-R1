"""Adapt completed batched endpoints to the existing predeclared paired analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROTOCOL = 'hf-sdpa-greedy-rep1p05-unmerged-fp32-lora-batch32-fixed512-v1'
ENDPOINTS = ['common-zero'] + [v+'-'+str(t) for v in ('G_POS','G_NEG','H_POS','H_NEG') for t in (15,25)]

def require(ok, message):
    if not ok: raise ValueError(message)

def read(ref):
    p=Path(ref['path']); b=p.read_bytes()
    require(hashlib.sha256(b).hexdigest()==ref['sha256'], 'Input SHA differs: '+str(p))
    return json.loads(b)

def convert(endpoint_rows, questions):
    require(set(endpoint_rows)==set(ENDPOINTS), 'All nine endpoints required')
    require(len(questions)==512 and len({q['question_id'] for q in questions})==512, 'Fixed512 required')
    records=[]
    for endpoint in ENDPOINTS:
        rr=endpoint_rows[endpoint]; index={r['question_id']:r for r in rr}
        require(len(rr)==len(index)==512 and set(index)=={q['question_id'] for q in questions}, 'Endpoint coverage differs')
        version,t=('source100',0) if endpoint=='common-zero' else (endpoint.rsplit('-',1)[0],int(endpoint.rsplit('-',1)[1]))
        for i,q in enumerate(questions):
            r=index[q['question_id']]; protocol=r['protocol']
            require(r['question']==q['question'] and r['answers']==q['answers'] and r['sibling']==0, 'Question/gold/greedy identity differs')
            require(type(r['reward']) is int and r['reward'] in (0,1), 'Strict binary EM required')
            expected=None if r['reward'] else 'final_invalid' if not protocol['protocol_valid_final_action'] else 'valid_gold_mismatch'
            require(protocol['failure_class']==expected and (not r['reward'] or protocol['protocol_valid_final_action']), 'Strict protocol classes differ')
            semantic=hashlib.sha256(json.dumps(q,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            records.append(dict(version=version,completed_outer=t,validation_index=i,question_id=q['question_id'],
                dataset=q['data_source'],row_semantic_sha256=semantic,primary=True,protocol_em=r['reward'],
                failure_class=expected,protocol_valid_final_action=protocol['protocol_valid_final_action'],
                search_calls=r['search_calls'],termination=r['termination']))
    return records

def main():
    p=argparse.ArgumentParser()
    for name in ('evaluation-complete','evaluation-complete-sha256','validation','validation-sha256','training-helper','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010').resolve()
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'HKUST4 CPU-only execution required')
    require(Path(a.output).resolve().is_relative_to(root), 'Large analysis remains on HKUST4')
    cref=dict(path=a.evaluation_complete,sha256=a.evaluation_complete_sha256);complete=read(cref)
    require(complete['accepted'] and complete['protocol']==PROTOCOL and complete['formal_endpoint_tests']==8
        and complete['common_baseline_evaluations']==1, 'Eight complete same-protocol tests required')
    validation=dict(path=a.validation,sha256=a.validation_sha256);b=Path(a.validation).read_bytes()
    require(hashlib.sha256(b).hexdigest()==a.validation_sha256, 'Validation SHA differs')
    questions=[json.loads(line) for line in b.splitlines() if line.strip()]
    summaries={s['endpoint']:s for s in complete['summaries']}
    require(len(complete['summaries'])==len(summaries)==9 and set(summaries)==set(ENDPOINTS), 'Missing/duplicate endpoint summaries')
    inputs=[];endpoint_rows={};directory=Path(a.evaluation_complete).parent
    for endpoint in ENDPOINTS:
        s=summaries[endpoint]
        require(s['accepted'] and s['protocol']==PROTOCOL and s['actual_worker_wait_returncodes']==[0]*8, 'Endpoint workers did not all exit0')
        rr=[]
        for rank in range(8):
            child=json.loads((directory/endpoint/('worker-%d-complete.json'%rank)).read_text())
            require(child['accepted'] and child['rank']==rank and all(child['request']['data'][k]==v for k,v in validation.items())
                and child['request']['checkpoint']==s['raw_checkpoint'], 'Shard model/data differs')
            ref=child['results'];bb=Path(ref['path']).read_bytes()
            require(hashlib.sha256(bb).hexdigest()==ref['sha256'], 'Shard SHA differs')
            shard=[json.loads(line) for line in bb.splitlines() if line.strip()]
            require(len(shard)==64, 'Incomplete64 shard');rr.extend(shard);inputs.append(ref)
        endpoint_rows[endpoint]=rr
        require(sum(r['reward'] for r in rr)==s['metrics']['overall']['strict_correct'], 'Saved EM does not reconstruct')
    # Reuse the unchanged exact McNemar, two-test Holm family and shared stratified bootstrap.
    require(hashlib.sha256((Path(a.training_helper)/'result_analysis.py').read_bytes()).hexdigest()
        == 'aac10c886b5a1ba4326b9ffe717bc8470bb72f6ea38a74dc352325d06ab0b84f', 'Original paired statistics changed')
    sys.path.insert(0,a.training_helper)
    from result_analysis import analyze
    result=analyze(convert(endpoint_rows,questions),repetitions=10000,seed=42)
    result.update(protocol=PROTOCOL,evaluation_complete=cref,validation=validation,shard_inputs=inputs,
        historical_HF239_not_used=True,train_relationships_reported_separately=True)
    with Path(a.output).open('x') as f: json.dump(result,f,sort_keys=True,indent=2)
    print(json.dumps(dict(accepted=True,output=a.output,formal_predictions=4608,protocol=PROTOCOL)))

if __name__=='__main__': main()
