"""Pure CPU summaries for one common batched HF protocol, never mixed with old HF EM."""
def require(ok,message):
    if not ok:raise ValueError(message)

def aggregate(rows,questions):
    expected={q['question_id']:q for q in questions}
    require(len(expected)==len(questions)==512,'Expected the fixed512 cohort')
    found={r['question_id']:r for r in rows}
    require(len(found)==len(rows)==512 and set(found)==set(expected),'Missing/duplicate/foreign test rows')
    for qid,r in found.items():
        require(r['question']==expected[qid]['question'] and r['answers']==expected[qid]['answers'],'Question/gold changed')
        require(r['reward'] in (0,1) and r['sibling']==0,'Exactly one strict greedy response per question')
    def count(values):
        hit=sum(r['reward'] for r in values)
        return dict(strict_correct=hit,denominator=len(values),em=hit/len(values),
            final_invalid=sum(not r['protocol']['protocol_valid_final_action'] for r in values),
            valid_gold_mismatch=sum(r['protocol']['failure_class']=='valid_gold_mismatch' for r in values),
            no_executed_search=sum(r['search_calls']==0 for r in values))
    datasets={d:count([r for qid,r in found.items() if expected[qid]['data_source']==d]) for d in ('nq','hotpotqa')}
    require([datasets[d]['denominator'] for d in ('nq','hotpotqa')]==[256,256],'Fixed NQ/Hotpot split changed')
    return dict(overall=count(rows),datasets=datasets)

def paired(rows,baseline):
    left={r['question_id']:r for r in baseline};right={r['question_id']:r for r in rows}
    require(len(left)==len(baseline)==len(right)==len(rows)==512 and set(left)==set(right),'Paired512 coverage differs')
    gained=[];lost=[];preserved=[];still_wrong=[]
    for qid,r in right.items():
        require(r['question']==left[qid]['question'] and r['answers']==left[qid]['answers'],'Paired question/gold differs')
        a,b=left[qid]['reward'],r['reward']
        (gained if not a and b else lost if a and not b else preserved if a and b else still_wrong).append(qid)
    lost_invalid=sum(not right[q]['protocol']['protocol_valid_final_action'] for q in lost)
    return dict(gained=gained,lost=lost,preserved_correct=preserved,still_wrong=still_wrong,
        net_hits=len(gained)-len(lost),delta_pp=(len(gained)-len(lost))*100/512,
        lost_final_invalid=lost_invalid,lost_valid_gold_mismatch=len(lost)-lost_invalid)
