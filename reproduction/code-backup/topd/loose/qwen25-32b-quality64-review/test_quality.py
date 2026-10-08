from pathlib import Path
import sys,hashlib,json,copy,re
import torch,pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'topd-terminal-fixed413'))
sys.path.insert(0,str(Path(__file__).parent))
import review_quality as m
from search_r1.baselines.bapo import answer_from_spans,token_f1,legal_action_records
from verl.utils.reward_score.qa_em import em_check
DECODE=m.decoder(Path(__file__).parents[1]/'topd-terminal-fixed413')
class Tokenizer:
    def decode(self,ids):return ''.join({1:'<answer>blue</answer>',2:'<answer>red</answer>',3:'trailing invalid text'}[int(x)] for x in ids)

def fixture(correct=26,observation=False):
    qs=[];qa=[];meta=dict(turns_stats=[1]*64,active_mask=[0]*64,valid_search_stats=[0]*64)
    responses=torch.ones((64,2),dtype=torch.long);responses[:,1]=2 if observation else 1
    masks=torch.ones((64,2),dtype=torch.long)
    masks[:,1]=0  # Second token is an observation or masked padding, never model output.
    tensors=dict(responses=responses,attention_mask=torch.ones_like(masks),info_mask=masks)
    tok=Tokenizer()
    for i in range(64):
        target='blue' if i<correct else 'red'
        raw=dict(prompt=[dict(content=str(i))],data_source='fixture',reward_model=dict(ground_truth=dict(target=[target])))
        uid=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest();qs.append(dict(sample_position=i,uid=uid,original_row=raw))
        spans=DECODE(tok,responses[i],masks[i].bool());answer=answer_from_spans(spans)
        qa.append(dict(source='fixture',f1=token_f1(answer,target) if answer else 0.,em=float(em_check(answer,[target])) if answer else 0.,answer_format=float(answer is not None),format_rate=float(answer is not None and legal_action_records([dict(text=s) for s in spans])),generated_tokens=int(masks[i].sum()),search_calls=0,sample_position=i,uid=uid,answer=answer,model_spans=spans,turns=1,unfinished=0,actual_search_calls=0))
    return tensors,qs,qa,meta,tok

def run(f):return m.review_rows(*f,DECODE,answer_from_spans,token_f1,em_check,legal_action_records)

@pytest.mark.parametrize('count,accepted',[(25,False),(26,True),(64,True)])
def test_exact64_gate_from_terminal_raw_tokens(count,accepted):
    # Only actual model mask admits the single terminal blue answer.
    result=run(fixture(count));assert result['correct']==count and result['candidate_quality_accepted']==accepted
    assert result['status'].endswith('UNREVIEWED')

def test_observation_answer_cannot_change_quality():
    f=fixture(26,True);assert run(f)['correct']==26
    f[2][0]['answer']='red'
    with pytest.raises(RuntimeError,match='QA differs'):run(f)

@pytest.mark.parametrize('bad',['mask','uid','duplicate','qa_count','wrong_em','wrong_tokens','order'])
def test_actual_quality_contradictions_rejected(bad):
    f=fixture();t,q,qa,meta,tok=f
    if bad=='mask':t['info_mask'][0,0]=2
    elif bad=='uid':q[0]['uid']='wrong'
    elif bad=='duplicate':q[1]['uid']=q[0]['uid']
    elif bad=='qa_count':qa.pop()
    elif bad=='wrong_em':qa[0]['em']=0.
    elif bad=='wrong_tokens':qa[0]['generated_tokens']=99
    else:q[0]['sample_position']=63
    with pytest.raises(RuntimeError):run(f)


def test_multiple_answers_same_model_span_not_terminal_legal():
    f=fixture(64);f[0]['info_mask'][:,1]=1
    # QA still claims previous single-answer success; raw multiple action span is invalid.
    with pytest.raises(RuntimeError,match='QA differs'):run(f)


def test_valid_answer_followed_by_model_tail_not_accepted():
    f=fixture(64);f[0]['info_mask'][:,1]=1;f[0]['responses'][:,1]=3
    with pytest.raises(RuntimeError,match='QA differs'):run(f)
