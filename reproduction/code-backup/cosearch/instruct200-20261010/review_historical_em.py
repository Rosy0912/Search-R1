"""Review historical mixed text without pretending unknown author boundaries are verified."""
import argparse,json,re
from pathlib import Path
from common_em import score,normalize,aliases,identity,VERSION

def review(root):
    root=Path(root);data=[json.loads(l) for l in (root/'data/validation.jsonl').read_text().splitlines()]
    old=[json.loads(l) for l in (root/'validation/5.jsonl').read_text().splitlines()]
    if len(old)!=512 or len(data)!=512:raise RuntimeError('Incomplete historical512')
    out=root/'historical-common-review';out.mkdir(exist_ok=True)
    legacy_correct=proxy_correct=identity_pass=0;disagreements=[]
    with (out/'iter5-per-question.jsonl').open('w') as f:
        for idx,(d,x) in enumerate(zip(data,old)):
            refs=d['answers'];question=d['question']
            # Verify row identity against the immutable input and all reference aliases.
            observed_question=x['input'].split('\nQuestion: ',1)[-1]
            if observed_question.endswith('\nassistant\n'):observed_question=observed_question[:-len('\nassistant\n')]
            observed_question=observed_question.strip()
            if observed_question!=question or sorted(aliases(x['gts']))!=sorted(aliases(refs)):raise RuntimeError('Historical question/reference identity differs at '+str(idx))
            identity_pass+=1
            found=re.findall(r'<answer>(.*?)</answer>',x['output'],re.S)
            legacy=int(bool(found) and any(normalize(found[-1])==normalize(a) for a in refs))
            proxy=score(x['output'],refs,True)
            legacy_correct+=legacy;proxy_correct+=proxy['em']
            record={'question_id':d['question_id'],'question':question,'references':refs,'question_answer_sha256':identity(question,refs),'old_text_rescore_em':legacy,'strict_terminal_text_proxy_em':proxy['em'],'strict_terminal_text_reason':proxy['answer_reason'],'online_common_em':None,'status':'NOT_FULLY_VERIFIED_MISSING_MODEL_RETRIEVAL_BOUNDARIES','scorer_version':VERSION,'old_native_reward':x['score']}
            if legacy!=proxy['em']:disagreements.append(record)
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    summary={'rows':512,'question_reference_identities_verified':identity_pass,'old_text_correct':legacy_correct,'old_text_em':legacy_correct/512,'strict_terminal_text_proxy_correct':proxy_correct,'strict_terminal_text_proxy_em':proxy_correct/512,'text_extraction_disagreements':len(disagreements),'boundary_unverified_rows':512,'online_common_em':None,'acceptance':'REQUIRES_NEW_GENERATION_FROM_THIS_HISTORICAL_CHECKPOINT_WITH_AUTHOR_BOUNDARIES','per_question_path':str(out/'iter5-per-question.jsonl')}
    (out/'iter5-disagreements.json').write_text(json.dumps(disagreements,ensure_ascii=False,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(review(a.root)))
