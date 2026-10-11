"""Compare old text heuristics with new same-checkpoint online common scoring."""
import json
from pathlib import Path

def compare(root):
    root=Path(root)
    old=root.parent/'cosearch-instruct200-20261010/historical-common-review/iter5-per-question.jsonl'
    before={x['question_answer_sha256']:x for x in map(json.loads,old.read_text().splitlines())}
    after=[json.loads(l) for l in (root/'common-historical-validation/5.jsonl').read_text().splitlines()]
    if len(before)!=512 or len(after)!=512 or {x['question_answer_sha256'] for x in after}!=set(before):raise RuntimeError('Historical comparison identities differ')
    records=[]
    for x in after:
        b=before[x['question_answer_sha256']]
        records.append({'question_id':b['question_id'],'question':x['question'],'references':x['references'],'old_text_rescore_em':b['old_text_rescore_em'],'old_terminal_text_proxy_em':b['strict_terminal_text_proxy_em'],'new_same_checkpoint_online_em':x['em'],'new_f1':x['f1'],'new_answer':x['answer'],'new_answer_reason':x['answer_reason'],'old_author_boundaries_verified':False,'difference':b['old_text_rescore_em']!=x['em'],'reason':('old mixed text extraction disagrees with terminal text legality; old author unverifiable' if b['old_text_rescore_em']!=b['strict_terminal_text_proxy_em'] else 'new generation or terminal answer differs; old author unverifiable') if b['old_text_rescore_em']!=x['em'] else 'same binary result; old author remains unverifiable'})
    folder=root/'common-historical-validation'
    with (folder/'5.old-new-per-question.jsonl').open('w') as f:
        for x in records:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    differences=[x for x in records if x['difference']]
    (folder/'5.disagreements.json').write_text(json.dumps(differences,ensure_ascii=False,indent=2)+'\n')
    summary={'checkpoint_step':5,'checkpoint_kind':'historical','questions':512,'old_text_correct':sum(x['old_text_rescore_em'] for x in records),'new_same_checkpoint_online_correct':sum(x['new_same_checkpoint_online_em'] for x in records),'differences':len(differences),'old_text_series':'historical_text_rescore_unverified','new_series':'online_common_scoring_regenerated_same_checkpoint','old_text_is_not_author_verified':True}
    (folder/'5.comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
