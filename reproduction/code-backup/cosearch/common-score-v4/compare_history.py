"""Retain row differences without attributing unknown old text to a model author."""
import json
from pathlib import Path
from score_saved_boundaries import read

def compare(root):
 root=Path(root);old=root.parent/'cosearch-instruct200-20261010/historical-common-review/iter5-per-question.jsonl';before={r['question_answer_sha256']:r for r in read(old)}
 after=read(root/'common-scoring-v4/common-historical-validation/5.jsonl');records=[]
 if len(after)!=512 or set(r['question_answer_sha256'] for r in after)!=set(before):raise RuntimeError('Comparison identity mismatch')
 for r in after:
  b=before[r['question_answer_sha256']];different=b['old_text_rescore_em']!=r['em']
  reason=('regenerated last MODEL action: '+r['answer_reason']+('; normalized answer alias mismatch' if r['answer_format_legal'] and not r['em'] else '; normalized answer alias matches' if r['em'] else '')+'; old mixed text author unverified, new generation may differ') if different else 'same binary result; old author still unverified'
  records.append({'question_id':b['question_id'],'question':r['question'],'references':r['references'],'old_text_rescore_em':b['old_text_rescore_em'],'new_boundary_common_em':r['em'],'new_f1':r['f1'],'new_answer':r['answer'],'new_answer_reason':r['answer_reason'],'difference':different,'reason':reason})
 folder=root/'common-scoring-v4/common-historical-validation'
 with (folder/'5.old-new-per-question.jsonl').open('w') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 diffs=[r for r in records if r['difference']];(folder/'5.disagreements.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2)+'\n')
 summary={'questions':512,'old_mixed_text_correct':sum(r['old_text_rescore_em'] for r in records),'old_mixed_text_rule':'last regex answer match on mixed output, shared normalization; author unknown','old_mixed_text_em':sum(r['old_text_rescore_em'] for r in records)/512,'new_same_checkpoint_common_correct':sum(r['new_boundary_common_em'] for r in records),'new_same_checkpoint_common_em':sum(r['new_boundary_common_em'] for r in records)/512,'disagreements':len(diffs),'old_predictions_unverified_author':512,'new_series':'online_common_boundary_score_regenerated_historical_checkpoint','old_series':'historical_text_rescore_unverified'}
 (folder/'5.comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
 import sys;print(json.dumps(compare(sys.argv[1])))
