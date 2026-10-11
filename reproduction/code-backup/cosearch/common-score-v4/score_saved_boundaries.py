"""Canonical EM on immutable model-boundary outputs; never mutate producer files."""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
from common_em import score,identity,summarize,VERSION

def read(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]
def canonical(rows,expected_count):
    # Native validation padding repeats existing input UIDs. Keep the first original,
    # never choose by EM, reward or answer legality; require exact known row count.
    seen={};kept=[];padding=[]
    for index,row in enumerate(rows):
        uid=row['question_uid']
        if uid in seen:
            if row['question_answer_sha256']!=seen[uid]:raise RuntimeError('UID identity conflict')
            padding.append(index);continue
        seen[uid]=row['question_answer_sha256'];kept.append(row)
    if len(kept)!=expected_count:raise RuntimeError('Incomplete original question coverage')
    if padding and (expected_count!=51713 or padding!=list(range(51713,51720))):raise RuntimeError('Unexpected padding, not native final-batch tail')
    result=[]
    for row in kept:
        actions=row.get('model_actions')
        if not isinstance(actions,list):raise RuntimeError('Missing explicit model boundaries')
        last=actions[-1] if actions else None
        if last is not None and last['author']!='main_model':raise RuntimeError('Wrong answer author')
        verified=bool(last)
        graded=score(last['text'] if last else None,row['references'],verified)
        graded['final_model_action_retained_in_training_tensor']=bool(last and last['fully_retained'])
        result.append(dict(row,producer_v1_em=row.get('em'),**graded))
    return result,len(padding)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)
def evaluation(root,folder,step,count):
    root=Path(root);source=root/folder/(str(step)+'.jsonl')
    rows,padding=canonical(read(source),count)
    data=root/'data'/('final.jsonl' if count==51713 else 'validation.jsonl')
    expected=[identity(x['question'],x['answers']) for x in read(data)]
    if collections.Counter(x['question_answer_sha256'] for x in rows)!=collections.Counter(expected):raise RuntimeError('Question/reference identity mismatch')
    summary=summarize(rows)
    if count==512 and {k:v['questions'] for k,v in summary['datasets'].items()}!={'nq':256,'hotpotqa':256}:raise RuntimeError('Not fixed NQ256 Hotpot256')
    if count==51713 and len(summary['datasets'])!=7:raise RuntimeError('Not seven final datasets')
    dest=root/'common-scoring-v4'/folder/(str(step)+'.jsonl');write_rows(dest,rows)
    summary.update(status='COMMON_BOUNDARY_EM_V4_FULL_IDENTITY_PASS',step=step,epoch=time.time(),producer_source_sha256=sha(source),per_question_sha256=sha(dest),native_padding_removed=padding,series_kind='online_common_boundary_score_historical_checkpoint' if 'historical' in folder else 'online_common_boundary_score',producer_config_path=str(root/folder/(str(step)+'.config.json')),scorer_sha256=sha(Path(__file__).with_name('common_em.py')))
    dest.with_suffix('.receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
def training(root,job,step):
    root=Path(root);folder=root/'common-training'/str(job);complete=folder/('completed-outer-%s.json'%step)
    if not complete.exists():raise RuntimeError('Not a complete outer')
    sources=sorted(folder.glob('step-%s-attempt-*.jsonl'%step));pools=[]
    for source in sources:
        # Five responses per question share UID. Training is not validation padding.
        raw=read(source)
        if len(raw)!=2560:raise RuntimeError('Training denominator must include all2560')
        rows=[]
        for row in raw:
            last=row['model_actions'][-1] if row['model_actions'] else None
            if last and last['author']!='main_model':raise RuntimeError('Wrong model author')
            result=score(last['text'] if last else None,row['references'],bool(last))
            result['final_model_action_retained_in_training_tensor']=bool(last and last['fully_retained'])
            rows.append(dict(row,producer_v1_em=row.get('em'),**result))
        dest=root/'common-scoring-v4/common-training'/str(job)/source.name;write_rows(dest,rows);pools.append(rows)
    if not pools:raise RuntimeError('Missing generated pool')
    summary={'status':'COMPLETE_OUTER_COMMON_EM_V4_PASS','job':str(job),'step':step,'scorer_version':VERSION,'selected_update_trajectories':summarize(pools[-1]),'all_actual_generated_responses':summarize(sum(pools,[])),'attempts':len(pools),'reranker':json.loads(complete.read_text())['reranker'],'producer_complete_receipt_sha256':sha(complete),'scorer_sha256':sha(Path(__file__).with_name('common_em.py'))}
    out=root/'common-scoring-v4/common-training'/str(job)/('completed-outer-%s.json'%step);out.write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--folder');p.add_argument('--step',type=int,required=True);p.add_argument('--count',type=int,default=512);p.add_argument('--job');a=p.parse_args()
    print(json.dumps(training(a.root,a.job,a.step) if a.job else evaluation(a.root,a.folder,a.step,a.count)))
