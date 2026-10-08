"""CPU-only actual64 raw re-evaluation; never issues independent approval."""
import argparse,ast,hashlib,inspect,json,re,sys
from pathlib import Path

def require(x,msg):
    if not x:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for block in iter(lambda:s.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def jsonable(v):
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
    if hasattr(v,'tolist'):return jsonable(v.tolist())
    if hasattr(v,'item'):return v.item()
    return v

def decoder(repo):
    p=Path(repo)/'verl/trainer/main_ppo.py'
    node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='decode_model_response_spans')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
    return ns[node.name]

def review_rows(tensors,questions,qa,meta,tokenizer,decode,answer_fn,f1_fn,em_fn,legal_fn):
    require(len(questions)==len(qa)==64 and len({q['uid'] for q in questions})==64,'Need actual64 uniqueUID questions and QA')
    require([q['sample_position'] for q in questions]==list(range(64)),'Question order differs')
    responses=tensors['responses'];attention=tensors['attention_mask'];info=tensors['info_mask']
    require(responses.ndim==2 and responses.shape[0]==64 and attention.shape[0]==info.shape[0]==64,'Final tensor rows differ')
    width=responses.shape[-1];require(width>0 and attention.shape==info.shape and attention.shape[-1]>=width,'Final mask shape differs')
    require(all(((mask==0)|(mask==1)).all().item() for mask in (attention,info)),'Nonbinary raw final mask')
    rows=[]
    for i,q in enumerate(questions):
        content=json.dumps(q['original_row'],sort_keys=True,ensure_ascii=False,allow_nan=False)
        require(hashlib.sha256(content.encode()).hexdigest()==q['uid'],'Question contentUID differs')
        mask=attention[i,-width:].bool() & info[i,-width:].bool();spans=decode(tokenizer,responses[i],mask)
        answer=answer_fn(spans);targets=q['original_row']['reward_model']['ground_truth']['target']
        if isinstance(targets,str):targets=[targets]
        row=dict(source=q['original_row']['data_source'],f1=max((f1_fn(answer,t) for t in targets),default=0.) if answer is not None else 0.,
                 em=float(em_fn(answer,targets)) if answer is not None else 0.,answer_format=float(answer is not None),
                 format_rate=float(answer is not None and legal_fn([dict(text=s) for s in spans])),generated_tokens=int(mask.sum()),
                 search_calls=sum(len(re.findall(r'<search>.*?</search>',s,re.DOTALL)) for s in spans),
                 sample_position=i,uid=q['uid'],answer=answer,model_spans=spans,turns=meta['turns_stats'][i],
                 unfinished=meta['active_mask'][i],actual_search_calls=meta['valid_search_stats'][i])
        require(jsonable(row)==qa[i],'Actual QA differs from source/tokens/mask re-evaluation at row'+str(i));rows.append(row)
    correct=sum(int(r['em']) for r in rows)
    return dict(status='QUALITY64_RAW_RECOMPUTED_UNREVIEWED',metric='EM',greedy=True,correct=correct,denominator=64,
                required_correct=26,threshold=.4,candidate_quality_accepted=correct>=26,next_path='TOPD_TEACHER_BACKEND_VALIDATION' if correct>=26 else 'SEARCH_G1_GRPO100',
                generated_model_tokens=sum(r['generated_tokens'] for r in rows),actual_search_calls=sum(r['actual_search_calls'] for r in rows),
                mean_f1=sum(r['f1'] for r in rows)/64,scope='QUALITY_ONLY_NOT_SELECTED_TEACHER_OR_TRAINING_APPROVAL')

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--binding',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path('/dataset/shaoshitong/search-r1-rl/topd');out=Path(a.output).resolve();require(root/'logs' in out.parents and not out.exists(),'Own exclusive output required')
    ev=Path(a.evaluation).resolve();repo=Path(a.repo).resolve();require(root/'logs' in ev.parents and repo==root/'repo-terminal-fixed413','Actual own evaluation/source only')
    binding=json.loads(Path(a.binding).read_text());require(binding['code_commit']=='413a10c50f32febaafc83db8170e2b005e0e979f','Source413 required')
    for name,expected in binding['source_files'].items():require(sha(repo/name)==expected,'Source drift '+name)
    config=json.loads((ev/'effective-config.json').read_text());require(config['binding_sha256']==sha(a.binding),'Actual binding differs')
    require(config['data_sha256']==binding['data_sha256']==sha(binding['data_path'])=='af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756','Actual64 dataSHA differs')
    require(config['sampling']==dict(n=1,best_of=1,temperature=0.,top_p=1.,top_k=-1,min_p=0.,max_tokens=500,detokenize=False,ignore_eos=False) and config['seed']==42 and config['max_turns']==4 and config['force_final_answer'] is False,'Greedy protocol differs')
    sys.path.insert(0,str(repo));from search_r1.baselines.bapo import answer_from_spans,token_f1,legal_action_records
    from verl.utils.reward_score.qa_em import em_check
    for fn in (answer_from_spans,token_f1,legal_action_records,em_check):require(repo in Path(inspect.getfile(fn)).resolve().parents,'Foreign scoring source imported')
    from transformers import AutoTokenizer
    model=Path(binding['model_directory']).resolve()
    for name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','config.json'):
        require(sha(model/name)==binding['model_files'][name],'Tokenizer/source model identity differs')
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    require(hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest()==config['tokenizer_vocab_sha256'],'Actual token-ID mapping differs')
    import pandas as pd
    expected=pd.read_parquet(binding['data_path']).sample(64,random_state=42)
    require(len(expected)==64,'Original64 source rows missing')
    qs=json.loads((ev/'questions.json').read_text())
    require([q['original_row'] for q in qs]==[jsonable(row.to_dict()) for _,row in expected.iterrows()],'Actual sample42 rows/order differ')
    from safetensors.torch import load_file
    tensors=load_file(str(ev/'final.safetensors'),device='cpu');meta=json.loads((ev/'final-meta.json').read_text());qa=[json.loads(x) for x in (ev/'qa.jsonl').read_text().splitlines()]
    result=json.loads((ev/'result.json').read_text());require(result['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING' and result['final_sha256']==sha(ev/'final.safetensors') and result['qa_sha256']==sha(ev/'qa.jsonl'),'Actual completion/rawSHA missing')
    reviewed=review_rows(tensors,qs,qa,meta,tok,decoder(repo),answer_from_spans,token_f1,em_check,legal_action_records)
    require(result['quality']['correct']==reviewed['correct'] and result['quality']['denominator']==64,'Original quality tally differs')
    reviewed.update(data_sha256=binding['data_sha256'],source_commit=binding['code_commit'],binding_sha256=sha(a.binding),
                    raw_files={name:dict(path=str(ev/name),sha256=sha(ev/name)) for name in ('questions.json','final.safetensors','final-meta.json','qa.jsonl','result.json','effective-config.json')})
    with out.open('x') as stream:json.dump(reviewed,stream,indent=2,allow_nan=False)
    print(json.dumps(reviewed))
if __name__=='__main__':main()
