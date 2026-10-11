"""Boundary evidence and metrics only; never alter rewards, filtering or PPO tensors."""
import functools,hashlib,json,os,time
from pathlib import Path
from common_em import score,identity,summarize,VERSION

def install_loop(cls):
    if getattr(cls,'_common_em_installed',False):return
    original_generate=cls._handle_generating_state
    @functools.wraps(original_generate)
    async def generating(self,data,*args,**kwargs):
        start=len(data.response_mask)
        state=await original_generate(self,data,*args,**kwargs)
        tokens=list(data.response_ids)
        action={'author':'main_model','text':self.tokenizer.decode(tokens,skip_special_tokens=True),'response_token_start':start,'response_token_end':start+len(tokens),'fully_retained':start+len(tokens)<=self.response_length}
        data.extra_fields.setdefault('common_model_actions',[]).append(action)
        # End offsets plus native response_mask distinguish model tokens and tool tokens.
        data.extra_fields['common_response_mask']=list(data.response_mask[:self.response_length])
        return state
    cls._handle_generating_state=generating
    original_run=cls.run
    @functools.wraps(original_run)
    async def run(self,sampling_params,**kwargs):
        result=await original_run(self,sampling_params,**kwargs)
        output=result[0] if isinstance(result,tuple) else result
        from question_metadata import with_question
        fixed=with_question(kwargs)
        output.extra_fields['common_response_mask']=list(output.response_mask)
        output.extra_fields.update(common_question=fixed['extra_info']['question'],common_references=fixed['reward_model']['ground_truth'],common_data_source=str(fixed.get('data_source','unknown')),common_boundary_version=VERSION)
        return result
    cls.run=run;cls._common_em_installed=True

def root():return Path(os.environ['COSEARCH_RUN_ROOT'])
def rows_from(batch):
    n=len(batch);nt=batch.non_tensor_batch
    required=('common_model_actions','common_question','common_references','common_data_source','common_boundary_version')
    if any(k not in nt for k in required):raise RuntimeError('Missing model boundary evidence')
    rows=[]
    rewards=batch.batch.get('rm_scores')
    for i in range(n):
        actions=nt['common_model_actions'][i];last=actions[-1] if actions else None
        verified=bool(last and last['author']=='main_model' and last['fully_retained'] and nt['common_boundary_version'][i]==VERSION)
        refs=nt['common_references'][i];question=nt['common_question'][i]
        row=dict(question=question,references=refs,data_source=str(nt['common_data_source'][i]),question_answer_sha256=identity(question,refs),model_actions=actions,response_mask=nt.get('common_response_mask',[None]*n)[i],**score(last['text'] if last else None,refs,verified))
        row['full_interaction_format_diagnostics']={k:str(nt[k][i]) for k in ('valid','json_correct','one_tool_call_per_assistant') if k in nt}
        row['algorithm_reward']=float(rewards[i].sum().item()) if rewards is not None else None
        rows.append(row)
    return rows

def write_rows(path,rows,append=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a' if append else 'w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def capture(result,step,validation,attempt):
    main=result if validation else result[0]
    rows=rows_from(main)
    job=os.environ['SLURM_JOB_ID']
    if validation:
        folder=os.environ['COSEARCH_COMMON_EVAL_FOLDER'];path=root()/folder/(str(step)+'.jsonl');write_rows(path,rows,True)
    else:
        if len(rows)!=2560:raise RuntimeError('Training EM must include all2560 responses')
        path=root()/'common-training'/job/('step-%s-attempt-%s.jsonl'%(step,attempt));write_rows(path,rows)
        summary=summarize(rows);summary.update(step=step,attempt=attempt,job=job,questions=len(set(x['question_answer_sha256'] for x in rows)))
        ranker=result[1];summary['reranker']={'trajectories':len(ranker),'algorithm_reward_sum':float(ranker.batch['rm_scores'].sum().item()) if len(ranker) else 0.0,'metric_kind':'native_ranking_reward_separate_from_main_em'}
        (path.with_suffix('.summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    return rows

def completed_outer(step):
    job=os.environ['SLURM_JOB_ID'];paths=sorted((root()/'common-training'/job).glob('step-%s-attempt-*.jsonl'%step))
    if not paths:raise RuntimeError('Missing complete training EM pool')
    pools=[[json.loads(l) for l in p.read_text().splitlines()] for p in paths]
    selected=summarize(pools[-1]);all_rows=sum(pools,[])
    result={'step':step,'job':job,'complete_outer':True,'all_actual_generated_responses':summarize(all_rows),'selected_update_trajectories':selected,'attempts':len(pools),'reranker':json.loads(paths[-1].with_suffix('.summary.json').read_text())['reranker']}
    (root()/'common-training'/job/('completed-outer-%s.json'%step)).write_text(json.dumps(result,indent=2)+'\n')
    return {'train-common/main_em':selected['em'],'train-common/main_correct':selected['correct'],'train-common/denominator':selected['trajectories'],'train-common/all_generated_em':result['all_actual_generated_responses']['em']}

def begin_validation(step,folder,config):
    os.environ['COSEARCH_COMMON_EVAL_FOLDER']=folder
    path=root()/folder/(str(step)+'.jsonl');path.parent.mkdir(exist_ok=True);path.write_text('')
    from omegaconf import OmegaConf
    evidence={'job':os.environ['SLURM_JOB_ID'],'checkpoint_step':step,'epoch':time.time(),'scorer_version':VERSION,'scorer_sha256':hashlib.sha256(Path(__file__).with_name('common_em.py').read_bytes()).hexdigest(),'resolved_config':OmegaConf.to_container(config,resolve=True),'reranker_retained':True,'algorithm_reward_is_not_em':True}
    for name in ['inputs-verified-at-launch.json','deployment-manifest.json','full-retriever-query.json','tools/tool.yaml','data/manifest.json']:
        p=root()/name
        if p.exists():evidence[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'content':p.read_text() if p.stat().st_size<250000 else None}
    evidence['source_checkpoint_kind']='historical_verified_checkpoint5' if folder=='common-historical-validation' else 'current_complete_outer'
    ck=root()/'checkpoints'/('global_step_'+str(step))/'full-sha-receipt.json'
    evidence['checkpoint_receipt_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None
    (root()/folder/(str(step)+'.config.json')).write_text(json.dumps(evidence,indent=2,default=str)+'\n')

def verify_validation(step,folder,expected_rows):
    from collections import Counter
    rows=[json.loads(l) for l in (root()/folder/(str(step)+'.jsonl')).read_text().splitlines()]
    data=root()/'data'/('final.jsonl' if expected_rows==51713 else 'validation.jsonl')
    expected=[]
    for l in data.read_text().splitlines():
        d=json.loads(l)
        expected.append(identity(d['question'],d['answers']))
    if len(rows)!=expected_rows or Counter(x['question_answer_sha256'] for x in rows)!=Counter(expected):raise RuntimeError('Common evaluation question/answer identity mismatch')
    if not all(x.get('model_actions') is not None for x in rows):raise RuntimeError('Missing model author boundaries')
    receipt=summarize(rows);receipt.update(step=step,job=os.environ['SLURM_JOB_ID'],status='ONLINE_COMMON_SCORER_FULL_IDENTITIES_PASS',path=str(root()/folder/(str(step)+'.jsonl')),per_question_sha256=hashlib.sha256((root()/folder/(str(step)+'.jsonl')).read_bytes()).hexdigest(),question_answer_manifest_sha256=hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest())
    if expected_rows==512:
        counts=sorted(x['questions'] for x in receipt['datasets'].values())
        if counts!=[256,256]:raise RuntimeError('Validation must be NQ256 plus Hotpot256')
    elif len(receipt['datasets'])!=7:raise RuntimeError('Final evaluation must cover seven datasets')
    (root()/folder/(str(step)+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
